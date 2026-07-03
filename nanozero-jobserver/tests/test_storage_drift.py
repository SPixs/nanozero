"""Tests for storage/drift.py + source_distribution — D.2 browser-cohort drift detection."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from nanozero_jobserver.storage.db import init_schema
from nanozero_jobserver.storage.drift import DriftThresholds, drift_report
from nanozero_jobserver.storage.replay_buffer import (
    Position,
    insert_positions,
    policy_resolution,
    source_distribution,
)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    p = tmp_path / "drift.db"
    init_schema(p)
    return p


def _pos(game_id: str, ply: int = 0, outcome: float = 0.0, mv: int = 30) -> Position:
    return Position(
        game_id=game_id,
        model_version=mv,
        ply=ply,
        fen="f",
        input_planes=b"\x00" * 8,
        policy_target=b"\x01" * 8,
        outcome=outcome,
    )


def _cohort(db: Path, source: str, games: int, plies: int, outcome: float) -> None:
    rows = [
        _pos(f"{source}-g{g}", ply=p, outcome=outcome) for g in range(games) for p in range(plies)
    ]
    insert_positions(db, rows, source=source)


def _balanced(db: Path, source: str, games: int = 10, plies: int = 6) -> None:
    """A balanced cohort : outcomes cycle +1/0/-1 across games."""
    rows = []
    for g in range(games):
        o = [1.0, 0.0, -1.0][g % 3]
        rows += [_pos(f"{source}-g{g}", ply=p, outcome=o) for p in range(plies)]
    insert_positions(db, rows, source=source)


# Seuils de test : min_browser_positions bas pour ne pas insérer 1000 lignes.
LOOSE = DriftThresholds(min_browser_positions=10)


# --- source_distribution -----------------------------------------------------


def test_source_distribution_features(db: Path) -> None:
    _cohort(db, "fleet", games=2, plies=5, outcome=1.0)  # 10 positions, 2 games, all wins
    d = source_distribution(db, 30, "fleet")
    assert d["n"] == 10
    assert d["n_games"] == 2
    assert d["positions_per_game"] == 5.0
    assert d["win_frac"] == 1.0
    assert d["draw_frac"] == 0.0
    assert d["loss_frac"] == 0.0
    assert d["mean_ply"] == 2.0  # plies 0..4


def test_source_distribution_absent_cohort(db: Path) -> None:
    d = source_distribution(db, 30, "browser")
    assert d["n"] == 0
    assert d["positions_per_game"] == 0.0
    assert d["outcome_mean"] == 0.0


def test_source_distribution_scopes_by_version_and_source(db: Path) -> None:
    _cohort(db, "fleet", 2, 5, 1.0)
    _cohort(db, "browser", 1, 5, 0.0)
    insert_positions(db, [_pos("other", outcome=1.0, mv=29)], source="fleet")  # autre version
    assert source_distribution(db, 30, "fleet")["n"] == 10
    assert source_distribution(db, 30, "browser")["n"] == 5
    assert source_distribution(db, 29, "fleet")["n"] == 1


# --- drift_report ------------------------------------------------------------


def test_drift_insufficient_when_browser_too_small(db: Path) -> None:
    _balanced(db, "fleet")
    _cohort(db, "browser", games=1, plies=5, outcome=0.0)  # n=5 < 10
    r = drift_report(db, 30, LOOSE)
    assert r["verdict"] == "insufficient"


def test_drift_ok_when_similar(db: Path) -> None:
    _balanced(db, "fleet")
    _balanced(db, "browser")  # même distribution exactement
    r = drift_report(db, 30, LOOSE)
    assert r["verdict"] == "ok", r["reasons"]
    assert r["deltas"]["draw_frac"] < 0.01


def test_drift_flags_all_draws_cohort(db: Path) -> None:
    _balanced(db, "fleet")  # draw_frac ≈ 0.33
    _cohort(db, "browser", games=10, plies=6, outcome=0.0)  # 100% draws
    r = drift_report(db, 30, LOOSE)
    assert r["verdict"] == "drift", r
    assert any("draw_frac" in reason for reason in r["reasons"])


def test_drift_flags_skewed_outcome(db: Path) -> None:
    _balanced(db, "fleet")  # outcome_mean ≈ 0.1
    _cohort(db, "browser", games=10, plies=6, outcome=1.0)  # tous gagnants → outcome_mean=1.0
    r = drift_report(db, 30, LOOSE)
    assert r["verdict"] == "drift", r
    assert any("outcome_mean" in reason for reason in r["reasons"])


def test_drift_warns_on_game_length_only(db: Path) -> None:
    # même distribution d'outcome, mais parties 3× plus longues côté browser → ratio ppg warn
    _balanced(db, "fleet", games=10, plies=6)
    _balanced(db, "browser", games=10, plies=20)  # ppg ratio = 20/6 ≈ 3.3 > 2.0
    r = drift_report(db, 30, LOOSE)
    assert r["verdict"] == "warn", r
    assert any("positions/partie" in reason for reason in r["reasons"])


# --- policy_resolution (sims effectifs) + règle de résolution ------------------


def _policy_blob(eff_sims: int) -> bytes:
    """Cible policy dense 4672 float32 dont la granularité normalisée encode ``eff_sims`` visites.

    Deux coups visités (eff_sims-1) et 1 fois → π=[(T-1)/T, 1/T], somme=1, min=1/T → sum/min = T.
    """
    arr = np.zeros(4672, dtype="<f4")
    arr[0] = (eff_sims - 1) / eff_sims
    arr[1] = 1.0 / eff_sims
    return arr.tobytes()


def _res_pos(game_id: str, ply: int, outcome: float, eff_sims: int, mv: int = 30) -> Position:
    return Position(
        game_id=game_id,
        model_version=mv,
        ply=ply,
        fen="f",
        input_planes=b"\x00" * 8,
        policy_target=_policy_blob(eff_sims),
        outcome=outcome,
    )


def _res_balanced(db: Path, source: str, eff_sims: int, games: int = 10, plies: int = 6) -> None:
    """Cohorte aux outcomes équilibrés (+1/0/-1) ET cible policy à ``eff_sims`` visites effectives."""
    rows = []
    for g in range(games):
        o = [1.0, 0.0, -1.0][g % 3]
        rows += [_res_pos(f"{source}-g{g}", p, o, eff_sims) for p in range(plies)]
    insert_positions(db, rows, source=source)


# Seuils de test : échantillon de résolution bas pour ne pas insérer des centaines de lignes.
RES = DriftThresholds(min_browser_positions=10, min_resolution_sample=5)


def test_policy_resolution_estimates_effective_sims(db: Path) -> None:
    _res_balanced(db, "browser", eff_sims=16)  # 60 positions, ~16 sims
    _res_balanced(db, "fleet", eff_sims=800)
    b = policy_resolution(db, 30, "browser")
    f = policy_resolution(db, 30, "fleet")
    assert b["n_sampled"] == 60
    assert abs(b["median_eff_sims"] - 16) < 1.0
    assert abs(f["median_eff_sims"] - 800) < 5.0


def test_policy_resolution_absent_cohort(db: Path) -> None:
    r = policy_resolution(db, 30, "browser")
    assert r["n_sampled"] == 0
    assert r["median_eff_sims"] == 0.0


def test_policy_resolution_ignores_nonconforming_blob(db: Path) -> None:
    _cohort(db, "browser", games=2, plies=5, outcome=0.0)  # _pos → BLOB 8 octets non conforme
    r = policy_resolution(db, 30, "browser")
    assert r["n_sampled"] == 0  # tous skippés (size != 4672), pas d'erreur


def test_drift_flags_low_sims_browser(db: Path) -> None:
    # outcomes équilibrés des DEUX côtés → seule la résolution policy peut flaguer.
    _balanced(db, "fleet")  # BLOB 8 octets → résolution fleet vide (ratio non testé)
    _res_balanced(db, "browser", eff_sims=16)  # 16 < 64 (floor drift)
    r = drift_report(db, 30, RES)
    assert r["verdict"] == "drift", r
    assert any("sims effectifs" in reason for reason in r["reasons"])
    assert abs(r["resolution"]["browser"]["median_eff_sims"] - 16) < 1.0


def test_drift_warns_on_modest_sims_browser(db: Path) -> None:
    _balanced(db, "fleet")
    _res_balanced(db, "browser", eff_sims=128)  # 64 ≤ 128 < 200 → warn
    r = drift_report(db, 30, RES)
    assert r["verdict"] == "warn", r
    assert any("sims effectifs" in reason for reason in r["reasons"])


def test_drift_resolution_can_be_disabled(db: Path) -> None:
    _balanced(db, "fleet")
    _res_balanced(db, "browser", eff_sims=16)  # flaguerait drift si la sonde tournait
    r = drift_report(db, 30, RES, include_policy_resolution=False)
    assert r["resolution"] is None
    assert r["verdict"] == "ok", r  # la règle de résolution ne s'applique pas
