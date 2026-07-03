"""Tests for storage/sprt.py — historique des décisions SPRT."""

from __future__ import annotations

from pathlib import Path

import pytest
from nanozero_jobserver.storage.db import init_schema
from nanozero_jobserver.storage.sprt import (
    latest_decision_for_candidate,
    list_sprt,
    record_sprt,
)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    p = tmp_path / "sprt.db"
    init_schema(p)
    return p


def test_record_and_list(db: Path) -> None:
    rid = record_sprt(
        db,
        candidate_version=31,
        baseline_version=29,
        decision="accepted",
        elo_estimate=52.0,
        games_played=677,
        wins=198,
        draws=312,
        losses=167,
        notes="continued 8x96",
    )
    assert rid > 0
    rows = list_sprt(db)
    assert len(rows) == 1
    r = rows[0]
    assert (r.candidate_version, r.baseline_version, r.decision) == (31, 29, "accepted")
    assert r.elo_estimate == 52.0 and r.wins == 198 and r.notes == "continued 8x96"
    assert r.created_at  # horodaté


def test_list_orders_most_recent_first(db: Path) -> None:
    record_sprt(db, 26, 25, "rejected")
    record_sprt(db, 31, 29, "accepted")
    assert [r.candidate_version for r in list_sprt(db)] == [31, 26]


def test_list_filter_by_model_version(db: Path) -> None:
    record_sprt(db, 31, 29, "accepted")  # 31 candidat
    record_sprt(db, 32, 31, "rejected")  # 31 baseline
    record_sprt(db, 40, 39, "accepted")  # 31 absent
    rows = list_sprt(db, model_version=31)
    assert len(rows) == 2
    assert all(31 in (r.candidate_version, r.baseline_version) for r in rows)


def test_record_invalid_decision_raises(db: Path) -> None:
    with pytest.raises(ValueError, match="decision"):
        record_sprt(db, 1, 2, "maybe")


def test_latest_decision_for_candidate(db: Path) -> None:
    assert latest_decision_for_candidate(db, 26) is None
    record_sprt(db, 26, 25, "inconclusive")
    record_sprt(db, 26, 25, "rejected")  # le plus récent
    assert latest_decision_for_candidate(db, 26) == "rejected"
