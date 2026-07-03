"""Tests du décompte des contributeurs (STORY-015 émission + STORY-016 compteurs).

Couvre : normalisation du token, dérivation machine fleet (par-machine, pas par-slot),
upsert idempotent (named monotone), fenêtre « actifs » horaire, comptage AU SUBMIT pour les
deux cohortes (browser via X-Contributor, fleet via claimed_by), et l'exposition season.
"""

from __future__ import annotations

from fastapi.testclient import TestClient
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app
from nanozero_jobserver.storage import db
from nanozero_jobserver.storage.contributors import (
    contributor_counts,
    fleet_machine_token,
    normalize_contributor_token,
    touch_contributor,
)
from nanozero_jobserver.storage.jobs import (
    create_and_claim_job,
    submit_job_with_positions,
)
from nanozero_jobserver.storage.models import promote_model, register_model

_TOK_A = "0123456789abcdef"
_TOK_B = "fedcba9876543210"


# -----------------------------------------------------------------------------
# Unités pures (pas de DB)
# -----------------------------------------------------------------------------


def test_normalize_contributor_token() -> None:
    assert normalize_contributor_token(_TOK_A) == _TOK_A
    assert normalize_contributor_token("ABCDEF0123456789") == "abcdef0123456789"  # lower
    assert normalize_contributor_token("  0123456789ABCDEF  ") == _TOK_A  # trim+lower
    assert normalize_contributor_token(None) is None
    assert normalize_contributor_token("") is None
    assert normalize_contributor_token("xyz") is None  # hors charset
    assert normalize_contributor_token("0123456789abcde") is None  # 15 (trop court)
    assert normalize_contributor_token("0123456789abcdef0") is None  # 17 (trop long)


def test_fleet_machine_token() -> None:
    # Par MACHINE : le suffixe numérique de slot/thread est retiré (AC-9).
    assert fleet_machine_token("ovh-gpu-4") == "ovh-gpu"
    assert fleet_machine_token("bugatti-night-10") == "bugatti-night"
    assert fleet_machine_token("ovh-21") == "ovh"
    assert fleet_machine_token("runpod3090-gpu-5") == "runpod3090-gpu"
    assert fleet_machine_token("w3090-night") == "w3090-night"  # pas de suffixe → inchangé


# -----------------------------------------------------------------------------
# Upsert + compteurs (DB réelle)
# -----------------------------------------------------------------------------


def _named(token: str) -> bool:
    with db.connect() as conn:
        return conn.execute(
            "SELECT named FROM contributors WHERE token = %s", (token,)
        ).fetchone()["named"]


def test_touch_upsert_idempotent_and_named_monotone(pg: str) -> None:
    with db.connect() as conn:
        touch_contributor(conn, _TOK_A, "browser", False)
        touch_contributor(conn, _TOK_A, "browser", True)  # même token → 1 ligne, named→TRUE
        touch_contributor(conn, _TOK_A, "browser", False)  # named ne repasse PAS à FALSE

    c = contributor_counts()
    assert c["total"] == 1
    assert c["total_by_source"] == {"browser": 1, "fleet": 0}
    assert _named(_TOK_A) is True


def test_counts_total_online_and_by_source(pg: str) -> None:
    with db.connect() as conn:
        touch_contributor(conn, _TOK_A, "browser", False)
        touch_contributor(conn, _TOK_B, "fleet", False)

    c = contributor_counts()
    assert c["total"] == 2
    assert c["online"] == 2
    assert c["total_by_source"] == {"browser": 1, "fleet": 1}
    assert c["online_by_source"] == {"browser": 1, "fleet": 1}


def test_online_window_excludes_stale(pg: str) -> None:
    with db.connect() as conn:
        touch_contributor(conn, _TOK_A, "browser", False)
        touch_contributor(conn, _TOK_B, "browser", False)
        # _TOK_B inactif depuis 2 h → hors fenêtre « actifs » (1 h) mais reste dans le cumul.
        conn.execute(
            "UPDATE contributors SET last_seen = NOW() - INTERVAL '2 hours' WHERE token = %s",
            (_TOK_B,),
        )
    c = contributor_counts(window_seconds=3600)
    assert c["total"] == 2
    assert c["online"] == 1
    assert c["online_by_source"]["browser"] == 1


# -----------------------------------------------------------------------------
# Comptage AU SUBMIT — fleet (claimed_by) et browser (X-Contributor)
# -----------------------------------------------------------------------------


def test_submit_fleet_counts_by_machine(pg: str) -> None:
    # Deux slots de la MÊME machine (ovh-gpu-4 / ovh-gpu-7) → 1 SEUL contributeur fleet.
    for wid in ("ovh-gpu-4", "ovh-gpu-7"):
        job = create_and_claim_job(worker_id=wid, model_version=1, num_sims=100)
        assert submit_job_with_positions(job.id, [], source="fleet") == 0

    c = contributor_counts()
    assert c["total_by_source"] == {"browser": 0, "fleet": 1}  # ×slots dédupliqué

    # Une autre machine → +1 fleet.
    job = create_and_claim_job(worker_id="bugatti-night-3", model_version=1, num_sims=100)
    submit_job_with_positions(job.id, [], source="fleet")
    assert contributor_counts()["total_by_source"]["fleet"] == 2


def test_submit_browser_requires_token(pg: str) -> None:
    # Browser AVEC token + pseudo → compté, named=TRUE.
    job = create_and_claim_job(worker_id="browser-aaaaaa-w0", model_version=1, num_sims=100)
    submit_job_with_positions(
        job.id, [], source="browser", contributor_token=_TOK_A, contributor_named=True
    )
    # Browser SANS token (vieux client) → NON compté (pas d'identité stable).
    job2 = create_and_claim_job(worker_id="browser-bbbbbb-w0", model_version=1, num_sims=100)
    submit_job_with_positions(job2.id, [], source="browser", contributor_token=None)

    c = contributor_counts()
    assert c["total_by_source"] == {"browser": 1, "fleet": 0}
    assert _named(_TOK_A) is True


# -----------------------------------------------------------------------------
# Endpoint /stats/season — exposition des nouveaux champs
# -----------------------------------------------------------------------------


def test_season_exposes_contributor_counts(pg: str) -> None:
    cfg = ServerConfig(
        host="127.0.0.1", port=8090, api_key="", database_url=pg, flusher_enabled=False
    )
    client = TestClient(create_app(cfg))

    register_model(version=1, name="gen-001", onnx_path="/m/1.onnx", sha256_onnx="aaa")
    assert promote_model(1) is True

    # 1 contributeur fleet (par machine) + 1 browser (token).
    job = create_and_claim_job(worker_id="ovh-gpu-1", model_version=1, num_sims=100)
    submit_job_with_positions(job.id, [], source="fleet")
    job2 = create_and_claim_job(worker_id="browser-xxxxxx-w0", model_version=1, num_sims=100)
    submit_job_with_positions(job2.id, [], source="browser", contributor_token=_TOK_A)

    body = client.get("/stats/season").json()
    assert body["contributors_total"] == 2
    assert body["contributors_online"] == 2
    assert body["contributors_total_by_source"] == {"browser": 1, "fleet": 1}
    assert body["active_contributors"] == body["contributors_online"]  # alias rétro-compat
