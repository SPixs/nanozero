"""On-demand autorefill (never-empty queue, Option C).

/jobs/claim synthesises a fresh job for the current champion when the pending
queue is empty AND autorefill is enabled AND self-play isn't paused — so the
fleet never idles for lack of work and no manual re-enqueue is needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app
from nanozero_jobserver.storage.control import (
    get_autorefill,
    set_autorefill,
    set_selfplay_paused,
)
from nanozero_jobserver.storage.jobs import count_jobs_by_status
from nanozero_jobserver.storage.models import promote_model, register_model

AUTH_KEY = "test-secret-key"
HEADERS = {"X-API-Key": AUTH_KEY, "X-Worker-Id": "gpu-1"}


@pytest.fixture()
def app_and_db(tmp_path: Path):
    cfg = ServerConfig(
        host="127.0.0.1", port=8090, api_key=AUTH_KEY, db_path=tmp_path / "t.db"
    )
    return create_app(cfg), cfg.db_path


@pytest.fixture()
def client(app_and_db) -> TestClient:
    return TestClient(app_and_db[0])


def _promote(db, version: int = 28) -> None:
    register_model(
        db, version=version, name=f"gen-{version}", onnx_path="/p", sha256_onnx="h"
    )
    promote_model(db, version)


# -----------------------------------------------------------------------------
# storage config round-trip
# -----------------------------------------------------------------------------


def test_autorefill_config_default_and_roundtrip(app_and_db) -> None:
    _, db = app_and_db
    assert get_autorefill(db) == (False, 800)  # off by default, 800 sims
    set_autorefill(db, enabled=True, num_sims=400)
    assert get_autorefill(db) == (True, 400)
    set_autorefill(db, enabled=False)  # disabling keeps the configured sims
    assert get_autorefill(db) == (False, 400)


# -----------------------------------------------------------------------------
# claim behaviour
# -----------------------------------------------------------------------------


def test_claim_empty_autorefill_off_returns_204(client, app_and_db) -> None:
    _, db = app_and_db
    _promote(db)  # champion exists, but autorefill stays off
    assert client.post("/jobs/claim", headers=HEADERS).status_code == 204


def test_claim_empty_autorefill_on_generates_for_champion(client, app_and_db) -> None:
    _, db = app_and_db
    _promote(db, version=28)
    set_autorefill(db, enabled=True, num_sims=800)

    resp = client.post("/jobs/claim", headers=HEADERS)

    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == 28  # the promoted champion
    assert body["num_sims"] == 800
    assert body["job_id"]
    # the synthesised job is recorded already-claimed (no transient pending row)
    assert count_jobs_by_status(db) == {"claimed": 1}


def test_claim_autorefill_on_but_no_champion_returns_204(client, app_and_db) -> None:
    _, db = app_and_db
    set_autorefill(db, enabled=True)  # no model promoted
    assert client.post("/jobs/claim", headers=HEADERS).status_code == 204
    assert count_jobs_by_status(db) == {}  # nothing minted


def test_claim_autorefill_respects_pause(client, app_and_db) -> None:
    _, db = app_and_db
    _promote(db)
    set_autorefill(db, enabled=True)
    set_selfplay_paused(db, True)
    assert client.post("/jobs/claim", headers=HEADERS).status_code == 204
    assert count_jobs_by_status(db) == {}  # paused server never auto-generates


def test_pending_jobs_served_before_autorefill(client, app_and_db) -> None:
    """Explicitly-enqueued jobs are still drained first (autorefill only fills gaps)."""
    from nanozero_jobserver.storage.jobs import create_job

    _, db = app_and_db
    _promote(db, version=28)
    set_autorefill(db, enabled=True)
    create_job(db, model_version=99, num_sims=200)  # a pre-enqueued job for mv99

    body = client.post("/jobs/claim", headers=HEADERS).json()
    assert body["model_version"] == 99  # the pending job, not an autorefill mv28


# -----------------------------------------------------------------------------
# config endpoint + stats surface
# -----------------------------------------------------------------------------


def test_autorefill_endpoint_configures(client, app_and_db) -> None:
    _, db = app_and_db
    resp = client.post(
        "/selfplay/autorefill", headers=HEADERS, json={"enabled": True, "num_sims": 600}
    )
    assert resp.status_code == 200
    assert resp.json() == {"autorefill_enabled": True, "autorefill_sims": 600}
    assert get_autorefill(db) == (True, 600)


def test_stats_selfplay_exposes_autorefill(client, app_and_db) -> None:
    _, db = app_and_db
    _promote(db)
    set_autorefill(db, enabled=True, num_sims=800)
    body = client.get("/stats/selfplay", headers=HEADERS).json()
    assert body["autorefill_enabled"] is True
    assert body["autorefill_sims"] == 800
