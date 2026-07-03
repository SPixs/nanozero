"""B.1 — defense-in-depth : 4 routes publiques (sans clé) + tout le reste key-gated.

Prouve notamment que les routes de CONTRÔLE self-play (que DevSrv pilote avec sa clé)
restent protégées (401 sans clé) — l'opérateur garde le contrôle.
"""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app
from nanozero_jobserver.storage.jobs import create_job
from nanozero_jobserver.submit_codec import PLANES_BYTES, POLICY_LEN

_POLICY_BYTES = POLICY_LEN * 4

AUTH_KEY = "test-secret-key"


@pytest.fixture()
def app_and_db(tmp_path: Path):
    cfg = ServerConfig(host="127.0.0.1", port=8090, api_key=AUTH_KEY, db_path=tmp_path / "test.db")
    return create_app(cfg), cfg.db_path


@pytest.fixture()
def client(app_and_db) -> TestClient:
    return TestClient(app_and_db[0])


# --- les 4 routes PUBLIQUES : servies SANS clé (pas de 401) ---


def test_claim_public(client: TestClient) -> None:
    assert client.post("/jobs/claim").status_code == 204  # file vide, mais PAS 401


def test_submit_public(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = client.post("/jobs/claim", headers={"X-Worker-Id": "browser-x"}).json()["job_id"]
    body = {
        "game_id": "g",
        "model_version": 1,
        "positions": [
            {
                "ply": 0,
                "fen": "f",
                "input_planes_b64": base64.b64encode(b"\x00" * PLANES_BYTES).decode(),
                "policy_target_b64": base64.b64encode(b"\x00" * _POLICY_BYTES).decode(),
                "outcome": 0.0,
            }
        ],
    }
    assert client.post(f"/jobs/{job_id}/submit", json=body).status_code == 200  # sans clé


def test_models_current_public(client: TestClient) -> None:
    assert client.get("/models/current").status_code != 401  # 404 si pas de champion, jamais 401


def test_model_download_public(client: TestClient) -> None:
    assert client.get("/models/0/download").status_code != 401  # 404 version inconnue, jamais 401


# --- tout le reste reste PROTÉGÉ : 401 sans clé (defense-in-depth) ---


def test_selfplay_pause_still_gated(client: TestClient) -> None:
    """Contrôle DevSrv : /selfplay/pause reste key-gated (l'opérateur garde la main)."""
    assert client.post("/selfplay/pause").status_code == 401


def test_selfplay_resume_still_gated(client: TestClient) -> None:
    assert client.post("/selfplay/resume").status_code == 401


def test_selfplay_autorefill_still_gated(client: TestClient) -> None:
    assert client.post("/selfplay/autorefill", json={"enabled": True}).status_code == 401


def test_enqueue_still_gated(client: TestClient) -> None:
    assert client.post("/jobs/enqueue", json={"count": 1, "model_version": 1, "num_sims": 100}).status_code == 401


def test_cancel_pending_still_gated(client: TestClient) -> None:
    assert client.post("/jobs/cancel_pending", json={}).status_code == 401


def test_models_register_still_gated(client: TestClient) -> None:
    body = {"version": 1, "name": "g", "onnx_path": "/x", "sha256_onnx": "0" * 64}
    assert client.post("/models/register", json=body).status_code == 401


def test_models_promote_still_gated(client: TestClient) -> None:
    assert client.post("/models/1/promote").status_code == 401


def test_models_list_still_gated(client: TestClient) -> None:
    assert client.get("/models").status_code == 401


def test_stats_selfplay_still_gated(client: TestClient) -> None:
    assert client.get("/stats/selfplay").status_code == 401


def test_admin_storage_still_gated(client: TestClient) -> None:
    assert client.get("/admin/storage").status_code == 401


def test_replay_sample_still_gated(client: TestClient) -> None:
    assert client.get("/replay/sample").status_code == 401
