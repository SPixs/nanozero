"""HTTP-level tests for /models/* endpoints (Phase 13.3)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app
from nanozero_jobserver.storage.models import promote_model, register_model

AUTH = {"X-API-Key": "k"}


@pytest.fixture()
def app_and_db(tmp_path: Path):
    cfg = ServerConfig(host="127.0.0.1", port=8090, api_key="k", db_path=tmp_path / "m.db")
    return create_app(cfg), cfg.db_path


@pytest.fixture()
def client(app_and_db) -> TestClient:
    return TestClient(app_and_db[0])


# -----------------------------------------------------------------------------
# /models/current
# -----------------------------------------------------------------------------


def test_current_404_when_nothing_promoted(client: TestClient) -> None:
    resp = client.get("/models/current", headers=AUTH)
    assert resp.status_code == 404


def test_current_returns_most_recent_promotion(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    register_model(db, version=1, name="v1", onnx_path="/p1", sha256_onnx="a")
    register_model(db, version=2, name="v2", onnx_path="/p2", sha256_onnx="b", parent_version=1)
    promote_model(db, 1)
    promote_model(db, 2)
    resp = client.get("/models/current", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["version"] == 2
    assert resp.json()["name"] == "v2"


def test_current_is_public_no_key(client: TestClient) -> None:
    """B.1 : /models/current est PUBLIQUE — pas de 401 sans clé (404 si pas de champion)."""
    assert client.get("/models/current").status_code != 401


# -----------------------------------------------------------------------------
# /models/{version}/download
# -----------------------------------------------------------------------------


def test_download_404_unknown_version(client: TestClient) -> None:
    resp = client.get("/models/42/download", headers=AUTH)
    assert resp.status_code == 404


def test_download_404_when_file_missing(client: TestClient, app_and_db, tmp_path) -> None:
    """Registered version but the .onnx file isn't on disk."""
    _, db = app_and_db
    register_model(
        db, version=1, name="v1", onnx_path=str(tmp_path / "missing.onnx"), sha256_onnx="a"
    )
    resp = client.get("/models/1/download", headers=AUTH)
    assert resp.status_code == 404


def test_download_streams_file(client: TestClient, app_and_db, tmp_path) -> None:
    """Happy path : registered model file is served as bytes."""
    _, db = app_and_db
    onnx_file = tmp_path / "real.onnx"
    onnx_file.write_bytes(b"FAKE_ONNX_CONTENT")
    register_model(db, version=1, name="v1", onnx_path=str(onnx_file), sha256_onnx="a")

    resp = client.get("/models/1/download", headers=AUTH)
    assert resp.status_code == 200
    assert resp.content == b"FAKE_ONNX_CONTENT"


# -----------------------------------------------------------------------------
# /models (list)
# -----------------------------------------------------------------------------


def test_list_returns_descending(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    register_model(db, version=1, name="v1", onnx_path="/p", sha256_onnx="a")
    register_model(db, version=2, name="v2", onnx_path="/p", sha256_onnx="b", parent_version=1)
    register_model(db, version=3, name="v3", onnx_path="/p", sha256_onnx="c", parent_version=2)
    resp = client.get("/models", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert [m["version"] for m in body] == [3, 2, 1]


# -----------------------------------------------------------------------------
# /models/register
# -----------------------------------------------------------------------------


def test_register_creates_201(client: TestClient) -> None:
    payload = {
        "version": 5,
        "name": "gen-005-trained",
        "onnx_path": "/foo/gen-005.onnx",
        "sha256_onnx": "abcdef" * 10,
    }
    resp = client.post("/models/register", headers=AUTH, json=payload)
    assert resp.status_code == 201
    assert resp.json()["version"] == 5
    assert resp.json()["promoted_at"] is None


def test_register_409_on_duplicate(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    register_model(db, version=1, name="v1", onnx_path="/p", sha256_onnx="a")

    payload = {"version": 1, "name": "v1-dup", "onnx_path": "/p", "sha256_onnx": "x"}
    resp = client.post("/models/register", headers=AUTH, json=payload)
    assert resp.status_code == 409


# -----------------------------------------------------------------------------
# /models/{version}/promote
# -----------------------------------------------------------------------------


def test_promote_404_unknown_version(client: TestClient) -> None:
    resp = client.post("/models/99/promote", headers=AUTH)
    assert resp.status_code == 404


def test_promote_marks_current(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    register_model(db, version=1, name="v1", onnx_path="/p", sha256_onnx="a")
    resp = client.post("/models/1/promote", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json() == {"status": "promoted", "version": 1}

    # /models/current now returns this model.
    cur = client.get("/models/current", headers=AUTH)
    assert cur.json()["version"] == 1


def test_promote_409_already_promoted(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    register_model(db, version=1, name="v1", onnx_path="/p", sha256_onnx="a")
    promote_model(db, 1)
    resp = client.post("/models/1/promote", headers=AUTH)
    assert resp.status_code == 409


def test_set_current_rollback(client: TestClient, app_and_db) -> None:
    """Rollback de promotion : set-current ramène le champion sur une version antérieure,
    LÀ où /promote refuserait (409 déjà promu) — sans SQL manuel (gen-026)."""
    _, db = app_and_db
    register_model(db, version=1, name="g1", onnx_path="/1", sha256_onnx="a")
    register_model(db, version=2, name="g2", onnx_path="/2", sha256_onnx="b")
    client.post("/models/1/promote", headers=AUTH)
    client.post("/models/2/promote", headers=AUTH)
    # set-current 1 = dernier appel → 1 redevient champion (robuste au tie de timestamp).
    resp = client.post("/models/1/set-current", headers=AUTH)
    assert resp.status_code == 200
    assert resp.json()["version"] == 1
    assert client.get("/models/current", headers=AUTH).json()["version"] == 1


def test_set_current_unknown_404(client: TestClient) -> None:
    assert client.post("/models/999/set-current", headers=AUTH).status_code == 404


def test_set_current_requires_auth(client: TestClient) -> None:
    assert client.post("/models/1/set-current").status_code == 401


def test_promote_blocked_when_sprt_rejected(client: TestClient, app_and_db) -> None:
    """Garde anti-footgun (gen-026) : un candidat SPRT 'rejected' n'est pas promu via /promote ;
    set-current force quand même."""
    _, db = app_and_db
    register_model(db, version=26, name="g26", onnx_path="/p", sha256_onnx="a")
    client.post(
        "/sprt/record",
        headers=AUTH,
        json={"candidate_version": 26, "baseline_version": 25, "decision": "rejected"},
    )
    resp = client.post("/models/26/promote", headers=AUTH)
    assert resp.status_code == 409
    assert "rejected" in resp.json()["detail"].lower()
    # set-current force malgré le rejet
    assert client.post("/models/26/set-current", headers=AUTH).status_code == 200
    assert client.get("/models/current", headers=AUTH).json()["version"] == 26
