"""HTTP-level tests for /replay/sample + /training/should_train + /stats."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app
from nanozero_jobserver.storage.jobs import claim_job, create_job
from nanozero_jobserver.storage.models import promote_model, register_model
from nanozero_jobserver.storage.replay_buffer import Position, insert_positions

AUTH = {"X-API-Key": "k"}


@pytest.fixture()
def app_and_db(tmp_path: Path):
    cfg = ServerConfig(host="127.0.0.1", port=8090, api_key="k", db_path=tmp_path / "r.db")
    return create_app(cfg), cfg.db_path


@pytest.fixture()
def client(app_and_db) -> TestClient:
    return TestClient(app_and_db[0])


def _mk_position(v: int = 1, ply: int = 0) -> Position:
    return Position(
        game_id="g",
        model_version=v,
        ply=ply,
        fen="startpos",
        input_planes=b"\x01" * 16,
        policy_target=b"\x02" * 8,
        outcome=0.0,
    )


# -----------------------------------------------------------------------------
# /replay/sample
# -----------------------------------------------------------------------------


def test_sample_returns_at_most_n(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    insert_positions(db, [_mk_position(v=1, ply=i) for i in range(100)])

    resp = client.get("/replay/sample?n=50&current_version=1&window=5", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["requested"] == 50
    assert body["returned"] == 50
    assert len(body["positions"]) == 50


def test_sample_returns_fewer_if_buffer_small(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    insert_positions(db, [_mk_position(v=1, ply=i) for i in range(5)])
    resp = client.get("/replay/sample?n=100&current_version=1", headers=AUTH)
    body = resp.json()
    assert body["returned"] == 5


def test_sample_window_filters_old_versions(client: TestClient, app_and_db) -> None:
    """Window=2 with current=10 → only v9 and v10 positions returned."""
    _, db = app_and_db
    for v in range(1, 11):
        insert_positions(db, [_mk_position(v=v, ply=i) for i in range(10)])

    resp = client.get("/replay/sample?n=100&current_version=10&window=2", headers=AUTH)
    body = resp.json()
    versions = {p["model_version"] for p in body["positions"]}
    assert versions == {9, 10}


def test_sample_uses_current_model_when_no_query(client: TestClient, app_and_db) -> None:
    """When current_version is omitted, fall back to the promoted model's version."""
    _, db = app_and_db
    register_model(db, version=7, name="v7", onnx_path="/p", sha256_onnx="h")
    promote_model(db, 7)
    insert_positions(db, [_mk_position(v=5, ply=i) for i in range(3)])
    insert_positions(db, [_mk_position(v=7, ply=i) for i in range(3)])

    resp = client.get("/replay/sample?n=100&window=5", headers=AUTH)  # no current_version
    body = resp.json()
    versions = {p["model_version"] for p in body["positions"]}
    # window=5 → versions [3, 7]
    assert versions.issubset({5, 7})


def test_sample_blob_b64_roundtrip(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    insert_positions(db, [_mk_position()])
    resp = client.get("/replay/sample?n=1&current_version=1", headers=AUTH)
    pos = resp.json()["positions"][0]
    assert base64.b64decode(pos["input_planes_b64"]) == b"\x01" * 16
    assert base64.b64decode(pos["policy_target_b64"]) == b"\x02" * 8


def test_sample_requires_auth(client: TestClient) -> None:
    assert client.get("/replay/sample?n=10&current_version=1").status_code == 401


def test_sample_validates_n_range(client: TestClient) -> None:
    resp = client.get("/replay/sample?n=0&current_version=1", headers=AUTH)
    assert resp.status_code == 422  # n must be >= 1


# -----------------------------------------------------------------------------
# /training/should_train
# -----------------------------------------------------------------------------


def test_should_train_false_below_threshold(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    insert_positions(db, [_mk_position(v=2, ply=i) for i in range(5)])
    resp = client.get("/training/should_train?threshold=10&since_version=1", headers=AUTH)
    body = resp.json()
    assert body["should_train"] is False
    assert body["new_positions"] == 5
    assert body["threshold"] == 10
    assert body["missing"] == 5  # 10 - 5, plus de calcul à la main
    assert body["since_version"] == 1


def test_should_train_true_above_threshold(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    insert_positions(db, [_mk_position(v=2, ply=i) for i in range(20)])
    resp = client.get("/training/should_train?threshold=10&since_version=1", headers=AUTH)
    assert resp.json()["should_train"] is True


def test_should_train_uses_current_model_when_no_query(client: TestClient, app_and_db) -> None:
    """When since_version is omitted, fall back to current promoted version."""
    _, db = app_and_db
    register_model(db, version=3, name="v3", onnx_path="/p", sha256_onnx="h")
    promote_model(db, 3)
    insert_positions(db, [_mk_position(v=4, ply=i) for i in range(8)])

    resp = client.get("/training/should_train?threshold=5", headers=AUTH)
    body = resp.json()
    assert body["new_positions"] == 8  # positions with v > 3
    assert body["should_train"] is True


# -----------------------------------------------------------------------------
# /stats
# -----------------------------------------------------------------------------


def test_stats_empty_returns_zero_counts(client: TestClient) -> None:
    resp = client.get("/stats", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["jobs"] == {}
    assert body["positions_total"] == 0
    assert body["models_registered"] == 0
    assert body["current_model_version"] is None


def test_stats_aggregates_counts(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    # 2 pending + 1 claimed
    for _ in range(3):
        create_job(db, model_version=1, num_sims=200)
    claim_job(db, "w")
    insert_positions(db, [_mk_position(v=1, ply=i) for i in range(7)])
    register_model(db, version=1, name="v1", onnx_path="/p", sha256_onnx="a")
    promote_model(db, 1)

    resp = client.get("/stats", headers=AUTH)
    body = resp.json()
    assert body["jobs"] == {"pending": 2, "claimed": 1}
    assert body["positions_total"] == 7
    assert body["models_registered"] == 1
    assert body["current_model_version"] == 1


def test_stats_requires_auth(client: TestClient) -> None:
    assert client.get("/stats").status_code == 401


def test_durable_new_positions_excludes_browser(tmp_path: Path) -> None:
    """B4-D2 : le déclencheur d'entraînement ignore la cohorte browser, durable ET live."""
    from nanozero_jobserver.api.replay import _durable_new_positions
    from nanozero_jobserver.storage.batches import insert_batch
    from nanozero_jobserver.storage.db import init_schema

    db = tmp_path / "d.db"
    init_schema(db)
    # Durable (batches) à mv5 : fleet 100 + browser 999.
    insert_batch(db, model_version=5, batch_idx=0, npz_path="/f.npz", n_positions=100, source="fleet")
    insert_batch(db, model_version=5, batch_idx=1, npz_path="/b.npz", n_positions=999, source="browser")
    # Live (positions non flushées) à mv5 : fleet 2 + browser 1.
    insert_positions(db, [_mk_position(5, 0), _mk_position(5, 1)], source="fleet")
    insert_positions(db, [_mk_position(5, 2)], source="browser")
    # should_train ne compte QUE le fleet : 100 (durable) + 2 (live) = 102 (browser ignoré).
    assert _durable_new_positions(db, since_version=5) == 102
