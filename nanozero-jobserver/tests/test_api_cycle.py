"""HTTP-level tests for GET /cycle/status (vue agrégée du cycle)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app
from nanozero_jobserver.storage.control import set_selfplay_target
from nanozero_jobserver.storage.models import promote_model, register_model
from nanozero_jobserver.storage.replay_buffer import Position, insert_positions

AUTH = {"X-API-Key": "k"}


@pytest.fixture()
def app_and_db(tmp_path: Path):
    cfg = ServerConfig(host="127.0.0.1", port=8090, api_key="k", db_path=tmp_path / "c.db")
    return create_app(cfg), cfg.db_path


@pytest.fixture()
def client(app_and_db) -> TestClient:
    return TestClient(app_and_db[0])


def _pos(mv: int, ply: int) -> Position:
    return Position(
        game_id="g",
        model_version=mv,
        ply=ply,
        fen="f",
        input_planes=b"\x00" * 8,
        policy_target=b"\x00" * 8,
        outcome=0.0,
    )


def _champion(db: Path, version: int = 5) -> None:
    register_model(db, version=version, name=f"g{version}", onnx_path="/p", sha256_onnx="a")
    promote_model(db, version)


def test_cycle_status_no_champion(client: TestClient) -> None:
    body = client.get("/cycle/status", headers=AUTH).json()
    assert body["phase"] == "no_champion"
    assert body["champion_version"] is None


def test_cycle_status_collecting_with_target(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    _champion(db, 5)
    insert_positions(db, [_pos(5, i) for i in range(10)], source="fleet")
    set_selfplay_target(db, 100)
    body = client.get("/cycle/status?ready_threshold=1000", headers=AUTH).json()
    assert body["phase"] == "collecting"
    assert body["champion_version"] == 5
    assert body["fleet_positions"] == 10
    assert body["target_positions"] == 100
    assert body["missing"] == 90
    assert body["ready_to_train"] is False


def test_cycle_status_ready_to_train(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    _champion(db, 5)
    insert_positions(db, [_pos(5, i) for i in range(10)], source="fleet")
    body = client.get("/cycle/status?ready_threshold=5", headers=AUTH).json()  # 10 >= 5
    assert body["ready_to_train"] is True
    assert body["phase"] == "ready_to_train"


def test_cycle_status_browser_excluded_from_fleet(client: TestClient, app_and_db) -> None:
    """Les positions browser ne comptent PAS dans fleet_positions (B4-D2)."""
    _, db = app_and_db
    _champion(db, 5)
    insert_positions(db, [_pos(5, i) for i in range(4)], source="fleet")
    insert_positions(db, [_pos(5, i) for i in range(9)], source="browser")
    body = client.get("/cycle/status", headers=AUTH).json()
    assert body["fleet_positions"] == 4
    assert body["browser_positions"] == 9


def test_cycle_status_paused(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    _champion(db, 5)
    client.post("/selfplay/pause", headers=AUTH)
    assert client.get("/cycle/status", headers=AUTH).json()["phase"] == "paused"


def test_cycle_status_requires_auth(client: TestClient) -> None:
    assert client.get("/cycle/status").status_code == 401
