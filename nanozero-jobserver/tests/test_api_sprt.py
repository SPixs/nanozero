"""HTTP-level tests for /sprt/record + /sprt/history."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app

AUTH = {"X-API-Key": "k"}


@pytest.fixture()
def client(tmp_path: Path) -> TestClient:
    cfg = ServerConfig(host="127.0.0.1", port=8090, api_key="k", db_path=tmp_path / "s.db")
    return TestClient(create_app(cfg))


def test_record_and_history(client: TestClient) -> None:
    r = client.post(
        "/sprt/record",
        headers=AUTH,
        json={
            "candidate_version": 31,
            "baseline_version": 29,
            "decision": "accepted",
            "elo_estimate": 52.0,
            "wins": 198,
        },
    )
    assert r.status_code == 201
    body = r.json()
    assert body["candidate_version"] == 31 and body["decision"] == "accepted"
    assert body["id"] > 0 and body["created_at"]

    hist = client.get("/sprt/history", headers=AUTH).json()
    assert len(hist) == 1 and hist[0]["candidate_version"] == 31


def test_history_filter_by_model(client: TestClient) -> None:
    client.post(
        "/sprt/record",
        headers=AUTH,
        json={"candidate_version": 31, "baseline_version": 29, "decision": "accepted"},
    )
    client.post(
        "/sprt/record",
        headers=AUTH,
        json={"candidate_version": 40, "baseline_version": 39, "decision": "accepted"},
    )
    rows = client.get("/sprt/history?model_version=29", headers=AUTH).json()
    assert len(rows) == 1 and rows[0]["baseline_version"] == 29


def test_record_invalid_decision_422(client: TestClient) -> None:
    r = client.post(
        "/sprt/record",
        headers=AUTH,
        json={"candidate_version": 1, "baseline_version": 2, "decision": "maybe"},
    )
    assert r.status_code == 422  # Literal rejette


def test_sprt_requires_auth(client: TestClient) -> None:
    assert (
        client.post(
            "/sprt/record",
            json={"candidate_version": 1, "baseline_version": 2, "decision": "accepted"},
        ).status_code
        == 401
    )
    assert client.get("/sprt/history").status_code == 401
