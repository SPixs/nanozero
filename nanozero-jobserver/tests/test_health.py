"""Smoke tests for the FastAPI app boot + /health endpoint (Phase 13.1)."""

from __future__ import annotations

import sqlite3

from fastapi.testclient import TestClient
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app


def _make_config(tmp_path):
    return ServerConfig(
        host="127.0.0.1",
        port=8080,
        api_key="",
        db_path=tmp_path / "test.db",
    )


def test_app_boots_and_exposes_health(tmp_path) -> None:
    app = create_app(_make_config(tmp_path))
    client = TestClient(app)

    resp = client.get("/health")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "ok"
    assert "version" in payload


def test_app_state_holds_config(tmp_path) -> None:
    """Endpoint handlers should be able to access config via app.state.config."""
    cfg = _make_config(tmp_path)
    app = create_app(cfg)

    assert app.state.config is cfg


def test_health_endpoint_has_meta_tag(tmp_path) -> None:
    """OpenAPI tagging — /health under 'meta' so it doesn't clutter business endpoints."""
    app = create_app(_make_config(tmp_path))
    schema = app.openapi()

    health_op = schema["paths"]["/health"]["get"]
    assert "meta" in health_op["tags"]


def test_health_healthy_has_empty_warnings(tmp_path) -> None:
    """#5 BMAD : DB saine → status ok + warnings vides + 200."""
    body = TestClient(create_app(_make_config(tmp_path))).get("/health").json()
    assert body["status"] == "ok"
    assert body["warnings"] == []


def test_health_warns_on_flusher_failure_without_failing_liveness(tmp_path) -> None:
    """#5 BMAD : un flusher bloqué = warning (200), PAS un 503 (≠ liveness)."""
    cfg = ServerConfig(
        host="127.0.0.1", port=8080, api_key="", db_path=tmp_path / "t.db", flusher_enabled=False
    )
    app = create_app(cfg)

    class _StubFlusher:
        consecutive_tick_failures = 3

    app.state.flusher = _StubFlusher()
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200  # capacité/flusher ≠ liveness
    body = resp.json()
    assert body["status"] == "warning"
    assert any("flusher" in w for w in body["warnings"])


def test_health_503_only_when_db_unreachable(tmp_path, monkeypatch) -> None:
    """#5 BMAD : 503 SEULEMENT si la DB ne répond pas (vraie panne de liveness)."""
    cfg = ServerConfig(
        host="127.0.0.1", port=8080, api_key="", db_path=tmp_path / "t.db", flusher_enabled=False
    )
    app = create_app(cfg)

    def _boom(*_a, **_k):
        raise sqlite3.OperationalError("db unreachable")

    monkeypatch.setattr("nanozero_jobserver.storage.db.connect", _boom)
    resp = TestClient(app).get("/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "down"
