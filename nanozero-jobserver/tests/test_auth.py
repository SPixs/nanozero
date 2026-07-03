"""Unit tests for API key authentication (Phase 13.1b)."""

from __future__ import annotations

from fastapi.testclient import TestClient
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app


def _make_config(tmp_path, api_key: str = ""):
    return ServerConfig(
        host="127.0.0.1",
        port=8090,
        api_key=api_key,
        db_path=tmp_path / "test.db",
    )


# -----------------------------------------------------------------------------
# Dev mode (api_key empty in config) — auth disabled
# -----------------------------------------------------------------------------


def test_whoami_dev_mode_no_header_allowed(tmp_path) -> None:
    """Dev mode (api_key=''): /whoami should accept calls without X-API-Key header."""
    app = create_app(_make_config(tmp_path, api_key=""))
    client = TestClient(app)

    resp = client.get("/whoami")

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["status"] == "authenticated"
    assert payload["auth_enabled"] is False


def test_whoami_dev_mode_any_header_allowed(tmp_path) -> None:
    """Dev mode: even a bogus X-API-Key header is allowed (auth is bypassed)."""
    app = create_app(_make_config(tmp_path, api_key=""))
    client = TestClient(app)

    resp = client.get("/whoami", headers={"X-API-Key": "literally-anything"})

    assert resp.status_code == 200


# -----------------------------------------------------------------------------
# Auth enabled (config.api_key non-empty)
# -----------------------------------------------------------------------------


def test_whoami_auth_enabled_correct_key(tmp_path) -> None:
    """Production mode: matching X-API-Key returns 200 with auth_enabled=true."""
    app = create_app(_make_config(tmp_path, api_key="secret-token-abc"))
    client = TestClient(app)

    resp = client.get("/whoami", headers={"X-API-Key": "secret-token-abc"})

    assert resp.status_code == 200
    payload = resp.json()
    assert payload["auth_enabled"] is True


def test_whoami_auth_enabled_wrong_key_rejected(tmp_path) -> None:
    """Production mode: wrong X-API-Key returns 401."""
    app = create_app(_make_config(tmp_path, api_key="secret-token-abc"))
    client = TestClient(app)

    resp = client.get("/whoami", headers={"X-API-Key": "wrong-token"})

    assert resp.status_code == 401
    assert "Invalid or missing" in resp.json()["detail"]


def test_whoami_auth_enabled_missing_header_rejected(tmp_path) -> None:
    """Production mode: absent X-API-Key returns 401."""
    app = create_app(_make_config(tmp_path, api_key="secret-token-abc"))
    client = TestClient(app)

    resp = client.get("/whoami")

    assert resp.status_code == 401


# -----------------------------------------------------------------------------
# Health remains public regardless of auth config
# -----------------------------------------------------------------------------


def test_health_unauthenticated_works_in_auth_mode(tmp_path) -> None:
    """/health must remain public — load balancers and watchdogs don't carry keys."""
    app = create_app(_make_config(tmp_path, api_key="secret-token-abc"))
    client = TestClient(app)

    resp = client.get("/health")

    assert resp.status_code == 200
