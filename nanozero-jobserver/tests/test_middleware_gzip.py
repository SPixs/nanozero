"""B.2 — cap anti zip-bomb du GzipRequestMiddleware.

Couvre : `_safe_gunzip` borné (unit) + intégration TestClient (413 sur zip-bomb,
pass-through légitime, non-gzip intact).
"""

from __future__ import annotations

import gzip
import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from nanozero_jobserver import middleware
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app
from nanozero_jobserver.middleware import _BodyTooLargeError, _safe_gunzip

AUTH_KEY = "k"


# --- unit : _safe_gunzip borne la SORTIE -------------------------------------


def test_safe_gunzip_within_cap() -> None:
    raw = b"x" * 500
    assert _safe_gunzip(gzip.compress(raw), max_out=10_000) == raw


def test_safe_gunzip_at_cap_boundary() -> None:
    raw = b"y" * 1000
    assert _safe_gunzip(gzip.compress(raw), max_out=1000) == raw  # exactement à la borne → OK


def test_safe_gunzip_zip_bomb_rejected() -> None:
    # ~5 Mo de zéros → gzip minuscule, dé-gzippé >> cap → rejet (sans matérialiser 5 Mo)
    bomb = gzip.compress(b"\x00" * (5 * 1024 * 1024))
    assert len(bomb) < 10_000  # le compressé est tout petit
    with pytest.raises(_BodyTooLargeError):
        _safe_gunzip(bomb, max_out=64 * 1024)


def test_safe_gunzip_not_gzip_raises_zliberror() -> None:
    with pytest.raises(zlib.error):
        _safe_gunzip(b"\xde\xad\xbe\xef not gzip", max_out=1_000_000)


# --- intégration : middleware dans l'app réelle ------------------------------


@pytest.fixture()
def app_and_db(tmp_path: Path):
    cfg = ServerConfig(host="127.0.0.1", port=8090, api_key=AUTH_KEY, db_path=tmp_path / "m.db")
    return create_app(cfg), cfg.db_path


@pytest.fixture()
def client(app_and_db) -> TestClient:
    return TestClient(app_and_db[0])


def test_gzip_bomb_request_returns_413(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(middleware, "MAX_REQUEST_BODY_BYTES", 1000)
    bomb = gzip.compress(b"\x00" * 50_000)  # dé-gzippé 50 000 > cap 1000
    resp = client.post(
        "/jobs/claim", content=bomb, headers={"Content-Encoding": "gzip", "X-Worker-Id": "z"}
    )
    assert resp.status_code == 413


def test_gzip_legit_request_passes_through(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(middleware, "MAX_REQUEST_BODY_BYTES", 1000)
    small = gzip.compress(b"x" * 100)  # dé-gzippé 100 < cap → traité par la route
    resp = client.post(
        "/jobs/claim", content=small, headers={"Content-Encoding": "gzip", "X-Worker-Id": "ok"}
    )
    assert resp.status_code != 413  # 204 (file vide) ou 200, jamais 413


def test_non_gzip_request_untouched(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(middleware, "MAX_REQUEST_BODY_BYTES", 1000)
    # Pas de Content-Encoding : le cap ne s'applique pas (pass-through), même corps > cap.
    resp = client.post("/jobs/claim", content=b"a" * 5000, headers={"X-Worker-Id": "raw"})
    assert resp.status_code != 413


def test_compressed_body_over_cap_returns_413(client: TestClient, monkeypatch) -> None:
    """Même un body COMPRESSÉ trop gros (avant dé-gzip) est rejeté au drain."""
    monkeypatch.setattr(middleware, "MAX_REQUEST_BODY_BYTES", 1000)
    # 2000 octets de données aléatoires gzippées : le compressé dépasse 1000 → 413 au drain.
    big_incompressible = gzip.compress(bytes(range(256)) * 2000)  # ~peu compressible, >1000 compressé
    assert len(big_incompressible) > 1000
    resp = client.post(
        "/jobs/claim",
        content=big_incompressible,
        headers={"Content-Encoding": "gzip", "X-Worker-Id": "big"},
    )
    assert resp.status_code == 413
