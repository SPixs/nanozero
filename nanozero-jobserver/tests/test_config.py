"""Tests for config.py — ADR-015 phase 13.6h additions."""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

import pytest
from nanozero_jobserver.config import (
    PSEUDO_MAX_LEN,
    ServerConfig,
    load_config,
    normalize_pseudo,
)


def _clean_env(monkeypatch) -> None:
    """Strip all NANOZERO_JOBSERVER_* env vars."""
    for k in list(os.environ):
        if k.startswith("NANOZERO_JOBSERVER_"):
            monkeypatch.delenv(k, raising=False)


def test_load_config_defaults_have_flusher_settings(monkeypatch) -> None:
    _clean_env(monkeypatch)
    cfg = load_config()
    assert cfg.flusher_enabled is True
    assert cfg.flush_threshold_positions == 100_000
    assert cfg.flush_retention_window == 2  # was 5, réduit 2026-05-22 (incident DB 128 GB)
    assert cfg.flush_tick_interval_seconds == 30.0
    assert cfg.npz_output_dir is None


def test_load_config_defaults_have_browser_and_watchdog_settings(monkeypatch) -> None:
    """#4 BMAD : les défauts du constructeur FlusherService sont désormais dans la config."""
    _clean_env(monkeypatch)
    cfg = load_config()
    assert cfg.flush_browser is True
    assert cfg.browser_flush_threshold == 25_000
    assert cfg.stale_claim_timeout_seconds == 3600
    assert cfg.stale_claim_max_per_tick == 500_000


def test_load_config_reads_browser_and_watchdog_env(monkeypatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("NANOZERO_JOBSERVER_FLUSH_BROWSER", "0")
    monkeypatch.setenv("NANOZERO_JOBSERVER_BROWSER_FLUSH_THRESHOLD", "5000")
    monkeypatch.setenv("NANOZERO_JOBSERVER_STALE_CLAIM_TIMEOUT", "600")
    monkeypatch.setenv("NANOZERO_JOBSERVER_STALE_CLAIM_MAX_PER_TICK", "1000")
    cfg = load_config()
    assert cfg.flush_browser is False
    assert cfg.browser_flush_threshold == 5000
    assert cfg.stale_claim_timeout_seconds == 600
    assert cfg.stale_claim_max_per_tick == 1000


def test_resolve_npz_output_dir_default(tmp_path: Path) -> None:
    """Default = db_path.parent / 'datasets'."""
    cfg = ServerConfig(host="0.0.0.0", port=8090, api_key="", db_path=tmp_path / "j.db")
    assert cfg.resolve_npz_output_dir() == tmp_path / "datasets"


def test_resolve_npz_output_dir_override(tmp_path: Path) -> None:
    """Explicit npz_output_dir wins over default."""
    explicit = tmp_path / "custom" / "dest"
    cfg = ServerConfig(
        host="0.0.0.0",
        port=8090,
        api_key="",
        db_path=tmp_path / "j.db",
        npz_output_dir=explicit,
    )
    assert cfg.resolve_npz_output_dir() == explicit


def test_env_flusher_enabled_disable(monkeypatch) -> None:
    _clean_env(monkeypatch)
    for falsey in ("0", "false", "False", ""):
        monkeypatch.setenv("NANOZERO_JOBSERVER_FLUSHER_ENABLED", falsey)
        cfg = load_config()
        assert cfg.flusher_enabled is False


def test_env_flusher_enabled_truthy(monkeypatch) -> None:
    _clean_env(monkeypatch)
    for truthy in ("1", "true", "True", "yes"):
        monkeypatch.setenv("NANOZERO_JOBSERVER_FLUSHER_ENABLED", truthy)
        cfg = load_config()
        assert cfg.flusher_enabled is True


def test_env_threshold_retention_tick_overrides(monkeypatch) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("NANOZERO_JOBSERVER_FLUSH_THRESHOLD", "50000")
    monkeypatch.setenv("NANOZERO_JOBSERVER_FLUSH_RETENTION_WINDOW", "10")
    monkeypatch.setenv("NANOZERO_JOBSERVER_FLUSH_TICK_INTERVAL", "5.0")
    cfg = load_config()
    assert cfg.flush_threshold_positions == 50_000
    assert cfg.flush_retention_window == 10
    assert cfg.flush_tick_interval_seconds == 5.0


def test_env_npz_output_dir_override(monkeypatch, tmp_path: Path) -> None:
    _clean_env(monkeypatch)
    monkeypatch.setenv("NANOZERO_JOBSERVER_NPZ_OUTPUT_DIR", str(tmp_path / "shards"))
    cfg = load_config()
    assert cfg.npz_output_dir == tmp_path / "shards"
    assert cfg.resolve_npz_output_dir() == tmp_path / "shards"


# -----------------------------------------------------------------------------
# STORY-007 — normalize_pseudo (autorité serveur)
# -----------------------------------------------------------------------------


def test_normalize_pseudo_none_returns_none() -> None:
    """Absence de pseudo (None) → None (anonyme)."""
    assert normalize_pseudo(None) is None


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Alice", "alice"),  # lowercase
        ("  bob  ", "bob"),  # strip
        ("X_y-9", "x_y-9"),  # charset complet [a-z0-9_-]
        ("a" * 40, "a" * PSEUDO_MAX_LEN),  # troncature 24
    ],
)
def test_normalize_pseudo_valid(raw: str, expected: str) -> None:
    assert normalize_pseudo(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",  # vide
        "   ",  # blancs → vide après strip
        "bad name",  # espace interne
        "héllo",  # accent non encodable en [a-z0-9_-]
        "with!bang",  # ponctuation
        "emoji😀",  # hors charset
    ],
)
def test_normalize_pseudo_invalid_returns_none(raw: str) -> None:
    """Tout pseudo hors charset → None (jamais d'exception)."""
    assert normalize_pseudo(raw) is None


@pytest.mark.parametrize("raw", ["fuck", "FuCk", "xxnazixx", "pedobear"])
def test_normalize_pseudo_blocklist_returns_none(raw: str) -> None:
    """Un terme offensant (sous-chaîne) → None silencieux."""
    assert normalize_pseudo(raw) is None


def test_normalize_pseudo_nfc_unification() -> None:
    """NFC : un 'é' tapé en deux codepoints (e + accent combinant) est unifié avant validation.

    Le 'é' n'est pas dans le charset → None de toute façon, mais on vérifie qu'on passe bien par
    NFC (forme canonique stable) sans lever. On teste aussi un cas où NFC change la longueur.
    """
    decomposed = "é"  # 'e' + U+0301 (accent aigu combinant) == 'é'
    assert unicodedata.normalize("NFC", decomposed) == "é"
    # 'é' hors charset → None (mais aucune exception, pipeline robuste).
    assert normalize_pseudo(decomposed) is None


def test_normalize_pseudo_never_raises() -> None:
    """Robustesse AC-6 : aucune entrée ne doit faire lever normalize_pseudo."""
    for raw in [None, "", "x" * 1000, "\x00\x01", "🙂" * 50, "a/b\\c"]:
        normalize_pseudo(raw)  # ne doit jamais lever
