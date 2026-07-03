"""Tests for storage/control.py — server_control kv flags."""

from __future__ import annotations

from pathlib import Path

import pytest
from nanozero_jobserver.storage.control import (
    get_control,
    get_selfplay_target,
    is_selfplay_paused,
    set_control,
    set_selfplay_paused,
    set_selfplay_target,
)
from nanozero_jobserver.storage.db import init_schema


def test_selfplay_target_set_get_clear(tmp_path: Path) -> None:
    from nanozero_jobserver.storage.db import init_schema as _init

    db = tmp_path / "t.db"
    _init(db)
    assert get_selfplay_target(db) == (None, "pause")  # défaut
    set_selfplay_target(db, 5_000_000, action="notify")
    assert get_selfplay_target(db) == (5_000_000, "notify")
    set_selfplay_target(db, 0)  # efface
    assert get_selfplay_target(db) == (None, "pause")
    set_selfplay_target(db, None)  # null efface aussi
    assert get_selfplay_target(db)[0] is None


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    p = tmp_path / "control.db"
    init_schema(p)
    return p


def test_get_control_missing_returns_none(db: Path) -> None:
    assert get_control(db, "nope") is None


def test_set_control_then_get(db: Path) -> None:
    set_control(db, "k", "v")
    assert get_control(db, "k") == "v"


def test_set_control_upserts(db: Path) -> None:
    set_control(db, "k", "v1")
    set_control(db, "k", "v2")
    assert get_control(db, "k") == "v2"


def test_selfplay_pause_defaults_false(db: Path) -> None:
    assert is_selfplay_paused(db) is False


def test_selfplay_pause_roundtrip(db: Path) -> None:
    set_selfplay_paused(db, True)
    assert is_selfplay_paused(db) is True
    set_selfplay_paused(db, False)
    assert is_selfplay_paused(db) is False
