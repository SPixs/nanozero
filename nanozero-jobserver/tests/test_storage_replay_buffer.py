"""Unit tests for storage/replay_buffer.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from nanozero_jobserver.storage.db import init_schema
from nanozero_jobserver.storage.replay_buffer import (
    Position,
    count_new_positions,
    count_positions,
    count_unflushed_by_version,
    insert_positions,
    sample_positions,
)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    """Empty initialized db for each test."""
    p = tmp_path / "rb.db"
    init_schema(p)
    return p


def _make_position(game_id: str = "g0", ply: int = 0, model_version: int = 1) -> Position:
    return Position(
        game_id=game_id,
        model_version=model_version,
        ply=ply,
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        input_planes=b"\x00" * 32,
        policy_target=b"\xff" * 16,
        outcome=0.5,
    )


def test_count_unflushed_by_version_source_filter(db: Path) -> None:
    """B4-D2 : le filtre source exclut la cohorte browser du compte par version."""
    insert_positions(db, [_make_position("f", 0), _make_position("f", 1)], source="fleet")
    insert_positions(db, [_make_position("b", 0)], source="browser")
    assert count_unflushed_by_version(db) == {1: 3}  # défaut None = toutes sources
    assert count_unflushed_by_version(db, source="fleet") == {1: 2}  # browser EXCLU
    assert count_unflushed_by_version(db, source="browser") == {1: 1}


def test_insert_empty_list_returns_zero(db: Path) -> None:
    assert insert_positions(db, []) == 0


def test_insert_single_position(db: Path) -> None:
    inserted = insert_positions(db, [_make_position()])
    assert inserted == 1
    assert count_positions(db) == 1


def test_insert_bulk(db: Path) -> None:
    positions = [_make_position(game_id=f"g{i}", ply=i) for i in range(50)]
    assert insert_positions(db, positions) == 50
    assert count_positions(db) == 50


def test_count_positions_with_min_filter(db: Path) -> None:
    # 30 positions of v1, 20 of v2, 10 of v3
    insert_positions(db, [_make_position(model_version=1, ply=i) for i in range(30)])
    insert_positions(db, [_make_position(model_version=2, ply=i) for i in range(20)])
    insert_positions(db, [_make_position(model_version=3, ply=i) for i in range(10)])

    assert count_positions(db) == 60
    assert count_positions(db, min_model_version=2) == 30
    assert count_positions(db, min_model_version=3) == 10
    assert count_positions(db, min_model_version=99) == 0


def test_sample_positions_window(db: Path) -> None:
    # Versions 1..10, 100 positions each
    for v in range(1, 11):
        insert_positions(db, [_make_position(model_version=v, ply=i) for i in range(100)])

    # window=3 with current=10 → versions [8, 10] → 300 positions eligible
    sample = sample_positions(db, n=200, current_model_version=10, window=3)
    assert len(sample) == 200
    assert all(8 <= p.model_version <= 10 for p in sample)


def test_sample_positions_returns_fewer_if_buffer_small(db: Path) -> None:
    insert_positions(db, [_make_position(model_version=1, ply=i) for i in range(5)])
    sample = sample_positions(db, n=100, current_model_version=1, window=5)
    assert len(sample) == 5


def test_sample_positions_n_zero_returns_empty(db: Path) -> None:
    insert_positions(db, [_make_position()])
    assert sample_positions(db, n=0, current_model_version=1) == []


def test_sample_positions_blobs_roundtrip(db: Path) -> None:
    """Bytes columns must survive the SQLite round-trip unchanged."""
    pos = Position(
        game_id="rt",
        model_version=1,
        ply=0,
        fen="startpos",
        input_planes=bytes(range(256)),
        policy_target=b"\xde\xad\xbe\xef",
        outcome=-1.0,
    )
    insert_positions(db, [pos])
    sample = sample_positions(db, n=1, current_model_version=1)
    assert len(sample) == 1
    got = sample[0]
    assert got.input_planes == pos.input_planes
    assert got.policy_target == pos.policy_target
    assert got.outcome == pos.outcome
    assert got.fen == pos.fen


def test_count_new_positions_since(db: Path) -> None:
    """Trainer trigger : count positions strictly newer than a model version."""
    insert_positions(db, [_make_position(model_version=v, ply=0) for v in [1, 1, 2, 3, 3, 3]])
    assert count_new_positions(db, since_model_version=0) == 6
    assert count_new_positions(db, since_model_version=1) == 4  # v2 + v3*3
    assert count_new_positions(db, since_model_version=2) == 3  # v3*3
    assert count_new_positions(db, since_model_version=3) == 0
