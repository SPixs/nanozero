"""Tests ADR-015 phase 13.6d : flush + purge queries on positions table."""

from __future__ import annotations

from pathlib import Path

import pytest
from nanozero_jobserver.storage.batches import insert_batch
from nanozero_jobserver.storage.db import init_schema
from nanozero_jobserver.storage.replay_buffer import (
    Position,
    count_unflushed_by_version,
    count_unflushed_positions,
    delete_flushed_old,
    insert_positions,
    iter_unflushed_positions,
    list_unflushed_model_versions,
    mark_positions_flushed,
)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.db"
    init_schema(p)
    return p


def _mk_pos(game_id: str, model_version: int, ply: int = 0) -> Position:
    return Position(
        game_id=game_id,
        model_version=model_version,
        ply=ply,
        fen=f"fen-{game_id}-{ply}",
        input_planes=b"\x00" * 32,
        policy_target=b"\x01" * 32,
        outcome=0.0,
    )


def test_count_unflushed_positions_empty(db_path: Path) -> None:
    assert count_unflushed_positions(db_path) == 0


def test_count_unflushed_positions_after_insert(db_path: Path) -> None:
    insert_positions(db_path, [_mk_pos("g1", 1), _mk_pos("g2", 1)])
    assert count_unflushed_positions(db_path) == 2


def test_count_unflushed_positions_filter_by_version(db_path: Path) -> None:
    insert_positions(
        db_path,
        [_mk_pos("g1", 1), _mk_pos("g2", 2), _mk_pos("g3", 2)],
    )
    assert count_unflushed_positions(db_path, model_version=1) == 1
    assert count_unflushed_positions(db_path, model_version=2) == 2
    assert count_unflushed_positions(db_path, model_version=3) == 0


def test_count_unflushed_by_version(db_path: Path) -> None:
    insert_positions(db_path, [_mk_pos("g1", 1), _mk_pos("g2", 2), _mk_pos("g3", 2)])
    assert count_unflushed_by_version(db_path) == {1: 1, 2: 2}


def test_count_unflushed_by_version_omits_flushed(db_path: Path) -> None:
    insert_positions(db_path, [_mk_pos("g1", 1), _mk_pos("g2", 1)])
    insert_batch(db_path, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=2)
    mark_positions_flushed(db_path, [1, 2], batch_id=1)
    # All flushed → version 1 drops out of the live map entirely.
    assert count_unflushed_by_version(db_path) == {}


def test_iter_unflushed_positions_fifo_order(db_path: Path) -> None:
    """Returned rows ordered by id ASC (FIFO insert order)."""
    positions = [_mk_pos(f"g{i}", 1, ply=i) for i in range(5)]
    insert_positions(db_path, positions)
    rows = iter_unflushed_positions(db_path, model_version=1, limit=10)
    assert [r.game_id for r in rows] == ["g0", "g1", "g2", "g3", "g4"]
    # ids increasing
    assert [r.id for r in rows] == sorted(r.id for r in rows)


def test_iter_unflushed_positions_respects_limit(db_path: Path) -> None:
    insert_positions(db_path, [_mk_pos(f"g{i}", 1) for i in range(10)])
    rows = iter_unflushed_positions(db_path, model_version=1, limit=3)
    assert len(rows) == 3


def test_iter_unflushed_positions_filters_by_version(db_path: Path) -> None:
    insert_positions(
        db_path,
        [_mk_pos("g1", 1), _mk_pos("g2", 2), _mk_pos("g3", 2)],
    )
    rows = iter_unflushed_positions(db_path, model_version=2, limit=10)
    assert {r.game_id for r in rows} == {"g2", "g3"}


def test_iter_unflushed_positions_skips_already_flushed(db_path: Path) -> None:
    insert_positions(db_path, [_mk_pos("g1", 1), _mk_pos("g2", 1)])
    batch_id = insert_batch(db_path, 1, 0, "/dummy.npz", n_positions=1)
    rows = iter_unflushed_positions(db_path, model_version=1, limit=10)
    ids_to_flush = [rows[0].id]
    mark_positions_flushed(db_path, ids_to_flush, batch_id)

    rows_after = iter_unflushed_positions(db_path, model_version=1, limit=10)
    assert len(rows_after) == 1
    assert rows_after[0].id != ids_to_flush[0]


def test_mark_positions_flushed_updates_flag_and_batch_id(db_path: Path) -> None:
    insert_positions(db_path, [_mk_pos("g1", 1), _mk_pos("g2", 1)])
    rows = iter_unflushed_positions(db_path, model_version=1, limit=10)
    batch_id = insert_batch(db_path, 1, 0, "/dummy.npz", n_positions=2)
    n = mark_positions_flushed(db_path, [r.id for r in rows], batch_id)
    assert n == 2
    assert count_unflushed_positions(db_path) == 0


def test_mark_positions_flushed_empty_returns_0(db_path: Path) -> None:
    assert mark_positions_flushed(db_path, [], batch_id=1) == 0


def test_delete_flushed_old(db_path: Path) -> None:
    """Purge flushed positions with model_version <= max."""
    insert_positions(
        db_path,
        [_mk_pos("g1", 1), _mk_pos("g2", 2), _mk_pos("g3", 3)],
    )
    # Flush all 3.
    rows = (
        iter_unflushed_positions(db_path, 1, 100)
        + iter_unflushed_positions(db_path, 2, 100)
        + iter_unflushed_positions(db_path, 3, 100)
    )
    b1 = insert_batch(db_path, 1, 0, "/a.npz", 1)
    b2 = insert_batch(db_path, 2, 0, "/b.npz", 1)
    b3 = insert_batch(db_path, 3, 0, "/c.npz", 1)
    mark_positions_flushed(db_path, [rows[0].id], b1)
    mark_positions_flushed(db_path, [rows[1].id], b2)
    mark_positions_flushed(db_path, [rows[2].id], b3)

    # Purge versions <= 2 ; v1 + v2 deleted, v3 kept.
    deleted = delete_flushed_old(db_path, max_model_version=2)
    assert deleted == 2

    # Verify v3 row still exists, flagged flushed.
    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM positions")
        assert cur.fetchone()[0] == 1


def test_delete_flushed_old_does_not_delete_unflushed(db_path: Path) -> None:
    """Purge MUST NOT touch unflushed positions even if they're old versions."""
    insert_positions(db_path, [_mk_pos("g1", 1)])
    deleted = delete_flushed_old(db_path, max_model_version=999)
    assert deleted == 0  # nothing deleted, position is unflushed
    assert count_unflushed_positions(db_path) == 1


def test_list_unflushed_model_versions_empty(db_path: Path) -> None:
    assert list_unflushed_model_versions(db_path) == []


def test_list_unflushed_model_versions_distinct_sorted(db_path: Path) -> None:
    insert_positions(
        db_path,
        [
            _mk_pos("g1", 2),
            _mk_pos("g2", 1),
            _mk_pos("g3", 1),
            _mk_pos("g4", 3),
            _mk_pos("g5", 2),
        ],
    )
    assert list_unflushed_model_versions(db_path) == [1, 2, 3]


def test_list_unflushed_model_versions_skips_flushed_versions(db_path: Path) -> None:
    """If all positions of a version are flushed, that version is NOT listed."""
    insert_positions(db_path, [_mk_pos("g1", 1), _mk_pos("g2", 2)])
    rows = iter_unflushed_positions(db_path, model_version=1, limit=10)
    batch_id = insert_batch(db_path, 1, 0, "/a.npz", 1)
    mark_positions_flushed(db_path, [rows[0].id], batch_id)
    assert list_unflushed_model_versions(db_path) == [2]
