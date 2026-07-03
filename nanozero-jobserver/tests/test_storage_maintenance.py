"""Tests for storage/maintenance.py — storage stats, checkpoint, VACUUM INTO."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from nanozero_jobserver.storage.batches import insert_batch
from nanozero_jobserver.storage.db import init_schema
from nanozero_jobserver.storage.maintenance import (
    count_purgeable,
    rebuild_compact,
    storage_stats,
    vacuum_into,
    wal_checkpoint,
)
from nanozero_jobserver.storage.replay_buffer import (
    Position,
    insert_positions,
    mark_positions_flushed,
)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    p = tmp_path / "maint.db"
    init_schema(p)
    return p


def _mk_pos(v: int, ply: int = 0) -> Position:
    return Position(
        game_id="g",
        model_version=v,
        ply=ply,
        fen="f",
        input_planes=b"\x00" * 64,
        policy_target=b"\x01" * 64,
        outcome=0.0,
    )


def test_storage_stats_basic(db: Path) -> None:
    insert_positions(db, [_mk_pos(v=1, ply=i) for i in range(3)])
    s = storage_stats(db)
    assert s.page_size > 0
    assert s.page_count > 0
    assert s.logical_bytes == s.page_count * s.page_size
    assert s.file_bytes > 0
    assert s.positions_total == 3
    assert s.positions_flushed == 0


def test_storage_stats_counts_flushed(db: Path) -> None:
    insert_positions(db, [_mk_pos(v=1, ply=i) for i in range(2)])
    insert_batch(db, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=2)
    mark_positions_flushed(db, [1, 2], batch_id=1)
    s = storage_stats(db)
    assert s.positions_flushed == 2


def test_count_purgeable(db: Path) -> None:
    insert_positions(db, [_mk_pos(v=1), _mk_pos(v=2), _mk_pos(v=3)])
    insert_batch(db, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=3)
    mark_positions_flushed(db, [1, 2], batch_id=1)  # v1,v2 flushed; v3 not
    # Only flushed rows with model_version <= bound count.
    assert count_purgeable(db, max_model_version=2) == 2
    assert count_purgeable(db, max_model_version=1) == 1
    assert count_purgeable(db, max_model_version=3) == 2  # v3 unflushed, excluded


def test_wal_checkpoint_returns_row(db: Path) -> None:
    insert_positions(db, [_mk_pos(v=1)])
    r = wal_checkpoint(db)
    assert set(r) == {"busy", "log", "checkpointed"}
    assert r["busy"] == 0


def _seed_for_compact(db: Path) -> None:
    from nanozero_jobserver.storage.control import set_selfplay_paused
    from nanozero_jobserver.storage.jobs import create_job
    from nanozero_jobserver.storage.models import promote_model, register_model

    register_model(db, version=1, name="gen-001", onnx_path="/p", sha256_onnx="h")
    promote_model(db, 1)
    create_job(db, model_version=1, num_sims=200)
    set_selfplay_paused(db, True)
    insert_batch(db, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=3)
    # 3 flushed (redundant, archived) + 2 unflushed (the live tail).
    insert_positions(db, [_mk_pos(v=1, ply=i) for i in range(3)])
    mark_positions_flushed(db, [1, 2, 3], batch_id=1)
    insert_positions(db, [_mk_pos(v=1, ply=i) for i in range(3, 5)])


def test_rebuild_compact_drops_flushed_keeps_metadata(db: Path, tmp_path: Path) -> None:
    _seed_for_compact(db)
    dest = tmp_path / "compact.db"
    r = rebuild_compact(db, dest, keep_unflushed=True)

    assert dest.exists()
    assert r["rows_copied"]["positions"] == 2  # only the 2 unflushed kept
    assert r["rows_copied"]["batches"] == 1
    assert r["rows_copied"]["models"] == 1
    assert r["rows_copied"]["jobs"] == 1
    assert r["rows_copied"]["server_control"] == 1

    # The copy is a valid standalone DB with metadata intact, flushed BLOBs gone.
    con = sqlite3.connect(dest)
    assert con.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 2
    assert con.execute("SELECT COUNT(*) FROM batches").fetchone()[0] == 1
    assert (
        con.execute("SELECT value FROM server_control WHERE key='selfplay_paused'").fetchone()[0]
        == "1"
    )
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    con.close()


def test_rebuild_compact_drop_all_positions(db: Path, tmp_path: Path) -> None:
    _seed_for_compact(db)
    dest = tmp_path / "compact-nopos.db"
    r = rebuild_compact(db, dest, keep_unflushed=False)
    assert r["rows_copied"]["positions"] == 0
    con = sqlite3.connect(dest)
    assert con.execute("SELECT COUNT(*) FROM positions").fetchone()[0] == 0
    assert con.execute("SELECT COUNT(*) FROM jobs").fetchone()[0] == 1  # metadata still there
    con.close()


def test_rebuild_compact_refuses_existing_dest(db: Path, tmp_path: Path) -> None:
    dest = tmp_path / "exists.db"
    dest.write_text("x")
    with pytest.raises(FileExistsError):
        rebuild_compact(db, dest)


def test_vacuum_into_writes_copy(db: Path, tmp_path: Path) -> None:
    insert_positions(db, [_mk_pos(v=1, ply=i) for i in range(10)])
    dest = tmp_path / "compacted.db"
    r = vacuum_into(db, dest)
    assert dest.exists()
    assert r["source_bytes"] > 0
    assert r["dest_bytes"] > 0
    # The copy is a valid, queryable DB with the same rows.
    init_schema(dest)  # no-op on an already-populated schema
    from nanozero_jobserver.storage.replay_buffer import count_positions

    assert count_positions(dest) == 10
