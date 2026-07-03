"""Tests for tools/migrate_existing_db.py — pre-ADR-015 DB migration."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

# Import migrate from the standalone tools/ script (not on sys.path by default).
_TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(_TOOLS))
from migrate_existing_db import migrate  # noqa: E402
from nanozero_jobserver.storage.npz_writer import (  # noqa: E402
    INPUT_PLANES_SHAPE,
    POLICY_LEN,
)


def _seed_pre_adr_015_db(db_path: Path, n_positions: int = 5, model_version: int = 1) -> None:
    """Create a DB with the OLD positions schema + N rows (no flushed_to_npz column)."""
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            """
            CREATE TABLE positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT NOT NULL,
                model_version INTEGER NOT NULL,
                ply INTEGER NOT NULL,
                fen TEXT NOT NULL,
                input_planes BLOB NOT NULL,
                policy_target BLOB NOT NULL,
                outcome REAL NOT NULL,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        ip_blob = np.zeros(INPUT_PLANES_SHAPE, dtype="<f4").tobytes()
        pt_blob = np.zeros(POLICY_LEN, dtype="<f4").tobytes()
        for i in range(n_positions):
            conn.execute(
                "INSERT INTO positions"
                " (game_id, model_version, ply, fen, input_planes, policy_target, outcome)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"g{i}", model_version, i, f"fen-{i}", ip_blob, pt_blob, 0.0),
            )
        conn.commit()


def test_migrate_empty_db_no_op(tmp_path: Path) -> None:
    db = tmp_path / "empty.db"
    # Empty DB but with schema applied
    from nanozero_jobserver.storage.db import init_schema

    init_schema(db)
    out = tmp_path / "datasets"
    shards = migrate(db_path=db, output_dir=out, shard_size=100)
    assert shards == 0


def test_migrate_pre_adr015_db_applies_schema(tmp_path: Path) -> None:
    """Migration adds flushed_to_npz + batches table on old DB."""
    db = tmp_path / "old.db"
    _seed_pre_adr_015_db(db, n_positions=3)
    out = tmp_path / "datasets"
    migrate(db_path=db, output_dir=out, shard_size=100)

    with sqlite3.connect(str(db)) as conn:
        cur = conn.execute("PRAGMA table_info(positions)")
        cols = {row[1] for row in cur.fetchall()}
        assert "flushed_to_npz" in cols
        assert "batch_id" in cols
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cur.fetchall()}
        assert "batches" in tables


def test_migrate_writes_npz_shard(tmp_path: Path) -> None:
    """3 positions migrated → 1 partial shard via force_flush."""
    db = tmp_path / "old.db"
    _seed_pre_adr_015_db(db, n_positions=3)
    out = tmp_path / "datasets"
    shards = migrate(db_path=db, output_dir=out, shard_size=100)
    assert shards == 1
    assert (out / "selfplay-gen001-batch-000.npz").exists()


def test_migrate_full_then_partial_shards(tmp_path: Path) -> None:
    """250 positions, shard_size 100 → 2 full (batch-0,1) + 1 partial (batch-2)."""
    db = tmp_path / "old.db"
    _seed_pre_adr_015_db(db, n_positions=250)
    out = tmp_path / "datasets"
    shards = migrate(db_path=db, output_dir=out, shard_size=100)
    assert shards == 3
    npz_files = sorted(out.glob("*.npz"))
    assert [f.name for f in npz_files] == [
        "selfplay-gen001-batch-000.npz",
        "selfplay-gen001-batch-001.npz",
        "selfplay-gen001-batch-002.npz",
    ]


def test_migrate_purge_deletes_positions(tmp_path: Path) -> None:
    """--purge → flushed rows removed from positions table."""
    db = tmp_path / "old.db"
    _seed_pre_adr_015_db(db, n_positions=3)
    out = tmp_path / "datasets"
    migrate(db_path=db, output_dir=out, shard_size=100, purge=True)
    with sqlite3.connect(str(db)) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM positions")
        assert cur.fetchone()[0] == 0


def test_migrate_no_purge_keeps_positions(tmp_path: Path) -> None:
    """Without --purge, positions remain (flushed_to_npz=1)."""
    db = tmp_path / "old.db"
    _seed_pre_adr_015_db(db, n_positions=3)
    out = tmp_path / "datasets"
    migrate(db_path=db, output_dir=out, shard_size=100, purge=False)
    with sqlite3.connect(str(db)) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM positions WHERE flushed_to_npz=1")
        assert cur.fetchone()[0] == 3


def test_migrate_missing_db_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        migrate(
            db_path=tmp_path / "nonexistent.db",
            output_dir=tmp_path / "datasets",
        )


def test_migrate_multi_version(tmp_path: Path) -> None:
    """Mixed model_versions → 1 shard per version (each below threshold)."""
    db = tmp_path / "old.db"
    _seed_pre_adr_015_db(db, n_positions=2, model_version=1)
    # Append more rows with version 2 directly.
    with sqlite3.connect(str(db)) as conn:
        ip_blob = np.zeros(INPUT_PLANES_SHAPE, dtype="<f4").tobytes()
        pt_blob = np.zeros(POLICY_LEN, dtype="<f4").tobytes()
        for i in range(3):
            conn.execute(
                "INSERT INTO positions"
                " (game_id, model_version, ply, fen, input_planes, policy_target, outcome)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (f"v2-g{i}", 2, i, f"fen-{i}", ip_blob, pt_blob, 0.0),
            )
        conn.commit()

    out = tmp_path / "datasets"
    shards = migrate(db_path=db, output_dir=out, shard_size=100)
    assert shards == 2  # 1 per version (both < threshold → partial)
    assert (out / "selfplay-gen001-batch-000.npz").exists()
    assert (out / "selfplay-gen002-batch-000.npz").exists()
