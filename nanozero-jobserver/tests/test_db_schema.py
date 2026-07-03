"""Tests for storage/db.py — schema + ADR-015 migration idempotence."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from nanozero_jobserver.storage.db import init_schema


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _tables(conn: sqlite3.Connection) -> set[str]:
    cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    return {row[0] for row in cur.fetchall()}


def _indexes(conn: sqlite3.Connection, table: str) -> set[str]:
    cur = conn.execute(f"PRAGMA index_list({table})")
    return {row[1] for row in cur.fetchall()}


def test_init_schema_creates_all_tables(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_schema(db)
    with sqlite3.connect(str(db)) as conn:
        assert _tables(conn) >= {"positions", "jobs", "models", "batches"}


def test_positions_has_flushed_to_npz_and_batch_id(tmp_path: Path) -> None:
    """ADR-015 : positions table has new columns post-migration."""
    db = tmp_path / "test.db"
    init_schema(db)
    with sqlite3.connect(str(db)) as conn:
        cols = _columns(conn, "positions")
        assert "flushed_to_npz" in cols
        assert "batch_id" in cols


def test_batches_table_schema(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_schema(db)
    with sqlite3.connect(str(db)) as conn:
        cols = _columns(conn, "batches")
        assert cols == {
            "id",
            "model_version",
            "batch_idx",
            "npz_path",
            "n_positions",
            "created_at",
            "source",  # chantier 1 cloisonnement : provenance du shard (fleet/browser)
        }


def test_init_schema_idempotent_double_call(tmp_path: Path) -> None:
    """Calling init_schema twice on the same DB must not raise."""
    db = tmp_path / "test.db"
    init_schema(db)
    init_schema(db)  # no error
    with sqlite3.connect(str(db)) as conn:
        cols = _columns(conn, "positions")
        # New columns present exactly once (not duplicated)
        assert sum(1 for c in cols if c == "flushed_to_npz") == 1


def test_migration_from_old_db(tmp_path: Path) -> None:
    """Simulate a pre-ADR-015 DB : positions table without flushed_to_npz.

    init_schema must add the missing columns idempotently.
    """
    db = tmp_path / "test.db"
    with sqlite3.connect(str(db)) as conn:
        # Create the OLD positions schema (no flushed_to_npz/batch_id).
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
        # Insert a sample row to ensure migration doesn't lose data.
        conn.execute(
            "INSERT INTO positions"
            " (game_id, model_version, ply, fen, input_planes, policy_target, outcome)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("g0", 1, 0, "fen0", b"\x00", b"\x00", 0.0),
        )
        conn.commit()

    init_schema(db)

    with sqlite3.connect(str(db)) as conn:
        cols = _columns(conn, "positions")
        assert "flushed_to_npz" in cols
        assert "batch_id" in cols
        # Existing data preserved
        cur = conn.execute("SELECT COUNT(*) FROM positions")
        assert cur.fetchone()[0] == 1
        # Default value applied to existing row
        cur = conn.execute("SELECT flushed_to_npz, batch_id FROM positions WHERE game_id='g0'")
        row = cur.fetchone()
        assert row[0] == 0  # False default
        assert row[1] is None


def test_flushed_index_exists(tmp_path: Path) -> None:
    db = tmp_path / "test.db"
    init_schema(db)
    with sqlite3.connect(str(db)) as conn:
        idx = _indexes(conn, "positions")
        assert "idx_positions_flushed" in idx


def test_batches_unique_constraint(tmp_path: Path) -> None:
    """batches table has UNIQUE(model_version, batch_idx)."""
    db = tmp_path / "test.db"
    init_schema(db)
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "INSERT INTO batches (model_version, batch_idx, npz_path, n_positions)"
            " VALUES (?, ?, ?, ?)",
            (1, 0, "/tmp/a.npz", 100),
        )
        # Duplicate (1, 0) raises IntegrityError.
        try:
            conn.execute(
                "INSERT INTO batches (model_version, batch_idx, npz_path, n_positions)"
                " VALUES (?, ?, ?, ?)",
                (1, 0, "/tmp/b.npz", 200),
            )
            raise AssertionError("Expected IntegrityError")
        except sqlite3.IntegrityError:
            pass  # expected
