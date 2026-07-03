"""Unit tests for storage/db.py — schema init + connection pragmas."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from nanozero_jobserver.storage.db import connect, init_schema


def test_init_schema_creates_tables(tmp_path: Path) -> None:
    db = tmp_path / "fresh.db"
    init_schema(db)

    with sqlite3.connect(str(db)) as conn:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        tables = {row[0] for row in cur.fetchall()}

    assert {"positions", "jobs", "models"}.issubset(tables)


def test_connect_sets_busy_timeout(tmp_path: Path) -> None:
    """#8 BMAD : la contention WAL fait ATTENDRE le lock (5 s) au lieu d'échouer direct."""
    db = tmp_path / "bt.db"
    init_schema(db)
    with connect(db) as conn:
        assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == 5000


def test_init_schema_idempotent(tmp_path: Path) -> None:
    """Calling init_schema twice must not raise (server-restart safety)."""
    db = tmp_path / "twice.db"
    init_schema(db)
    init_schema(db)  # must not raise

    with sqlite3.connect(str(db)) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'")
        assert int(cur.fetchone()[0]) >= 3


def test_init_schema_enables_wal(tmp_path: Path) -> None:
    """WAL mode is critical for concurrent reader/writer access."""
    db = tmp_path / "wal.db"
    init_schema(db)
    with sqlite3.connect(str(db)) as conn:
        cur = conn.execute("PRAGMA journal_mode")
        assert cur.fetchone()[0] == "wal"


def test_init_schema_creates_parent_dirs(tmp_path: Path) -> None:
    """If the .db path's parent directory doesn't exist, init_schema creates it."""
    db = tmp_path / "deep" / "nested" / "store.db"
    assert not db.parent.exists()
    init_schema(db)
    assert db.exists()


def test_connect_row_factory_returns_dict_like(tmp_path: Path) -> None:
    """connect() should set row_factory=sqlite3.Row so we get column-by-name access."""
    db = tmp_path / "rows.db"
    init_schema(db)
    with connect(db) as conn:
        cur = conn.execute(
            "INSERT INTO models (version, name, onnx_path, sha256_onnx) VALUES (1, 'x', '/x', 'abc')"
        )
        assert cur.rowcount == 1
    with connect(db) as conn:
        row = conn.execute("SELECT version, name FROM models WHERE version=1").fetchone()
        assert row["version"] == 1
        assert row["name"] == "x"


def test_connect_rolls_back_on_exception(tmp_path: Path) -> None:
    """If the with-block raises, the transaction must be rolled back."""
    db = tmp_path / "rollback.db"
    init_schema(db)
    try:
        with connect(db) as conn:
            conn.execute(
                "INSERT INTO models (version, name, onnx_path, sha256_onnx)"
                " VALUES (1, 'first', '/x', 'aaa')"
            )
            raise RuntimeError("simulated failure mid-transaction")
    except RuntimeError:
        pass

    with connect(db) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM models")
        assert int(cur.fetchone()[0]) == 0
