"""Tests for storage/batches.py — CRUD on the batches NPZ-archive table."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from nanozero_jobserver.storage.batches import (
    batch_timestamps,
    batches_summary,
    count_batches,
    count_batches_by_version,
    insert_batch,
    list_batches,
    next_batch_idx,
    sum_positions_by_version,
)
from nanozero_jobserver.storage.db import init_schema


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.db"
    init_schema(p)
    return p


def test_insert_batch_returns_row_id(db_path: Path) -> None:
    row_id = insert_batch(db_path, model_version=5, batch_idx=0, npz_path="/a.npz", n_positions=100)
    assert row_id > 0


def test_batches_summary_rollup(db_path: Path) -> None:
    """#A/B tooling : rollup n_shards/n_positions par gen, filtrable par source (D.3)."""
    insert_batch(
        db_path, model_version=1, batch_idx=0, npz_path="/f.npz", n_positions=100, source="fleet"
    )
    insert_batch(
        db_path, model_version=1, batch_idx=1, npz_path="/b.npz", n_positions=40, source="browser"
    )
    insert_batch(
        db_path, model_version=2, batch_idx=0, npz_path="/f2.npz", n_positions=50, source="fleet"
    )
    alls = {r["model_version"]: r for r in batches_summary(db_path)}
    assert alls[1]["n_shards"] == 2 and alls[1]["n_positions"] == 140
    assert alls[2]["n_positions"] == 50

    browser = batches_summary(db_path, source="browser")
    assert len(browser) == 1
    assert browser[0]["model_version"] == 1 and browser[0]["n_positions"] == 40

    fleet = {r["model_version"]: r for r in batches_summary(db_path, source="fleet")}
    assert fleet[1]["n_positions"] == 100 and fleet[2]["n_positions"] == 50


def test_count_batches_by_version(db_path: Path) -> None:
    """#focus stats : compte des batches par version en UNE requête (fix N+1)."""
    insert_batch(db_path, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=10)
    insert_batch(db_path, model_version=1, batch_idx=1, npz_path="/b.npz", n_positions=20)
    insert_batch(db_path, model_version=2, batch_idx=0, npz_path="/c.npz", n_positions=5)
    assert count_batches_by_version(db_path) == {1: 2, 2: 1}
    assert count_batches_by_version(db_path) == {
        v: count_batches(db_path, model_version=v) for v in (1, 2)
    }  # cohérent avec l'ancien per-version


def test_sum_positions_by_version_source_filter(db_path: Path) -> None:
    """B4-D2 : le filtre source exclut les shards browser du compte durable par version."""
    insert_batch(
        db_path, model_version=1, batch_idx=0, npz_path="/f.npz", n_positions=100, source="fleet"
    )
    insert_batch(
        db_path, model_version=1, batch_idx=1, npz_path="/b.npz", n_positions=40, source="browser"
    )
    assert sum_positions_by_version(db_path) == {1: 140}  # défaut None = toutes sources
    assert sum_positions_by_version(db_path, source="fleet") == {1: 100}  # browser EXCLU
    assert sum_positions_by_version(db_path, source="browser") == {1: 40}


def test_insert_batch_assigns_increasing_ids(db_path: Path) -> None:
    id1 = insert_batch(db_path, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=10)
    id2 = insert_batch(db_path, model_version=1, batch_idx=1, npz_path="/b.npz", n_positions=20)
    assert id2 == id1 + 1


def test_insert_batch_duplicate_raises(db_path: Path) -> None:
    """UNIQUE(model_version, batch_idx) constraint."""
    insert_batch(db_path, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=10)
    with pytest.raises(sqlite3.IntegrityError):
        insert_batch(db_path, model_version=1, batch_idx=0, npz_path="/b.npz", n_positions=20)


def test_next_batch_idx_empty_returns_0(db_path: Path) -> None:
    assert next_batch_idx(db_path, model_version=1) == 0


def test_next_batch_idx_after_inserts(db_path: Path) -> None:
    insert_batch(db_path, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=10)
    assert next_batch_idx(db_path, model_version=1) == 1
    insert_batch(db_path, model_version=1, batch_idx=1, npz_path="/b.npz", n_positions=20)
    assert next_batch_idx(db_path, model_version=1) == 2


def test_next_batch_idx_per_version_independent(db_path: Path) -> None:
    """Different model_versions have independent sequences."""
    insert_batch(db_path, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=10)
    insert_batch(db_path, model_version=1, batch_idx=1, npz_path="/b.npz", n_positions=10)
    insert_batch(db_path, model_version=2, batch_idx=0, npz_path="/c.npz", n_positions=10)
    assert next_batch_idx(db_path, model_version=1) == 2
    assert next_batch_idx(db_path, model_version=2) == 1
    assert next_batch_idx(db_path, model_version=3) == 0


def test_list_batches_empty(db_path: Path) -> None:
    assert list_batches(db_path) == []


def test_list_batches_filter_by_version(db_path: Path) -> None:
    insert_batch(db_path, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=10)
    insert_batch(db_path, model_version=2, batch_idx=0, npz_path="/b.npz", n_positions=20)
    insert_batch(db_path, model_version=2, batch_idx=1, npz_path="/c.npz", n_positions=30)

    all_batches = list_batches(db_path)
    assert len(all_batches) == 3

    v2 = list_batches(db_path, model_version=2)
    assert len(v2) == 2
    assert all(b.model_version == 2 for b in v2)


def test_list_batches_ordering(db_path: Path) -> None:
    """Default order = model_version DESC, batch_idx ASC."""
    insert_batch(db_path, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=10)
    insert_batch(db_path, model_version=2, batch_idx=0, npz_path="/b.npz", n_positions=20)
    insert_batch(db_path, model_version=2, batch_idx=1, npz_path="/c.npz", n_positions=30)
    rows = list_batches(db_path)
    assert [(r.model_version, r.batch_idx) for r in rows] == [(2, 0), (2, 1), (1, 0)]


def test_list_batches_limit(db_path: Path) -> None:
    for i in range(5):
        insert_batch(db_path, model_version=1, batch_idx=i, npz_path=f"/a{i}.npz", n_positions=10)
    rows = list_batches(db_path, limit=3)
    assert len(rows) == 3


def test_count_batches(db_path: Path) -> None:
    assert count_batches(db_path) == 0
    insert_batch(db_path, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=10)
    assert count_batches(db_path) == 1
    insert_batch(db_path, model_version=2, batch_idx=0, npz_path="/b.npz", n_positions=20)
    assert count_batches(db_path) == 2
    assert count_batches(db_path, model_version=1) == 1
    assert count_batches(db_path, model_version=2) == 1
    assert count_batches(db_path, model_version=3) == 0


def test_sum_positions_by_version(db_path: Path) -> None:
    insert_batch(db_path, model_version=5, batch_idx=0, npz_path="/a.npz", n_positions=200)
    insert_batch(db_path, model_version=5, batch_idx=1, npz_path="/b.npz", n_positions=100)
    insert_batch(db_path, model_version=6, batch_idx=0, npz_path="/c.npz", n_positions=50)
    sums = sum_positions_by_version(db_path)
    assert sums == {5: 300, 6: 50}


def test_sum_positions_by_version_empty(db_path: Path) -> None:
    assert sum_positions_by_version(db_path) == {}


def test_batch_timestamps_ordered(db_path: Path) -> None:
    for i in range(3):
        insert_batch(db_path, model_version=9, batch_idx=i, npz_path=f"/{i}.npz", n_positions=10)
    ts = batch_timestamps(db_path, model_version=9)
    assert len(ts) == 3
    assert ts == sorted(ts)  # ascending by batch_idx → ascending in time
    assert batch_timestamps(db_path, model_version=99) == []
