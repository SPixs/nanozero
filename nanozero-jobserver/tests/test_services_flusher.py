"""Tests services/flusher.py — FlusherService background flush + purge."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from nanozero_jobserver.services.flusher import FlusherService
from nanozero_jobserver.storage.batches import count_batches, list_batches
from nanozero_jobserver.storage.db import init_schema
from nanozero_jobserver.storage.npz_writer import INPUT_PLANES_SHAPE, POLICY_LEN
from nanozero_jobserver.storage.replay_buffer import (
    Position,
    count_unflushed_positions,
    insert_positions,
)


@pytest.fixture()
def db_path(tmp_path: Path) -> Path:
    p = tmp_path / "test.db"
    init_schema(p)
    return p


@pytest.fixture()
def output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "datasets"
    d.mkdir()
    return d


def _mk_position(game_id: str, model_version: int, ply: int = 0) -> Position:
    ip = np.zeros(INPUT_PLANES_SHAPE, dtype="<f4")
    pt = np.zeros(POLICY_LEN, dtype="<f4")
    return Position(
        game_id=game_id,
        model_version=model_version,
        ply=ply,
        fen="fen",
        input_planes=ip.tobytes(),
        policy_target=pt.tobytes(),
        outcome=0.0,
    )


# ---------------------------------------------------------------- constructor


def test_init_validates_threshold(db_path: Path, output_dir: Path) -> None:
    with pytest.raises(ValueError, match="flush_threshold"):
        FlusherService(db_path, output_dir, flush_threshold=0)


def test_consecutive_tick_failures_tracks_and_resets(
    db_path: Path, output_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#8 BMAD : un tick qui lève incrémente le compteur de santé ; un succès le remet à 0."""
    flusher = FlusherService(db_path, output_dir, flush_threshold=100)
    assert flusher.consecutive_tick_failures == 0

    def _boom() -> int:
        raise RuntimeError("disque plein")

    monkeypatch.setattr(flusher, "tick", _boom)
    flusher._tick_guarded()
    flusher._tick_guarded()
    assert flusher.consecutive_tick_failures == 2

    monkeypatch.setattr(flusher, "tick", lambda: 0)  # tick sain → reset
    flusher._tick_guarded()
    assert flusher.consecutive_tick_failures == 0


def test_init_validates_retention(db_path: Path, output_dir: Path) -> None:
    with pytest.raises(ValueError, match="retention_window"):
        FlusherService(db_path, output_dir, retention_window=-1)


def test_init_validates_tick_interval(db_path: Path, output_dir: Path) -> None:
    with pytest.raises(ValueError, match="tick_interval_seconds"):
        FlusherService(db_path, output_dir, tick_interval_seconds=0)


# --------------------------------------------------------------------- tick()


def test_tick_below_threshold_no_flush(db_path: Path, output_dir: Path) -> None:
    insert_positions(db_path, [_mk_position(f"g{i}", 1) for i in range(50)])
    flusher = FlusherService(db_path, output_dir, flush_threshold=100)
    written = flusher.tick()
    assert written == 0
    assert count_batches(db_path) == 0
    assert count_unflushed_positions(db_path) == 50


def test_tick_at_threshold_flushes_one_shard(db_path: Path, output_dir: Path) -> None:
    insert_positions(db_path, [_mk_position(f"g{i}", 1) for i in range(100)])
    flusher = FlusherService(db_path, output_dir, flush_threshold=100)
    written = flusher.tick()
    assert written == 1
    assert count_batches(db_path) == 1
    assert count_unflushed_positions(db_path) == 0
    # NPZ file actually created
    shards = list(output_dir.glob("*.npz"))
    assert len(shards) == 1
    assert shards[0].name == "selfplay-gen001-batch-000.npz"


def test_tick_well_above_threshold_multiple_shards(db_path: Path, output_dir: Path) -> None:
    """250 positions, threshold 100 → 2 shards of 100 (50 leftover unflushed)."""
    insert_positions(db_path, [_mk_position(f"g{i}", 1) for i in range(250)])
    flusher = FlusherService(db_path, output_dir, flush_threshold=100)
    written = flusher.tick()
    assert written == 2
    assert count_batches(db_path) == 2
    assert count_unflushed_positions(db_path) == 50
    batches = list_batches(db_path, model_version=1)
    assert [b.batch_idx for b in batches] == [0, 1]


def test_tick_separate_shards_per_version(db_path: Path, output_dir: Path) -> None:
    """Version 1 (full) + version 2 (full) → 2 shards, one per version."""
    insert_positions(
        db_path,
        [_mk_position(f"g{i}", 1) for i in range(100)]
        + [_mk_position(f"g{i}", 2) for i in range(100)],
    )
    flusher = FlusherService(db_path, output_dir, flush_threshold=100)
    written = flusher.tick()
    assert written == 2
    assert (output_dir / "selfplay-gen001-batch-000.npz").exists()
    assert (output_dir / "selfplay-gen002-batch-000.npz").exists()


def test_tick_consecutive_calls_increment_batch_idx(db_path: Path, output_dir: Path) -> None:
    """3 ticks, 100 positions each → batches 0, 1, 2."""
    flusher = FlusherService(db_path, output_dir, flush_threshold=100)
    for k in range(3):
        insert_positions(
            db_path,
            [_mk_position(f"k{k}-g{i}", 1) for i in range(100)],
        )
        flusher.tick()
    batches = list_batches(db_path, model_version=1)
    assert [b.batch_idx for b in batches] == [0, 1, 2]


# --------------------------------------------------------------- force_flush()


def test_force_flush_below_threshold(db_path: Path, output_dir: Path) -> None:
    """force_flush ignores threshold, flushes partial shard."""
    insert_positions(db_path, [_mk_position(f"g{i}", 1) for i in range(30)])
    flusher = FlusherService(db_path, output_dir, flush_threshold=1000)
    written = flusher.force_flush()
    assert written == 1
    assert count_unflushed_positions(db_path) == 0


def test_force_flush_empty_db_returns_0(db_path: Path, output_dir: Path) -> None:
    flusher = FlusherService(db_path, output_dir)
    assert flusher.force_flush() == 0


# ------------------------------------------------------------------ purge


def test_purge_no_current_version_no_op(db_path: Path, output_dir: Path) -> None:
    """If get_current_model_version returns None, no purge happens."""
    insert_positions(db_path, [_mk_position(f"g{i}", 1) for i in range(100)])
    flusher = FlusherService(
        db_path,
        output_dir,
        flush_threshold=100,
        retention_window=5,
        get_current_model_version=lambda: None,
    )
    flusher.tick()
    # 100 rows flushed but not purged.
    # Just check rows still in DB (count_positions through replay_buffer or
    # by direct query).
    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM positions")
        assert cur.fetchone()[0] == 100


def test_purge_deletes_old_flushed(db_path: Path, output_dir: Path) -> None:
    """current_version=10, retention=2 → purge versions <= 8."""
    # Insert across versions 1, 5, 9 (all old enough to be purged after).
    insert_positions(
        db_path,
        [_mk_position(f"v1-g{i}", 1) for i in range(100)]
        + [_mk_position(f"v5-g{i}", 5) for i in range(100)]
        + [_mk_position(f"v9-g{i}", 9) for i in range(100)],
    )
    flusher = FlusherService(
        db_path,
        output_dir,
        flush_threshold=100,
        retention_window=2,
        get_current_model_version=lambda: 10,
    )
    flusher.tick()
    # Tick flushed all 3 versions. Purge condition : max_version=10-2=8.
    # v1, v5 deleted; v9 kept (9 > 8).
    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute("SELECT DISTINCT model_version FROM positions ORDER BY model_version")
        remaining = [row[0] for row in cur.fetchall()]
    assert remaining == [9]


def test_purge_keeps_unflushed_even_if_old(db_path: Path, output_dir: Path) -> None:
    """Purge MUST NOT delete unflushed rows, even if old."""
    insert_positions(db_path, [_mk_position(f"g{i}", 1) for i in range(50)])  # under threshold
    flusher = FlusherService(
        db_path,
        output_dir,
        flush_threshold=100,
        retention_window=0,
        get_current_model_version=lambda: 10,
    )
    flusher.tick()
    # Nothing flushed (50 < 100), nothing purged (only flushed are purgeable).
    assert count_unflushed_positions(db_path) == 50


# ------------------------------------------------------------------ start/stop


def test_start_stop_no_deadlock(db_path: Path, output_dir: Path) -> None:
    """Spawn thread, sleep briefly, stop. Must not hang."""
    flusher = FlusherService(db_path, output_dir, tick_interval_seconds=0.05)
    flusher.start()
    time.sleep(0.1)  # let _loop run once or twice
    flusher.stop(timeout=2.0)
    assert flusher._thread is not None
    assert not flusher._thread.is_alive()


def test_start_idempotent(db_path: Path, output_dir: Path) -> None:
    """Two start() calls don't spawn 2 threads."""
    flusher = FlusherService(db_path, output_dir, tick_interval_seconds=0.05)
    flusher.start()
    t1 = flusher._thread
    flusher.start()  # idempotent : same thread
    assert flusher._thread is t1
    flusher.stop(timeout=2.0)


def test_stop_drains_remaining_unflushed(db_path: Path, output_dir: Path) -> None:
    """stop() should force_flush the buffer."""
    flusher = FlusherService(
        db_path, output_dir, flush_threshold=10_000, tick_interval_seconds=0.05
    )
    flusher.start()
    # Insert below threshold while loop is running.
    insert_positions(db_path, [_mk_position(f"g{i}", 1) for i in range(50)])
    time.sleep(0.1)
    flusher.stop(timeout=2.0)
    # After stop, force_flush should have drained the 50 unflushed.
    assert count_unflushed_positions(db_path) == 0


def test_tick_purges_stale_claims(db_path: Path, output_dir: Path) -> None:
    """Wiring : tick() runs the stale-claim watchdog (deletes dead 'claimed' rows)."""
    import sqlite3

    from nanozero_jobserver.storage.jobs import (
        claim_job,
        count_jobs_by_status,
        create_job,
    )

    create_job(db_path, model_version=1, num_sims=200)
    claim_job(db_path, "w-dead")
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE jobs SET claimed_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-2 hours')"
        )

    flusher = FlusherService(db_path, output_dir, stale_claim_timeout_seconds=3600)
    flusher.tick()
    assert count_jobs_by_status(db_path).get("claimed", 0) == 0


def test_tick_keeps_stale_claims_when_disabled(db_path: Path, output_dir: Path) -> None:
    """stale_claim_timeout_seconds=0 disables the watchdog."""
    import sqlite3

    from nanozero_jobserver.storage.jobs import (
        claim_job,
        count_jobs_by_status,
        create_job,
    )

    create_job(db_path, model_version=1, num_sims=200)
    claim_job(db_path, "w-dead")
    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(
            "UPDATE jobs SET claimed_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-2 hours')"
        )

    flusher = FlusherService(db_path, output_dir, stale_claim_timeout_seconds=0)
    flusher.tick()
    assert count_jobs_by_status(db_path).get("claimed", 0) == 1


# -----------------------------------------------------------------------------
# Chantier 1 — flush PAR-SOURCE (cloisonnement fleet vs browser)
# -----------------------------------------------------------------------------


def test_flush_separates_browser_and_fleet_shards(db_path: Path, output_dir: Path) -> None:
    """Fleet → 'selfplay-gen*' ; browser → 'browser-gen*' séparés ; le glob training n'attrape que fleet."""
    insert_positions(db_path, [_mk_position(f"f{i}", 7) for i in range(12)], source="fleet")
    insert_positions(db_path, [_mk_position(f"b{i}", 7) for i in range(10)], source="browser")
    flusher = FlusherService(db_path, output_dir, flush_threshold=10, browser_flush_threshold=5)
    flusher.tick()

    names = sorted(p.name for p in output_dir.glob("*.npz"))
    assert any(n.startswith("selfplay-gen007") for n in names), names
    assert any(n.startswith("browser-gen007") for n in names), names
    # Quarantine par NOMMAGE : le glob d'entraînement ne ramasse aucun shard browser.
    assert all("browser" not in p.name for p in output_dir.glob("selfplay-gen007*"))
    # Le browser a quitté le HOT cache (10 >= 2× seuil 5 → 2 shards, 0 restant) → fix du bloat.
    assert count_unflushed_positions(db_path, source="browser") == 0
    assert count_unflushed_positions(db_path, source="fleet") == 2  # 12 - 10 (un shard fleet)


def test_flush_browser_disabled_keeps_quarantine(db_path: Path, output_dir: Path) -> None:
    """flush_browser=False = quarantine stricte historique : browser jamais flushé."""
    insert_positions(db_path, [_mk_position(f"f{i}", 7) for i in range(12)], source="fleet")
    insert_positions(db_path, [_mk_position(f"b{i}", 7) for i in range(20)], source="browser")
    flusher = FlusherService(db_path, output_dir, flush_threshold=10, flush_browser=False)
    flusher.tick()

    names = [p.name for p in output_dir.glob("*.npz")]
    assert any(n.startswith("selfplay-gen007") for n in names)
    assert not any(n.startswith("browser-") for n in names)  # aucun shard browser
    assert count_unflushed_positions(db_path, source="browser") == 20  # reste en quarantine HOT cache


def test_flush_tags_batches_by_source(db_path: Path, output_dir: Path) -> None:
    """Le registre durable `batches` trace la provenance de chaque shard."""
    import sqlite3

    insert_positions(db_path, [_mk_position(f"f{i}", 7) for i in range(12)], source="fleet")
    insert_positions(db_path, [_mk_position(f"b{i}", 7) for i in range(10)], source="browser")
    FlusherService(db_path, output_dir, flush_threshold=10, browser_flush_threshold=5).tick()

    with sqlite3.connect(str(db_path)) as conn:
        by_src = dict(conn.execute("SELECT source, COUNT(*) FROM batches GROUP BY source").fetchall())
    assert by_src.get("fleet", 0) == 1
    assert by_src.get("browser", 0) == 2


def test_check_selfplay_target_auto_pauses_and_clears(db_path: Path, output_dir: Path) -> None:
    """Cible atteinte → auto-pause + cible effacée (one-shot)."""
    from nanozero_jobserver.storage.control import (
        get_selfplay_target,
        is_selfplay_paused,
        set_selfplay_target,
    )

    flusher = FlusherService(
        db_path, output_dir, flush_threshold=1000, get_current_model_version=lambda: 1
    )
    insert_positions(db_path, [_mk_position(f"g{i}", 1) for i in range(3)])  # 3 fleet live, mv1
    set_selfplay_target(db_path, 3)  # cible atteinte pile
    assert is_selfplay_paused(db_path) is False
    flusher._check_selfplay_target()
    assert is_selfplay_paused(db_path) is True  # auto-pause
    assert get_selfplay_target(db_path)[0] is None  # one-shot : effacée


def test_check_selfplay_target_below_does_not_pause(db_path: Path, output_dir: Path) -> None:
    from nanozero_jobserver.storage.control import is_selfplay_paused, set_selfplay_target

    flusher = FlusherService(
        db_path, output_dir, flush_threshold=1000, get_current_model_version=lambda: 1
    )
    insert_positions(db_path, [_mk_position(f"g{i}", 1) for i in range(2)])
    set_selfplay_target(db_path, 1000)  # loin
    flusher._check_selfplay_target()
    assert is_selfplay_paused(db_path) is False
