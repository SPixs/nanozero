"""ADR-015 migration tool : flush pre-existing positions to NPZ shards.

Usage:
  python tools/migrate_existing_db.py \\
    --db $HOME/nanozero-night/jobserver.db \\
    --output-dir $HOME/nanozero-night/datasets \\
    [--shard-size 100000] \\
    [--purge] \\
    [--vacuum]

What it does :
  1. Apply schema migration (idempotent init_schema) : add flushed_to_npz
     column + batches table on pre-ADR-015 DBs.
  2. Use FlusherService to write NPZ shards from existing unflushed positions.
     Default shard size 100k, configurable via --shard-size.
  3. If --purge : DELETE flushed positions to shrink the DB.
  4. If --vacuum : VACUUM the SQLite file to reclaim disk space (slow on large
     DBs, but reclaims actual MB).
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

from nanozero_jobserver.services.flusher import FlusherService
from nanozero_jobserver.storage.db import init_schema
from nanozero_jobserver.storage.replay_buffer import (
    count_unflushed_positions,
    delete_flushed_old,
)

LOG = logging.getLogger("migrate")


def migrate(
    db_path: Path,
    output_dir: Path,
    shard_size: int = 100_000,
    purge: bool = False,
    vacuum: bool = False,
) -> int:
    """Run the migration. Returns total shards written.

    Args:
        db_path: SQLite DB to migrate.
        output_dir: target directory for NPZ shards.
        shard_size: positions per shard (threshold + partial drain target).
        purge: if True, DELETE flushed rows after migration.
        vacuum: if True, VACUUM the DB to reclaim disk space.

    Returns:
        Total shards written.
    """
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found : {db_path}")

    LOG.info("Applying schema migration to %s", db_path)
    init_schema(db_path)

    initial_n = count_unflushed_positions(db_path)
    LOG.info("Unflushed positions to migrate : %d", initial_n)
    if initial_n == 0:
        LOG.info("Nothing to do.")
        return 0

    flusher = FlusherService(
        db_path=db_path,
        output_dir=output_dir,
        flush_threshold=shard_size,
        retention_window=0,
        get_current_model_version=None,  # disable purge inside FlusherService
    )

    total_shards = 0
    # Drain full shards in chunks.
    while count_unflushed_positions(db_path) >= shard_size:
        shards = flusher.tick()
        total_shards += shards
        remaining = count_unflushed_positions(db_path)
        LOG.info("Wrote %d shard(s) — remaining unflushed : %d", shards, remaining)
        if shards == 0:
            LOG.warning("tick() wrote 0 shards but threshold reached — break")
            break

    # Final partial drain (last shard < shard_size).
    shards = flusher.force_flush()
    total_shards += shards
    LOG.info("Drained remaining positions via force_flush() : %d final shard(s)", shards)

    if purge:
        # Purge ALL flushed (delete_flushed_old with very-high cutoff).
        deleted = delete_flushed_old(db_path, max_model_version=10**9)
        LOG.info("Purged %d flushed rows from positions table", deleted)

    if vacuum:
        LOG.info("VACUUM (this may take a while on large DBs)...")
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute("VACUUM")
        LOG.info("VACUUM complete")

    final_n = count_unflushed_positions(db_path)
    LOG.info(
        "Migration complete. Total shards : %d. Unflushed remaining : %d",
        total_shards,
        final_n,
    )
    return int(total_shards)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path, help="SQLite DB path")
    parser.add_argument(
        "--output-dir", required=True, type=Path, help="Target directory for NPZ shards"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100_000,
        help="Positions per NPZ shard (default 100_000)",
    )
    parser.add_argument(
        "--purge",
        action="store_true",
        help="DELETE flushed positions from SQLite after migration",
    )
    parser.add_argument(
        "--vacuum",
        action="store_true",
        help="Run VACUUM to reclaim disk space (slow on large DBs)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        stream=sys.stderr,
    )
    migrate(
        db_path=args.db,
        output_dir=args.output_dir,
        shard_size=args.shard_size,
        purge=args.purge,
        vacuum=args.vacuum,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
