"""Atomic NPZ shard writer for ADR-015 write-through cache.

Builds NPZ files compatible with nanozero_training.data.dataset.NpzDataset :
  required keys : input_planes, policy_target, value_target, turn, ply
  required meta : _meta_n_samples (NpzDataset reads it in __init__)

Workers POST positions with side-to-move POV outcome already applied
(cf. ADR-001 + worker/GamePlayer.java:216). Storage layer keeps that
convention : `outcome` field == value_target POV side-to-move.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np

from nanozero_jobserver.storage.replay_buffer import PositionRow

INPUT_PLANES_SHAPE = (119, 8, 8)
POLICY_LEN = 4672


def atomic_write_npz_shard(
    output_dir: Path,
    model_version: int,
    batch_idx: int,
    positions: list[PositionRow],
    name_prefix: str = "selfplay",
) -> Path:
    """Build and atomically write one NPZ shard from a list of PositionRows.

    File naming : ``{name_prefix}-gen{model_version:03d}-batch-{batch_idx:03d}.npz``.
    Chantier 1 cloisonnement : prefix='selfplay' (fleet, corpus d'entraînement) vs
    'browser' (cohorte navigateur, séparée — le glob training ne ramasse que 'selfplay-gen*').

    Format (compat NpzDataset, cf. nanozero_training/data/dataset.py:35) :
        input_planes  : (N, 119, 8, 8) float32, decoded from BLOB '<f4'
        policy_target : (N, 4672)     float32, decoded from BLOB '<f4'
        value_target  : (N,)          float32, == outcome (side-to-move POV)
        turn          : (N,)          int64,   == ply % 2
        ply           : (N,)          int64
        _meta_n_samples       : int64
        _meta_model_version   : int64
        _meta_batch_idx       : int64

    Atomic write : tmp file in same directory + os.replace (POSIX atomic).
    Trainer readers never see a partial file.

    Args:
        output_dir: target directory. Created if absent.
        model_version: gen number for filename (1 → gen-001).
        batch_idx: per-version sequence (0 → batch-000).
        positions: rows to pack. MUST be non-empty.

    Returns:
        Absolute path of the written NPZ.

    Raises:
        ValueError: if positions is empty or BLOB sizes don't match expected shapes.
        OSError: on filesystem I/O failures.
    """
    if not positions:
        raise ValueError("Cannot write empty NPZ shard")

    n = len(positions)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_name = f"{name_prefix}-gen{model_version:03d}-batch-{batch_idx:03d}.npz"
    target_path = output_dir / target_name

    # Decode BLOBs into numpy arrays.
    input_planes = np.empty((n, *INPUT_PLANES_SHAPE), dtype=np.float32)
    policy_target = np.empty((n, POLICY_LEN), dtype=np.float32)
    value_target = np.empty(n, dtype=np.float32)
    turn = np.empty(n, dtype=np.int64)
    ply = np.empty(n, dtype=np.int64)

    for i, p in enumerate(positions):
        ip = np.frombuffer(p.input_planes, dtype="<f4")
        if ip.size != np.prod(INPUT_PLANES_SHAPE):
            raise ValueError(
                f"Position {p.id} : input_planes BLOB size {ip.size} != "
                f"{int(np.prod(INPUT_PLANES_SHAPE))} expected"
            )
        input_planes[i] = ip.reshape(INPUT_PLANES_SHAPE)

        pt = np.frombuffer(p.policy_target, dtype="<f4")
        if pt.size != POLICY_LEN:
            raise ValueError(
                f"Position {p.id} : policy_target BLOB size {pt.size} != " f"{POLICY_LEN} expected"
            )
        policy_target[i] = pt

        value_target[i] = p.outcome  # already POV side-to-move (ADR-001)
        ply[i] = p.ply
        turn[i] = p.ply % 2

    # Atomic write : same-dir tmp file + os.replace.
    with tempfile.NamedTemporaryFile(
        dir=output_dir,
        prefix=f".{target_name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tmp_path.open("wb") as f:
            np.savez_compressed(
                f,
                input_planes=input_planes,
                policy_target=policy_target,
                value_target=value_target,
                turn=turn,
                ply=ply,
                _meta_n_samples=np.int64(n),
                _meta_model_version=np.int64(model_version),
                _meta_batch_idx=np.int64(batch_idx),
            )
        os.replace(tmp_path, target_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return target_path
