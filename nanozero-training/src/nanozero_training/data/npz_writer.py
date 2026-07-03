"""Atomic batched writer for self-play .npz files (ADR-002).

Writes a list of Samples to a single .npz file using atomic_write_npz
(write tmp + rename pattern from utils.atomic_io).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from nanozero_training.data.sample import Sample
from nanozero_training.utils.atomic_io import atomic_write_npz

FORMAT_VERSION = "1.0"
WRITER_VERSION = "1.0.0"


def write_batch(
    samples: list[Sample],
    path: str | Path,
    gen: int,
    batch_idx: int,
    n_games: int,
    extra_meta: dict[str, Any] | None = None,
) -> None:
    """Atomically write a batch of samples to a `.npz` file (ADR-002).

    Args:
        samples: list of validated Sample instances. Must be non-empty.
        path: target .npz file path.
        gen: generation index (1-based, eg. 1 for selfplay during gen-001).
        batch_idx: batch index within the generation (0-based).
        n_games: number of self-play games that produced these samples.
        extra_meta: optional extra metadata (will be _meta_<key>=<value>).

    Raises:
        ValueError: if samples is empty.
        OSError: on I/O errors.
    """
    if not samples:
        raise ValueError("Cannot write empty batch")

    n = len(samples)

    # Stack arrays (N, ...).
    input_planes = np.stack([s.input_planes for s in samples], axis=0)
    policy_target = np.stack([s.policy_target for s in samples], axis=0)
    value_target = np.array([s.value_target for s in samples], dtype=np.float32)
    turn = np.array([s.turn for s in samples], dtype=np.int8)
    ply = np.array([s.ply for s in samples], dtype=np.int16)

    # Metadata as 0-d arrays.
    meta: dict[str, Any] = {
        "_meta_gen": np.array(gen),
        "_meta_batch": np.array(batch_idx),
        "_meta_n_samples": np.array(n),
        "_meta_n_games": np.array(n_games),
        "_meta_export_date": np.array(datetime.now(timezone.utc).isoformat()),
        "_meta_writer_version": np.array(WRITER_VERSION),
        "_meta_format_version": np.array(FORMAT_VERSION),
    }
    if extra_meta is not None:
        for k, v in extra_meta.items():
            meta[f"_meta_{k}"] = np.array(v)

    atomic_write_npz(
        path,
        input_planes=input_planes,
        policy_target=policy_target,
        value_target=value_target,
        turn=turn,
        ply=ply,
        **meta,
    )


def make_batch_filename(gen: int, batch_idx: int) -> str:
    """Standard naming convention for selfplay-data .npz files.

    Returns:
        'selfplay-genNNN-batch-MMM.npz' (zero-padded 3 digits each).
    """
    return f"selfplay-gen{gen:03d}-batch-{batch_idx:03d}.npz"
