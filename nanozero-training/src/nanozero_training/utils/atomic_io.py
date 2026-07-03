"""Atomic I/O helpers for resilience (ADR-009).

All file writes for critical artifacts (.npz, .yaml, .csv, .pt) must use these
helpers to guarantee no half-written files on crash/abort.

Pattern: write to tmp file in same directory, then atomic rename to target.
POSIX rename(2) is atomic on same filesystem; NTFS ReplaceFileW provides similar
atomicity. The atomic rename ensures readers never see a partial file.
"""

from __future__ import annotations

import csv
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

LOG = logging.getLogger(__name__)

# Phase 12 hotfix-007 — Windows-specific retry on file rename collision.
# Background : os.replace() on Windows fails with WinError 5 (PermissionError)
# if the destination file is currently open by ANOTHER process for reading
# (e.g., monitoring dashboard tailing run_state.yaml via Flask SSE).
# On POSIX this never happens (rename(2) always succeeds even if file open).
# Fix : retry briefly with exponential backoff capped at ~100ms to absorb the
# typical reader window without significantly delaying writes.
_REPLACE_MAX_RETRIES = 10
_REPLACE_BASE_DELAY_S = 0.02
_REPLACE_MAX_DELAY_S = 0.1


def _replace_with_retry(src: Path, dst: Path) -> None:
    """Cross-platform os.replace with Windows-aware retry on PermissionError."""
    for attempt in range(_REPLACE_MAX_RETRIES):
        try:
            os.replace(src, dst)
            if attempt > 0:
                LOG.debug("atomic rename succeeded on attempt %d for %s", attempt + 1, dst)
            return
        except PermissionError as e:
            if attempt == _REPLACE_MAX_RETRIES - 1:
                LOG.error(
                    "atomic rename %s -> %s failed after %d retries: %s",
                    src,
                    dst,
                    _REPLACE_MAX_RETRIES,
                    e,
                )
                raise
            delay = min(_REPLACE_BASE_DELAY_S * (2**attempt), _REPLACE_MAX_DELAY_S)
            time.sleep(delay)


def atomic_write_npz(path: str | Path, **arrays: Any) -> None:
    """Atomically write a `.npz` file with the given arrays.

    Uses np.savez_compressed in a tmp file, then atomic rename.

    Args:
        path: Target file path. Must end with .npz.
        **arrays: Named arrays to save (np.savez_compressed kwargs format).

    Raises:
        OSError: On I/O errors during write or rename.
        ValueError: If path doesn't end with .npz.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"Path must end with .npz, got: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".npz.tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        # Pass a binary file handle (not a path) so np.savez_compressed doesn't
        # silently append ".npz" to a tmp path that doesn't end with ".npz".
        with tmp_path.open("wb") as f:
            np.savez_compressed(f, **arrays)
        _replace_with_retry(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_npz_uncompressed(path: str | Path, **arrays: Any) -> None:
    """Same as atomic_write_npz but uses np.savez (uncompressed).

    Used for model exports where load speed matters more than disk size.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"Path must end with .npz, got: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".npz.tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tmp_path.open("wb") as f:
            np.savez(f, **arrays)
        _replace_with_retry(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_torch(obj: object, path: str | Path) -> None:
    """Atomically save a PyTorch object via torch.save + rename.

    Used for training checkpoints (optimizer state, scheduler state, RNG state,
    full model state_dict) consumed back via torch.load for resume.
    NOT for engine inference — those go through export_to_npz (ADR-002 nn).

    Args:
        obj: any picklable object (typically dict of state_dicts).
        path: target file path (typically .pt).

    Raises:
        OSError: on I/O errors during write or rename.
    """
    import torch  # local import — atomic_io reste utilisable sans torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".pt.tmp",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(obj, tmp_path)
        _replace_with_retry(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_write_yaml(path: str | Path, data: dict[str, Any]) -> None:
    """Atomically write a YAML file from a dict.

    Uses yaml.safe_dump in a tmp file, then atomic rename.

    Args:
        path: Target file path.
        data: Dict to serialize as YAML.

    Raises:
        OSError: On I/O errors.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tmp_fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=".yaml.tmp",
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)
        _replace_with_retry(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def atomic_append_csv(
    path: str | Path,
    row: list[Any],
    header: list[str] | None = None,
) -> None:
    """Append a row to a CSV file (creates with header if file doesn't exist).

    Note: this is NOT fully atomic for the append operation itself — writing
    a single line is small enough that POSIX systems usually keep it atomic
    via OS buffering. For full atomicity on overwrite use atomic_write_yaml-style
    pattern.

    Args:
        path: Target CSV path.
        row: Row values (must be CSV-safe — no embedded newlines).
        header: If file doesn't exist yet, write this header line first.

    Raises:
        OSError: On I/O errors.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists and header is not None:
            writer.writerow(header)
        writer.writerow(row)


def atomic_rename(src: str | Path, dst: str | Path) -> None:
    """Atomically rename a file (or directory).

    Args:
        src: Source path (must exist).
        dst: Destination path (will be replaced if exists).

    Raises:
        OSError: If rename fails (eg. cross-filesystem on POSIX).
    """
    _replace_with_retry(Path(src), Path(dst))


def atomic_remove_if_exists(path: str | Path) -> bool:
    """Remove a file if it exists, return True if removed, False if not found.

    Raises:
        OSError: On other I/O errors (permissions, etc.).
    """
    path = Path(path)
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True
