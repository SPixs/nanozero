"""Anti-regression tests for cross-platform dtype consistency in export_npz meta.

Bug context: np.array(int) without explicit dtype produces int32 on Windows
(default for system 'int') vs int64 on Linux. NpzReader.java accepts only
<i8 for int, <f4 for float, <U/|S for string. Any other dtype raises
IllegalArgumentException at .npz load time.

Detected during phase 12-deploy W3090 validation 2026-05-14. Fix forces
np.int64 for all meta int (and bool cast to int64), np.float32 for float.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from nanozero_training.network.export_npz import export_to_npz
from nanozero_training.network.init import init_kaiming_standard
from nanozero_training.network.resnet import NanoZeroResNet


def _make_model() -> NanoZeroResNet:
    model = NanoZeroResNet()  # default 8 blocks x 96 channels = 42 weight tensors
    init_kaiming_standard(model, seed=42)
    return model


def test_export_npz_meta_int_is_i8_cross_platform(tmp_path: Path) -> None:
    """Custom meta int values must be stored as int64 (<i8) regardless of OS.

    Without explicit dtype, np.array(int) uses platform default:
    int32 on Windows, int64 on Linux. This breaks Java NpzReader which
    only accepts <i8 for int.
    """
    out = tmp_path / "test.npz"
    export_to_npz(
        _make_model(),
        out,
        meta={"seed": 42, "n_blocks": 2, "channels": 16, "generation": 1},
    )
    with np.load(out) as data:
        assert (
            data["_meta_seed"].dtype == np.int64
        ), f"_meta_seed: expected int64, got {data['_meta_seed'].dtype}"
        assert data["_meta_n_blocks"].dtype == np.int64
        assert data["_meta_channels"].dtype == np.int64
        assert data["_meta_generation"].dtype == np.int64
        # training_step (already explicit int64 pre-fix) must remain int64
        assert data["_meta_training_step"].dtype == np.int64


def test_export_npz_meta_float_is_f4_cross_platform(tmp_path: Path) -> None:
    """Custom meta float values must be stored as float32 (<f4).

    NpzReader.java accepts only <f4 for floats. Python float (which is
    float64 under the hood) without explicit dtype would produce <f8.
    """
    out = tmp_path / "test.npz"
    export_to_npz(
        _make_model(),
        out,
        meta={"learning_rate": 0.001, "elo_diff": 11.3},
    )
    with np.load(out) as data:
        assert data["_meta_learning_rate"].dtype == np.float32
        assert data["_meta_elo_diff"].dtype == np.float32


def test_export_npz_meta_bool_cast_to_int64(tmp_path: Path) -> None:
    """Bool must be cast to int64 (NpzReader refuses |b1 dtype).

    isinstance ordering matters: `bool` check MUST precede `int` check,
    since `isinstance(True, int)` is True in Python. The fix uses
    int(bool_value) cast to ensure semantic 0/1 preservation.
    """
    out = tmp_path / "test.npz"
    export_to_npz(
        _make_model(),
        out,
        meta={"is_promoted": True, "value_ok": False},
    )
    with np.load(out) as data:
        # bool → int64 cast (NpzReader refuses |b1)
        assert data["_meta_is_promoted"].dtype == np.int64
        assert data["_meta_value_ok"].dtype == np.int64
        # Value semantics preserved (True → 1, False → 0)
        assert int(data["_meta_is_promoted"].item()) == 1
        assert int(data["_meta_value_ok"].item()) == 0


def test_export_npz_meta_str_passthrough(tmp_path: Path) -> None:
    """String meta values pass through to <U... (NpzReader accepts <U/|S)."""
    out = tmp_path / "test.npz"
    export_to_npz(
        _make_model(),
        out,
        meta={"author": "nanozero", "version": "1.0.0"},
    )
    with np.load(out) as data:
        # Strings: kind 'U' (unicode) or 'S' (bytes), both accepted by NpzReader
        assert data["_meta_author"].dtype.kind in ("U", "S")
        assert data["_meta_version"].dtype.kind in ("U", "S")
        assert str(data["_meta_author"].item()) == "nanozero"
        assert str(data["_meta_version"].item()) == "1.0.0"
