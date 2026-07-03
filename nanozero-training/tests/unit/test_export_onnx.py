"""Tests for network/export_onnx.py — denormal cleaning + companion export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from nanozero_training.network.export_npz import export_to_npz
from nanozero_training.network.export_onnx import (
    _FLOAT32_MIN_NORMAL,
    _zero_denormals_in_module,
    export_onnx_companion,
)
from nanozero_training.network.init import init_kaiming_standard
from nanozero_training.network.resnet import NanoZeroResNet
from torch import nn


def _make_module_with(values: list[float]) -> nn.Module:
    """Build a 1-parameter module whose flat weight equals `values`."""
    m = nn.Linear(len(values), 1, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.tensor(values, dtype=torch.float32).reshape(1, -1))
    return m


# ----------------------------------------------------------- _zero_denormals_in_module


def test_zero_denormals_returns_zero_when_no_denormals() -> None:
    m = _make_module_with([1.0, -0.5, 1e-30, -1e-30, 0.0])
    assert _zero_denormals_in_module(m) == 0


def test_zero_denormals_counts_and_zeros_subnormals() -> None:
    # 2 normal, 3 denormals (1e-40, -2e-40, 5e-41), 1 zero.
    m = _make_module_with([1.0, -0.5, 1e-40, -2e-40, 5e-41, 0.0])
    n = _zero_denormals_in_module(m)
    assert n == 3
    flat = m.weight.flatten().tolist()
    assert flat[0] == pytest.approx(1.0)
    assert flat[1] == pytest.approx(-0.5)
    assert flat[2] == 0.0
    assert flat[3] == 0.0
    assert flat[4] == 0.0
    assert flat[5] == 0.0


def test_zero_denormals_preserves_existing_zeros() -> None:
    m = _make_module_with([0.0] * 10)
    assert _zero_denormals_in_module(m) == 0


def test_zero_denormals_does_not_touch_non_float32() -> None:
    """Parameters with non-float32 dtypes must be left alone."""

    class IntModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.idx = nn.Parameter(
                torch.tensor([0, 1, 2, 3], dtype=torch.int64), requires_grad=False
            )

    m = IntModule()
    assert _zero_denormals_in_module(m) == 0
    assert m.idx.tolist() == [0, 1, 2, 3]


def test_zero_denormals_boundary_exactly_at_min_normal() -> None:
    """Values exactly at FLT_MIN are NORMAL, not denormal — must be preserved."""
    m = _make_module_with([_FLOAT32_MIN_NORMAL, -_FLOAT32_MIN_NORMAL])
    assert _zero_denormals_in_module(m) == 0
    flat = m.weight.flatten().tolist()
    assert flat[0] == pytest.approx(_FLOAT32_MIN_NORMAL, rel=1e-6)
    assert flat[1] == pytest.approx(-_FLOAT32_MIN_NORMAL, rel=1e-6)


def test_zero_denormals_counts_across_multiple_parameters() -> None:
    """Module with several parameters : denormals from all of them are tallied."""

    class TwoLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(2, 1, bias=True)
            self.fc2 = nn.Linear(2, 1, bias=True)
            with torch.no_grad():
                self.fc1.weight.copy_(torch.tensor([[1e-40, 1.0]]))
                self.fc1.bias.copy_(torch.tensor([2e-40]))
                self.fc2.weight.copy_(torch.tensor([[1.0, 1.0]]))
                self.fc2.bias.copy_(torch.tensor([0.0]))

    m = TwoLayer()
    assert _zero_denormals_in_module(m) == 2
    assert m.fc1.weight.tolist() == [[0.0, 1.0]]
    assert m.fc1.bias.tolist() == [0.0]


# ------------------------------------------------------- export_onnx_companion integration


def test_export_onnx_companion_runs_end_to_end(tmp_path: Path) -> None:
    """End-to-end : init small net → export .npz → export_onnx_companion → valid .onnx."""
    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)

    npz_path = tmp_path / "model.npz"
    export_to_npz(model, npz_path)

    onnx_path = export_onnx_companion(npz_path)

    assert onnx_path.exists()
    assert onnx_path.suffix == ".onnx"
    assert onnx_path.stat().st_size > 0


def test_export_onnx_companion_no_denormals_in_kaiming_init(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A freshly Kaiming-initialised net has no denormals — no warning emitted."""
    import logging

    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)

    npz_path = tmp_path / "model.npz"
    export_to_npz(model, npz_path)

    with caplog.at_level(logging.WARNING, logger="nanozero_training.network.export_onnx"):
        export_onnx_companion(npz_path)
    assert not any(
        "Zeroed" in rec.message for rec in caplog.records
    ), "Kaiming-init should produce no denormals"


def test_export_onnx_companion_missing_npz_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        export_onnx_companion(tmp_path / "ghost.npz")


def test_zero_denormals_preserves_dtype() -> None:
    """After cleaning, parameter dtype must still be float32 (no upcast)."""
    m = _make_module_with([1.0, 1e-40, -2e-40])
    _zero_denormals_in_module(m)
    assert m.weight.dtype == torch.float32


def test_zero_denormals_with_numpy_array_data() -> None:
    """Sanity : weight init from a numpy array containing denormals is detected."""
    arr = np.array([1.0, 1e-40, -2e-40, 0.0, 0.5], dtype=np.float32)
    m = _make_module_with(arr.tolist())
    assert _zero_denormals_in_module(m) == 2
