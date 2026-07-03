"""Tests for data/npz_writer.py — atomic batched writer (ADR-002)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from nanozero_training.data.npz_writer import (
    FORMAT_VERSION,
    WRITER_VERSION,
    make_batch_filename,
    write_batch,
)
from nanozero_training.data.sample import (
    BOARD_SIZE,
    N_INPUT_PLANES,
    N_POLICY,
    Sample,
    make_valid_sample_arrays,
)


def _make_sample(ply: int = 0, turn: int = 0, value: float = 0.0) -> Sample:
    args = make_valid_sample_arrays()
    args["ply"] = ply
    args["turn"] = turn
    args["value_target"] = value
    return Sample(**args)


def test_write_batch_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "batch.npz"
    write_batch(
        samples=[_make_sample()],
        path=target,
        gen=1,
        batch_idx=0,
        n_games=1,
    )
    assert target.exists()


def test_write_batch_empty_raises(tmp_path: Path) -> None:
    target = tmp_path / "batch.npz"
    with pytest.raises(ValueError, match="empty batch"):
        write_batch(samples=[], path=target, gen=1, batch_idx=0, n_games=0)


def test_write_batch_single_sample(tmp_path: Path) -> None:
    target = tmp_path / "batch.npz"
    write_batch(
        samples=[_make_sample(ply=5, turn=1, value=1.0)],
        path=target,
        gen=2,
        batch_idx=3,
        n_games=1,
    )
    with np.load(target) as data:
        assert data["input_planes"].shape == (1, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
        assert data["policy_target"].shape == (1, N_POLICY)
        assert data["value_target"].shape == (1,)
        assert data["turn"].shape == (1,)
        assert data["ply"].shape == (1,)
        assert int(data["ply"].item()) == 5
        assert int(data["turn"].item()) == 1
        assert float(data["value_target"].item()) == 1.0


def test_write_batch_n_samples_stack_shape(tmp_path: Path) -> None:
    n = 100
    samples = [_make_sample(ply=i) for i in range(n)]
    target = tmp_path / "batch.npz"
    write_batch(samples=samples, path=target, gen=1, batch_idx=0, n_games=10)
    with np.load(target) as data:
        assert data["input_planes"].shape == (n, N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
        assert data["policy_target"].shape == (n, N_POLICY)
        assert data["value_target"].shape == (n,)
        assert data["turn"].shape == (n,)
        assert data["ply"].shape == (n,)
        # ply preserved per sample
        np.testing.assert_array_equal(data["ply"], np.arange(n, dtype=np.int16))


def test_write_batch_dtypes_preserved(tmp_path: Path) -> None:
    samples = [_make_sample()]
    target = tmp_path / "batch.npz"
    write_batch(samples=samples, path=target, gen=1, batch_idx=0, n_games=1)
    with np.load(target) as data:
        assert data["input_planes"].dtype == np.float32
        assert data["policy_target"].dtype == np.float32
        assert data["value_target"].dtype == np.float32
        assert data["turn"].dtype == np.int8
        assert data["ply"].dtype == np.int16


def test_write_batch_roundtrip_arrays(tmp_path: Path) -> None:
    samples = [
        _make_sample(ply=0, turn=0, value=-1.0),
        _make_sample(ply=1, turn=1, value=0.0),
        _make_sample(ply=2, turn=0, value=1.0),
    ]
    target = tmp_path / "batch.npz"
    write_batch(samples=samples, path=target, gen=1, batch_idx=0, n_games=1)
    with np.load(target) as data:
        # Sample 0
        np.testing.assert_array_equal(data["input_planes"][0], samples[0].input_planes)
        np.testing.assert_array_equal(data["policy_target"][0], samples[0].policy_target)
        assert float(data["value_target"][0]) == -1.0
        assert int(data["turn"][0]) == 0
        assert int(data["ply"][0]) == 0
        # Sample 2
        assert float(data["value_target"][2]) == 1.0
        assert int(data["turn"][2]) == 0
        assert int(data["ply"][2]) == 2


def test_write_batch_roundtrip_metadata(tmp_path: Path) -> None:
    target = tmp_path / "batch.npz"
    write_batch(
        samples=[_make_sample()],
        path=target,
        gen=7,
        batch_idx=42,
        n_games=250,
    )
    with np.load(target) as data:
        assert int(data["_meta_gen"].item()) == 7
        assert int(data["_meta_batch"].item()) == 42
        assert int(data["_meta_n_samples"].item()) == 1
        assert int(data["_meta_n_games"].item()) == 250
        assert str(data["_meta_writer_version"].item()) == WRITER_VERSION
        assert str(data["_meta_format_version"].item()) == FORMAT_VERSION
        # _meta_export_date is an ISO 8601 string
        export_date = str(data["_meta_export_date"].item())
        assert export_date.startswith("20")  # ISO 8601 century
        assert "T" in export_date


def test_write_batch_extra_meta(tmp_path: Path) -> None:
    target = tmp_path / "batch.npz"
    write_batch(
        samples=[_make_sample()],
        path=target,
        gen=1,
        batch_idx=0,
        n_games=1,
        extra_meta={"worker_seed": 12345, "model_hash": "abc123"},
    )
    with np.load(target) as data:
        assert int(data["_meta_worker_seed"].item()) == 12345
        assert str(data["_meta_model_hash"].item()) == "abc123"


def test_write_batch_atomic_on_failure(tmp_path: Path) -> None:
    """Mock crash mid-write : no half-written file should remain."""
    target = tmp_path / "batch.npz"
    with (
        patch(
            "nanozero_training.utils.atomic_io.np.savez_compressed",
            side_effect=RuntimeError("simulated crash"),
        ),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        write_batch(
            samples=[_make_sample()],
            path=target,
            gen=1,
            batch_idx=0,
            n_games=1,
        )
    assert not target.exists()
    assert list(tmp_path.glob(".*.npz.tmp")) == []


def test_make_batch_filename() -> None:
    assert make_batch_filename(3, 17) == "selfplay-gen003-batch-017.npz"
    assert make_batch_filename(1, 0) == "selfplay-gen001-batch-000.npz"
    assert make_batch_filename(999, 999) == "selfplay-gen999-batch-999.npz"
