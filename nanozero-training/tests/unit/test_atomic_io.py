"""Tests for utils/atomic_io.py — atomic write helpers (ADR-009)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml
from nanozero_training.utils.atomic_io import (
    atomic_append_csv,
    atomic_remove_if_exists,
    atomic_rename,
    atomic_write_npz,
    atomic_write_npz_uncompressed,
    atomic_write_torch,
    atomic_write_yaml,
)

# ----- atomic_write_npz -----


def test_atomic_write_npz_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "out.npz"
    atomic_write_npz(target, arr=np.array([1, 2, 3], dtype=np.int32))
    assert target.exists()
    loaded = np.load(target)
    assert "arr" in loaded.files
    np.testing.assert_array_equal(loaded["arr"], np.array([1, 2, 3], dtype=np.int32))


def test_atomic_write_npz_invalid_extension(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    with pytest.raises(ValueError, match="must end with .npz"):
        atomic_write_npz(target, arr=np.array([0]))


def test_atomic_write_npz_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "dir" / "out.npz"
    atomic_write_npz(target, arr=np.array([42]))
    assert target.exists()


def test_atomic_write_npz_no_partial_file_on_failure(tmp_path: Path) -> None:
    """If write fails mid-flight, no .npz file should be left behind."""
    target = tmp_path / "test.npz"
    with (
        patch(
            "nanozero_training.utils.atomic_io.np.savez_compressed",
            side_effect=RuntimeError("simulated crash"),
        ),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        atomic_write_npz(target, data=np.array([1, 2, 3]))

    assert not target.exists(), "Target file should not exist after failure"
    leftover_tmp = list(tmp_path.glob(".*.npz.tmp"))
    assert len(leftover_tmp) == 0, f"Leftover tmp files: {leftover_tmp}"


def test_atomic_write_npz_uncompressed_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "model.npz"
    a = np.random.default_rng(seed=42).standard_normal((5, 3)).astype(np.float32)
    atomic_write_npz_uncompressed(target, weights=a)
    assert target.exists()
    loaded = np.load(target)
    np.testing.assert_array_equal(loaded["weights"], a)


def test_atomic_write_npz_uncompressed_invalid_extension(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end with .npz"):
        atomic_write_npz_uncompressed(tmp_path / "x.bin", arr=np.array([0]))


def test_atomic_write_npz_uncompressed_no_partial_file_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "model.npz"
    with (
        patch(
            "nanozero_training.utils.atomic_io.np.savez",
            side_effect=RuntimeError("boom"),
        ),
        pytest.raises(RuntimeError, match="boom"),
    ):
        atomic_write_npz_uncompressed(target, arr=np.array([1]))
    assert not target.exists()
    assert list(tmp_path.glob(".*.npz.tmp")) == []


# ----- atomic_write_yaml -----


def test_atomic_write_yaml_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "state.yaml"
    data = {"run_id": "test", "current_gen": 3, "phases": ["selfplay", "train"]}
    atomic_write_yaml(target, data)
    assert target.exists()
    with target.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded == data


def test_atomic_write_yaml_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "config.yaml"
    atomic_write_yaml(target, {"k": 1})
    assert target.exists()


def test_atomic_write_yaml_no_partial_file_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "state.yaml"
    with (
        patch(
            "nanozero_training.utils.atomic_io.yaml.safe_dump",
            side_effect=RuntimeError("yaml-boom"),
        ),
        pytest.raises(RuntimeError, match="yaml-boom"),
    ):
        atomic_write_yaml(target, {"x": 1})
    assert not target.exists()
    assert list(tmp_path.glob(".*.yaml.tmp")) == []


# ----- atomic_append_csv -----


def test_atomic_append_csv_creates_file_with_header(tmp_path: Path) -> None:
    target = tmp_path / "metrics.csv"
    atomic_append_csv(target, ["2026-05-13", 1, 0.5], header=["date", "gen", "loss"])
    atomic_append_csv(target, ["2026-05-14", 2, 0.42], header=["date", "gen", "loss"])
    content = target.read_text(encoding="utf-8")
    assert content.startswith("date,gen,loss\r\n") or content.startswith("date,gen,loss\n")
    assert "2026-05-13,1,0.5" in content
    assert "2026-05-14,2,0.42" in content
    # Header appears only once
    assert content.count("date,gen,loss") == 1


def test_atomic_append_csv_no_header_when_already_exists(tmp_path: Path) -> None:
    target = tmp_path / "metrics.csv"
    target.write_text("h1,h2\nrow0,0\n", encoding="utf-8")
    atomic_append_csv(target, ["row1", 1], header=["h1", "h2"])
    content = target.read_text(encoding="utf-8")
    assert content.count("h1,h2") == 1  # No duplicate header


# ----- atomic_rename / atomic_remove_if_exists -----


def test_atomic_rename_basic(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("hello", encoding="utf-8")
    atomic_rename(src, dst)
    assert not src.exists()
    assert dst.exists()
    assert dst.read_text(encoding="utf-8") == "hello"


def test_atomic_rename_replaces_existing(tmp_path: Path) -> None:
    src = tmp_path / "src.txt"
    dst = tmp_path / "dst.txt"
    src.write_text("new", encoding="utf-8")
    dst.write_text("old", encoding="utf-8")
    atomic_rename(src, dst)
    assert dst.read_text(encoding="utf-8") == "new"


def test_atomic_remove_if_exists_present(tmp_path: Path) -> None:
    f = tmp_path / "to-remove.txt"
    f.write_text("bye", encoding="utf-8")
    assert atomic_remove_if_exists(f) is True
    assert not f.exists()


def test_atomic_remove_if_exists_absent(tmp_path: Path) -> None:
    assert atomic_remove_if_exists(tmp_path / "missing.txt") is False


# ----- atomic_write_torch -----


def test_atomic_write_torch_roundtrip(tmp_path: Path) -> None:
    target = tmp_path / "state.pt"
    obj = {"x": torch.tensor([1, 2, 3]), "y": 42}
    atomic_write_torch(obj, target)
    assert target.exists()
    loaded = torch.load(target, weights_only=False)
    assert torch.equal(loaded["x"], torch.tensor([1, 2, 3]))
    assert loaded["y"] == 42


def test_atomic_write_torch_no_partial_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "state.pt"
    with (
        patch("torch.save", side_effect=RuntimeError("simulated crash")),
        pytest.raises(RuntimeError, match="simulated crash"),
    ):
        atomic_write_torch({"x": torch.tensor([0])}, target)
    assert not target.exists()
    leftover_tmp = list(tmp_path.glob(".*.pt.tmp"))
    assert leftover_tmp == []


def test_atomic_write_torch_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "state.pt"
    atomic_write_torch({"x": 1}, target)
    assert target.exists()
