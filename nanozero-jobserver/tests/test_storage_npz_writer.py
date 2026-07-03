"""Tests storage/npz_writer.py — atomic NPZ shard writer compat NpzDataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from nanozero_jobserver.storage.npz_writer import (
    INPUT_PLANES_SHAPE,
    POLICY_LEN,
    atomic_write_npz_shard,
)
from nanozero_jobserver.storage.replay_buffer import PositionRow


def _mk_row(
    row_id: int,
    game_id: str = "g0",
    model_version: int = 1,
    ply: int = 0,
    outcome: float = 0.0,
) -> PositionRow:
    """PositionRow with valid-sized BLOBs (float32 little-endian)."""
    input_planes = np.zeros(INPUT_PLANES_SHAPE, dtype="<f4")
    policy_target = np.zeros(POLICY_LEN, dtype="<f4")
    return PositionRow(
        id=row_id,
        game_id=game_id,
        model_version=model_version,
        ply=ply,
        fen="fen",
        input_planes=input_planes.tobytes(),
        policy_target=policy_target.tobytes(),
        outcome=outcome,
    )


def test_write_creates_file_with_correct_name(tmp_path: Path) -> None:
    rows = [_mk_row(1)]
    p = atomic_write_npz_shard(tmp_path, model_version=5, batch_idx=2, positions=rows)
    assert p.name == "selfplay-gen005-batch-002.npz"
    assert p.exists()


def test_write_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "deep" / "nested" / "datasets"
    atomic_write_npz_shard(target, model_version=1, batch_idx=0, positions=[_mk_row(1)])
    assert target.exists()


def test_npz_has_all_required_keys(tmp_path: Path) -> None:
    rows = [_mk_row(i) for i in range(3)]
    p = atomic_write_npz_shard(tmp_path, 1, 0, rows)
    with np.load(p) as d:
        expected_keys = {
            "input_planes",
            "policy_target",
            "value_target",
            "turn",
            "ply",
            "_meta_n_samples",
            "_meta_model_version",
            "_meta_batch_idx",
        }
        assert set(d.files) == expected_keys


def test_npz_compatible_with_npz_dataset_meta(tmp_path: Path) -> None:
    """_meta_n_samples key expected by NpzDataset.__init__ (data/dataset.py:41)."""
    rows = [_mk_row(i) for i in range(7)]
    p = atomic_write_npz_shard(tmp_path, 1, 0, rows)
    with np.load(p) as d:
        assert int(d["_meta_n_samples"]) == 7


def test_npz_shapes(tmp_path: Path) -> None:
    rows = [_mk_row(i) for i in range(5)]
    p = atomic_write_npz_shard(tmp_path, 1, 0, rows)
    with np.load(p) as d:
        assert d["input_planes"].shape == (5, 119, 8, 8)
        assert d["input_planes"].dtype == np.float32
        assert d["policy_target"].shape == (5, 4672)
        assert d["policy_target"].dtype == np.float32
        assert d["value_target"].shape == (5,)
        assert d["value_target"].dtype == np.float32
        assert d["turn"].shape == (5,)
        assert d["turn"].dtype == np.int64
        assert d["ply"].shape == (5,)
        assert d["ply"].dtype == np.int64


def test_value_target_equals_outcome(tmp_path: Path) -> None:
    """outcome (already POV side-to-move) → value_target verbatim."""
    rows = [
        _mk_row(1, outcome=1.0),
        _mk_row(2, outcome=-1.0),
        _mk_row(3, outcome=0.0),
    ]
    p = atomic_write_npz_shard(tmp_path, 1, 0, rows)
    with np.load(p) as d:
        assert list(d["value_target"]) == [1.0, -1.0, 0.0]


def test_turn_derived_from_ply(tmp_path: Path) -> None:
    """turn = ply % 2 (ply=0 → WHITE=0, ply=1 → BLACK=1)."""
    rows = [_mk_row(i, ply=i) for i in range(4)]
    p = atomic_write_npz_shard(tmp_path, 1, 0, rows)
    with np.load(p) as d:
        assert list(d["turn"]) == [0, 1, 0, 1]
        assert list(d["ply"]) == [0, 1, 2, 3]


def test_meta_values(tmp_path: Path) -> None:
    rows = [_mk_row(i) for i in range(3)]
    p = atomic_write_npz_shard(tmp_path, model_version=7, batch_idx=42, positions=rows)
    with np.load(p) as d:
        assert int(d["_meta_n_samples"]) == 3
        assert int(d["_meta_model_version"]) == 7
        assert int(d["_meta_batch_idx"]) == 42


def test_empty_positions_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty NPZ shard"):
        atomic_write_npz_shard(tmp_path, 1, 0, [])


def test_invalid_input_planes_size_raises(tmp_path: Path) -> None:
    bad = PositionRow(
        id=1,
        game_id="g",
        model_version=1,
        ply=0,
        fen="fen",
        input_planes=b"\x00" * 100,  # WRONG size
        policy_target=b"\x00" * (POLICY_LEN * 4),
        outcome=0.0,
    )
    with pytest.raises(ValueError, match="input_planes BLOB size"):
        atomic_write_npz_shard(tmp_path, 1, 0, [bad])


def test_invalid_policy_target_size_raises(tmp_path: Path) -> None:
    bad = PositionRow(
        id=1,
        game_id="g",
        model_version=1,
        ply=0,
        fen="fen",
        input_planes=b"\x00" * (119 * 8 * 8 * 4),
        policy_target=b"\x00" * 100,  # WRONG size
        outcome=0.0,
    )
    with pytest.raises(ValueError, match="policy_target BLOB size"):
        atomic_write_npz_shard(tmp_path, 1, 0, [bad])


def test_atomic_write_no_tmp_leftover_on_success(tmp_path: Path) -> None:
    """After success, no .tmp file in output_dir."""
    rows = [_mk_row(1)]
    atomic_write_npz_shard(tmp_path, 1, 0, rows)
    tmps = list(tmp_path.glob(".*tmp*")) + list(tmp_path.glob("*.tmp"))
    assert tmps == []


def test_atomic_write_no_tmp_leftover_on_failure(tmp_path: Path) -> None:
    """On exception (bad BLOB), tmp file cleaned up."""
    bad = PositionRow(
        id=1,
        game_id="g",
        model_version=1,
        ply=0,
        fen="fen",
        input_planes=b"\x00" * 100,
        policy_target=b"\x00" * (POLICY_LEN * 4),
        outcome=0.0,
    )
    with pytest.raises(ValueError):
        atomic_write_npz_shard(tmp_path, 1, 0, [bad])
    tmps = list(tmp_path.glob(".*.tmp"))
    assert tmps == []
