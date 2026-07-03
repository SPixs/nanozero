"""Unit tests for train/dataloader.py."""

from __future__ import annotations

from pathlib import Path

import pytest
from nanozero_training.data.npz_writer import make_batch_filename, write_batch
from nanozero_training.data.sample import Sample, make_valid_sample_arrays
from nanozero_training.train.config import TrainConfig
from nanozero_training.train.dataloader import (
    make_train_dataloader,
    make_train_dataloader_for_gen,
)


def _make_valid_sample() -> Sample:
    return Sample(**make_valid_sample_arrays())


def _write_batch_at(tmp_path: Path, gen: int, batch: int, n_samples: int = 5) -> Path:
    """Write a tiny .npz batch file in tmp_path."""
    target = tmp_path / make_batch_filename(gen, batch)
    samples = [_make_valid_sample() for _ in range(n_samples)]
    write_batch(samples, target, gen=gen, batch_idx=batch, n_games=1)
    return target


def test_make_train_dataloader_basic(tmp_path: Path) -> None:
    """50 samples + batch_size=16 -> 4 batches (16, 16, 16, 2) drop_last=False."""
    from nanozero_training.data.dataset import NpzDataset

    path = _write_batch_at(tmp_path, gen=1, batch=0, n_samples=50)
    dataset = NpzDataset(path)
    config = TrainConfig(batch_size=16, num_workers=0, pin_memory=False, shuffle=False)
    dl = make_train_dataloader(dataset, config)

    batches = list(dl)
    assert len(batches) == 4
    assert batches[0]["input_planes"].shape[0] == 16
    assert batches[-1]["input_planes"].shape[0] == 2


def test_make_train_dataloader_yields_tensor_dicts(tmp_path: Path) -> None:
    """1st batch -> dict avec 5 keys, tensors PyTorch."""
    from nanozero_training.data.dataset import NpzDataset

    path = _write_batch_at(tmp_path, gen=1, batch=0, n_samples=4)
    dataset = NpzDataset(path)
    config = TrainConfig(batch_size=2, num_workers=0, pin_memory=False, shuffle=False)
    dl = make_train_dataloader(dataset, config)
    first = next(iter(dl))
    assert set(first.keys()) == {
        "input_planes",
        "policy_target",
        "value_target",
        "turn",
        "ply",
    }
    import torch

    for v in first.values():
        assert isinstance(v, torch.Tensor)


def test_make_train_dataloader_shapes(tmp_path: Path) -> None:
    """Shapes batch après collate."""
    from nanozero_training.data.dataset import NpzDataset

    path = _write_batch_at(tmp_path, gen=1, batch=0, n_samples=4)
    dataset = NpzDataset(path)
    config = TrainConfig(batch_size=2, num_workers=0, pin_memory=False, shuffle=False)
    dl = make_train_dataloader(dataset, config)
    first = next(iter(dl))
    assert first["input_planes"].shape == (2, 119, 8, 8)
    assert first["policy_target"].shape == (2, 4672)
    assert first["value_target"].shape == (2,)
    assert first["turn"].shape == (2,)
    assert first["ply"].shape == (2,)


def test_make_train_dataloader_for_gen_uses_replay_window(tmp_path: Path) -> None:
    """3 gens written, replay_window=2, current_gen=3 -> load only gens 2-3."""
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=3)
    _write_batch_at(tmp_path, gen=2, batch=0, n_samples=3)
    _write_batch_at(tmp_path, gen=3, batch=0, n_samples=3)

    config = TrainConfig(
        batch_size=4, num_workers=0, pin_memory=False, shuffle=False, replay_window=2
    )
    dl = make_train_dataloader_for_gen(datasets_dir=tmp_path, current_gen=3, config=config)

    # 2 gens x 3 samples = 6 samples total, batch_size=4 -> 2 batches (4, 2).
    batches = list(dl)
    assert len(batches) == 2
    total = sum(b["input_planes"].shape[0] for b in batches)
    assert total == 6


def test_make_train_dataloader_for_gen_raises_if_no_files(tmp_path: Path) -> None:
    """Empty dir -> ValueError propagé depuis load_replay_buffer."""
    config = TrainConfig(batch_size=4, num_workers=0, pin_memory=False, replay_window=10)
    with pytest.raises(ValueError, match="No batch files"):
        make_train_dataloader_for_gen(datasets_dir=tmp_path, current_gen=1, config=config)


def test_make_train_dataloader_for_gen_lazy_path(tmp_path: Path) -> None:
    """lazy_dataset=True : dispatches to LazyNpzDataset, yields all samples."""
    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=10)
    _write_batch_at(tmp_path, gen=2, batch=0, n_samples=10)
    config = TrainConfig(
        batch_size=4,
        num_workers=0,
        pin_memory=False,
        replay_window=2,
        lazy_dataset=True,
        lazy_buffer_size=5,
    )
    dl = make_train_dataloader_for_gen(datasets_dir=tmp_path, current_gen=2, config=config)
    batches = list(dl)
    total = sum(b["input_planes"].shape[0] for b in batches)
    assert total == 20  # 2 gens x 10 samples


def test_make_train_dataloader_lazy_forces_shuffle_false_at_loader_level(
    tmp_path: Path,
) -> None:
    """IterableDataset incompatible avec DataLoader shuffle=True : doit être désactivé."""
    from nanozero_training.data.dataset import LazyNpzDataset

    _write_batch_at(tmp_path, gen=1, batch=0, n_samples=4)
    ds = LazyNpzDataset(paths=sorted(tmp_path.glob("*.npz")), buffer_size=2, shuffle=True)
    # config.shuffle=True devrait être ignoré pour IterableDataset (sinon torch raise).
    config = TrainConfig(batch_size=2, num_workers=0, pin_memory=False, shuffle=True)
    dl = make_train_dataloader(ds, config)
    batches = list(dl)
    total = sum(b["input_planes"].shape[0] for b in batches)
    assert total == 4
