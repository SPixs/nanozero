"""DataLoader helpers pour training replay buffer."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from nanozero_training.data.dataset import (
    BatchedNpzDataset,
    load_replay_buffer,
    load_replay_buffer_batched,
    load_replay_buffer_lazy,
)
from nanozero_training.train.config import TrainConfig


def make_train_dataloader(
    dataset: Dataset[dict[str, torch.Tensor]],
    config: TrainConfig,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Build a PyTorch DataLoader pour training.

    Args:
        dataset: typically result of load_replay_buffer(...) or
                 load_replay_buffer_lazy(...).
        config: TrainConfig (batch_size, num_workers, shuffle, pin_memory).

    Returns:
        DataLoader iterating over batches de dicts avec keys :
        'input_planes' (B, 119, 8, 8), 'policy_target' (B, 4672),
        'value_target' (B,), 'turn' (B,), 'ply' (B,).

    Note: IterableDataset forces shuffle=False at the DataLoader level (the
    LazyNpzDataset performs its own shuffling internally via shard order
    randomization + rolling shuffle buffer).
    """
    is_iterable = isinstance(dataset, IterableDataset)
    # (2026-06-08) persistent_workers + prefetch_factor : accélère le dataloader
    # quand num_workers>0 (évite le re-spawn des workers à chaque epoch + précharge
    # plus de batchs pour ne pas affamer le GPU). Invalides pour num_workers=0 →
    # gardés derrière le test.
    extra: dict[str, Any] = {}
    if config.num_workers > 0:
        extra["persistent_workers"] = True
        extra["prefetch_factor"] = 4
    if isinstance(dataset, BatchedNpzDataset):
        # Le dataset yield des batches déjà empilés → DataLoader ne re-batch PAS
        # (batch_size=None) et collate=identité par défaut. pin_memory pinne le batch.
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            **extra,
        )
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False if is_iterable else config.shuffle,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False,
        **extra,
    )


def make_train_dataloader_for_gen(
    datasets_dir: str | Path,
    current_gen: int,
    config: TrainConfig,
) -> DataLoader[dict[str, torch.Tensor]]:
    """Convenience : load replay buffer pour current_gen + window, return DataLoader.

    Honors ``config.lazy_dataset`` : when True uses LazyNpzDataset (streaming,
    bounded RAM via ``config.lazy_buffer_size``) ; when False uses the eager
    NpzDataset / ConcatDataset path (v1.0.0 default, preserved bit-for-bit).

    Args:
        datasets_dir: répertoire contenant selfplay-genNNN-batch-MMM.npz.
        current_gen: index de génération.
        config: TrainConfig (replay_window + lazy_dataset + lazy_buffer_size utilisés).

    Returns:
        DataLoader sur le replay buffer combiné.

    Raises:
        FileNotFoundError: si datasets_dir absent.
        ValueError: si aucun batch file pour la fenêtre.
    """
    replay: Dataset[dict[str, torch.Tensor]]
    if config.batched_dataset:
        replay = load_replay_buffer_batched(
            datasets_dir=datasets_dir,
            current_gen=current_gen,
            batch_size=config.batch_size,
            window=config.replay_window,
            shuffle=config.shuffle,
            seed=config.train_seed,
            browser_fraction=config.browser_fraction,
        )
    elif config.lazy_dataset:
        replay = load_replay_buffer_lazy(
            datasets_dir=datasets_dir,
            current_gen=current_gen,
            window=config.replay_window,
            buffer_size=config.lazy_buffer_size,
            shuffle=config.shuffle,
            seed=config.train_seed,
            browser_fraction=config.browser_fraction,
        )
    else:
        replay = load_replay_buffer(
            datasets_dir=datasets_dir,
            current_gen=current_gen,
            window=config.replay_window,
            browser_fraction=config.browser_fraction,
            seed=config.train_seed,
        )
    return make_train_dataloader(replay, config)
