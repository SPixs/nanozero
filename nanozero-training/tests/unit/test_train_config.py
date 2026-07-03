"""Unit tests for train/config.py."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from nanozero_training.train.config import TrainConfig


def test_train_config_defaults_match_spec() -> None:
    """Defaults alignés SPEC §4."""
    cfg = TrainConfig()
    assert cfg.batch_size == 256
    assert cfg.learning_rate == 1e-3
    assert cfg.total_epochs == 10
    assert cfg.l2_reg == 5e-5  # (2026-05-21) was 1e-4 — diagnostic denormals
    assert cfg.lr_min_ratio == 0.05  # (2026-05-21) eta_min floor cosine
    assert cfg.replay_window == 10
    assert cfg.num_workers == 4
    assert cfg.shuffle is True
    assert cfg.pin_memory is True
    assert cfg.use_cosine_schedule is True
    assert cfg.train_seed == 42


def test_train_config_max_grad_norm_default_generous() -> None:
    """Default Lc0-style : 10000.0 = safety net, jamais activé en flow normal."""
    cfg = TrainConfig()
    assert cfg.max_grad_norm == 10000.0


def test_train_config_immutable() -> None:
    cfg = TrainConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.batch_size = 128  # type: ignore[misc]


def test_train_config_custom_values() -> None:
    cfg = TrainConfig(
        batch_size=128,
        learning_rate=5e-4,
        total_epochs=20,
        replay_window=5,
        max_grad_norm=10.0,
        train_seed=999,
    )
    assert cfg.batch_size == 128
    assert cfg.learning_rate == 5e-4
    assert cfg.total_epochs == 20
    assert cfg.replay_window == 5
    assert cfg.max_grad_norm == 10.0
    assert cfg.train_seed == 999
    # Defaults non spécifiés préservés
    assert cfg.num_workers == 4


def test_train_config_replay_window_default_10() -> None:
    """SPEC §4 : default 10 générations replay."""
    cfg = TrainConfig()
    assert cfg.replay_window == 10
