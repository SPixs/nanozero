"""Unit tests for selfplay/config.py."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
from nanozero_training.selfplay.config import SelfplayConfig


def test_selfplay_config_defaults_match_spec() -> None:
    """Defaults alignés SPEC-training §4."""
    cfg = SelfplayConfig()
    # Worker single-game (phase 4)
    assert cfg.mcts_sims == 400
    assert cfg.temperature == 1.0
    assert cfg.temperature_switch_ply == 30
    assert cfg.dirichlet_alpha == 300
    assert cfg.dirichlet_epsilon == 250
    assert cfg.go_timeout_seconds == 60.0
    assert cfg.worker_seed == 42
    assert cfg.max_game_plies == 400
    # Orchestrator multi-parties (phase 5)
    assert cfg.games_per_batch == 250
    assert cfg.target_games_per_gen == 500
    assert cfg.worker_restart_every == 1000
    assert cfg.max_consecutive_crashes == 5


def test_selfplay_config_orchestrator_params_overridable() -> None:
    """Override orchestrator params for tests."""
    cfg = SelfplayConfig(
        games_per_batch=10,
        target_games_per_gen=20,
        worker_restart_every=100,
        max_consecutive_crashes=3,
    )
    assert cfg.games_per_batch == 10
    assert cfg.target_games_per_gen == 20
    assert cfg.worker_restart_every == 100
    assert cfg.max_consecutive_crashes == 3


def test_selfplay_config_max_game_plies_overridable() -> None:
    cfg = SelfplayConfig(max_game_plies=20)
    assert cfg.max_game_plies == 20
    # Autres defaults préservés
    assert cfg.mcts_sims == 400


def test_selfplay_config_frozen() -> None:
    cfg = SelfplayConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.mcts_sims = 800  # type: ignore[misc]


def test_selfplay_config_custom_values() -> None:
    cfg = SelfplayConfig(
        mcts_sims=800,
        temperature=0.5,
        temperature_switch_ply=10,
        dirichlet_alpha=500,
        worker_seed=12345,
    )
    assert cfg.mcts_sims == 800
    assert cfg.temperature == 0.5
    assert cfg.temperature_switch_ply == 10
    assert cfg.dirichlet_alpha == 500
    # Defaults preserved pour les non-spécifiés.
    assert cfg.dirichlet_epsilon == 250
    assert cfg.worker_seed == 12345
