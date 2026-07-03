"""Unit tests for W3090 production configs (phase 1.0.0-12-prep)."""

from __future__ import annotations

from pathlib import Path

from nanozero_training.config.loader import load_config
from nanozero_training.config.run_config import RunConfig

CONFIGS_DIR = Path(__file__).resolve().parent.parent.parent / "configs"
W3090_RUN_YAML = CONFIGS_DIR / "w3090-run.yaml"
W3090_MINI_YAML = CONFIGS_DIR / "w3090-mini.yaml"


def test_w3090_run_yaml_loadable() -> None:
    """w3090-run.yaml loads into a valid RunConfig (all __post_init__ pass)."""
    cfg = load_config(W3090_RUN_YAML)
    assert isinstance(cfg, RunConfig)


def test_w3090_run_max_generations_5() -> None:
    """Production run targets exactly 5 generations (ADR-013 §Décision)."""
    cfg = load_config(W3090_RUN_YAML)
    assert cfg.max_generations == 5


def test_w3090_run_paths_windows_style() -> None:
    """Paths use forward slashes (cross-platform via pathlib on Windows)."""
    cfg = load_config(W3090_RUN_YAML)
    assert "\\" not in cfg.paths.run_root
    assert "\\" not in cfg.paths.uci_jar
    assert cfg.paths.run_root.startswith("S:/nano2/")


def test_w3090_run_monitor_host_0_0_0_0() -> None:
    """Monitor accepts external Tailscale connections (vs 127.0.0.1 local-only)."""
    cfg = load_config(W3090_RUN_YAML)
    assert cfg.monitor.host == "0.0.0.0"
    assert cfg.monitor.enabled is True


def test_w3090_mini_yaml_loadable() -> None:
    """w3090-mini.yaml loads into a valid RunConfig."""
    cfg = load_config(W3090_MINI_YAML)
    assert isinstance(cfg, RunConfig)


def test_w3090_mini_max_generations_1() -> None:
    """Mini targets 1 generation only (smoke + reproducibility)."""
    cfg = load_config(W3090_MINI_YAML)
    assert cfg.max_generations == 1


def test_w3090_mini_smaller_workload() -> None:
    """Mini config is strictly smaller than prod on every workload dimension (sanity)."""
    mini = load_config(W3090_MINI_YAML)
    prod = load_config(W3090_RUN_YAML)

    assert mini.max_generations < prod.max_generations
    assert mini.selfplay.mcts_sims < prod.selfplay.mcts_sims
    assert mini.selfplay.target_games_per_gen < prod.selfplay.target_games_per_gen
    assert mini.train.batch_size < prod.train.batch_size
    assert mini.train.total_epochs < prod.train.total_epochs
    assert mini.eval_fastchess.max_games < prod.eval_fastchess.max_games
