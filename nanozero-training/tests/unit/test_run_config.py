"""Unit tests for config/run_config — RunConfig + MonitorConfig + PathsConfig."""

from __future__ import annotations

from pathlib import Path

import pytest
from nanozero_training.config.run_config import (
    MonitorConfig,
    PathsConfig,
    RunConfig,
)
from nanozero_training.eval.fastchess_runner import FastchessConfig
from nanozero_training.selfplay.config import SelfplayConfig
from nanozero_training.train.config import TrainConfig


def test_run_config_defaults() -> None:
    cfg = RunConfig()
    assert cfg.max_generations == 100
    assert cfg.run_id == ""
    assert isinstance(cfg.selfplay, SelfplayConfig)
    assert isinstance(cfg.train, TrainConfig)
    assert isinstance(cfg.eval_fastchess, FastchessConfig)
    assert isinstance(cfg.monitor, MonitorConfig)
    assert isinstance(cfg.paths, PathsConfig)


def test_run_config_max_generations_zero_raises() -> None:
    with pytest.raises(ValueError, match="max_generations must be > 0"):
        RunConfig(max_generations=0)
    with pytest.raises(ValueError, match="max_generations must be > 0"):
        RunConfig(max_generations=-5)


def test_monitor_config_port_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="monitor.port must be in"):
        MonitorConfig(port=0)
    with pytest.raises(ValueError, match="monitor.port must be in"):
        MonitorConfig(port=70000)


def test_monitor_config_negative_flush_interval_raises() -> None:
    with pytest.raises(ValueError, match="csv_flush_interval_seconds must be > 0"):
        MonitorConfig(csv_flush_interval_seconds=0.0)
    with pytest.raises(ValueError, match="csv_flush_interval_seconds must be > 0"):
        MonitorConfig(csv_flush_interval_seconds=-1.0)


def test_monitor_config_sse_poll_interval_default_and_validation() -> None:
    """Phase 10 : sse_poll_interval_seconds default 0.5 + must be > 0."""
    cfg = MonitorConfig()
    assert cfg.sse_poll_interval_seconds == 0.5
    with pytest.raises(ValueError, match="sse_poll_interval_seconds must be > 0"):
        MonitorConfig(sse_poll_interval_seconds=0.0)
    with pytest.raises(ValueError, match="sse_poll_interval_seconds must be > 0"):
        MonitorConfig(sse_poll_interval_seconds=-0.1)


def test_monitor_config_valid_defaults() -> None:
    m = MonitorConfig()
    assert m.enabled is True
    assert m.port == 5000
    assert m.host == "127.0.0.1"


def test_paths_config_empty_run_root_raises() -> None:
    with pytest.raises(ValueError, match="run_root cannot be empty"):
        PathsConfig(run_root="")


def test_paths_config_resolve_relative(tmp_path: Path) -> None:
    p = PathsConfig(run_root=str(tmp_path), datasets_dir="data")
    resolved = p.resolve("datasets_dir")
    assert resolved == tmp_path / "data"


def test_paths_config_resolve_absolute_keeps_absolute(tmp_path: Path) -> None:
    abs_path = "/var/lib/nanozero/data"
    p = PathsConfig(run_root=str(tmp_path), datasets_dir=abs_path)
    resolved = p.resolve("datasets_dir")
    assert resolved == Path(abs_path)


def test_paths_config_resolve_unknown_field_raises() -> None:
    p = PathsConfig()
    with pytest.raises(AttributeError):
        p.resolve("nonexistent_field")


def test_paths_config_resolve_empty_field_raises() -> None:
    p = PathsConfig()  # uci_jar default empty
    with pytest.raises(ValueError, match="paths.uci_jar is empty"):
        p.resolve("uci_jar")


def test_run_config_composes_sub_configs() -> None:
    """Override one sub-config, verify others retain defaults."""
    cfg = RunConfig(train=TrainConfig(batch_size=128))
    assert cfg.train.batch_size == 128
    assert cfg.selfplay.mcts_sims == 400  # SelfplayConfig default
    assert cfg.monitor.port == 5000  # MonitorConfig default


def test_paths_config_resolve_all_default_fields(tmp_path: Path) -> None:
    """Resolve all non-empty default fields without crashing."""
    p = PathsConfig(run_root=str(tmp_path))
    for field_name in ("datasets_dir", "models_dir", "monitoring_dir", "versions_yaml", "pgn_path"):
        resolved = p.resolve(field_name)
        assert resolved.is_absolute()
