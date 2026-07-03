"""Unit tests for config/loader — load_config + compute_config_hash + merge_cli_overrides."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from nanozero_training.config.loader import (
    compute_config_hash,
    load_config,
    merge_cli_overrides,
)
from nanozero_training.config.run_config import RunConfig


def _write_yaml(path: Path, data: dict | None) -> None:
    if data is None:
        path.write_text("", encoding="utf-8")
    else:
        path.write_text(yaml.safe_dump(data), encoding="utf-8")


def test_load_config_returns_run_config(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write_yaml(p, {"selfplay": {"mcts_sims": 200}})
    cfg = load_config(p)
    assert isinstance(cfg, RunConfig)
    assert cfg.selfplay.mcts_sims == 200


def test_load_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        load_config(tmp_path / "missing.yaml")


def test_load_config_malformed_yaml_raises(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("not: valid: yaml: : :", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed YAML"):
        load_config(p)


def test_load_config_root_not_dict_raises(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n- c\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a dict"):
        load_config(p)


def test_load_config_empty_uses_defaults(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    _write_yaml(p, None)
    cfg = load_config(p)
    assert cfg.max_generations == 100
    assert cfg.selfplay.mcts_sims == 400  # SelfplayConfig default


def test_load_config_unknown_field_raises_typeerror(tmp_path: Path) -> None:
    p = tmp_path / "unknown.yaml"
    _write_yaml(p, {"selfplay": {"foo_bar_baz": 1}})
    with pytest.raises(TypeError):
        load_config(p)


def test_load_config_validation_propagates(tmp_path: Path) -> None:
    p = tmp_path / "bad-port.yaml"
    _write_yaml(p, {"monitor": {"port": -1}})
    with pytest.raises(ValueError, match="monitor.port must be in"):
        load_config(p)


def test_load_config_max_generations_zero_validation(tmp_path: Path) -> None:
    p = tmp_path / "bad-maxgens.yaml"
    _write_yaml(p, {"max_generations": 0})
    with pytest.raises(ValueError, match="max_generations must be > 0"):
        load_config(p)


def test_compute_config_hash_deterministic() -> None:
    content = "max_generations: 5\nselfplay:\n  mcts_sims: 100\n"
    h1 = compute_config_hash(content)
    h2 = compute_config_hash(content)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex = 64 chars


def test_compute_config_hash_differs_on_change() -> None:
    h1 = compute_config_hash("max_generations: 5\n")
    h2 = compute_config_hash("max_generations: 6\n")
    assert h1 != h2


def test_compute_config_hash_includes_whitespace() -> None:
    """2 YAML fonctionnellement identiques mais avec whitespace différent -> hashes différents."""
    h1 = compute_config_hash("max_generations: 5\n")
    h2 = compute_config_hash("max_generations:  5\n")  # 2 espaces
    assert h1 != h2


def test_merge_cli_overrides_flat_key(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write_yaml(p, {"train": {"batch_size": 64}})
    cfg = load_config(p)
    merged = merge_cli_overrides(cfg, batch_size=128)
    assert merged.train.batch_size == 128


def test_merge_cli_overrides_dotted_key(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write_yaml(p, {"selfplay": {"mcts_sims": 100}})
    cfg = load_config(p)
    merged = merge_cli_overrides(cfg, **{"selfplay.mcts_sims": 200})
    assert merged.selfplay.mcts_sims == 200


def test_merge_cli_overrides_none_values_ignored(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write_yaml(p, {"train": {"batch_size": 64}})
    cfg = load_config(p)
    merged = merge_cli_overrides(cfg, batch_size=None, total_epochs=None)
    assert merged.train.batch_size == 64
    assert merged.train.total_epochs == cfg.train.total_epochs


def test_merge_cli_overrides_validation_applied(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write_yaml(p, {})
    cfg = load_config(p)
    with pytest.raises(ValueError, match="monitor.port must be in"):
        merge_cli_overrides(cfg, **{"monitor.port": 70000})


def test_merge_cli_overrides_top_level_keys(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write_yaml(p, {"max_generations": 10})
    cfg = load_config(p)
    merged = merge_cli_overrides(cfg, max_generations=20)
    assert merged.max_generations == 20


def test_merge_cli_overrides_no_overrides_returns_same(tmp_path: Path) -> None:
    p = tmp_path / "c.yaml"
    _write_yaml(p, {})
    cfg = load_config(p)
    merged = merge_cli_overrides(cfg)
    assert merged is cfg
