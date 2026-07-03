"""CLI smoke tests via click.testing.CliRunner (in-process)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml
from click.testing import CliRunner
from nanozero_training.cli import cli


def _write_minimal_config(path: Path, **overrides: Any) -> None:
    """Write a minimal valid YAML config file. Overrides are dotted key paths."""
    data: dict[str, Any] = {
        "max_generations": 5,
        "selfplay": {"mcts_sims": 50},
        "eval": {"uci_jar": "/dummy/uci.jar"},
        "paths": {
            "run_root": str(path.parent),
            "datasets_dir": "datasets",
            "models_dir": "models",
            "monitoring_dir": "monitoring",
            "uci_jar": "/dummy/uci.jar",
        },
    }
    for key, value in overrides.items():
        if "." in key:
            section, field = key.split(".", 1)
            data.setdefault(section, {})[field] = value
        else:
            data[key] = value
    path.write_text(yaml.safe_dump(data))


@pytest.fixture()
def cfg_path(tmp_path: Path) -> Path:
    p = tmp_path / "test-config.yaml"
    _write_minimal_config(p)
    return p


def test_cli_help_lists_all_commands() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    expected = (
        "selfplay",
        "train",
        "eval",
        "versions",
        "status",
        "resume",
        "run",
        "monitor",
        "generate-gen0",
    )
    for command in expected:
        assert command in result.output


def test_cli_status_no_state_shows_empty(cfg_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--config", str(cfg_path)])
    assert result.exit_code == 0
    assert "No active run" in result.output
    assert "No versions.yaml" in result.output


def test_cli_selfplay_no_longer_stubbed(cfg_path: Path) -> None:
    """Phase 11 wiring : selfplay stub removed, command actually dispatches."""
    runner = CliRunner()
    result = runner.invoke(cli, ["selfplay", "--config", str(cfg_path), "--model", "gen-001-init"])
    # Stub message must be gone (phase 11 wired this command).
    assert "phase 1.0.0-11" not in result.output


def test_cli_train_no_longer_stubbed(cfg_path: Path) -> None:
    """Phase 11 wiring : train stub removed."""
    runner = CliRunner()
    result = runner.invoke(
        cli, ["train", "--config", str(cfg_path), "--base-model", "gen-001-init"]
    )
    assert "phase 1.0.0-11" not in result.output


def test_cli_eval_no_longer_stubbed(cfg_path: Path) -> None:
    """Phase 11 wiring : eval stub removed."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "eval",
            "--config",
            str(cfg_path),
            "--challenger",
            "gen-002-trained",
            "--baseline",
            "gen-001-init",
        ],
    )
    assert "phase 1.0.0-11" not in result.output


def test_cli_run_no_longer_stubbed(cfg_path: Path) -> None:
    """Phase 11 wiring : run stub removed."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--config", str(cfg_path)])
    assert "phase 1.0.0-11" not in result.output


def test_cli_monitor_disabled_aborts(tmp_path: Path) -> None:
    """Phase 10 : monitor.enabled=false -> exit non-zero avec message."""
    p = tmp_path / "c.yaml"
    _write_minimal_config(p, **{"monitor.enabled": False})
    runner = CliRunner()
    result = runner.invoke(cli, ["monitor", "--config", str(p)])
    assert result.exit_code != 0
    assert "disabled" in result.output.lower()


def test_cli_versions_list_empty(cfg_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["versions", "list", "--config", str(cfg_path)])
    assert result.exit_code == 0
    assert "No versions.yaml" in result.output


def test_cli_invalid_config_path_fails() -> None:
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--config", "/nonexistent.yaml"])
    assert result.exit_code != 0


def test_cli_invalid_yaml_content_fails(tmp_path: Path) -> None:
    bad = tmp_path / "bad.yaml"
    bad.write_text("not: valid: yaml: : :", encoding="utf-8")
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--config", str(bad)])
    # Either yaml parse fails or schema validation fails — either way non-zero
    assert result.exit_code != 0


def test_cli_config_validation_negative_port_fails(tmp_path: Path) -> None:
    p = tmp_path / "bad-port.yaml"
    _write_minimal_config(p, **{"monitor.port": -1})
    runner = CliRunner()
    result = runner.invoke(cli, ["status", "--config", str(p)])
    assert result.exit_code != 0


def test_cli_versions_promote_requires_force(tmp_path: Path) -> None:
    """Manual promote without --force should fail."""
    p = tmp_path / "c.yaml"
    _write_minimal_config(p)
    runner = CliRunner()
    # Need to accept the confirmation prompt via input.
    result = runner.invoke(
        cli,
        ["versions", "promote", "--config", str(p), "gen-002-trained"],
        input="y\n",
    )
    assert result.exit_code != 0
    # Either confirmation prompt rejected without input, or --force missing.


def test_cli_compute_config_hash_displayed_in_selfplay_stub(cfg_path: Path) -> None:
    """selfplay stub displays config_hash[:16] before raising."""
    runner = CliRunner()
    result = runner.invoke(cli, ["selfplay", "--config", str(cfg_path), "--model", "gen-001-init"])
    # Stub raises but echoes config_hash before
    assert "config_hash=" in result.output
