"""CLI wiring tests : verify each command dispatches to the correct phase_runner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner
from nanozero_training.cli import cli


def _write_minimal_config(tmp_path: Path) -> Path:
    """Write a minimal valid YAML config."""
    data: dict[str, object] = {
        "max_generations": 3,
        "selfplay": {"mcts_sims": 50},
        "eval": {"uci_jar": "/dummy/uci.jar"},
        "paths": {
            "run_root": str(tmp_path / "run"),
            "datasets_dir": "datasets",
            "models_dir": "models",
            "monitoring_dir": "monitoring",
            "uci_jar": "/dummy/uci.jar",
        },
        "monitor": {"enabled": False},
    }
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


@pytest.fixture()
def cfg_path(tmp_path: Path) -> Path:
    return _write_minimal_config(tmp_path)


def _make_mocks() -> dict[str, MagicMock]:
    """Patch all bootstrap + phase_runner symbols used by CLI commands."""
    patches = {
        "bootstrap": patch("nanozero_training.cli._bootstrap_managers"),
        "selfplay_phase": patch("nanozero_training.pipeline.phase_runner.run_selfplay_phase"),
        "train_phase": patch("nanozero_training.pipeline.phase_runner.run_train_phase"),
        "eval_phase": patch("nanozero_training.pipeline.phase_runner.run_eval_phase"),
        "promote_phase": patch("nanozero_training.pipeline.phase_runner.run_promote_phase"),
        "orch": patch("nanozero_training.pipeline.orchestrator.PipelineOrchestrator"),
    }
    mocks = {name: p.start() for name, p in patches.items()}
    # Set up _bootstrap_managers return value (sm, vm, abort_flag, writer)
    sm = MagicMock()
    sm.state.current_gen = 0
    sm.state.run_id = "test-run"
    sm.state.status = "in_progress"
    sm.state.phase = "idle"
    sm.detect_existing_run.return_value = False
    vm = MagicMock()
    mocks["bootstrap"].return_value = (sm, vm, {"requested": False}, None)
    mocks["sm"] = sm
    mocks["vm"] = vm
    return mocks


def test_cli_selfplay_calls_run_selfplay_phase(cfg_path: Path) -> None:
    mocks = _make_mocks()
    try:
        runner = CliRunner()
        runner.invoke(cli, ["selfplay", "--config", str(cfg_path), "--model", "gen-001-init"])
        # Imported lazily inside selfplay command -> patch at the import location
        # Since selfplay does `from ... phase_runner import run_selfplay_phase`,
        # we patched at the module level which is sufficient.
        mocks["selfplay_phase"].assert_called_once()
    finally:
        patch.stopall()


def test_cli_train_calls_run_train_phase(cfg_path: Path) -> None:
    mocks = _make_mocks()
    try:
        mocks["train_phase"].return_value = "gen-001-trained"
        runner = CliRunner()
        runner.invoke(cli, ["train", "--config", str(cfg_path), "--base-model", "gen-001-init"])
        mocks["train_phase"].assert_called_once()
    finally:
        patch.stopall()


def test_cli_eval_calls_run_eval_phase(cfg_path: Path) -> None:
    mocks = _make_mocks()
    try:
        from nanozero_training.eval.sprt_result import SPRTResult, SPRTStatus

        mocks["eval_phase"].return_value = SPRTResult(
            status=SPRTStatus.H1_ACCEPTED,
            llr=2.95,
            games_played=100,
            wins=60,
            losses=30,
            draws=10,
            elo_diff=10.0,
        )
        runner = CliRunner()
        runner.invoke(
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
            input="n\n",  # decline promote confirm
        )
        mocks["eval_phase"].assert_called_once()
    finally:
        patch.stopall()


def test_cli_run_calls_orchestrator_run(cfg_path: Path) -> None:
    mocks = _make_mocks()
    try:
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--config", str(cfg_path)])
        # PipelineOrchestrator is imported inside run() — the patch on
        # nanozero_training.pipeline.orchestrator.PipelineOrchestrator
        # affects the symbol that run() will pick up.
        mocks["orch"].assert_called_once()
        mocks["orch"].return_value.run.assert_called_once()
        assert result.exit_code == 0
    finally:
        patch.stopall()


def test_cli_run_refuses_existing_run(cfg_path: Path) -> None:
    """Phase 11 Q2 : run command refuse si existing run."""
    import click

    with patch("nanozero_training.cli._bootstrap_managers") as mock_bootstrap:
        mock_bootstrap.side_effect = click.ClickException("An existing run is in progress.")
        runner = CliRunner()
        result = runner.invoke(cli, ["run", "--config", str(cfg_path)])
        assert result.exit_code != 0
        assert "existing run" in result.output.lower()


def test_cli_resume_calls_orchestrator_resume(cfg_path: Path) -> None:
    mocks = _make_mocks()
    try:
        # Make detect_existing_run return True (so resume actually runs)
        mocks["sm"].detect_existing_run.return_value = True
        mocks["sm"].state.status = "in_progress"
        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--config", str(cfg_path)])
        mocks["orch"].return_value.resume.assert_called_once()
        assert result.exit_code == 0
    finally:
        patch.stopall()


def test_cli_resume_no_existing_run_completed_no_op(cfg_path: Path) -> None:
    """status=completed -> echo "already completed" + return."""
    mocks = _make_mocks()
    try:
        mocks["sm"].detect_existing_run.return_value = False
        mocks["sm"].state.status = "completed"
        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--config", str(cfg_path)])
        assert "already completed" in result.output.lower()
        # orchestrator.resume() NOT called
        mocks["orch"].return_value.resume.assert_not_called()
    finally:
        patch.stopall()


def test_cli_resume_aborted_prompts_user(cfg_path: Path) -> None:
    """status=aborted -> click.confirm prompts user."""
    mocks = _make_mocks()
    try:
        mocks["sm"].detect_existing_run.return_value = False
        mocks["sm"].state.status = "aborted"
        runner = CliRunner()
        # Send "n" to decline -> orchestrator.resume NOT called
        result = runner.invoke(cli, ["resume", "--config", str(cfg_path)], input="n\n")
        assert "Resume anyway" in result.output
        mocks["orch"].return_value.resume.assert_not_called()
    finally:
        patch.stopall()
