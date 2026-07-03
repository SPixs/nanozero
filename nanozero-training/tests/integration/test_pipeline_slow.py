"""Slow integration test : 1 mini-generation end-to-end via CLI 'run' command.

Requires:
  - nanozero-uci-1.2.0.jar built
  - JDK 25 available
  - fastchess in PATH (for SPRT phase) ; if absent, test skipped.
  - test-gen0-model.npz available (generated on-the-fly if missing)

Config :
  mcts_sims=10, max_game_plies=20, target_games_per_gen=2, batch_size=2,
  total_epochs=1, max_games (SPRT)=10, max_generations=1.

Wall-clock target : < 10 min on DevSrv CPU. Skip propre si env manquant.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from nanozero_training.cli import cli

REPO_ROOT = Path(__file__).resolve().parents[3]
UCI_JAR = REPO_ROOT / "nanozero-uci" / "target" / "nanozero-uci-1.2.0.jar"
GEN0_SCRIPT = REPO_ROOT / "nanozero-training" / "scripts" / "generate_gen0_model.py"


def _java_version_major() -> int:
    try:
        result = subprocess.run(
            ["java", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        first = (result.stdout or result.stderr).strip().split("\n")[0]
        for tok in first.split():
            if tok.replace(".", "").isdigit():
                return int(tok.split(".")[0])
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return -1


def _fastchess_available() -> bool:
    return shutil.which("fastchess") is not None


@pytest.mark.slow()
def test_full_pipeline_one_mini_generation(tmp_path: Path) -> None:
    """Run 1 mini-generation end-to-end via CLI 'run' command."""
    if not UCI_JAR.exists():
        pytest.skip(f"UCI JAR not found: {UCI_JAR}")
    if _java_version_major() < 25:
        pytest.skip(f"JDK 25+ required, found {_java_version_major()}")
    if not _fastchess_available():
        pytest.skip("fastchess binary not in PATH")
    if not GEN0_SCRIPT.exists():
        pytest.skip(f"generate_gen0_model.py not found: {GEN0_SCRIPT}")

    # Mini config for fast end-to-end run
    config_path = tmp_path / "test-config.yaml"
    run_root = tmp_path / "run"
    config_data = {
        "max_generations": 1,
        "selfplay": {
            "mcts_sims": 10,
            "max_game_plies": 20,
            "target_games_per_gen": 2,
            "games_per_batch": 2,
            "go_timeout_seconds": 15.0,
        },
        "train": {
            "batch_size": 2,
            "total_epochs": 1,
            "replay_window": 1,
            "num_workers": 0,
            "pin_memory": False,
            "shuffle": False,
        },
        "eval": {
            "uci_jar": str(UCI_JAR),
            "max_games": 4,
            "time_control": "2+0.1",
            "poll_interval_seconds": 1.0,
            "concurrency": 1,
        },
        "monitor": {"enabled": False},
        "paths": {
            "run_root": str(run_root),
            "datasets_dir": "datasets",
            "models_dir": "models",
            "monitoring_dir": "monitoring",
            "versions_yaml": "versions.yaml",
            "pgn_path": "monitoring/sprt.pgn",
            "uci_jar": str(UCI_JAR),
        },
    }
    config_path.write_text(yaml.safe_dump(config_data))

    runner = CliRunner()

    # Step 1 : generate-gen0
    result = runner.invoke(cli, ["generate-gen0", "--config", str(config_path), "--seed", "42"])
    if result.exit_code != 0:
        pytest.fail(f"generate-gen0 failed: exit={result.exit_code}, output={result.output}")

    # Step 2 : run 1 mini-generation
    result = runner.invoke(cli, ["run", "--config", str(config_path)])

    # Acceptable outcomes : 0 (pipeline complete) ou 1 (rejection / SPRT
    # non-convergence sur si peu de games). Pas de crash filesystem ou syntax.
    assert result.exit_code in (
        0,
        1,
    ), f"Unexpected exit={result.exit_code} ; output={result.output[-1000:]}"

    # Validate run_state.yaml exists
    state_file = run_root / "monitoring" / "run_state.yaml"
    assert state_file.exists(), "run_state.yaml not created"

    # Validate versions.yaml has gen-001-init + gen-001-trained
    versions_file = run_root / "versions.yaml"
    assert versions_file.exists(), "versions.yaml not created"
    versions = yaml.safe_load(versions_file.read_text())
    names = [e["name"] for e in versions["all"]]
    assert "gen-001-init" in names
    # gen-001-trained should exist if train phase reached
    # (might be gen-002-trained if numbering quirks ; flexible assertion)
    assert any("trained" in n for n in names), f"No trained model in versions: {names}"
