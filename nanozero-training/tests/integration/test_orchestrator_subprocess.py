"""Slow integration: orchestrator with real UCI subprocess.

Requires JAR nanozero-uci-1.2.0.jar + JDK 25.
Test : 10 parties splittées en 2 batches de 5, validation :
- Batch rotation (batch 0 puis batch 1).
- State persisté on-disk via RunStateManager.
- 2 fichiers .npz produits.
- Compatible avec phase 1.0.0-11 (pipeline assembly avec resume).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from nanozero_training.selfplay import (
    SelfplayConfig,
    SelfplayOrchestrator,
    UciClient,
)
from nanozero_training.state.manager import RunStateManager

REPO_ROOT = Path(__file__).resolve().parents[3]
UCI_JAR = REPO_ROOT / "nanozero-uci" / "target" / "nanozero-uci-1.2.0.jar"
TEST_MODEL_DIR = Path(__file__).resolve().parents[1] / "fixtures"
TEST_MODEL = TEST_MODEL_DIR / "test-gen0-model.npz"


def _java_version_major() -> int | None:
    try:
        result = subprocess.run(
            ["java", "--version"],
            capture_output=True,
            text=True,
            timeout=10.0,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or "") + (result.stderr or "")
    for line in output.splitlines():
        for tok in line.split():
            if tok and tok[0].isdigit():
                major_str = tok.split(".")[0]
                if major_str.isdigit():
                    return int(major_str)
    return None


def _ensure_test_model() -> Path:
    if TEST_MODEL.exists():
        return TEST_MODEL
    TEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "nanozero-training" / "scripts" / "generate_gen0_model.py"),
            "--seed",
            "42",
            "--output",
            str(TEST_MODEL),
        ],
        check=True,
        capture_output=True,
    )
    return TEST_MODEL


@pytest.fixture()
def env() -> Path:
    if not UCI_JAR.exists():
        pytest.skip(f"UCI JAR not found: {UCI_JAR}")
    java_major = _java_version_major()
    if java_major is None or java_major < 25:
        pytest.skip(f"JDK 25+ required, found {java_major}")
    return _ensure_test_model()


@pytest.mark.slow()
def test_orchestrator_two_minibatches_end_to_end(env: Path, tmp_path: Path) -> None:
    """Run 10 parties splittées en 2 batches de 5, validation full state + files."""
    model = env

    state_mgr = RunStateManager(monitoring_dir=tmp_path / "monitoring")
    state_mgr.create_new(
        run_id="test-orch-001",
        config_path="dummy.yaml",
        config_hash="dummy",
        max_generations=1,
        target_games_per_gen=10,
    )

    client = UciClient()
    client.start(UCI_JAR, model, dirichlet_seed=42, handshake_timeout=60.0)

    abort_flag = {"requested": False}

    config = SelfplayConfig(
        mcts_sims=50,  # fast pour tests
        max_game_plies=20,
        go_timeout_seconds=30.0,
        games_per_batch=5,
        target_games_per_gen=10,
        worker_restart_every=1000,
        max_consecutive_crashes=5,
        worker_seed=42,
    )

    datasets_dir = tmp_path / "datasets"
    orch = SelfplayOrchestrator(
        uci_client=client,
        config=config,
        state_manager=state_mgr,
        abort_flag=abort_flag,
        datasets_dir=datasets_dir,
    )

    try:
        orch.run_generation(gen=1, target_games=10)
    finally:
        client.quit()

    # Validate state.
    assert state_mgr.state.selfplay.completed_games == 10
    assert state_mgr.state.selfplay.completed_batches == 2

    # Validate batch files.
    batch_files = sorted(datasets_dir.glob("selfplay-gen001-batch-*.npz"))
    assert len(batch_files) == 2, f"Expected 2 batches, found {batch_files}"

    # Validate state file persisté on-disk via RunStateManager.
    assert (tmp_path / "monitoring" / "run_state.yaml").exists()
