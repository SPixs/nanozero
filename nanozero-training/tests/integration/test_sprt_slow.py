"""Slow integration: SPRT gen0 vs gen0 via real fastchess subprocess.

Requires:
  - nanozero-uci-1.2.0.jar built (mvn -pl nanozero-uci -am package -DskipTests)
  - JDK 25 available
  - fastchess binary in PATH (install via scripts/install-fastchess-w3090.md)
  - test-gen0-model.npz available (regenerated on-the-fly si manquant)

Q4 décision actée : ce test valide le PIPELINE, pas la décision SPRT.
SPRT gen0 vs gen0 avec max_games=30 ne va probablement pas converger
-> on accepte MAX_GAMES_REACHED comme issue valide.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from nanozero_training.eval import (
    FastchessConfig,
    FastchessRunner,
    SPRTStatus,
    count_games_in_pgn,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
UCI_JAR = REPO_ROOT / "nanozero-uci" / "target" / "nanozero-uci-1.2.0.jar"
TEST_MODEL_DIR = Path(__file__).resolve().parents[1] / "fixtures"
TEST_MODEL = TEST_MODEL_DIR / "test-gen0-model.npz"


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


@pytest.mark.slow()
def test_sprt_gen0_vs_gen0_pipeline(tmp_path: Path) -> None:
    """Run SPRT gen0 vs gen0 with max_games=30. Valide pipeline + parsing."""
    if not UCI_JAR.exists():
        pytest.skip(f"UCI JAR not found: {UCI_JAR}")
    if _java_version_major() < 25:
        pytest.skip(f"JDK 25+ required, found {_java_version_major()}")
    if not _fastchess_available():
        pytest.skip(
            "fastchess binary not in PATH " "(install per scripts/install-fastchess-w3090.md)"
        )

    model = _ensure_test_model()
    pgn_path = tmp_path / "sprt.pgn"

    config = FastchessConfig(
        uci_jar=str(UCI_JAR),
        elo_low=0.0,
        elo_high=20.0,
        alpha=0.05,
        beta=0.05,
        max_games=30,
        time_control="2+0.1",
        concurrency=1,
        pgn_output=str(pgn_path),
        poll_interval_seconds=5.0,
    )
    runner = FastchessRunner(config)

    result = runner.run_sprt(
        challenger_npz=model,
        baseline_npz=model,
        state_manager=None,
        timeout_seconds=600.0,
    )

    # Q4: accept MAX_GAMES_REACHED comme outcome valide
    assert result.status in (
        SPRTStatus.H0_ACCEPTED,
        SPRTStatus.H1_ACCEPTED,
        SPRTStatus.MAX_GAMES_REACHED,
    ), f"Unexpected SPRT status: {result.status} (raw: {result.raw_output[-500:]})"

    assert result.games_played > 0
    assert result.wins + result.losses + result.draws <= result.games_played + 2

    assert pgn_path.exists()
    assert count_games_in_pgn(pgn_path) > 0
