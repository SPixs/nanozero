"""Slow integration test : play_one_game with real UCI subprocess.

Requires JAR nanozero-uci-1.2.0.jar built + JDK 25 available.
Skip propre si JAR absent ou Java < 25 (cohérent pattern phase 4-b).

mcts_sims=50 + max_plies=20 -> ~10-30 sec/partie sur CPU.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from nanozero_training.selfplay import (
    SelfplayConfig,
    UciClient,
    play_one_game,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
UCI_JAR = REPO_ROOT / "nanozero-uci" / "target" / "nanozero-uci-1.2.0.jar"
TEST_MODEL_DIR = Path(__file__).resolve().parents[1] / "fixtures"
TEST_MODEL = TEST_MODEL_DIR / "test-gen0-model.npz"


def _java_version_major() -> int | None:
    """Retourne la version majeure de `java` dans le PATH, ou None si introuvable."""
    try:
        result = subprocess.run(
            ["java", "--version"], capture_output=True, text=True, timeout=10.0, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or "") + (result.stderr or "")
    for line in output.splitlines():
        parts = line.split()
        for tok in parts:
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
def started_client() -> Iterator[UciClient]:
    """Start a real UciClient ; skip si JAR absent ou Java < 25."""
    if not UCI_JAR.exists():
        pytest.skip(
            f"UCI JAR not found at {UCI_JAR}. "
            "Build via 'mvn -pl nanozero-uci -am package -DskipTests'."
        )
    java_major = _java_version_major()
    if java_major is None:
        pytest.skip("`java` not found in PATH.")
    if java_major < 25:
        pytest.skip(
            f"`java` in PATH is JDK {java_major} ; nanozero-uci-1.2.0.jar requires JDK 25+."
        )
    model = _ensure_test_model()
    client = UciClient()
    client.start(UCI_JAR, model, dirichlet_seed=42, handshake_timeout=60.0)
    try:
        yield client
    finally:
        client.quit()


@pytest.mark.slow()
def test_play_one_game_real_uci_returns_samples(started_client: UciClient) -> None:
    """End-to-end : 1 partie complète mcts_sims=50, max_plies=20. Valide samples."""
    config = SelfplayConfig(
        mcts_sims=50,
        max_game_plies=20,
        temperature_switch_ply=5,
        worker_seed=42,
        go_timeout_seconds=30.0,
    )
    rng = np.random.default_rng(42)

    samples = play_one_game(started_client, config, rng)

    assert len(samples) >= 1, "Expected at least 1 sample"
    assert len(samples) <= 20, f"Expected ≤ max_plies=20 samples, got {len(samples)}"

    # Tous samples validés (Sample.__post_init__ ran).
    for s in samples:
        assert s.input_planes.shape == (119, 8, 8)
        assert s.input_planes.dtype == np.float32
        assert s.policy_target.shape == (4672,)
        assert s.policy_target.dtype == np.float32
        assert s.value_target in (-1.0, 0.0, 1.0)
        assert s.turn in (0, 1)
        assert s.ply >= 0

    # Plies monotones, première ply=0, unicité.
    plies = [s.ply for s in samples]
    assert plies == sorted(plies)
    assert plies[0] == 0
    assert len(plies) == len(set(plies))

    # Turn alternés à partir de white.
    assert samples[0].turn == 0
    for i, s in enumerate(samples):
        assert s.turn == i % 2

    # Value targets cohérents : si outcome non-draw, alternance des signs.
    if samples[0].value_target != 0.0 and len(samples) >= 2:
        assert samples[0].value_target == -samples[1].value_target


@pytest.mark.slow()
def test_play_one_game_two_consecutive_games_independent(started_client: UciClient) -> None:
    """2 invocations sur même client : fresh chess.Board() pour chacune."""
    config = SelfplayConfig(
        mcts_sims=50,
        max_game_plies=10,
        temperature_switch_ply=5,
        worker_seed=42,
        go_timeout_seconds=30.0,
    )

    samples1 = play_one_game(started_client, config, np.random.default_rng(42))
    samples2 = play_one_game(started_client, config, np.random.default_rng(43))

    # Les deux parties commencent au ply 0 (fresh board, pas de pollution).
    assert samples1[0].ply == 0
    assert samples2[0].ply == 0
    # Et white-to-move au ply 0.
    assert samples1[0].turn == 0
    assert samples2[0].turn == 0
