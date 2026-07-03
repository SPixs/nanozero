"""Slow integration tests : real UCI subprocess (requires built JAR).

Skip propre si nanozero-uci-1.2.0.jar absent (cohérent pattern parity tests).
Le modèle de test est généré on-the-fly via scripts/generate_gen0_model.py
(seed=42, déterministe, ~5-10 MB), caché dans tests/fixtures/.

Le .npz généré n'est PAS committé (trop volumineux, regenable rapidement).
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from nanozero_training.selfplay.uci_client import UciClient

REPO_ROOT = Path(__file__).resolve().parents[3]
UCI_JAR = REPO_ROOT / "nanozero-uci" / "target" / "nanozero-uci-1.2.0.jar"

TEST_MODEL_DIR = Path(__file__).resolve().parents[1] / "fixtures"
TEST_MODEL = TEST_MODEL_DIR / "test-gen0-model.npz"


def _ensure_test_model() -> Path:
    """Generate test-gen0-model.npz once if absent (seed=42, déterministe)."""
    if TEST_MODEL.exists():
        return TEST_MODEL
    TEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    script = REPO_ROOT / "nanozero-training" / "scripts" / "generate_gen0_model.py"
    subprocess.run(
        [
            sys.executable,
            str(script),
            "--seed",
            "42",
            "--output",
            str(TEST_MODEL),
        ],
        check=True,
        capture_output=True,
    )
    return TEST_MODEL


def _java_version_major() -> int | None:
    """Retourne la version majeure de `java` dans le PATH, ou None si introuvable."""
    try:
        result = subprocess.run(
            ["java", "--version"], capture_output=True, text=True, timeout=10.0, check=False
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = (result.stdout or "") + (result.stderr or "")
    # Format attendu : 'openjdk 25.0.1 ...' ou 'java 25.0.1 ...'
    for line in output.splitlines():
        parts = line.split()
        for tok in parts:
            if tok and tok[0].isdigit():
                # eg "25.0.10" ou "21.0.10"
                major_str = tok.split(".")[0]
                if major_str.isdigit():
                    return int(major_str)
    return None


@pytest.fixture()
def uci_client() -> Iterator[UciClient]:
    """Start a real UciClient. Skip si JAR absent ou Java < 25 (JAR compilé JDK 25)."""
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
            f"`java` in PATH is JDK {java_major} ; nanozero-uci-1.2.0.jar requires JDK 25+. "
            "Set JAVA_HOME and PATH to a JDK 25 install for slow integration tests."
        )
    model = _ensure_test_model()
    client = UciClient()
    client.start(UCI_JAR, model, dirichlet_seed=42, handshake_timeout=60.0)
    try:
        yield client
    finally:
        client.quit()


@pytest.mark.slow()
def test_subprocess_handshake_succeeds(uci_client: UciClient) -> None:
    """Start + handshake (uciok + readyok) sans timeout/crash."""
    assert uci_client.is_alive()


@pytest.mark.slow()
def test_subprocess_go_nodes_startpos_returns_visits(uci_client: UciClient) -> None:
    """`position startpos` + `go nodes 100` retourne non-empty visits + bestmove."""
    uci_client.new_game(timeout=10.0)
    result = uci_client.go_nodes("position startpos", nodes=100, timeout=60.0)
    assert result.bestmove is not None, "Expected non-null bestmove on startpos"
    assert result.visits, "Expected non-empty visits dict on startpos"
    assert all(isinstance(c, int) and c >= 0 for c in result.visits.values())
    assert (
        result.bestmove in result.visits
    ), f"bestmove {result.bestmove} should be in visits {list(result.visits.keys())}"


@pytest.mark.slow()
def test_subprocess_two_consecutive_games(uci_client: UciClient) -> None:
    """new_game + go + new_game + go : 2 résultats indépendants."""
    uci_client.new_game(timeout=10.0)
    r1 = uci_client.go_nodes("position startpos", nodes=50, timeout=30.0)
    uci_client.new_game(timeout=10.0)
    r2 = uci_client.go_nodes("position startpos", nodes=50, timeout=30.0)
    assert r1.bestmove is not None
    assert r2.bestmove is not None
    # Dirichlet noise rend les distributions probablement (mais pas strictement)
    # différentes — pas d'assertion stricte.


@pytest.mark.slow()
def test_subprocess_position_with_moves(uci_client: UciClient) -> None:
    """`position startpos moves e2e4 e7e5` fonctionne."""
    uci_client.new_game(timeout=10.0)
    result = uci_client.go_nodes("position startpos moves e2e4 e7e5", nodes=50, timeout=30.0)
    assert result.bestmove is not None
    assert result.visits, "Expected non-empty visits after 2 plies"
