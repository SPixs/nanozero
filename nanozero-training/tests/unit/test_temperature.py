"""Unit tests for selfplay/temperature.py."""

from __future__ import annotations

import chess
import numpy as np
import pytest
from nanozero_training.selfplay.temperature import (
    TAU_MIN,
    argmax_move,
    select_move,
    temperature_sample,
)


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# ----- argmax_move -----


def test_argmax_single_move() -> None:
    board = chess.Board()
    move = argmax_move({"e2e4": 10}, board)
    assert move == chess.Move.from_uci("e2e4")


def test_argmax_max_count_wins() -> None:
    board = chess.Board()
    move = argmax_move({"e2e4": 50, "d2d4": 30, "g1f3": 20}, board)
    assert move == chess.Move.from_uci("e2e4")


def test_argmax_tie_break_lexicographic() -> None:
    """2+ moves même count → uci_str plus petit gagne (a < e dans tri lex)."""
    board = chess.Board()
    # Tie : a2a3 et e2e4 ont même count. 'a2a3' < 'e2e4' lexicographiquement.
    move = argmax_move({"e2e4": 50, "a2a3": 50, "d2d4": 30}, board)
    assert move == chess.Move.from_uci("a2a3")


def test_argmax_empty_raises() -> None:
    board = chess.Board()
    with pytest.raises(ValueError, match="empty visits"):
        argmax_move({}, board)


# ----- temperature_sample -----


def test_temperature_sample_returns_chess_move() -> None:
    board = chess.Board()
    move = temperature_sample({"e2e4": 10, "d2d4": 5}, board, tau=1.0, rng=_rng(42))
    assert isinstance(move, chess.Move)
    assert move.uci() in {"e2e4", "d2d4"}


def test_temperature_sample_tau_1_proportional() -> None:
    """Avec tau=1.0 et 10000 samples, fréquence ≈ proportion exacte."""
    board = chess.Board()
    visits = {"e2e4": 80, "d2d4": 20}
    rng = _rng(42)
    counts = {"e2e4": 0, "d2d4": 0}
    n_samples = 10_000
    for _ in range(n_samples):
        m = temperature_sample(visits, board, tau=1.0, rng=rng)
        counts[m.uci()] += 1
    freq_e2e4 = counts["e2e4"] / n_samples
    # Tolerance ~3 stddev : sqrt(p(1-p)/n) approx 0.004 pour p=0.8, n=10000.
    assert 0.78 <= freq_e2e4 <= 0.82, f"frequency e2e4 = {freq_e2e4}, expected ≈ 0.8"


def test_temperature_sample_tau_low_concentrates_max() -> None:
    """Avec tau=0.1 (concentration forte), fréquence du max >> proportion brute."""
    board = chess.Board()
    visits = {"e2e4": 80, "d2d4": 20}
    rng = _rng(42)
    counts = {"e2e4": 0, "d2d4": 0}
    n_samples = 1_000
    for _ in range(n_samples):
        m = temperature_sample(visits, board, tau=0.1, rng=rng)
        counts[m.uci()] += 1
    freq_e2e4 = counts["e2e4"] / n_samples
    # tau=0.1 → visits**10. ratio e2e4/d2d4 = (80/20)**10 ≈ 1M, donc e2e4 should dominate ≫ 99%.
    assert freq_e2e4 > 0.95


def test_temperature_sample_tau_below_min_clamped() -> None:
    """tau < TAU_MIN clamp à TAU_MIN (pas de crash)."""
    board = chess.Board()
    visits = {"e2e4": 80, "d2d4": 20}
    rng = _rng(42)
    # tau=0.01 < TAU_MIN=0.05 → clamp à 0.05. Pas de overflow attendu.
    move = temperature_sample(visits, board, tau=0.01, rng=rng)
    assert isinstance(move, chess.Move)


def test_temperature_sample_zero_tau_raises() -> None:
    board = chess.Board()
    with pytest.raises(ValueError, match="tau must be > 0"):
        temperature_sample({"e2e4": 10}, board, tau=0.0, rng=_rng())


def test_temperature_sample_negative_tau_raises() -> None:
    board = chess.Board()
    with pytest.raises(ValueError, match="tau must be > 0"):
        temperature_sample({"e2e4": 10}, board, tau=-1.0, rng=_rng())


def test_temperature_sample_empty_raises() -> None:
    board = chess.Board()
    with pytest.raises(ValueError, match="empty visits"):
        temperature_sample({}, board, tau=1.0, rng=_rng())


def test_temperature_sample_all_zero_visits_raises() -> None:
    board = chess.Board()
    with pytest.raises(ValueError, match="All visits are zero"):
        temperature_sample({"e2e4": 0, "d2d4": 0}, board, tau=1.0, rng=_rng())


def test_temperature_sample_reproducible_with_seed() -> None:
    """Même seed → même séquence de moves samplés."""
    board = chess.Board()
    visits = {"e2e4": 60, "d2d4": 40, "g1f3": 20}
    rng_a = _rng(42)
    rng_b = _rng(42)
    moves_a = [temperature_sample(visits, board, tau=1.0, rng=rng_a) for _ in range(20)]
    moves_b = [temperature_sample(visits, board, tau=1.0, rng=rng_b) for _ in range(20)]
    assert moves_a == moves_b


def test_temperature_min_constant() -> None:
    """Sanity check TAU_MIN constant."""
    assert TAU_MIN == 0.05


# ----- select_move -----


def test_select_move_early_ply_samples() -> None:
    """ply < temperature_switch_ply → utilise temperature_sample."""
    board = chess.Board()
    visits = {"e2e4": 50, "d2d4": 30, "g1f3": 20}
    # Pas d'assertion directe sur le coup retourné (peut varier),
    # mais le RNG doit être consommé (vs argmax qui n'y touche pas).
    rng = _rng(42)
    state_before = rng.bit_generator.state
    select_move(visits, board, ply=10, temperature_switch_ply=30, tau=1.0, rng=rng)
    state_after = rng.bit_generator.state
    assert state_before != state_after  # RNG consommé


def test_select_move_late_ply_argmax() -> None:
    """ply >= temperature_switch_ply → utilise argmax_move (RNG inchangé)."""
    board = chess.Board()
    visits = {"e2e4": 50, "d2d4": 30, "g1f3": 20}
    rng = _rng(42)
    state_before = rng.bit_generator.state
    move = select_move(visits, board, ply=50, temperature_switch_ply=30, tau=1.0, rng=rng)
    state_after = rng.bit_generator.state
    assert state_before == state_after  # RNG inchangé (argmax pas de tirage)
    assert move == chess.Move.from_uci("e2e4")  # argmax du dict


def test_select_move_at_switch_ply_uses_argmax() -> None:
    """ply == temperature_switch_ply → argmax (frontière inclusive vers argmax)."""
    board = chess.Board()
    visits = {"e2e4": 50, "d2d4": 30}
    rng = _rng(42)
    state_before = rng.bit_generator.state
    move = select_move(visits, board, ply=30, temperature_switch_ply=30, tau=1.0, rng=rng)
    state_after = rng.bit_generator.state
    assert state_before == state_after
    assert move == chess.Move.from_uci("e2e4")
