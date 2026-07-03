"""Move selection with temperature for AlphaZero self-play (SPEC-training §4)."""

from __future__ import annotations

import chess
import numpy as np

# Tau lower bound to avoid overflow on counts ** (1/tau) for tau very close to 0.
# Practical: SPEC §4 uses tau=1.0 then argmax (tau→0). Clamp never activated in
# default flow but defensive against config typos / future low-tau experiments.
TAU_MIN = 0.05


def temperature_sample(
    visits: dict[str, int],
    board: chess.Board,
    tau: float,
    rng: np.random.Generator,
) -> chess.Move:
    """Sample a move according to visits ** (1/tau) normalized.

    Args:
        visits: {uci_move: count}, non-empty with at least one count > 0.
        board: current chess.Board (currently unused — accepted for API uniformity
               with argmax_move / select_move and future hardening).
        tau: temperature in (0, inf). Use argmax_move directly for tau ≈ 0.
        rng: numpy RNG (np.random.default_rng(seed)) for reproducibility.

    Returns:
        Sampled chess.Move (parsed from UCI string ; legality unchecked).

    Raises:
        ValueError: si visits empty, tau <= 0, ou all counts == 0.

    Numerical notes:
        - tau < TAU_MIN clamp à TAU_MIN (protection overflow visits**(1/tau)).
        - Calculs en float64 pour stabilité, normalisation finale.
    """
    _ = board  # API uniformity, currently unused
    if not visits:
        raise ValueError("Cannot sample from empty visits dict")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")

    tau_safe = max(tau, TAU_MIN)

    ucis = list(visits.keys())
    counts = np.array([visits[u] for u in ucis], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        raise ValueError("All visits are zero — cannot sample")

    weighted = counts ** (1.0 / tau_safe)
    probs = weighted / weighted.sum()
    chosen_idx = rng.choice(len(ucis), p=probs)
    return chess.Move.from_uci(ucis[chosen_idx])


def argmax_move(visits: dict[str, int], board: chess.Board) -> chess.Move:
    """Return the move with maximum visit count (deterministic tie-break).

    Tie-break: lexicographic order on UCI strings (smallest first).

    Args:
        visits: {uci_move: count}, non-empty.
        board: current chess.Board (unused — API uniformity).

    Returns:
        chess.Move with max count, tie-broken lexicographically.

    Raises:
        ValueError: si visits empty.
    """
    _ = board  # API uniformity, currently unused
    if not visits:
        raise ValueError("Cannot argmax over empty visits dict")
    # Sort by (-count, uci) : count décroissant, UCI ascendant en cas d'égalité.
    sorted_items = sorted(visits.items(), key=lambda kv: (-kv[1], kv[0]))
    best_uci, _ = sorted_items[0]
    return chess.Move.from_uci(best_uci)


def select_move(
    visits: dict[str, int],
    board: chess.Board,
    ply: int,
    temperature_switch_ply: int,
    tau: float,
    rng: np.random.Generator,
) -> chess.Move:
    """High-level move selection (SPEC-training §4 convention).

    - plies [0, temperature_switch_ply) : temperature_sample avec `tau`.
    - plies ≥ temperature_switch_ply : argmax_move (tau → 0).

    Args:
        visits: {uci_move: count}, non-empty avec count > 0.
        board: current chess.Board.
        ply: 0-indexed ply number depuis le début de la partie.
        temperature_switch_ply: ply à partir duquel on bascule en argmax
                                (défaut SPEC : 30).
        tau: température pour la phase sampling (défaut SPEC : 1.0).
        rng: numpy RNG.

    Returns:
        chess.Move sélectionné.
    """
    if ply < temperature_switch_ply:
        return temperature_sample(visits, board, tau, rng)
    return argmax_move(visits, board)
