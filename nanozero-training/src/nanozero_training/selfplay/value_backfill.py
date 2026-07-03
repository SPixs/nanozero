"""Backfill value_target into samples after game outcome determined.

A Sample collected during a game has no value_target initially (game outcome
unknown mid-game). After game ends, we know the outcome from white's POV
(-1.0 / 0.0 / +1.0). For each sample, value_target is set to outcome relative
to the side-to-move at that sample's position :
- sample.turn == 0 (white to move) : value_target = outcome_white
- sample.turn == 1 (black to move) : value_target = -outcome_white

Cohérent SPEC-training §6.2 et §5.1.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from nanozero_training.data.sample import Sample

# Partial sample tuple : (input_planes, policy_target, turn, ply).
PartialSample = tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], int, int]


def backfill_value_targets(
    samples_partial: list[PartialSample],
    outcome_white: float,
) -> list[Sample]:
    """Build final Sample list with value_target filled from game outcome.

    Args:
        samples_partial: list de tuples (input_planes, policy_target, turn, ply)
                         collectés pendant la partie (value pending).
        outcome_white: outcome from white POV. ∈ {-1.0, 0.0, +1.0}.

    Returns:
        list de Sample validés (Sample.__post_init__ ran on each).

    Raises:
        ValueError: si outcome_white n'est pas dans {-1.0, 0.0, +1.0}.

    Note : Sample frozen → reconstruction de nouveaux Sample (pas de mutation
    in-place). Validation via Sample.__post_init__ (shape/dtype/range phase 2).
    """
    if outcome_white not in (-1.0, 0.0, 1.0):
        raise ValueError(f"outcome_white must be in (-1.0, 0.0, 1.0), got {outcome_white}")

    samples: list[Sample] = []
    for input_planes, policy_target, turn, ply in samples_partial:
        value_target = outcome_white if turn == 0 else -outcome_white
        samples.append(
            Sample(
                input_planes=input_planes,
                policy_target=policy_target,
                value_target=value_target,
                turn=turn,
                ply=ply,
            )
        )
    return samples


def outcome_white_from_result(result_str: str) -> float:
    """Convert chess.Board.result(claim_draw=True) string to white-POV outcome.

    Args:
        result_str: "1-0", "0-1", "1/2-1/2", ou "*" (game not over).

    Returns:
        +1.0 (white wins), -1.0 (black wins), 0.0 (draw).

    Raises:
        ValueError: si result_str == "*" (game ongoing) ou format inconnu.
    """
    if result_str == "1-0":
        return 1.0
    if result_str == "0-1":
        return -1.0
    if result_str == "1/2-1/2":
        return 0.0
    raise ValueError(f"Cannot determine outcome from result {result_str!r}")
