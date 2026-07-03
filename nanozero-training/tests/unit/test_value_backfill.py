"""Unit tests for selfplay/value_backfill.py."""

from __future__ import annotations

import numpy as np
import pytest
from nanozero_training.data.sample import BOARD_SIZE, N_INPUT_PLANES, N_POLICY
from nanozero_training.selfplay.value_backfill import (
    backfill_value_targets,
    outcome_white_from_result,
)


def _make_partial(turn: int, ply: int) -> tuple:
    planes = np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    policy = np.zeros(N_POLICY, dtype=np.float32)
    policy[0] = 1.0
    return (planes, policy, turn, ply)


# ----- backfill_value_targets -----


def test_backfill_white_wins() -> None:
    """outcome_white=+1.0, samples alternés turn 0/1 -> values +1/-1 alternés."""
    partials = [_make_partial(turn=i % 2, ply=i) for i in range(4)]
    samples = backfill_value_targets(partials, outcome_white=1.0)
    assert len(samples) == 4
    assert samples[0].value_target == 1.0  # turn=0
    assert samples[1].value_target == -1.0  # turn=1
    assert samples[2].value_target == 1.0
    assert samples[3].value_target == -1.0


def test_backfill_black_wins() -> None:
    """outcome_white=-1.0, values opposés (turn=0 -> -1, turn=1 -> +1)."""
    partials = [_make_partial(turn=i % 2, ply=i) for i in range(4)]
    samples = backfill_value_targets(partials, outcome_white=-1.0)
    assert samples[0].value_target == -1.0
    assert samples[1].value_target == 1.0
    assert samples[2].value_target == -1.0
    assert samples[3].value_target == 1.0


def test_backfill_draw() -> None:
    """outcome_white=0.0, all value_targets == 0.0 (both signs)."""
    partials = [_make_partial(turn=i % 2, ply=i) for i in range(4)]
    samples = backfill_value_targets(partials, outcome_white=0.0)
    for s in samples:
        assert s.value_target == 0.0


def test_backfill_empty_list() -> None:
    """[] -> [] sans crash."""
    assert backfill_value_targets([], outcome_white=1.0) == []


def test_backfill_invalid_outcome_raises() -> None:
    with pytest.raises(ValueError, match="outcome_white must be in"):
        backfill_value_targets([_make_partial(0, 0)], outcome_white=0.5)


def test_backfill_returns_validated_samples() -> None:
    """Samples retournés passent Sample.__post_init__ (shape/dtype/range checks)."""
    partials = [_make_partial(turn=0, ply=0), _make_partial(turn=1, ply=1)]
    samples = backfill_value_targets(partials, outcome_white=1.0)
    for s in samples:
        # Shape valide
        assert s.input_planes.shape == (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
        assert s.policy_target.shape == (N_POLICY,)
        # Value dans range valide
        assert s.value_target in (-1.0, 0.0, 1.0)


def test_backfill_preserves_ply_and_turn() -> None:
    """ply et turn préservés dans le Sample final."""
    partials = [_make_partial(turn=0, ply=5), _make_partial(turn=1, ply=6)]
    samples = backfill_value_targets(partials, outcome_white=1.0)
    assert samples[0].ply == 5
    assert samples[0].turn == 0
    assert samples[1].ply == 6
    assert samples[1].turn == 1


# ----- outcome_white_from_result -----


def test_outcome_white_from_result_white_wins() -> None:
    assert outcome_white_from_result("1-0") == 1.0


def test_outcome_white_from_result_black_wins() -> None:
    assert outcome_white_from_result("0-1") == -1.0


def test_outcome_white_from_result_draw() -> None:
    assert outcome_white_from_result("1/2-1/2") == 0.0


def test_outcome_white_from_result_ongoing_raises() -> None:
    with pytest.raises(ValueError, match="Cannot determine outcome"):
        outcome_white_from_result("*")


def test_outcome_white_from_result_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Cannot determine outcome"):
        outcome_white_from_result("unknown-format")
