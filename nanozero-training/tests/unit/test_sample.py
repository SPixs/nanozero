"""Tests for data/sample.py — Sample dataclass + validation (ADR-002)."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest
from nanozero_training.data.sample import (
    BOARD_SIZE,
    N_INPUT_PLANES,
    N_POLICY,
    Sample,
    make_valid_sample_arrays,
)


def test_sample_valid_construction() -> None:
    s = Sample(**make_valid_sample_arrays())
    assert s.input_planes.shape == (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert s.policy_target.shape == (N_POLICY,)
    assert s.value_target == 0.0
    assert s.turn == 0
    assert s.ply == 0


def test_sample_frozen_immutable() -> None:
    s = Sample(**make_valid_sample_arrays())
    with pytest.raises(FrozenInstanceError):
        s.turn = 1  # type: ignore[misc]


def test_sample_input_planes_wrong_shape() -> None:
    args = make_valid_sample_arrays()
    args["input_planes"] = np.zeros((118, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="input_planes shape"):
        Sample(**args)


def test_sample_input_planes_wrong_dtype() -> None:
    args = make_valid_sample_arrays()
    args["input_planes"] = np.zeros((119, 8, 8), dtype=np.float64)
    with pytest.raises(ValueError, match="input_planes dtype"):
        Sample(**args)


def test_sample_input_planes_not_ndarray() -> None:
    args = make_valid_sample_arrays()
    args["input_planes"] = [[0.0] * 8] * 119  # type: ignore[assignment]
    with pytest.raises(ValueError, match="input_planes must be np.ndarray"):
        Sample(**args)


def test_sample_policy_target_wrong_shape() -> None:
    args = make_valid_sample_arrays()
    args["policy_target"] = np.zeros(4671, dtype=np.float32)
    args["policy_target"][0] = 1.0
    with pytest.raises(ValueError, match="policy_target shape"):
        Sample(**args)


def test_sample_policy_target_wrong_dtype() -> None:
    args = make_valid_sample_arrays()
    args["policy_target"] = np.zeros(N_POLICY, dtype=np.float64)
    args["policy_target"][0] = 1.0
    with pytest.raises(ValueError, match="policy_target dtype"):
        Sample(**args)


def test_sample_policy_target_not_ndarray() -> None:
    args = make_valid_sample_arrays()
    args["policy_target"] = [0.0] * N_POLICY  # type: ignore[assignment]
    with pytest.raises(ValueError, match="policy_target must be np.ndarray"):
        Sample(**args)


def test_sample_policy_target_negative() -> None:
    args = make_valid_sample_arrays()
    policy = np.zeros(N_POLICY, dtype=np.float32)
    policy[0] = 1.5
    policy[1] = -0.5
    args["policy_target"] = policy
    with pytest.raises(ValueError, match="non-negative"):
        Sample(**args)


def test_sample_policy_target_sum_invalid() -> None:
    args = make_valid_sample_arrays()
    policy = np.zeros(N_POLICY, dtype=np.float32)
    policy[0] = 0.5  # sum != 1.0 and != 0.0
    args["policy_target"] = policy
    with pytest.raises(ValueError, match="sum to"):
        Sample(**args)


def test_sample_policy_target_terminal_sum_zero() -> None:
    """Terminal position : policy all zeros is valid."""
    args = make_valid_sample_arrays()
    args["policy_target"] = np.zeros(N_POLICY, dtype=np.float32)
    # No raise expected
    s = Sample(**args)
    assert float(s.policy_target.sum()) == 0.0


def test_sample_value_target_invalid() -> None:
    args = make_valid_sample_arrays()
    args["value_target"] = 0.5
    with pytest.raises(ValueError, match="value_target must be one of"):
        Sample(**args)


def test_sample_value_target_accepts_int() -> None:
    """int 1 should be accepted as 1.0 via float() conversion."""
    args = make_valid_sample_arrays()
    args["value_target"] = 1  # int, not float
    s = Sample(**args)
    assert s.value_target == 1


def test_sample_value_target_not_numeric() -> None:
    args = make_valid_sample_arrays()
    args["value_target"] = "0.0"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="value_target must be int or float"):
        Sample(**args)


def test_sample_turn_invalid() -> None:
    args = make_valid_sample_arrays()
    args["turn"] = 2
    with pytest.raises(ValueError, match="turn must be 0 or 1"):
        Sample(**args)


def test_sample_turn_not_int() -> None:
    args = make_valid_sample_arrays()
    args["turn"] = "white"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="turn must be int"):
        Sample(**args)


def test_sample_ply_negative() -> None:
    args = make_valid_sample_arrays()
    args["ply"] = -1
    with pytest.raises(ValueError, match="ply must be >= 0"):
        Sample(**args)


def test_sample_ply_not_int() -> None:
    args = make_valid_sample_arrays()
    args["ply"] = 1.5  # type: ignore[assignment]
    with pytest.raises(ValueError, match="ply must be int"):
        Sample(**args)


def test_make_valid_sample_arrays_helper() -> None:
    """Sanity check helper produces dict that constructs a valid Sample."""
    args = make_valid_sample_arrays()
    assert set(args.keys()) == {"input_planes", "policy_target", "value_target", "turn", "ply"}
    s = Sample(**args)
    assert isinstance(s, Sample)
