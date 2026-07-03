"""Sample dataclass for self-play data (ADR-002).

A Sample represents one training instance: a position (input planes), its
MCTS policy target (visit distribution at root), the game outcome (value
target), and metadata (turn, ply).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

# Constants from SPEC-nn §3.
N_INPUT_PLANES = 119
BOARD_SIZE = 8
N_POLICY = 4672
VALID_VALUE_TARGETS = (-1.0, 0.0, 1.0)
VALID_TURNS = (0, 1)


@dataclass(frozen=True)
class Sample:
    """One self-play training sample.

    Attributes:
        input_planes: shape (119, 8, 8) float32 — position encoding
        policy_target: shape (4672,) float32 — visit distribution sum ≈ 1.0
                       (or all zeros if position is terminal)
        value_target: in {-1.0, 0.0, +1.0} — outcome from the side-to-move's
                      perspective
        turn: 0 (white to move) or 1 (black to move)
        ply: >= 0, position depth in the game

    Validation occurs in __post_init__: raises ValueError on schema mismatch.
    """

    input_planes: npt.NDArray[np.float32]
    policy_target: npt.NDArray[np.float32]
    value_target: float
    turn: int
    ply: int

    def __post_init__(self) -> None:
        self._validate_input_planes()
        self._validate_policy_target()
        self._validate_value_target()
        self._validate_turn()
        self._validate_ply()

    def _validate_input_planes(self) -> None:
        if not isinstance(self.input_planes, np.ndarray):
            raise ValueError(
                f"input_planes must be np.ndarray, got {type(self.input_planes).__name__}"
            )
        expected_shape = (N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE)
        if self.input_planes.shape != expected_shape:
            raise ValueError(
                f"input_planes shape must be {expected_shape}, got {self.input_planes.shape}"
            )
        if self.input_planes.dtype != np.float32:
            raise ValueError(f"input_planes dtype must be float32, got {self.input_planes.dtype}")

    def _validate_policy_target(self) -> None:
        if not isinstance(self.policy_target, np.ndarray):
            raise ValueError(
                f"policy_target must be np.ndarray, got {type(self.policy_target).__name__}"
            )
        if self.policy_target.shape != (N_POLICY,):
            raise ValueError(
                f"policy_target shape must be ({N_POLICY},), got {self.policy_target.shape}"
            )
        if self.policy_target.dtype != np.float32:
            raise ValueError(f"policy_target dtype must be float32, got {self.policy_target.dtype}")
        if not bool(np.all(self.policy_target >= 0)):
            raise ValueError("policy_target must be non-negative")
        # Allow sum ≈ 1.0 (training) or sum == 0.0 (terminal position).
        s = float(self.policy_target.sum())
        valid_training = math.isclose(s, 1.0, abs_tol=1e-5)
        valid_terminal = math.isclose(s, 0.0, abs_tol=1e-7)
        if not (valid_training or valid_terminal):
            raise ValueError(
                f"policy_target must sum to ≈ 1.0 (training) or 0.0 (terminal), got {s}"
            )

    def _validate_value_target(self) -> None:
        if not isinstance(self.value_target, int | float):
            raise ValueError(
                f"value_target must be int or float, got {type(self.value_target).__name__}"
            )
        if float(self.value_target) not in VALID_VALUE_TARGETS:
            raise ValueError(
                f"value_target must be one of {VALID_VALUE_TARGETS}, got {self.value_target}"
            )

    def _validate_turn(self) -> None:
        if not isinstance(self.turn, int) or isinstance(self.turn, bool):
            raise ValueError(f"turn must be int, got {type(self.turn).__name__}")
        if self.turn not in VALID_TURNS:
            raise ValueError(f"turn must be 0 or 1, got {self.turn}")

    def _validate_ply(self) -> None:
        if not isinstance(self.ply, int) or isinstance(self.ply, bool):
            raise ValueError(f"ply must be int, got {type(self.ply).__name__}")
        if self.ply < 0:
            raise ValueError(f"ply must be >= 0, got {self.ply}")


def make_valid_sample_arrays() -> dict[str, Any]:
    """Helper for tests: produce valid arrays/values for a Sample."""
    planes = np.zeros((N_INPUT_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
    policy = np.zeros(N_POLICY, dtype=np.float32)
    policy[0] = 1.0  # peaked at index 0, sums to 1.0
    return {
        "input_planes": planes,
        "policy_target": policy,
        "value_target": 0.0,
        "turn": 0,
        "ply": 0,
    }
