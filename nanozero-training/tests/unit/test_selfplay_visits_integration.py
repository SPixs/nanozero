"""Integration tests : parse_visits_line -> visits_to_policy_target chain.

Vérifie que la sortie du parser UCI alimente correctement le builder de
policy target consommé par le training loop (phase 1.0.0-6).
"""

from __future__ import annotations

import chess
import numpy as np
import pytest
from nanozero_training.network.move_encoding import visits_to_policy_target
from nanozero_training.selfplay import parse_visits_line  # public API


def test_visits_parser_to_policy_target_full_chain() -> None:
    """UCI line -> parsed visits -> normalized policy target (sum=1.0)."""
    line = "info string visits e2e4 80 d2d4 15 g1f3 5"
    visits = parse_visits_line(line)
    assert visits == {"e2e4": 80, "d2d4": 15, "g1f3": 5}

    board = chess.Board()
    target = visits_to_policy_target(visits, board)
    assert target.shape == (4672,)
    assert target.dtype == np.float32
    assert float(target.sum()) == pytest.approx(1.0, abs=1e-5)
    # 3 indices non-zero.
    assert int(np.count_nonzero(target)) == 3


def test_visits_parser_terminal_to_policy_target() -> None:
    """Terminal position : visits vide -> policy target all zeros."""
    line = "info string visits"
    visits = parse_visits_line(line)
    assert visits == {}

    board = chess.Board()
    target = visits_to_policy_target(visits, board)
    assert target.shape == (4672,)
    assert float(target.sum()) == 0.0


def test_selfplay_package_public_api() -> None:
    """Sanity check : symbols exportés au top-level du package selfplay."""
    from nanozero_training import selfplay

    assert callable(selfplay.parse_visits_line)
    assert callable(selfplay.temperature_sample)
    assert callable(selfplay.argmax_move)
    assert callable(selfplay.select_move)
    # UciClient class
    assert isinstance(selfplay.UciClient, type)
    # Exceptions
    assert issubclass(selfplay.UciTimeoutError, RuntimeError)
    assert issubclass(selfplay.UciCrashError, RuntimeError)
    # UciResult dataclass : verify fields via dataclasses introspection.
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(selfplay.UciResult)}
    assert field_names == {"visits", "bestmove"}
