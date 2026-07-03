"""Integration: eval/__init__ public API exposes expected symbols."""

from __future__ import annotations


def test_eval_package_exports() -> None:
    from nanozero_training.eval import (
        FastchessConfig,
        FastchessRunner,
        SPRTResult,
        SPRTStatus,
        count_games_in_pgn,
        parse_sprt_stdout,
    )

    # Cohérence avec EvalState.last_decision (persistance YAML).
    assert SPRTStatus.H1_ACCEPTED.value == "h1_accepted"
    assert SPRTStatus.H0_ACCEPTED.value == "h0_accepted"
    assert SPRTStatus.MAX_GAMES_REACHED.value == "max_games"
    assert SPRTStatus.ERROR.value == "error"

    # Sanity de chaque symbol importable.
    assert FastchessConfig.__name__ == "FastchessConfig"
    assert FastchessRunner.__name__ == "FastchessRunner"
    assert SPRTResult.__name__ == "SPRTResult"
    assert callable(count_games_in_pgn)
    assert callable(parse_sprt_stdout)
