"""Evaluation module: SPRT testing via fastchess subprocess (ADR-005)."""

from nanozero_training.eval.fastchess_runner import FastchessConfig, FastchessRunner
from nanozero_training.eval.pgn_utils import count_games_in_pgn
from nanozero_training.eval.sprt_result import (
    SPRTResult,
    SPRTStatus,
    parse_sprt_stdout,
)

__all__ = [
    "FastchessConfig",
    "FastchessRunner",
    "SPRTResult",
    "SPRTStatus",
    "count_games_in_pgn",
    "parse_sprt_stdout",
]
