"""Unit tests for eval/sprt_result — SPRTResult + parse_sprt_stdout."""

from __future__ import annotations

import dataclasses

import pytest
from nanozero_training.eval.sprt_result import SPRTResult, SPRTStatus, parse_sprt_stdout


def test_parse_h1_accepted() -> None:
    out = (
        "Score of challenger vs baseline: 80 - 20 - 30  [0.730] 130\n"
        "LLR: 2.95 (-2.94, 2.94) [0.00, 5.00]\n"
        "H1 accepted\n"
    )
    res = parse_sprt_stdout(out)
    assert res.status == SPRTStatus.H1_ACCEPTED


def test_parse_h0_accepted() -> None:
    out = (
        "Score of challenger vs baseline: 20 - 80 - 30  [0.270] 130\n"
        "LLR: -2.95 (-2.94, 2.94) [0.00, 5.00]\n"
        "H0 accepted\n"
    )
    res = parse_sprt_stdout(out)
    assert res.status == SPRTStatus.H0_ACCEPTED


def test_parse_hypothesis_test_passed_alias() -> None:
    out = (
        "Score of challenger vs baseline: 50 - 30 - 20  [0.600] 100\n"
        "Hypothesis test passed (LLR=2.95 > 2.94)\n"
    )
    res = parse_sprt_stdout(out)
    assert res.status == SPRTStatus.H1_ACCEPTED


def test_parse_hypothesis_test_failed_alias() -> None:
    out = (
        "Score of challenger vs baseline: 30 - 50 - 20  [0.400] 100\n"
        "Hypothesis test failed (LLR=-2.95 < -2.94)\n"
    )
    res = parse_sprt_stdout(out)
    assert res.status == SPRTStatus.H0_ACCEPTED


def test_parse_score_extracts_wld() -> None:
    out = "Score of A vs B: 23 - 18 - 39  [0.525] 80\n"
    res = parse_sprt_stdout(out)
    assert res.wins == 23
    assert res.losses == 18
    assert res.draws == 39
    assert res.games_played == 80


def test_parse_uses_last_score() -> None:
    out = (
        "Score of A vs B: 1 - 0 - 0  [1.000] 1\n"
        "Score of A vs B: 5 - 3 - 2  [0.600] 10\n"
        "Score of A vs B: 23 - 18 - 39  [0.525] 80\n"
    )
    res = parse_sprt_stdout(out)
    assert res.games_played == 80
    assert res.wins == 23


def test_parse_llr_extracts() -> None:
    out = "Score of A vs B: 10 - 10 - 0  [0.500] 20\n" "LLR: 1.23 (-2.94, 2.94) [0.00, 5.00]\n"
    res = parse_sprt_stdout(out)
    assert res.llr == 1.23


def test_parse_elo_extracts() -> None:
    out = (
        "Score of A vs B: 50 - 30 - 20  [0.600] 100\n"
        "Elo difference: 17.4 +/- 65.5, LOS: 70.2 %\n"
    )
    res = parse_sprt_stdout(out)
    assert res.elo_diff == 17.4


def test_parse_elo_current_fastchess_format() -> None:
    """Phase 12 hotfix-009: current fastchess format "Elo: X" (without 'difference').

    Negative value + presence of "nElo: ..." after must not confuse the regex
    (negative lookbehind \\bn rejects nElo)."""
    out = (
        "LLR: -0.73 (-24.7%) (-2.94, 2.94) [0.00, 5.00]\n"
        "Elo: -19.97 +/- 16.46, nElo: -40.53 +/- 33.31\n"
    )
    res = parse_sprt_stdout(out)
    assert res.elo_diff == -19.97
    assert res.llr == -0.73


def test_parse_elo_doesnt_match_nelo() -> None:
    """Negative lookbehind on 'Elo:' regex must not match 'nElo:'."""
    out = "nElo: 99.0 +/- 5.0\n"
    res = parse_sprt_stdout(out)
    assert res.elo_diff is None


def test_parse_games_current_fastchess_format() -> None:
    """Phase 12 hotfix-012: current fastchess Results block format.

    fastchess no longer emits "Score of X vs Y: 23 - 18 - 39 [0.525] 80" — instead
    emits "Games: N, Wins: W, Losses: L, Draws: D, Points: P (X.X %)" in a Results
    block. Parser must extract from new format AND preserve compatibility with old.
    """
    out = (
        "Results of challenger vs baseline (3+0.03, NULL, NULL, UHO_4060_v2.epd):\n"
        "Elo: 3.47 +/- 11.76, nElo: 20.13 +/- 68.10\n"
        "LOS: 71.88 %, DrawRatio: 94.00 %, PairsRatio: 2.00\n"
        "Games: 100, Wins: 2, Losses: 1, Draws: 97, Points: 50.5 (50.50 %)\n"
        "Ptnml(0-2): [0, 1, 47, 2, 0], WL/DD Ratio: 0.00\n"
        "LLR: 0.07 (2.5%) (-2.94, 2.94) [0.00, 5.00]\n"
    )
    res = parse_sprt_stdout(out)
    assert res.games_played == 100
    assert res.wins == 2
    assert res.losses == 1
    assert res.draws == 97
    assert res.llr == 0.07
    assert res.elo_diff == 3.47
    # Status was RUNNING (no H1/H0), promoted to MAX_GAMES_REACHED by games_played>0.
    assert res.status == SPRTStatus.MAX_GAMES_REACHED


def test_parse_games_takes_priority_over_score() -> None:
    """If both old and new formats are present, the new Games: format wins (more recent)."""
    out = (
        "Score of A vs B: 100 - 50 - 25  [0.643] 175\n"
        "Games: 100, Wins: 2, Losses: 1, Draws: 97, Points: 50.5 (50.50 %)\n"
    )
    res = parse_sprt_stdout(out)
    assert res.games_played == 100
    assert res.wins == 2


def test_parse_h1_from_llr_bounds_only_no_string() -> None:
    """Phase 12 hotfix-014: fastchess never emits "H1 accepted" string.

    Parser must infer H1_ACCEPTED from LLR >= upper_bound, even without explicit
    H1 string in stdout. Format observed: "LLR: 2.97 (...) (-2.94, 2.94) [0.00, 5.00]"
    """
    out = (
        "Games: 1828, Wins: 75, Losses: 24, Draws: 1729, Points: 939.5 (51.39 %)\n"
        "LLR: 2.97 (100.0%) (-2.94, 2.94) [0.00, 5.00]\n"
    )
    res = parse_sprt_stdout(out)
    assert (
        res.status == SPRTStatus.H1_ACCEPTED
    ), f"LLR 2.97 >= upper bound 2.94 must be H1, got {res.status}"
    assert res.llr == 2.97
    assert res.games_played == 1828


def test_parse_h0_from_llr_bounds_only() -> None:
    """LLR <= lower_bound → H0_ACCEPTED, no explicit H0 string needed."""
    out = (
        "Games: 500, Wins: 20, Losses: 100, Draws: 380, Points: 210 (42.00 %)\n"
        "LLR: -3.05 (100.0%) (-2.94, 2.94) [0.00, 5.00]\n"
    )
    res = parse_sprt_stdout(out)
    assert res.status == SPRTStatus.H0_ACCEPTED


def test_parse_llr_between_bounds_stays_max_games() -> None:
    """LLR strictly between bounds + games > 0 → MAX_GAMES_REACHED (not H0/H1)."""
    out = (
        "Games: 2000, Wins: 94, Losses: 44, Draws: 1862, Points: 1025 (51.25 %)\n"
        "LLR: 2.47 (84.0%) (-2.94, 2.94) [0.00, 5.00]\n"
    )
    res = parse_sprt_stdout(out)
    assert (
        res.status == SPRTStatus.MAX_GAMES_REACHED
    ), f"LLR 2.47 between bounds, games>0 must be MAX_GAMES, got {res.status}"


def test_parse_llr_at_upper_bound_exact() -> None:
    """Edge case: LLR exactly at upper bound (>=) → H1_ACCEPTED."""
    out = (
        "Games: 1000, Wins: 60, Losses: 30, Draws: 910\n"
        "LLR: 2.94 (100.0%) (-2.94, 2.94) [0.00, 5.00]\n"
    )
    res = parse_sprt_stdout(out)
    assert res.status == SPRTStatus.H1_ACCEPTED


def test_parse_h1_string_overrides_llr_bounds() -> None:
    """If both an explicit "H1 accepted" string AND LLR-vs-bounds match, prefer
    explicit string (backward compat older fastchess that did emit it)."""
    out = (
        "Games: 1828, Wins: 75, Losses: 24, Draws: 1729\n"
        "LLR: 2.47 (84.0%) (-2.94, 2.94) [0.00, 5.00]\n"  # LLR below bound
        "H1 accepted\n"  # but explicit string says H1
    )
    res = parse_sprt_stdout(out)
    assert res.status == SPRTStatus.H1_ACCEPTED


def test_parse_no_data_returns_error() -> None:
    res = parse_sprt_stdout("")
    assert res.status == SPRTStatus.ERROR
    assert res.games_played == 0
    assert res.llr is None
    assert res.elo_diff is None


def test_parse_games_but_no_decision_returns_max_games() -> None:
    out = "Score of A vs B: 15 - 12 - 3  [0.550] 30\n" "LLR: 0.42 (-2.94, 2.94) [0.00, 5.00]\n"
    res = parse_sprt_stdout(out)
    assert res.status == SPRTStatus.MAX_GAMES_REACHED
    assert res.games_played == 30


def test_parse_raw_output_truncated() -> None:
    long_stdout = "x" * 5000
    res = parse_sprt_stdout(long_stdout)
    assert len(res.raw_output) <= 2000


def test_parse_dataclass_frozen() -> None:
    res = parse_sprt_stdout("")
    with pytest.raises(dataclasses.FrozenInstanceError):
        res.games_played = 99  # type: ignore[misc]


def test_sprt_status_string_values() -> None:
    """Cohérence avec eval.last_decision YAML."""
    assert SPRTStatus.H1_ACCEPTED.value == "h1_accepted"
    assert SPRTStatus.H0_ACCEPTED.value == "h0_accepted"
    assert SPRTStatus.MAX_GAMES_REACHED.value == "max_games"
    assert SPRTStatus.ERROR.value == "error"
    assert SPRTStatus.RUNNING.value == "running"


def test_sprt_result_dataclass_fields() -> None:
    """Sanity check sur la signature du dataclass."""
    res = SPRTResult(
        status=SPRTStatus.H1_ACCEPTED,
        llr=2.95,
        games_played=130,
        wins=80,
        losses=20,
        draws=30,
        elo_diff=42.0,
        raw_output="tail",
    )
    assert res.status == SPRTStatus.H1_ACCEPTED
    assert res.raw_output == "tail"
