"""SPRT result types and fastchess stdout parser."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class SPRTStatus(str, Enum):
    """SPRT decision status.

    Values are persisted in run_state.yaml eval.last_decision field (string).
    """

    RUNNING = "running"
    H0_ACCEPTED = "h0_accepted"  # Challenger NOT proven better
    H1_ACCEPTED = "h1_accepted"  # Challenger proven better (promote!)
    MAX_GAMES_REACHED = "max_games"  # Inconclusive: max games hit before LLR converged
    ERROR = "error"  # fastchess crashed or unparseable output


@dataclass(frozen=True)
class SPRTResult:
    """Result of an SPRT test run.

    Fields:
        status: terminal status (one of H0/H1/MAX_GAMES/ERROR).
        llr: last reported Log Likelihood Ratio (None if not reported).
        games_played: total games played.
        wins: wins for challenger.
        losses: losses for challenger.
        draws: draws.
        elo_diff: estimated Elo difference challenger - baseline (None if not reported).
        raw_output: tail of stdout (last 2KB, for debugging).
    """

    status: SPRTStatus
    llr: float | None
    games_played: int
    wins: int
    losses: int
    draws: int
    elo_diff: float | None
    raw_output: str = ""


# Regex patterns matching typical fastchess stdout output.
# Examples of lines parsed (current fastchess format observed 2026-05-16):
#   "Score of challenger vs baseline: 23 - 18 - 39  [0.525] 80"           (older fastchess)
#   "Games: 100, Wins: 2, Losses: 1, Draws: 97, Points: 50.5 (50.50 %)"   (current fastchess)
#   "Elo: -19.97 +/- 16.46, nElo: -40.53 +/- 33.31"                       (current fastchess)
#   "LLR: -0.73 (-24.7%) (-2.94, 2.94) [0.00, 5.00]"                      (current fastchess)
#   "H1 accepted" / "Hypothesis test passed"
#   "H0 accepted" / "Hypothesis test failed"
#
# Phase 12 hotfix-009 — Updated _ELO_RE to match the current fastchess format
# ("Elo: X +/- Y") which dropped the "difference" word.
# Phase 12 hotfix-012 — Added _GAMES_RE for the current fastchess Results block
# ("Games: N, Wins: W, Losses: L, Draws: D"). Older "Score of X vs Y" line no
# longer emitted by current fastchess. PGN fallback in fastchess_runner is the
# ultimate safety net.

_SCORE_RE = re.compile(r"Score of \S+ vs \S+:\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s+\[[\d.]+\]\s+(\d+)")
_GAMES_RE = re.compile(r"Games:\s*(\d+),\s*Wins:\s*(\d+),\s*Losses:\s*(\d+),\s*Draws:\s*(\d+)")
_LLR_RE = re.compile(r"LLR:\s*(-?[\d.]+)")
# Phase 12 hotfix-014 — fastchess does NOT emit "H1 accepted" / "H0 accepted" strings.
# It only writes the LLR line with bounds : "LLR: 2.97 (...) (-2.94, 2.94) [0.00, 5.00]".
# It's up to the caller to compare LLR against bounds to determine the SPRT verdict.
# Capture LLR + both bounds (lower=H0, upper=H1) from the same line.
_LLR_WITH_BOUNDS_RE = re.compile(
    r"LLR:\s*(-?[\d.]+)\s*(?:\([^)]+\)\s*)?\((-?[\d.]+)\s*,\s*(-?[\d.]+)\)"
)
_ELO_RE = re.compile(r"\bElo(?:\s+difference)?:\s*(-?[\d.]+)")
_H1_ACCEPTED_RE = re.compile(r"\bH1 accepted\b|Hypothesis test passed", re.IGNORECASE)
_H0_ACCEPTED_RE = re.compile(r"\bH0 accepted\b|Hypothesis test failed", re.IGNORECASE)


def parse_sprt_stdout(stdout: str) -> SPRTResult:
    """Parse fastchess stdout into an SPRTResult.

    Args:
        stdout: full or partial fastchess stdout text.

    Returns:
        SPRTResult with best-effort extraction. status defaults to MAX_GAMES_REACHED
        if no explicit H0/H1 found but games were played. ERROR if nothing parseable.
    """
    # Status detection : H1 prioritaire sur H0 (cas pathologique improbable mais sûr).
    if _H1_ACCEPTED_RE.search(stdout):
        status = SPRTStatus.H1_ACCEPTED
    elif _H0_ACCEPTED_RE.search(stdout):
        status = SPRTStatus.H0_ACCEPTED
    else:
        status = SPRTStatus.RUNNING

    # Try current fastchess format first ("Games: N, Wins: W, Losses: L, Draws: D"),
    # fall back to older "Score of X vs Y" format if not found.
    games_matches = list(_GAMES_RE.finditer(stdout))
    if games_matches:
        last = games_matches[-1]
        games_played = int(last.group(1))
        wins = int(last.group(2))
        losses = int(last.group(3))
        draws = int(last.group(4))
    else:
        score_matches = list(_SCORE_RE.finditer(stdout))
        if score_matches:
            last = score_matches[-1]
            wins = int(last.group(1))
            losses = int(last.group(2))
            draws = int(last.group(3))
            games_played = int(last.group(4))
        else:
            wins = losses = draws = games_played = 0

    llr_matches = list(_LLR_RE.finditer(stdout))
    llr = float(llr_matches[-1].group(1)) if llr_matches else None

    elo_matches = list(_ELO_RE.finditer(stdout))
    elo = float(elo_matches[-1].group(1)) if elo_matches else None

    # Phase 12 hotfix-014 — Status from LLR vs bounds.
    # fastchess never emits "H1 accepted" / "H0 accepted" — only the LLR line with
    # bounds. We need to compare LLR ourselves to determine the SPRT verdict.
    # Done BEFORE the RUNNING→MAX_GAMES upgrade so we catch H1/H0 first.
    bounds_matches = list(_LLR_WITH_BOUNDS_RE.finditer(stdout))
    if bounds_matches and llr is not None and status == SPRTStatus.RUNNING:
        last = bounds_matches[-1]
        lower_bound = float(last.group(2))
        upper_bound = float(last.group(3))
        if llr >= upper_bound:
            status = SPRTStatus.H1_ACCEPTED
        elif llr <= lower_bound:
            status = SPRTStatus.H0_ACCEPTED

    # Promote RUNNING to MAX_GAMES_REACHED if games_played > 0 but no H0/H1 decision.
    if status == SPRTStatus.RUNNING and games_played > 0:
        status = SPRTStatus.MAX_GAMES_REACHED

    # Si absolument rien de parsable.
    if games_played == 0 and llr is None and elo is None and status == SPRTStatus.RUNNING:
        status = SPRTStatus.ERROR

    return SPRTResult(
        status=status,
        llr=llr,
        games_played=games_played,
        wins=wins,
        losses=losses,
        draws=draws,
        elo_diff=elo,
        raw_output=stdout[-2000:],
    )
