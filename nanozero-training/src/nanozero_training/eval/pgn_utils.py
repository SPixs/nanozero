"""PGN helpers for SPRT resume / progress tracking."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PgnSprtCounts:
    """W/L/D counts from a fastchess PGN, from POV of the first engine name
    matching `challenger_name` (or 'challenger' literal if not set)."""

    total: int
    challenger_wins: int
    challenger_losses: int
    challenger_draws: int


def parse_sprt_counts_from_pgn(
    path: str | Path,
    challenger_name: str = "challenger",
) -> PgnSprtCounts:
    """Walk a fastchess PGN and tally W/L/D for the `challenger_name` engine.

    Used as a fallback when fastchess stdout parsing fails to extract the
    "Score of X vs Y" summary line (Bug #7 hotfix-008).

    Args:
        path: PGN file path. Returns zeros if file doesn't exist.
        challenger_name: engine name set via fastchess `-engine ... name=X`.
            Default 'challenger' matches the convention used in fastchess_runner.

    Returns:
        PgnSprtCounts. POV of challenger_name (so a '1-0' game where
        challenger plays White counts as a challenger_wins).
    """
    path = Path(path)
    if not path.exists():
        return PgnSprtCounts(0, 0, 0, 0)

    white_re = re.compile(r'^\[White "(.+)"\]')
    result_re = re.compile(r'^\[Result "(.+)"\]')

    ch_w = ch_l = ch_d = 0
    current_white: str | None = None
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                mw = white_re.match(stripped)
                if mw:
                    current_white = mw.group(1)
                    continue
                mr = result_re.match(stripped)
                if mr and current_white is not None:
                    r = mr.group(1)
                    challenger_is_white = current_white == challenger_name
                    if r == "1-0":
                        if challenger_is_white:
                            ch_w += 1
                        else:
                            ch_l += 1
                    elif r == "0-1":
                        if challenger_is_white:
                            ch_l += 1
                        else:
                            ch_w += 1
                    elif r == "1/2-1/2":
                        ch_d += 1
                    current_white = None
    except OSError:
        return PgnSprtCounts(0, 0, 0, 0)

    return PgnSprtCounts(
        total=ch_w + ch_l + ch_d,
        challenger_wins=ch_w,
        challenger_losses=ch_l,
        challenger_draws=ch_d,
    )


def count_games_in_pgn(path: str | Path) -> int:
    """Count number of games in a PGN file by counting [Result "..."] tags.

    A PGN file contains one or more games, each with metadata tags including
    [Result "1-0"], [Result "0-1"], [Result "1/2-1/2"], or [Result "*"].
    We count these tag occurrences as a proxy for game count.

    Args:
        path: PGN file path. Tolerates non-existent file (returns 0).

    Returns:
        Number of games. Returns 0 if file doesn't exist or is empty.

    Notes:
        Robuste face aux fichiers tronqués (un Result manquant ne casse pas).
        Ne parse pas le PGN complet (pas de python-chess.pgn) — trop lourd
        pour un simple compte. Le pattern [Result "X"] est suffisant et standard.
    """
    path = Path(path)
    if not path.exists():
        return 0

    count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("[Result ") and stripped.endswith("]"):
                    count += 1
    except OSError:
        return 0
    return count
