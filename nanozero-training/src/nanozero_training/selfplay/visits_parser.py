"""Parser for UCI `info string visits` lines (SPEC-uci §6.5).

Format figé v1.2.0 :
    info string visits <move1> <count1> <move2> <count2> ... <moveN> <countN>

Cas particuliers :
- Position terminale : `info string visits` sans pairs (préfixe seul).
- `go nodes 0` ou search interrompue : ligne potentiellement vide.
- simulationsCount=0 : selon SPEC §6.5, la ligne N'EST PAS émise (UCI no-op
  défensif). Le parser ne devrait donc pas voir cette ligne en pratique,
  mais reste défensif si elle apparaissait.
"""

from __future__ import annotations

_PREFIX = "info string visits"
_PREFIX_TOKENS = ("info", "string", "visits")
UCI_MIN_LEN = 4  # eg. e2e4
UCI_MAX_LEN = 5  # eg. e7e8q (promotion)


def parse_visits_line(line: str) -> dict[str, int]:
    """Parse an `info string visits` line into {uci_move: count}.

    Args:
        line: full UCI stdout line, eg. 'info string visits e2e4 12 d2d4 8'
              or 'info string visits' (terminal position).
              Trailing newline / whitespace toléré.

    Returns:
        dict {uci_move: count}. Empty if terminal position.

    Raises:
        ValueError: si préfixe absent, nombre impair de tokens après préfixe,
                    count non-int, ou format UCI invalide (longueur < 4 ou > 5).
    """
    stripped = line.strip()
    tokens = stripped.split()
    if len(tokens) < len(_PREFIX_TOKENS) or tuple(tokens[: len(_PREFIX_TOKENS)]) != _PREFIX_TOKENS:
        raise ValueError(f"Line does not start with '{_PREFIX}': {line!r}")

    payload = tokens[len(_PREFIX_TOKENS) :]
    if not payload:
        return {}  # Terminal position.

    if len(payload) % 2 != 0:
        raise ValueError(
            f"Odd number of tokens after prefix ({len(payload)}) — expected pairs (uci, count): "
            f"{line!r}"
        )

    result: dict[str, int] = {}
    for i in range(0, len(payload), 2):
        uci = payload[i]
        count_str = payload[i + 1]

        # Validate UCI move format : 4 chars (e2e4) ou 5 chars (e7e8q promotion).
        if not (UCI_MIN_LEN <= len(uci) <= UCI_MAX_LEN):
            raise ValueError(
                f"Invalid UCI move format at position {i}: {uci!r} "
                f"(expected {UCI_MIN_LEN}-{UCI_MAX_LEN} chars)"
            )
        try:
            count = int(count_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid count at position {i + 1}: {count_str!r} (expected integer)"
            ) from e

        result[uci] = count

    return result
