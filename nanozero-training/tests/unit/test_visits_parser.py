"""Unit tests for selfplay/visits_parser.py."""

from __future__ import annotations

import pytest
from nanozero_training.selfplay.visits_parser import parse_visits_line


def test_parse_empty_terminal() -> None:
    """Position terminale : préfixe seul → dict vide."""
    assert parse_visits_line("info string visits") == {}


def test_parse_single_move() -> None:
    assert parse_visits_line("info string visits e2e4 12") == {"e2e4": 12}


def test_parse_multiple_moves() -> None:
    line = "info string visits e2e4 234 d2d4 187 g1f3 156 c2c4 89"
    result = parse_visits_line(line)
    assert result == {"e2e4": 234, "d2d4": 187, "g1f3": 156, "c2c4": 89}


def test_parse_with_promotion() -> None:
    """Promotions UCI : 5 chars (e7e8q, e7e8n, etc.)."""
    line = "info string visits e7e8q 5 e7e8n 3 e7e8b 1 e7e8r 0"
    result = parse_visits_line(line)
    assert result == {"e7e8q": 5, "e7e8n": 3, "e7e8b": 1, "e7e8r": 0}


def test_parse_extra_whitespace() -> None:
    """Whitespace multiple toléré (split par défaut Python)."""
    line = "info string visits   e2e4   12   d2d4   8"
    assert parse_visits_line(line) == {"e2e4": 12, "d2d4": 8}


def test_parse_trailing_newline() -> None:
    """Trailing newline stripped."""
    assert parse_visits_line("info string visits e2e4 12\n") == {"e2e4": 12}


def test_parse_leading_whitespace() -> None:
    """Leading whitespace toléré."""
    assert parse_visits_line("  info string visits e2e4 12") == {"e2e4": 12}


def test_parse_zero_count() -> None:
    """count=0 acceptable (selon SPEC : seulement count > 0 émis, mais robuste si présent)."""
    assert parse_visits_line("info string visits e2e4 0") == {"e2e4": 0}


def test_parse_invalid_prefix_wrong_words() -> None:
    with pytest.raises(ValueError, match="does not start with"):
        parse_visits_line("info string foo e2e4 12")


def test_parse_invalid_prefix_too_short() -> None:
    with pytest.raises(ValueError, match="does not start with"):
        parse_visits_line("info string")


def test_parse_invalid_prefix_empty_line() -> None:
    with pytest.raises(ValueError, match="does not start with"):
        parse_visits_line("")


def test_parse_invalid_prefix_other_info() -> None:
    """Lignes 'info depth ...' ne doivent pas être confondues."""
    with pytest.raises(ValueError, match="does not start with"):
        parse_visits_line("info depth 12 nodes 1024")


def test_parse_missing_count_odd_tokens() -> None:
    """Nombre impair de tokens après préfixe → ValueError."""
    with pytest.raises(ValueError, match="Odd number of tokens"):
        parse_visits_line("info string visits e2e4")


def test_parse_invalid_count_non_integer() -> None:
    with pytest.raises(ValueError, match="Invalid count"):
        parse_visits_line("info string visits e2e4 abc")


def test_parse_invalid_count_float() -> None:
    """Float refused even if syntactically a number."""
    with pytest.raises(ValueError, match="Invalid count"):
        parse_visits_line("info string visits e2e4 1.5")


def test_parse_invalid_uci_too_short() -> None:
    with pytest.raises(ValueError, match="Invalid UCI move format"):
        parse_visits_line("info string visits e2 12")


def test_parse_invalid_uci_too_long() -> None:
    with pytest.raises(ValueError, match="Invalid UCI move format"):
        parse_visits_line("info string visits e7e8qx 12")
