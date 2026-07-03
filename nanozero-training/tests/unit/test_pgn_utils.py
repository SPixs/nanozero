"""Unit tests for eval/pgn_utils.count_games_in_pgn."""

from __future__ import annotations

from pathlib import Path

from nanozero_training.eval.pgn_utils import count_games_in_pgn, parse_sprt_counts_from_pgn


def test_count_empty_file_returns_0(tmp_path: Path) -> None:
    pgn = tmp_path / "empty.pgn"
    pgn.write_text("", encoding="utf-8")
    assert count_games_in_pgn(pgn) == 0


def test_count_nonexistent_file_returns_0(tmp_path: Path) -> None:
    assert count_games_in_pgn(tmp_path / "missing.pgn") == 0


def test_count_one_game(tmp_path: Path) -> None:
    pgn = tmp_path / "one.pgn"
    pgn.write_text(
        '[Event "test"]\n[Site "?"]\n[Result "1-0"]\n\n1. e4 e5 1-0\n',
        encoding="utf-8",
    )
    assert count_games_in_pgn(pgn) == 1


def test_count_multiple_games(tmp_path: Path) -> None:
    pgn = tmp_path / "three.pgn"
    pgn.write_text(
        '[Event "a"]\n[Result "1-0"]\n\n1. e4 1-0\n\n'
        '[Event "b"]\n[Result "0-1"]\n\n1. e4 0-1\n\n'
        '[Event "c"]\n[Result "1/2-1/2"]\n\n1. e4 1/2-1/2\n',
        encoding="utf-8",
    )
    assert count_games_in_pgn(pgn) == 3


def test_count_ignores_other_tags(tmp_path: Path) -> None:
    pgn = tmp_path / "other.pgn"
    pgn.write_text(
        '[Event "test"]\n[Site "?"]\n[White "A"]\n[Black "B"]\n[Result "1-0"]\n',
        encoding="utf-8",
    )
    assert count_games_in_pgn(pgn) == 1


def test_count_handles_draw_result(tmp_path: Path) -> None:
    pgn = tmp_path / "draw.pgn"
    pgn.write_text('[Result "1/2-1/2"]\n', encoding="utf-8")
    assert count_games_in_pgn(pgn) == 1


def test_count_handles_unknown_result(tmp_path: Path) -> None:
    pgn = tmp_path / "ongoing.pgn"
    pgn.write_text('[Result "*"]\n', encoding="utf-8")
    assert count_games_in_pgn(pgn) == 1


def test_parse_sprt_counts_challenger_pov(tmp_path: Path) -> None:
    """Regression Bug #7 hotfix-008: PGN-based fallback for SPRT W/L/D when
    fastchess stdout parsing fails. Counts must be from challenger POV regardless
    of who plays White each round."""
    pgn = tmp_path / "sprt.pgn"
    pgn.write_text(
        # Round 1 game 1: challenger=White, wins
        '[Event "Fastchess Tournament"]\n[White "challenger"]\n[Black "baseline"]\n[Result "1-0"]\n\n1. e4 1-0\n\n'  # noqa: E501
        # Round 1 game 2: challenger=Black, wins (Result 0-1)
        '[Event "Fastchess Tournament"]\n[White "baseline"]\n[Black "challenger"]\n[Result "0-1"]\n\n1. e4 0-1\n\n'  # noqa: E501
        # Round 2 game 1: challenger=White, loses
        '[Event "Fastchess Tournament"]\n[White "challenger"]\n[Black "baseline"]\n[Result "0-1"]\n\n1. e4 0-1\n\n'  # noqa: E501
        # Round 2 game 2: challenger=Black, draw
        '[Event "Fastchess Tournament"]\n[White "baseline"]\n[Black "challenger"]\n[Result "1/2-1/2"]\n\n1. e4 1/2-1/2\n',  # noqa: E501
        encoding="utf-8",
    )
    counts = parse_sprt_counts_from_pgn(pgn, challenger_name="challenger")
    assert counts.total == 4
    assert counts.challenger_wins == 2  # game 1 (W) + game 2 (B)
    assert counts.challenger_losses == 1  # game 3 (W lost)
    assert counts.challenger_draws == 1  # game 4


def test_parse_sprt_counts_nonexistent(tmp_path: Path) -> None:
    counts = parse_sprt_counts_from_pgn(tmp_path / "absent.pgn")
    assert counts.total == 0
    assert counts.challenger_wins == 0


def test_parse_sprt_counts_custom_name(tmp_path: Path) -> None:
    """challenger_name parameter respected — POV swaps if name differs."""
    pgn = tmp_path / "named.pgn"
    pgn.write_text(
        '[White "mybot"]\n[Black "stockfish"]\n[Result "1-0"]\n\n1. e4 1-0\n',
        encoding="utf-8",
    )
    counts_mybot = parse_sprt_counts_from_pgn(pgn, challenger_name="mybot")
    assert counts_mybot.challenger_wins == 1
    counts_sf = parse_sprt_counts_from_pgn(pgn, challenger_name="stockfish")
    assert counts_sf.challenger_losses == 1


def test_count_handles_utf8_errors_replace(tmp_path: Path) -> None:
    """Fichier avec bytes invalides UTF-8 — ne raise pas (errors='replace')."""
    pgn = tmp_path / "bad.pgn"
    pgn.write_bytes(b'[Event "ok"]\n\xff\xfe\xfd\n[Result "1-0"]\n')
    assert count_games_in_pgn(pgn) == 1
