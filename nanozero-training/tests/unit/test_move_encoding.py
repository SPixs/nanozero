"""Unit tests for network/move_encoding.py.

Tests fonctionnels + roundtrip. Le test de parité bit-perfect vs Java
(1000 fixtures CSV) vit dans tests/integration/test_move_encoding_parity.py
(slow opt-in).
"""

from __future__ import annotations

import chess
import numpy as np
import pytest
from nanozero_training.network.move_encoding import (
    POLICY_INDICES,
    POLICY_PLANES,
    decode_move,
    encode_move,
    visits_to_policy_target,
)

# ----- encode_move -----


def test_encode_move_returns_valid_index() -> None:
    """Any legal move from startpos produces an index in [0, 4672)."""
    board = chess.Board()
    for move in board.legal_moves:
        idx = encode_move(move, board)
        assert 0 <= idx < POLICY_INDICES


def test_encode_e2e4_from_startpos() -> None:
    """e2e4 from startpos : queen-style direction N (idx 0), distance 2, from=e2=12.
    plane = 0 * 7 + (2-1) = 1, index = 12 * 73 + 1 = 877.
    """
    board = chess.Board()
    move = chess.Move.from_uci("e2e4")
    assert encode_move(move, board) == 12 * POLICY_PLANES + 1


def test_encode_b1c3_from_startpos() -> None:
    """b1c3 (knight) : from=b1=1, dRank=+2, dFile=+1 → knight delta 0, plane=56.
    Index = 1 * 73 + 56 = 129.
    """
    board = chess.Board()
    move = chess.Move.from_uci("b1c3")
    assert encode_move(move, board) == 1 * POLICY_PLANES + 56


def test_encode_promotion_queen_treated_as_queen_style() -> None:
    """Promotion en dame : traitée comme queen-style (pas un plan underpromo)."""
    # Position avec pion blanc sur e7, prêt à promouvoir.
    board = chess.Board("8/4P3/8/8/8/8/8/k6K w - - 0 1")
    move = chess.Move.from_uci("e7e8q")
    idx = encode_move(move, board)
    # plane queen-style N (dir 0), distance 1, plane=0; from=e7=52, index = 52*73 + 0 = 3796.
    assert idx == 52 * POLICY_PLANES + 0


def test_encode_promotion_knight_underpromotion() -> None:
    """Underpromotion en cavalier : plane 64..72."""
    board = chess.Board("8/4P3/8/8/8/8/8/k6K w - - 0 1")
    move = chess.Move.from_uci("e7e8n")
    idx = encode_move(move, board)
    # direction_index = dFile+1 = 0+1 = 1 (push), promo_index=0 (N).
    # plane = 64 + 1*3 + 0 = 67. from=e7=52. Index = 52*73 + 67.
    assert idx == 52 * POLICY_PLANES + 67


def test_encode_promotion_bishop_underpromotion() -> None:
    board = chess.Board("8/4P3/8/8/8/8/8/k6K w - - 0 1")
    move = chess.Move.from_uci("e7e8b")
    # promo_index=1 (B), direction=push (1), plane = 64 + 1*3 + 1 = 68.
    assert encode_move(move, board) == 52 * POLICY_PLANES + 68


def test_encode_promotion_rook_underpromotion() -> None:
    board = chess.Board("8/4P3/8/8/8/8/8/k6K w - - 0 1")
    move = chess.Move.from_uci("e7e8r")
    # promo_index=2 (R), direction=push (1), plane = 64 + 1*3 + 2 = 69.
    assert encode_move(move, board) == 52 * POLICY_PLANES + 69


def test_encode_castling_kingside_white() -> None:
    """White O-O : king e1g1, traité comme queen-style E (dir 2), distance 2."""
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    move = chess.Move.from_uci("e1g1")
    # dRank=0, dFile=+2 → queen-style direction E (idx 2), distance 2.
    # plane = 2 * 7 + 1 = 15. from=e1=4. Index = 4*73 + 15.
    assert encode_move(move, board) == 4 * POLICY_PLANES + 15


def test_encode_en_passant() -> None:
    """En passant : queen-style diagonale (dir NE ou NW), distance 1."""
    # Position avec EP possible : black pawn vient de jouer c7c5, white pawn sur d5.
    board = chess.Board("rnbqkbnr/pp1ppppp/8/2pPP3/8/8/PPP2PPP/RNBQKBNR w KQkq c6 0 3")
    move = chess.Move.from_uci("d5c6")
    # dRank=+1, dFile=-1 → NW (idx 7), distance 1, plane = 7*7 + 0 = 49.
    # from=d5=35. Index = 35*73 + 49.
    assert encode_move(move, board) == 35 * POLICY_PLANES + 49


def test_encode_black_to_move_mirror() -> None:
    """Black move : XOR 56 sur from/to → encoding en perspective P1 normalisée."""
    # Black to move, joue e7e5 — perspective P1 (mirror) = e2e4.
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))  # white plays e4
    move = chess.Move.from_uci("e7e5")  # black to move
    idx = encode_move(move, board)
    # from_p1 = e7 ^ 56 = 52 ^ 56 = 12 (= e2). dRank_p1 = +2, dFile_p1 = 0 → dir N, distance 2.
    # plane = 1, index = 12 * 73 + 1 = 877 (identique à e2e4 par symétrie attendue).
    assert idx == 12 * POLICY_PLANES + 1


# ----- decode_move -----


def test_decode_inverse_of_encode_legal_moves_startpos() -> None:
    """Roundtrip decode(encode(m)) == m pour tous coups légaux de startpos."""
    board = chess.Board()
    for move in board.legal_moves:
        idx = encode_move(move, board)
        decoded = decode_move(idx, board)
        assert decoded == move, f"Roundtrip mismatch: {move} -> {idx} -> {decoded}"


def test_decode_inverse_of_encode_legal_moves_midgame() -> None:
    """Roundtrip sur position midgame (Sicilian)."""
    board = chess.Board("r1bqkbnr/pp1ppppp/2n5/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    for move in board.legal_moves:
        idx = encode_move(move, board)
        decoded = decode_move(idx, board)
        assert decoded == move, f"Roundtrip mismatch: {move} -> {idx} -> {decoded}"


def test_decode_inverse_of_encode_promotion_position() -> None:
    """Roundtrip sur position de promotion."""
    board = chess.Board("8/4P3/8/8/8/8/8/k6K w - - 0 1")
    for move in board.legal_moves:
        idx = encode_move(move, board)
        decoded = decode_move(idx, board)
        assert decoded == move, f"Roundtrip mismatch: {move} -> {idx} -> {decoded}"


def test_decode_inverse_of_encode_castling_position() -> None:
    """Roundtrip sur position avec castling possible."""
    board = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    for move in board.legal_moves:
        idx = encode_move(move, board)
        decoded = decode_move(idx, board)
        assert decoded == move, f"Roundtrip mismatch: {move} -> {idx} -> {decoded}"


def test_decode_invalid_index_raises() -> None:
    board = chess.Board()
    with pytest.raises(ValueError, match="hors plage"):
        decode_move(POLICY_INDICES, board)
    with pytest.raises(ValueError, match="hors plage"):
        decode_move(-1, board)


# ----- visits_to_policy_target -----


def test_visits_to_policy_empty_terminal() -> None:
    """Visits vide → policy target all zeros."""
    board = chess.Board()
    target = visits_to_policy_target({}, board)
    assert target.shape == (POLICY_INDICES,)
    assert target.dtype == np.float32
    assert float(target.sum()) == 0.0


def test_visits_to_policy_normalizes_sum_1() -> None:
    """Visits non-vide → sum ≈ 1.0."""
    board = chess.Board()
    visits = {"e2e4": 50, "d2d4": 30, "g1f3": 20}
    target = visits_to_policy_target(visits, board)
    assert target.shape == (POLICY_INDICES,)
    assert target.dtype == np.float32
    assert float(target.sum()) == pytest.approx(1.0, abs=1e-6)


def test_visits_to_policy_assigns_correct_indices() -> None:
    """Chaque visit count se retrouve à l'index encode_move correspondant."""
    board = chess.Board()
    visits = {"e2e4": 70, "d2d4": 30}
    target = visits_to_policy_target(visits, board)
    idx_e2e4 = encode_move(chess.Move.from_uci("e2e4"), board)
    idx_d2d4 = encode_move(chess.Move.from_uci("d2d4"), board)
    assert target[idx_e2e4] == pytest.approx(0.7, abs=1e-6)
    assert target[idx_d2d4] == pytest.approx(0.3, abs=1e-6)
    # Toutes les autres entrées doivent être 0.
    target[idx_e2e4] = 0
    target[idx_d2d4] = 0
    assert float(target.sum()) == 0.0


def test_visits_to_policy_zero_total_returns_zeros() -> None:
    """Visits avec total=0 (edge case) → all zeros."""
    board = chess.Board()
    target = visits_to_policy_target({"e2e4": 0}, board)
    assert float(target.sum()) == 0.0
