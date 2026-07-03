"""Unit tests for network/position_encoding.py.

Tests fonctionnels + invariant no-mutation. Test parité bit-perfect vs Java
(100 fixtures binary) vit dans tests/integration/test_position_encoding_parity.py
(slow opt-in).
"""

from __future__ import annotations

import chess
import numpy as np
from nanozero_training.network.position_encoding import (
    BOARD_SIZE,
    N_HISTORY_LENGTH,
    N_HISTORY_PLANES,
    N_PLANES,
    N_PLANES_PER_TIMESTAMP,
    encode_position,
)


def test_encode_position_shape_dtype() -> None:
    board = chess.Board()
    planes = encode_position(board)
    assert planes.shape == (N_PLANES, BOARD_SIZE, BOARD_SIZE)
    assert planes.dtype == np.float32


def test_encode_position_constants() -> None:
    """Sanity check sur les constantes alignées Java."""
    assert N_PLANES == 119
    assert N_HISTORY_LENGTH == 8
    assert N_PLANES_PER_TIMESTAMP == 14
    assert N_HISTORY_PLANES == 112


def test_encode_position_no_mutation_startpos() -> None:
    """encode_position ne mute pas le board (FEN identique avant/après)."""
    board = chess.Board()
    fen_before = board.fen()
    moves_before = list(board.move_stack)
    encode_position(board)
    assert board.fen() == fen_before
    assert list(board.move_stack) == moves_before


def test_encode_position_no_mutation_with_history() -> None:
    """encode_position ne mute pas avec history."""
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6"):
        board.push(chess.Move.from_uci(uci))
    fen_before = board.fen()
    moves_before = [m.uci() for m in board.move_stack]
    encode_position(board)
    assert board.fen() == fen_before
    assert [m.uci() for m in board.move_stack] == moves_before


def test_encode_position_startpos_p1_pawns() -> None:
    """Startpos : 8 pions P1 (blancs) sur rank 1 (rank index 1)."""
    board = chess.Board()
    planes = encode_position(board)
    # P1 pawns (plane 0 du timestamp 0).
    pawn_plane = planes[0]
    # Pions blancs sur a2..h2 (rank 1, file 0..7).
    assert pawn_plane[1].sum() == 8.0
    assert pawn_plane[1].min() == 1.0
    # Reste du plan = 0.
    pawn_plane_no_rank1 = pawn_plane.copy()
    pawn_plane_no_rank1[1] = 0
    assert pawn_plane_no_rank1.sum() == 0.0


def test_encode_position_startpos_p2_pawns() -> None:
    """Startpos : 8 pions P2 (noirs) sur rank 6 (index 6)."""
    board = chess.Board()
    planes = encode_position(board)
    p2_pawn_plane = planes[6]  # P2 pawn = plane 6 du timestamp 0
    assert p2_pawn_plane[6].sum() == 8.0


def test_encode_position_history_padded_zeros_at_startpos() -> None:
    """Startpos : timestamps t=1..7 = zeros (pas d'history)."""
    board = chess.Board()
    planes = encode_position(board)
    for t in range(1, N_HISTORY_LENGTH):
        ts_planes = planes[t * N_PLANES_PER_TIMESTAMP : (t + 1) * N_PLANES_PER_TIMESTAMP]
        assert ts_planes.sum() == 0.0, f"Timestamp t={t} expected zeros, got sum={ts_planes.sum()}"


def test_encode_position_history_filled_after_8_plies() -> None:
    """Après 8 plies : tous timestamps t=0..7 ont du contenu."""
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6"):
        board.push(chess.Move.from_uci(uci))
    planes = encode_position(board)
    for t in range(N_HISTORY_LENGTH):
        ts_planes = planes[t * N_PLANES_PER_TIMESTAMP : (t + 1) * N_PLANES_PER_TIMESTAMP]
        assert ts_planes.sum() > 0, f"Timestamp t={t} expected non-zero (8+ plies of history)"


def test_encode_position_color_plane_white() -> None:
    """Plan 112 = 1.0 si WHITE to move."""
    board = chess.Board()
    planes = encode_position(board)
    assert planes[112].min() == 1.0
    assert planes[112].max() == 1.0


def test_encode_position_color_plane_black() -> None:
    """Plan 112 = 0.0 si BLACK to move."""
    board = chess.Board()
    board.push(chess.Move.from_uci("e2e4"))  # now BLACK to move
    planes = encode_position(board)
    assert planes[112].min() == 0.0
    assert planes[112].max() == 0.0


def test_encode_position_castling_rights_startpos() -> None:
    """Startpos : tous les 4 castling planes = 1.0 (KQkq)."""
    board = chess.Board()
    planes = encode_position(board)
    # planes 114..117 = P1 K, P1 Q, P2 K, P2 Q.
    for i in (114, 115, 116, 117):
        assert planes[i].max() == 1.0, f"Plane {i} (castling) expected 1.0 at startpos"


def test_encode_position_value_range_in_0_1() -> None:
    """Tous les values dans [0.0, 1.0]."""
    board = chess.Board()
    for uci in ("e2e4", "e7e5", "g1f3"):
        board.push(chess.Move.from_uci(uci))
    planes = encode_position(board)
    assert planes.min() >= 0.0
    assert planes.max() <= 1.0


def test_encode_position_terminal_no_crash() -> None:
    """Position mate ne crash pas l'encoder."""
    # Fool's mate
    board = chess.Board()
    for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
        board.push(chess.Move.from_uci(uci))
    assert board.is_checkmate()
    planes = encode_position(board)
    assert planes.shape == (N_PLANES, BOARD_SIZE, BOARD_SIZE)
