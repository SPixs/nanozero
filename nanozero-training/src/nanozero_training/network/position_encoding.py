"""Position encoding AlphaZero 119 planes (Python) — parité bit-perfect vs Java.

Reproduit BIT-POUR-BIT `nanozero-board/src/main/java/.../GameState.java::toPlanes`
(cf. SPEC-board §7). Conformité ADR-010 training (duplication contrôlée).

Format figé : 119 plans = 8 timestamps x 14 plans + 7 plans constants.

Layout 119 plans (cf. GameState.encodeTimestamp + toPlanes constants) :
- Plans 0..111 : 8 timestamps (t=0 current, t=1 = -1 ply, ..., t=7 = -7 plies)
  Chaque timestamp = 14 plans :
    - Plans 0..5 : P1 pieces (PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING) — P1 = side
                   to move courant (indépendant du snapshot)
    - Plans 6..11 : P2 pieces (idem)
    - Plan 12 : repetition >= 1 occurrence stricte dans history (excluant snapshot)
    - Plan 13 : repetition >= 2 occurrences strictes
- Plans 112..118 : constants (basés sur position courante uniquement)
    - 112 : Color (1.0 si WHITE to move, 0.0 si BLACK)
    - 113 : full move count / 99 (clamp)
    - 114..117 : castling rights P1_kingside, P1_queenside, P2_kingside, P2_queenside
    - 118 : halfmove clock / 100

Inversion verticale (Long.reverseBytes Java ≡ swap bytes du long bitboard) si
sideToMove==BLACK : flip rank-wise pour mettre P1 en bas en perspective normalisée.
"""

from __future__ import annotations

import chess
import numpy as np
import numpy.typing as npt

# Constantes alignées Java GameState.
N_HISTORY_LENGTH = 8
N_PLANES_PER_TIMESTAMP = 14  # 6 P1 + 6 P2 + 2 repetition
N_HISTORY_PLANES = N_HISTORY_LENGTH * N_PLANES_PER_TIMESTAMP  # 112
N_AUXILIARY_PLANES = 7
N_PLANES = N_HISTORY_PLANES + N_AUXILIARY_PLANES  # 119
BOARD_SIZE = 8
SQUARES_PER_PLANE = BOARD_SIZE * BOARD_SIZE  # 64

# Ordre piece types Java PieceType : PAWN=0, KNIGHT=1, BISHOP=2, ROOK=3, QUEEN=4, KING=5.
_PIECE_TYPES_ORDER: tuple[int, ...] = (
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
)


def _piece_bitboard(board: chess.Board, color: bool, piece_type: int) -> int:
    """Bitboard 64-bit (long) des pièces de couleur+type données.

    Convention LSB=a1=0, identique nanozero-board (cf. ADR-008 board).
    `chess.Board.pieces(piece_type, color)` retourne un SquareSet, conversion
    via int().
    """
    return int(board.pieces(piece_type, color))


def _reverse_bytes_uint64(bb: int) -> int:
    """Reproduit `Long.reverseBytes` Java : swap les 8 bytes du long 64-bit.

    Effet géométrique : flip vertical de l'échiquier (rank 0 ↔ rank 7).
    Each byte = 1 rank, swap byte order = swap rank order.
    """
    result = 0
    for i in range(8):
        byte = (bb >> (i * 8)) & 0xFF
        result |= byte << ((7 - i) * 8)
    return result


def _bitboard_to_plane(
    bb: int,
    dest: npt.NDArray[np.float32],
    plane_idx: int,
) -> None:
    """Écrit 1.0 sur les 64 cases du plan où le bit correspondant est à 1.

    Layout dest : shape (N_PLANES, ROWS, COLS) row-major. dest[plane_idx, rank, file]
    = 1.0f si bit (rank*8+file) à 1 dans bb, 0.0f sinon.
    """
    # Itère sur les 64 bits ; dest[plane_idx] est déjà initialisé à 0 (np.zeros).
    for sq in range(64):
        if (bb >> sq) & 1:
            rank = sq // 8
            file = sq % 8
            dest[plane_idx, rank, file] = 1.0


def _encode_timestamp(
    snapshot_board: chess.Board,
    dest: npt.NDArray[np.float32],
    timestamp_offset: int,
    us_color: bool,
    them_color: bool,
    need_flip: bool,
    repetition_count: int,
) -> None:
    """Encode 14 plans pour un timestamp donné (cohérent GameState.encodeTimestamp).

    Args:
        snapshot_board: position au timestamp (current ou history-popped).
        dest: buffer (119, 8, 8).
        timestamp_offset: index du 1er plan de ce timestamp dans dest (0, 14, 28, ...).
        us_color: side-to-move COURANT (chess.WHITE ou chess.BLACK, indépendant du snapshot).
        them_color: opposé.
        need_flip: True si us == BLACK (mirror).
        repetition_count: nombre d'occurrences strictes du hash du snapshot dans l'historique
                         (excluant le snapshot lui-même).
    """
    # 6 plans P1 (PAWN..KING).
    for i, piece_type in enumerate(_PIECE_TYPES_ORDER):
        bb = _piece_bitboard(snapshot_board, us_color, piece_type)
        if need_flip:
            bb = _reverse_bytes_uint64(bb)
        _bitboard_to_plane(bb, dest, timestamp_offset + i)

    # 6 plans P2 (PAWN..KING).
    for i, piece_type in enumerate(_PIECE_TYPES_ORDER):
        bb = _piece_bitboard(snapshot_board, them_color, piece_type)
        if need_flip:
            bb = _reverse_bytes_uint64(bb)
        _bitboard_to_plane(bb, dest, timestamp_offset + 6 + i)

    # 2 plans répétition. Java : rep >= 1 et rep >= 2 où rep = occurrences strictes
    # dans l'historique (EXCLUANT le snapshot lui-même).
    if repetition_count >= 1:
        dest[timestamp_offset + 12, :, :] = 1.0
    if repetition_count >= 2:
        dest[timestamp_offset + 13, :, :] = 1.0


_PositionKey = tuple[int, int, int, int, int, int, int, int, bool, int, int | None]


def _count_repetitions(
    target_key: _PositionKey,
    history_keys: list[_PositionKey],
    current_halfmove_clock: int,
) -> int:
    """Compte occurrences de target_key dans history_keys (cohérent Java).

    Reproduit `GameState.countRepetitions` Java :
    - Scan window = historyPositions[limit..historySize-1] où
      limit = max(0, historySize - currentPosition.halfmoveClock).
    - Le scan ne tronque PAS le snapshot lui-même (s'il est dans la fenêtre,
      il est compté).
    - Le halfmove_clock utilisé pour limit est celui de la POSITION COURANTE
      (pas du snapshot), reflet du principe "aucune répétition ne peut
      traverser un coup irréversible".

    Args:
        target_key: clé identifiant la position (cf. _position_key).
        history_keys: list des keys des snapshots historiques (ordre chronologique :
                     index 0 = state avant 1er coup = startpos, index n-1 = state
                     avant dernier coup).
        current_halfmove_clock: halfmove_clock de la position COURANTE (pas du snapshot).

    Returns:
        Nombre d'occurrences dans la scan window.
    """
    count = 0
    n = len(history_keys)
    limit = max(0, n - current_halfmove_clock)
    for i in range(n - 1, limit - 1, -1):
        if history_keys[i] == target_key:
            count += 1
    return count


def _position_key(board: chess.Board) -> _PositionKey:
    """Clé d'égalité de position pour comptage répétitions.

    Reproduit la sémantique Zobrist Java (équivalence pour répétitions FIDE) :
    pieces + side to move + castling rights + en passant square pertinent.
    Le halfmove_clock et fullmove_number sont EXCLUS (cohérent §7.7 board).

    Note : `chess.Board._transposition_key()` (méthode privée python-chess)
    retourne exactement cette clé. On la duplique manuellement pour éviter
    de dépendre d'une API privée.
    """
    return (
        board.pawns,
        board.knights,
        board.bishops,
        board.rooks,
        board.queens,
        board.kings,
        board.occupied_co[chess.WHITE],
        board.occupied_co[chess.BLACK],
        board.turn,
        board.clean_castling_rights(),
        board.ep_square if board.has_legal_en_passant() else None,
    )


def encode_position(board: chess.Board) -> npt.NDArray[np.float32]:  # noqa: PLR0912, PLR0915
    """Encode position to 119 planes float32 (8, 8) per plane.

    Reproduit bit-pour-bit `GameState.toPlanes(dest, offset)` Java.

    Invariant : NO MUTATION du board en entrée. Le board est traversé via pop/push
    avec save+restore garanti par try/finally.

    Args:
        board: chess.Board (mutable, mais sera restauré à l'identique en sortie).

    Returns:
        np.ndarray shape (119, 8, 8) dtype float32. Layout :
        - planes[t*14:(t+1)*14] pour t in [0, 8) : timestamps (t=0 current, t=7 -7 plies).
        - planes[112..118] : constants.
    """
    out = np.zeros((N_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    us = board.turn
    them = not us
    need_flip = us == chess.BLACK

    # Reconstituer les snapshots historiques via pop/push.
    # Java : historyPositions[i] pour i < historySize = state AVANT i-ème move.
    # snapshot t=0 = currentPosition. snapshot t=k>0 = historyPositions[historySize - k].
    # Donc snapshot t=k pour k>0 = état après (historySize - k) coups joués = currentPosition
    # avec k coups annulés.
    # Pop snapshots into a list of immutable copies.
    # snapshots[0] = currentPosition. snapshots[k] = k plies ago.
    snapshots: list[chess.Board] = []
    popped: list[chess.Move] = []
    try:
        snapshots.append(board.copy(stack=False))
        for _k in range(1, N_HISTORY_LENGTH):
            if len(board.move_stack) == 0:
                break
            popped.append(board.pop())
            snapshots.append(board.copy(stack=False))
    finally:
        # Restore exact state via re-push in reverse order.
        for mv in reversed(popped):
            board.push(mv)

    # Pour le calcul des répétitions, on a besoin des keys de TOUT l'historique (avant le
    # snapshot courant) y compris au-delà des 8 plies stockés en plans. Calculer via re-walk.
    # On reconstruit la liste complète des keys + halfmove_clocks de l'historique en jouant
    # depuis startpos via les moves de board.move_stack — mais ça suppose qu'on connaît
    # startpos. Simplification cohérente Java : on ne dispose en pratique que des
    # `history_size` derniers snapshots. On utilise donc board._stack qui contient les
    # _BoardState de python-chess (history + current).
    #
    # Java GameState gère son propre historyPositions[]. Côté Python, on reconstruit
    # l'historique via pop tous les coups disponibles.
    all_history_keys: list[_PositionKey] = []
    all_history_halfmove: list[int] = []
    popped_full: list[chess.Move] = []
    try:
        while len(board.move_stack) > 0:
            popped_full.append(board.pop())
            all_history_keys.append(_position_key(board))
            all_history_halfmove.append(board.halfmove_clock)
        # all_history_keys (avant reverse) : index 0 = state juste avant currentPosition,
        # index N-1 = startpos. Java historyPositions[0] = startpos,
        # historyPositions[N-1] = juste avant currentPosition. → reverse.
        all_history_keys.reverse()
        all_history_halfmove.reverse()
    finally:
        for mv in reversed(popped_full):
            board.push(mv)

    # Encoder les 8 timestamps.
    # snapshots[0] = currentPosition, snapshots[k] = k plies ago.
    # Java : t=0 → currentPosition, t=k>0 → historyPositions[historySize - k].
    # historyPositions[historySize - k] (Java) = snapshot AVANT le k-ème dernier coup
    # appliqué = snapshot k plies en arrière depuis currentPosition.
    # Donc snapshots[k] (notre liste) ≡ historyPositions[historySize - k] (Java) pour k>0.
    current_halfmove_clock = board.halfmove_clock  # halfmove_clock de la position courante

    for t in range(N_HISTORY_LENGTH):
        timestamp_offset = t * N_PLANES_PER_TIMESTAMP
        if t >= len(snapshots):
            # Profondeur insuffisante : 14 plans à 0 (déjà via np.zeros init).
            continue

        snapshot = snapshots[t]
        snapshot_key = _position_key(snapshot)

        # Java GameState.countRepetitions :
        # - scan window = historyPositions[limit..historySize-1] où
        #   limit = max(0, historySize - current.halfmoveClock).
        # - Le scan inclut le snapshot lui-même s'il est dans la fenêtre.
        # - Le halfmove_clock utilisé est celui de currentPosition, PAS du snapshot.
        rep = _count_repetitions(snapshot_key, all_history_keys, current_halfmove_clock)

        _encode_timestamp(snapshot, out, timestamp_offset, us, them, need_flip, rep)

    # Plans constants (112..118).
    const_offset = N_HISTORY_PLANES

    # 112 : Color (1.0 si WHITE to move).
    if board.turn == chess.WHITE:
        out[const_offset, :, :] = 1.0

    # 113 : Move count / 99 (clamp).
    move_count_norm = min(board.fullmove_number, 99) / 99.0
    out[const_offset + 1, :, :] = move_count_norm

    # 114..117 : castling rights — mapping P1/P2 dépendant de sideToMove (§7.4).
    cr = board.clean_castling_rights()
    if us == chess.WHITE:
        p1_ks_mask = chess.BB_H1
        p1_qs_mask = chess.BB_A1
        p2_ks_mask = chess.BB_H8
        p2_qs_mask = chess.BB_A8
    else:
        p1_ks_mask = chess.BB_H8
        p1_qs_mask = chess.BB_A8
        p2_ks_mask = chess.BB_H1
        p2_qs_mask = chess.BB_A1

    if cr & p1_ks_mask:
        out[const_offset + 2, :, :] = 1.0
    if cr & p1_qs_mask:
        out[const_offset + 3, :, :] = 1.0
    if cr & p2_ks_mask:
        out[const_offset + 4, :, :] = 1.0
    if cr & p2_qs_mask:
        out[const_offset + 5, :, :] = 1.0

    # 118 : halfmove clock / 100.
    out[const_offset + 6, :, :] = board.halfmove_clock / 100.0

    return out
