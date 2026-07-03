"""Move encoding AlphaZero (Python) — parité bit-perfect vs nanozero-nn MoveEncoding Java.

Reproduit BIT-POUR-BIT `nanozero-nn/src/main/java/.../MoveEncoding.java::encode`
(cf. SPEC §3.5, §5.5 nn). Conformité ADR-010 training (duplication contrôlée
des conventions Java côté Python self-play).

Format figé : 73 plans par from-square x 64 from-squares = 4672 indices.

- Plans 0..55 : queen-style (8 directions x 7 distances)
- Plans 56..63 : knight (8 deltas)
- Plans 64..72 : underpromotions (3 directions x 3 sous-pièces N/B/R)

Index final : `from * POLICY_PLANES + plane_index`. Mirror XOR 56 si
`board.turn == BLACK` (perspective P1 normalisée — le réseau prédit toujours
du point de vue du côté au trait).
"""

from __future__ import annotations

import chess
import numpy as np
import numpy.typing as npt

# Constants (figées, alignées MoveEncoding.java).
N_FROM_SQUARES = 64
N_QUEEN_PLANES = 56  # 8 directions x 7 distances
N_KNIGHT_PLANES = 8
N_UNDERPROMOTION_PLANES = 9  # 3 directions x 3 pieces (N/B/R)
POLICY_PLANES = N_QUEEN_PLANES + N_KNIGHT_PLANES + N_UNDERPROMOTION_PLANES  # 73
POLICY_INDICES = POLICY_PLANES * N_FROM_SQUARES  # 4672

# Queen-style directions (deltaRank, deltaFile), ordre NORMATIF aligné Java
# `MoveEncoding.QUEEN_DELTAS` : N, NE, E, SE, S, SW, W, NW.
QUEEN_DELTAS: tuple[tuple[int, int], ...] = (
    (+1, 0),
    (+1, +1),
    (0, +1),
    (-1, +1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (+1, -1),
)

# Knight deltas (deltaRank, deltaFile), ordre NORMATIF aligné Java
# `MoveEncoding.KNIGHT_DELTAS`.
KNIGHT_DELTAS: tuple[tuple[int, int], ...] = (
    (+2, +1),
    (+1, +2),
    (-1, +2),
    (-2, +1),
    (-2, -1),
    (-1, -2),
    (+1, -2),
    (+2, -1),
)

# Underpromotion file deltas P1 : capture-left, push, capture-right.
UNDERPROMO_FILE_DELTAS: tuple[int, ...] = (-1, 0, +1)

# Mapping python-chess.Move.promotion -> AlphaZero promo index (Java Move.promo convention).
# Java: 0=KNIGHT, 1=BISHOP, 2=ROOK, 3=QUEEN (cf. Move.pieceTypeToPromo).
# python-chess: KNIGHT=2, BISHOP=3, ROOK=4, QUEEN=5.
_CHESS_PROMOTION_TO_PROMO_INDEX: dict[int, int] = {
    chess.KNIGHT: 0,
    chess.BISHOP: 1,
    chess.ROOK: 2,
    chess.QUEEN: 3,
}
_PROMO_INDEX_TO_CHESS_PROMOTION: dict[int, int] = {
    v: k for k, v in _CHESS_PROMOTION_TO_PROMO_INDEX.items()
}

# Lookup tables précalculées (cohérent Java static init).
# QUEEN_DIR_LUT[(signR+1)*3 + (signF+1)] -> index direction queen (0..7) ou -1.
_QUEEN_DIR_LUT: list[int] = [-1] * 9
for _i, (_dr, _df) in enumerate(QUEEN_DELTAS):
    _QUEEN_DIR_LUT[(_dr + 1) * 3 + (_df + 1)] = _i

# KNIGHT_DELTA_LUT[(dR+2)*5 + (dF+2)] -> index knight (0..7) ou -1.
_KNIGHT_DELTA_LUT: list[int] = [-1] * 25
for _i, (_dr, _df) in enumerate(KNIGHT_DELTAS):
    _KNIGHT_DELTA_LUT[(_dr + 2) * 5 + (_df + 2)] = _i


def _signum(x: int) -> int:
    """Reproduit Integer.signum Java : -1, 0, +1."""
    return (x > 0) - (x < 0)


def _is_knight_shape(d_rank: int, d_file: int) -> bool:
    a_r = abs(d_rank)
    a_f = abs(d_file)
    return (a_r == 1 and a_f == 2) or (a_r == 2 and a_f == 1)


def encode_move(move: chess.Move, board: chess.Board) -> int:
    """Encode a chess.Move to an AlphaZero policy index in [0, 4672).

    Reproduit bit-pour-bit `MoveEncoding.encode(int move, int sideToMove)` Java.

    Args:
        move: chess.Move à encoder. Promotion via move.promotion (chess.QUEEN,
              chess.ROOK, chess.BISHOP, chess.KNIGHT ou None).
        board: chess.Board pour déterminer le side-to-move (mirror XOR 56 si BLACK).

    Returns:
        Index in [0, 4672).

    Raises:
        ValueError: si delta du coup ne correspond à aucun pattern (queen,
                    knight, underpromo) — coup invalide pour AlphaZero encoding.
    """
    from_sq = move.from_square
    to_sq = move.to_square
    promotion = move.promotion  # None or chess.PAWN/KNIGHT/BISHOP/ROOK/QUEEN

    # Mirror XOR 56 si BLACK to move (perspective P1).
    if board.turn == chess.BLACK:
        from_sq ^= 56
        to_sq ^= 56

    # chess.SQUARES indexe LSB=a1=0 ; rank = sq // 8, file = sq % 8.
    d_rank = (to_sq // 8) - (from_sq // 8)
    d_file = (to_sq % 8) - (from_sq % 8)

    plane_index: int

    if promotion is not None and promotion != chess.QUEEN:
        # Underpromotion (plans 64..72). promo : 0=N, 1=B, 2=R.
        direction_index = d_file + 1  # capture-left=0, push=1, capture-right=2
        if direction_index < 0 or direction_index > 2:
            raise ValueError(f"Underpromotion deltaFile invalide : {d_file} (attendu : -1, 0, +1)")
        promo_index = _CHESS_PROMOTION_TO_PROMO_INDEX[promotion]
        plane_index = 64 + direction_index * 3 + promo_index
    elif _is_knight_shape(d_rank, d_file):
        knight_idx = _KNIGHT_DELTA_LUT[(d_rank + 2) * 5 + (d_file + 2)]
        plane_index = 56 + knight_idx
    else:
        # Queen-style (inclut promotion en dame, EP, castling, normal slider/king/pawn-push).
        sign_r = _signum(d_rank)
        sign_f = _signum(d_file)
        direction = _QUEEN_DIR_LUT[(sign_r + 1) * 3 + (sign_f + 1)]
        if direction < 0:
            raise ValueError(
                f"Coup inencodable : delta queen-style invalide (d_rank={d_rank}, d_file={d_file})"
            )
        distance = max(abs(d_rank), abs(d_file))
        if distance < 1 or distance > 7:
            raise ValueError(f"Distance queen-style hors plage : {distance} (attendu : 1..7)")
        plane_index = direction * 7 + (distance - 1)

    return from_sq * POLICY_PLANES + plane_index


def decode_move(index: int, board: chess.Board) -> chess.Move:  # noqa: PLR0912
    """Decode an AlphaZero policy index back to a chess.Move in the given position.

    Reproduit `MoveEncoding.decode(int policyIndex, Position position)` Java.

    Args:
        index: in [0, 4672).
        board: chess.Board pour résoudre le type de coup (NORMAL/EN_PASSANT/
               CASTLING/PROMOTION en dame).

    Returns:
        chess.Move correspondant. Légalité dans la position N'EST PAS vérifiée
        (cohérent contrat Java : decode produit un Move syntaxiquement valide,
        la légalité est responsabilité du caller via board.is_legal).

    Raises:
        ValueError: si index hors [0, 4672).
    """
    if index < 0 or index >= POLICY_INDICES:
        raise ValueError(f"index hors plage : {index} (attendu : 0..{POLICY_INDICES - 1})")

    from_p1 = index // POLICY_PLANES
    plane = index % POLICY_PLANES

    is_underpromo = plane >= 64
    is_knight = (not is_underpromo) and plane >= 56

    promo_index: int | None = None

    if is_underpromo:
        direction_index = (plane - 64) // 3
        promo_index = (plane - 64) % 3  # 0=N, 1=B, 2=R
        d_file_p1 = UNDERPROMO_FILE_DELTAS[direction_index]
        d_rank_p1 = +1  # P1 avance toujours vers le haut en perspective normalisée
    elif is_knight:
        d_rank_p1, d_file_p1 = KNIGHT_DELTAS[plane - 56]
    else:
        direction = plane // 7
        distance = plane % 7 + 1
        d_r, d_f = QUEEN_DELTAS[direction]
        d_rank_p1 = d_r * distance
        d_file_p1 = d_f * distance

    to_p1 = from_p1 + d_rank_p1 * 8 + d_file_p1

    # Mirror retour si BLACK to move.
    if board.turn == chess.BLACK:
        from_sq = from_p1 ^ 56
        to_sq = to_p1 ^ 56
    else:
        from_sq = from_p1
        to_sq = to_p1

    if is_underpromo or _is_knight_shape_pre(plane):
        # Underpromotion ou knight : promotion explicite ou aucune.
        if is_underpromo:
            promotion: int | None = _PROMO_INDEX_TO_CHESS_PROMOTION[promo_index]  # type: ignore[index]
        else:
            promotion = None
    else:
        # Queen-style : déterminer type via la pièce qui bouge et la géométrie.
        piece = board.piece_at(from_sq)
        if piece is None:
            # Pas de pièce sur from_sq : coup syntaxiquement non interprétable
            # (probablement un index décodant un coup illégal dans la position).
            # On retourne quand même un Move "brut" sans promotion pour cohérence
            # avec le contrat Java (decode produit un Move ; caller vérifie légalité).
            promotion = None
        elif piece.piece_type == chess.PAWN and chess.square_rank(to_sq) in (0, 7):
            promotion = chess.QUEEN
        else:
            promotion = None

    return chess.Move(from_sq, to_sq, promotion=promotion)


def _is_knight_shape_pre(plane: int) -> bool:
    """Helper : un plane in [56, 64) est knight-shape."""
    return 56 <= plane < 64


def visits_to_policy_target(
    visits: dict[str, int],
    board: chess.Board,
) -> npt.NDArray[np.float32]:
    """Convert {uci_move: visit_count} to normalized policy target (4672,) float32.

    Args:
        visits: dict UCI move -> visit count (extrait `info string visits` UCI v1.2.0+).
        board: chess.Board pour encoder chaque move dans la position.

    Returns:
        np.ndarray (4672,) float32, sum ≈ 1.0 (training) OU sum == 0.0 (terminal /
        visits vide).
    """
    target = np.zeros(POLICY_INDICES, dtype=np.float32)
    if not visits:
        return target  # Terminal ou simulationsCount=0
    total = sum(visits.values())
    if total == 0:
        return target
    for uci_move, count in visits.items():
        move = chess.Move.from_uci(uci_move)
        idx = encode_move(move, board)
        target[idx] = count / total
    return target
