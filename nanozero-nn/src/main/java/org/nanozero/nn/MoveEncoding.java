package org.nanozero.nn;

import org.nanozero.board.Color;
import org.nanozero.board.Move;
import org.nanozero.board.MoveType;
import org.nanozero.board.Piece;
import org.nanozero.board.PieceType;
import org.nanozero.board.Position;
import org.nanozero.board.Square;

/**
 * Mapping bidirectionnel {@code Move ↔ index policy AlphaZero (4672 indices)} (cf. SPEC §3.5,
 * §4.2.5, §5.5).
 *
 * <p>Le format est figé : 73 plans par case d'origine × 64 cases = 4672 indices.
 *
 * <ul>
 *   <li>plans 0..55 : queen-style (8 directions × 7 distances)
 *   <li>plans 56..63 : knight (8 deltas)
 *   <li>plans 64..72 : underpromotions (3 directions × 3 sous-pièces N/B/R)
 * </ul>
 *
 * <p>L'inversion de perspective (XOR 56 sur les cases) est appliquée si {@code sideToMove == BLACK}
 * pour ramener le coup en perspective P1 (le réseau prédit toujours du point de vue de P1, le côté
 * au trait courant ; cf. SPEC §3.5 et §7.3 SPEC-board).
 *
 * <p>Invariants :
 *
 * <ul>
 *   <li>{@code I-ME-1} : pour tout coup légal {@code m} dans une position {@code p} avec côté au
 *       trait {@code s = p.sideToMove()}, {@code decode(encode(m, s), p) == m}.
 *   <li>{@code I-ME-2} : {@code decodePolicy} produit une distribution dont la somme des {@code n}
 *       premières entrées est exactement {@code 1.0} (modulo erreur flottante &lt; 1e-6).
 * </ul>
 *
 * <p>Classe non instanciable, conforme aux conventions du module.
 */
public final class MoveEncoding {

  /** Nombre total d'indices de policy (4672). */
  public static final int POLICY_INDICES = 4672;

  /** Nombre de plans par case d'origine (73). */
  public static final int POLICY_PLANES = 73;

  private MoveEncoding() {
    throw new AssertionError("Non-instantiable");
  }

  // -----------------------------------------------------------------------------------------
  // Tables figées (cf. SPEC §5.5.2)
  // -----------------------------------------------------------------------------------------

  /**
   * Directions queen-style en {@code (deltaRank, deltaFile)} : N, NE, E, SE, S, SW, W, NW. Ordre
   * normatif §5.5.2.
   */
  static final int[][] QUEEN_DELTAS = {
    {+1, 0}, {+1, +1}, {0, +1}, {-1, +1},
    {-1, 0}, {-1, -1}, {0, -1}, {+1, -1}
  };

  /** Deltas du cavalier en {@code (deltaRank, deltaFile)}, ordre normatif §5.5.2. */
  static final int[][] KNIGHT_DELTAS = {
    {+2, +1}, {+1, +2}, {-1, +2}, {-2, +1},
    {-2, -1}, {-1, -2}, {+1, -2}, {+2, -1}
  };

  /**
   * Deltas de fichier pour underpromotion : 3 directions (capture-left, push, capture-right) du
   * point de vue P1.
   */
  static final int[] UNDERPROMO_FILE_DELTAS = {-1, 0, +1};

  /** Pièces de sous-promotion (KNIGHT, BISHOP, ROOK), index 0..2. */
  static final int[] UNDERPROMO_PIECES = {PieceType.KNIGHT, PieceType.BISHOP, PieceType.ROOK};

  // Lookup tables précalculées au static init pour éviter les recherches linéaires en hot path.

  /**
   * {@code QUEEN_DIR_LUT[(signRank+1)*3 + (signFile+1)]} retourne l'index queen-direction (0..7) ou
   * {@code -1} si la combinaison de signes est invalide ({@code (0,0)}).
   */
  private static final int[] QUEEN_DIR_LUT = new int[9];

  /**
   * {@code KNIGHT_DELTA_LUT[(dR+2)*5 + (dF+2)]} retourne l'index knight-delta (0..7) ou {@code -1}
   * si {@code (dR, dF)} n'est pas un saut de cavalier.
   */
  private static final int[] KNIGHT_DELTA_LUT = new int[25];

  static {
    java.util.Arrays.fill(QUEEN_DIR_LUT, -1);
    for (int i = 0; i < QUEEN_DELTAS.length; i++) {
      int dR = QUEEN_DELTAS[i][0];
      int dF = QUEEN_DELTAS[i][1];
      QUEEN_DIR_LUT[(dR + 1) * 3 + (dF + 1)] = i;
    }
    java.util.Arrays.fill(KNIGHT_DELTA_LUT, -1);
    for (int i = 0; i < KNIGHT_DELTAS.length; i++) {
      int dR = KNIGHT_DELTAS[i][0];
      int dF = KNIGHT_DELTAS[i][1];
      KNIGHT_DELTA_LUT[(dR + 2) * 5 + (dF + 2)] = i;
    }
  }

  // -----------------------------------------------------------------------------------------
  // Encode
  // -----------------------------------------------------------------------------------------

  /**
   * Encode un coup en index policy AlphaZero. L'indice dépend du from-square et du type de coup
   * (queen-style, knight-style, ou underpromotion). L'inversion de perspective ({@code XOR 56}) est
   * appliquée si {@code sideToMove == BLACK} (cf. SPEC §5.5.3).
   *
   * @param move coup encodé selon le format {@link Move} 16-bit
   * @param sideToMove côté au trait courant ({@link Color#WHITE} ou {@link Color#BLACK})
   * @return index dans {@code [0..4671]}
   * @throws IllegalArgumentException si le coup est inencodable (delta ne correspondant à aucun
   *     type)
   */
  public static int encode(int move, int sideToMove) {
    int from = Move.from(move);
    int to = Move.to(move);
    int type = Move.type(move);
    int promo = Move.promo(move);

    if (sideToMove == Color.BLACK) {
      from ^= 56;
      to ^= 56;
    }

    int dRank = Square.rank(to) - Square.rank(from);
    int dFile = Square.file(to) - Square.file(from);

    int planeIndex;
    if (type == MoveType.PROMOTION && promo != 3) {
      // Underpromotion (planes 64-72). promo : 0=N, 1=B, 2=R.
      int directionIndex = dFile + 1;
      if (directionIndex < 0 || directionIndex > 2) {
        throw new IllegalArgumentException(
            "Underpromotion deltaFile invalide : " + dFile + " (attendu : -1, 0, +1)");
      }
      planeIndex = 64 + directionIndex * 3 + promo;
    } else if (isKnightShape(dRank, dFile)) {
      int knightIdx = KNIGHT_DELTA_LUT[(dRank + 2) * 5 + (dFile + 2)];
      planeIndex = 56 + knightIdx;
    } else {
      // Queen-style (inclut promotion en dame, EP, castling, normal slider/king/pawn-push).
      int signR = Integer.signum(dRank);
      int signF = Integer.signum(dFile);
      int direction = QUEEN_DIR_LUT[(signR + 1) * 3 + (signF + 1)];
      if (direction < 0) {
        throw new IllegalArgumentException(
            "Coup inencodable : delta queen-style invalide (dRank="
                + dRank
                + ", dFile="
                + dFile
                + ")");
      }
      int distance = Math.max(Math.abs(dRank), Math.abs(dFile));
      if (distance < 1 || distance > 7) {
        throw new IllegalArgumentException(
            "Distance queen-style hors plage : " + distance + " (attendu : 1..7)");
      }
      planeIndex = direction * 7 + (distance - 1);
    }

    return from * POLICY_PLANES + planeIndex;
  }

  // -----------------------------------------------------------------------------------------
  // Decode
  // -----------------------------------------------------------------------------------------

  /**
   * Décode un index policy en coup, dans le contexte d'une position donnée. La position est
   * nécessaire pour distinguer les types de coup (NORMAL, EN_PASSANT, CASTLING, PROMOTION en dame).
   * Conforme §5.5.4.
   *
   * @param policyIndex index dans {@code [0..4671]}
   * @param position position courante pour résoudre le type de coup
   * @return coup encodé selon le format {@link Move} 16-bit
   * @throws IllegalArgumentException si {@code policyIndex} est hors plage
   */
  public static int decode(int policyIndex, Position position) {
    if (policyIndex < 0 || policyIndex >= POLICY_INDICES) {
      throw new IllegalArgumentException(
          "policyIndex hors plage : "
              + policyIndex
              + " (attendu : 0.."
              + (POLICY_INDICES - 1)
              + ")");
    }
    int sideToMove = position.sideToMove();
    int fromP1 = policyIndex / POLICY_PLANES;
    int plane = policyIndex % POLICY_PLANES;

    int dRankP1;
    int dFileP1;
    int type;
    int promo;
    boolean isUnderpromo = plane >= 64;
    boolean isKnight = !isUnderpromo && plane >= 56;

    if (isUnderpromo) {
      int directionIndex = (plane - 64) / 3;
      int pieceIndex = (plane - 64) % 3;
      dFileP1 = UNDERPROMO_FILE_DELTAS[directionIndex];
      // P1 (côté au trait) avance toujours vers le haut en perspective normalisée.
      dRankP1 = +1;
      type = MoveType.PROMOTION;
      promo = pieceIndex; // 0=N, 1=B, 2=R
    } else if (isKnight) {
      int[] delta = KNIGHT_DELTAS[plane - 56];
      dRankP1 = delta[0];
      dFileP1 = delta[1];
      type = MoveType.NORMAL;
      promo = 0;
    } else {
      int direction = plane / 7;
      int distance = plane % 7 + 1;
      int[] dir = QUEEN_DELTAS[direction];
      dRankP1 = dir[0] * distance;
      dFileP1 = dir[1] * distance;
      // type/promo déterminés ci-dessous selon la position
      type = -1;
      promo = 0;
    }

    int toP1 = fromP1 + dRankP1 * 8 + dFileP1;
    int from;
    int to;
    if (sideToMove == Color.BLACK) {
      from = fromP1 ^ 56;
      to = toP1 ^ 56;
    } else {
      from = fromP1;
      to = toP1;
    }

    if (!isUnderpromo && !isKnight) {
      // Queen-style : type déterminé par la pièce qui bouge et la géométrie du coup.
      long fromBB = 1L << from;
      int movingPieceType = pieceTypeAt(position, fromBB, sideToMove);
      int absFileDiff = Math.abs(Square.file(to) - Square.file(from));

      if (movingPieceType == PieceType.KING && absFileDiff == 2) {
        type = MoveType.CASTLING;
      } else if (movingPieceType == PieceType.PAWN && to == position.epSquare()) {
        type = MoveType.EN_PASSANT;
      } else if (movingPieceType == PieceType.PAWN
          && (Square.rank(to) == 0 || Square.rank(to) == 7)) {
        type = MoveType.PROMOTION;
        promo = 3; // queen
      } else {
        type = MoveType.NORMAL;
        promo = 0;
      }
    }

    return Move.encode(from, to, type, promo);
  }

  /**
   * Retourne le type de pièce ({@link PieceType#PAWN}..{@link PieceType#KING}) du côté donné qui
   * occupe la case bit-encodée par {@code sqBB}, ou {@link PieceType#NONE} si vide.
   */
  private static int pieceTypeAt(Position pos, long sqBB, int color) {
    for (int pt = 0; pt < PieceType.NB_PIECE_TYPES; pt++) {
      if ((pos.pieceBB(Piece.make(color, pt)) & sqBB) != 0L) {
        return pt;
      }
    }
    return PieceType.NONE;
  }

  /** Indique si {@code (dRank, dFile)} est un saut de cavalier (L-shape 1×2 ou 2×1). */
  private static boolean isKnightShape(int dRank, int dFile) {
    int aR = Math.abs(dRank);
    int aF = Math.abs(dFile);
    return (aR == 1 && aF == 2) || (aR == 2 && aF == 1);
  }

  // -----------------------------------------------------------------------------------------
  // decodePolicy : softmax masqué sur coups légaux
  // -----------------------------------------------------------------------------------------

  /**
   * Applique un softmax masqué sur les coups légaux. Conforme §5.5.5.
   *
   * <p>Étapes :
   *
   * <ol>
   *   <li>Extraire les logits aux indices des coups légaux ({@code dest[i] = logits[encode(...)]})
   *   <li>Softmax stable (subtract max avant exp pour éviter overflow)
   *   <li>Normalisation à somme = 1
   * </ol>
   *
   * <p>Validation stricte des arguments avant toute logique : {@link IllegalArgumentException} avec
   * message localisé pour chaque échec. Au-delà, le contrat de {@link #encode(int, int)} prend le
   * relais sur {@code legalMoves[i]} et {@code sideToMove} (validation existant phase 1).
   *
   * <p><strong>Zéro allocation</strong> : aucun {@code new}. Le caller MCTS réutilise les buffers
   * {@code legalMoves} et {@code dest} entre nodes.
   *
   * @param logits logits bruts d'une position : {@code float[POLICY_INDICES]} (4672 entrées)
   * @param legalMoves liste de coups légaux générés via {@code MoveGen}
   * @param numLegalMoves nombre de coups légaux dans le buffer (≥ 1, ≤ {@code legalMoves.length})
   * @param sideToMove côté au trait pour l'encodage de perspective ({@code Color.WHITE} ou {@code
   *     Color.BLACK})
   * @param dest buffer destinataire : {@code float[≥ numLegalMoves]}
   * @throws IllegalArgumentException si un argument viole le contrat (null, longueurs, bornes)
   */
  public static void decodePolicy(
      float[] logits, int[] legalMoves, int numLegalMoves, int sideToMove, float[] dest) {
    if (logits == null) {
      throw new IllegalArgumentException("logits must not be null");
    }
    if (logits.length != POLICY_INDICES) {
      throw new IllegalArgumentException(
          "logits length must be " + POLICY_INDICES + ", got " + logits.length);
    }
    if (legalMoves == null) {
      throw new IllegalArgumentException("legalMoves must not be null");
    }
    if (numLegalMoves < 1) {
      throw new IllegalArgumentException(
          "decodePolicy requires at least 1 legal move (received "
              + numLegalMoves
              + "); caller must check terminal state before invoking");
    }
    if (numLegalMoves > legalMoves.length) {
      throw new IllegalArgumentException(
          "numLegalMoves ("
              + numLegalMoves
              + ") exceeds legalMoves buffer length ("
              + legalMoves.length
              + ")");
    }
    if (dest == null) {
      throw new IllegalArgumentException("dest must not be null");
    }
    if (dest.length < numLegalMoves) {
      throw new IllegalArgumentException(
          "dest length (" + dest.length + ") must be >= numLegalMoves (" + numLegalMoves + ")");
    }
    // sideToMove validé indirectement par encode() au premier coup (lève IAE si invalide).

    // 1. Extraire les logits aux indices des coups légaux.
    for (int i = 0; i < numLegalMoves; i++) {
      int idx = encode(legalMoves[i], sideToMove);
      dest[i] = logits[idx];
    }
    // 2. Softmax stable : subtract max avant exp.
    float maxLogit = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < numLegalMoves; i++) {
      if (dest[i] > maxLogit) {
        maxLogit = dest[i];
      }
    }
    float sum = 0f;
    for (int i = 0; i < numLegalMoves; i++) {
      float e = (float) Math.exp(dest[i] - maxLogit);
      dest[i] = e;
      sum += e;
    }
    // 3. Normalisation.
    float invSum = 1f / sum;
    for (int i = 0; i < numLegalMoves; i++) {
      dest[i] *= invSum;
    }
  }
}
