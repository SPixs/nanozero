package org.nanozero.board;

import java.io.PrintStream;

/**
 * État courant d'une position d'échecs (cf. SPEC §4.2.1, §5.1, ADR-005, ADR-006).
 *
 * <p>{@code Position} tient uniquement l'état immédiat (bitboards de pièces, occupancy, côté au
 * trait, droits de roque, EP square, compteurs, hash Zobrist) — pas d'historique. La gestion des
 * historiques (positions précédentes pour {@code unapplyLastMove}, plans NN, détection de
 * répétition) relève de {@link GameState} (cf. ADR-005).
 *
 * <p>Les champs internes sont package-private (pas de modificateur d'accès) pour permettre l'accès
 * direct aux <em>friends</em> du même package ({@code MoveGen}, {@code Fen}, {@code GameState},
 * {@code Zobrist}) sans surcoût d'invocation. Les consommateurs externes au package DOIVENT
 * utiliser exclusivement les getters publics, qui sont triviaux et inlinés par le JIT C2 après
 * warmup (cf. ADR-006).
 *
 * <p>{@code Position} est <strong>mutable</strong> et <strong>non thread-safe</strong>. Toute
 * mutation par un thread invalide les lectures concurrentes par un autre. Le pattern usuel en MCTS
 * multi-thread est de confiner une instance par thread (cf. SPEC §10.2).
 *
 * <p>Invariants à maintenir après tout {@code applyMove} (cf. §4.2.1) :
 *
 * <ul>
 *   <li>I-Pos-1 : {@code occupancyBB[2] == occupancyBB[0] | occupancyBB[1]}
 *   <li>I-Pos-2 : {@code (occupancyBB[0] & occupancyBB[1]) == 0}
 *   <li>I-Pos-3 : pour {@code c ∈ {0,1}} : {@code occupancyBB[c]} = OR des {@code
 *       pieceBB[c*6..c*6+5]}
 *   <li>I-Pos-4 : {@code popcount(pieceBB[WHITE_KING]) == popcount(pieceBB[BLACK_KING]) == 1}
 *   <li>I-Pos-5 : {@code zobristHash == Zobrist.computeFull(this)} (vérifiable en assertion)
 * </ul>
 */
public final class Position {

  // ---------------------------------------------------------------------------------------------
  // Tables statiques internes pour applyMove (cf. SPEC §5.1)
  // ---------------------------------------------------------------------------------------------

  /**
   * Mask de droits de roque à appliquer après chaque coup, indexé par square. {@code rights &
   * MASK[from] & MASK[to]} retire correctement les droits dans tous les cas (roi qui bouge de
   * E1/E8, tour qui bouge d'un coin, capture d'une tour adverse sur un coin). Toute autre case n'a
   * aucun impact : MASK = -1 (tous bits à 1).
   */
  private static final int[] CASTLING_RIGHTS_MASK = new int[64];

  static {
    java.util.Arrays.fill(CASTLING_RIGHTS_MASK, -1);
    CASTLING_RIGHTS_MASK[Square.A1] = ~Castling.WHITE_QUEENSIDE;
    CASTLING_RIGHTS_MASK[Square.E1] = ~Castling.WHITE_BOTH;
    CASTLING_RIGHTS_MASK[Square.H1] = ~Castling.WHITE_KINGSIDE;
    CASTLING_RIGHTS_MASK[Square.A8] = ~Castling.BLACK_QUEENSIDE;
    CASTLING_RIGHTS_MASK[Square.E8] = ~Castling.BLACK_BOTH;
    CASTLING_RIGHTS_MASK[Square.H8] = ~Castling.BLACK_KINGSIDE;
  }

  // ---------------------------------------------------------------------------------------------
  // État interne (package-private — voir Javadoc de classe)
  // ---------------------------------------------------------------------------------------------

  /** Bitboards de pièces, indexés par {@code color * 6 + pieceType} (cf. ADR-009). */
  final long[] pieceBB = new long[Piece.NB_PIECES];

  /** Occupancy par couleur : index 0 = WHITE, 1 = BLACK, 2 = BOTH (union, cf. SPEC §4.2.1). */
  final long[] occupancyBB = new long[3];

  int sideToMove;
  int castlingRights;
  int epSquare;
  int halfmoveClock;
  int fullmoveNumber;
  long zobristHash;

  // ---------------------------------------------------------------------------------------------
  // Constructeurs
  // ---------------------------------------------------------------------------------------------

  /**
   * Construit une {@code Position} vide : tous les bitboards à zéro, {@link Color#WHITE} au trait,
   * aucun droit de roque, pas d'EP, compteurs initialisés ({@code halfmoveClock = 0}, {@code
   * fullmoveNumber = 1}), hash Zobrist à zéro.
   *
   * <p>Cette construction NE produit PAS la position de départ d'une partie d'échecs — elle produit
   * un échiquier vide. Pour obtenir la position de départ standard, utiliser {@code
   * Fen.parse(Fen.STARTPOS)} une fois le module {@code Fen} disponible (phase 3).
   */
  public Position() {
    this.sideToMove = Color.WHITE;
    this.castlingRights = Castling.NONE;
    this.epSquare = Square.NONE;
    this.halfmoveClock = 0;
    this.fullmoveNumber = 1;
    this.zobristHash = 0L;
  }

  // ---------------------------------------------------------------------------------------------
  // Getters publics (triviaux, inlinés par le JIT)
  // ---------------------------------------------------------------------------------------------

  /**
   * Retourne la référence interne au tableau des bitboards de pièces, de longueur 12.
   *
   * <p>Le tableau est indexé par {@code color * 6 + pieceType} (cf. {@link Piece#make}). Le contenu
   * N'EST PAS recopié : toute mutation par l'appelant corrompt l'état et viole les invariants. Les
   * consommateurs externes au package {@code org.nanozero.board} DOIVENT considérer ce tableau
   * comme strictement en lecture seule.
   */
  public long[] pieceBB() {
    return pieceBB;
  }

  /** Bitboard de toutes les pièces du type donné, indexé par {@link Piece}. */
  public long pieceBB(int piece) {
    return pieceBB[piece];
  }

  /**
   * Retourne la référence interne au tableau d'occupancy de longueur 3 : {@code [WHITE, BLACK,
   * BOTH]}. Mêmes contraintes que {@link #pieceBB()} : lecture seule pour les consommateurs
   * externes.
   */
  public long[] occupancyBB() {
    return occupancyBB;
  }

  /** Bitboard d'occupancy pour la couleur donnée ({@link Color#WHITE} ou {@link Color#BLACK}). */
  public long occupancyBB(int color) {
    return occupancyBB[color];
  }

  /** Bitboard d'occupancy totale (équivaut à {@code occupancyBB(0) | occupancyBB(1)}). */
  public long allOccupancy() {
    return occupancyBB[2];
  }

  /** Côté au trait : {@link Color#WHITE} ou {@link Color#BLACK}. */
  public int sideToMove() {
    return sideToMove;
  }

  /** Droits de roque, bitmask sur 4 bits (cf. {@link Castling}). */
  public int castlingRights() {
    return castlingRights;
  }

  /** Case d'EP disponible {@code 0..63}, ou {@link Square#NONE} (= -1) si absente. */
  public int epSquare() {
    return epSquare;
  }

  /**
   * Compteur des 50 coups : nombre de demi-coups depuis la dernière capture, avance de pion ou EP.
   * Sert au calcul de la règle de nullité et à la borne basse de la fenêtre de répétition (cf. SPEC
   * §5.2.3, §6.3).
   */
  public int halfmoveClock() {
    return halfmoveClock;
  }

  /** Numéro de coup complet : incrémenté après chaque coup des noirs. */
  public int fullmoveNumber() {
    return fullmoveNumber;
  }

  /** Hash Zobrist incrémental, cohérent avec l'état courant (cf. SPEC §5.6). */
  public long zobristHash() {
    return zobristHash;
  }

  // ---------------------------------------------------------------------------------------------
  // Mutation contrôlée
  // ---------------------------------------------------------------------------------------------

  /**
   * Copie l'état complet de {@code other} dans {@code this} via {@link System#arraycopy} et
   * assignations primitives. Zéro allocation, conforme à la cible §9.2 ({@code < 30 ns}).
   *
   * @param other position source à copier
   */
  public void copyFrom(Position other) {
    System.arraycopy(other.pieceBB, 0, this.pieceBB, 0, Piece.NB_PIECES);
    System.arraycopy(other.occupancyBB, 0, this.occupancyBB, 0, 3);
    this.sideToMove = other.sideToMove;
    this.castlingRights = other.castlingRights;
    this.epSquare = other.epSquare;
    this.halfmoveClock = other.halfmoveClock;
    this.fullmoveNumber = other.fullmoveNumber;
    this.zobristHash = other.zobristHash;
  }

  /**
   * Alloue une nouvelle {@code Position} et y copie l'état courant. NE DOIT PAS être appelé en hot
   * path : utiliser {@link #copyFrom(Position)} sur une instance scratch préallouée.
   *
   * @return nouvelle instance avec le même état
   */
  public Position copy() {
    Position p = new Position();
    p.copyFrom(this);
    return p;
  }

  /**
   * Applique un coup encodé (cf. {@link Move}, format §3.4) à la position courante, mettant à jour
   * incrémentalement les bitboards, le hash Zobrist, les compteurs, les droits de roque, l'EP
   * square et le côté au trait. Conforme à l'algorithme normatif §5.1 du SPEC.
   *
   * <p>Précondition : le coup DOIT être légal pour la position courante (généré par {@link
   * MoveGen#generateMoves}). En mode {@code -ea} (assertions activées), les invariants {@code
   * I-Pos-1} à {@code I-Pos-5} sont vérifiés en sortie ; en production, ils sont supposés.
   *
   * <p>L'EP square est positionné <em>inconditionnellement</em> après tout double-push de pion,
   * conformément à §5.1 step 5 NORMAL et à la convention X-FEN utilisée par chesslib. Cela garantit
   * la cohérence FEN avec les outils externes ; la sémantique de répétition s'aligne sur ce choix.
   *
   * @param move coup encodé selon le format {@link Move} 16-bit
   */
  public void applyMove(int move) {
    int from = Move.from(move);
    int to = Move.to(move);
    int type = Move.type(move);
    int promo = Move.promo(move);
    int us = sideToMove;
    int them = us ^ 1;

    int movingPiece = pieceAt(from);
    int capturedPiece = (type == MoveType.EN_PASSANT) ? Piece.NONE : pieceAt(to);
    int movingType = movingPiece % 6;

    // 3. XOR-out de l'ancien EP square si présent.
    if (epSquare != Square.NONE) {
      zobristHash ^= Zobrist.enPassantFile(Square.file(epSquare));
    }
    epSquare = Square.NONE;

    // 4. Halfmove clock : reset sur pawn move, capture ou EP, sinon increment.
    if (movingType == PieceType.PAWN
        || capturedPiece != Piece.NONE
        || type == MoveType.EN_PASSANT) {
      halfmoveClock = 0;
    } else {
      halfmoveClock += 1;
    }

    // 5. Application selon le type de coup.
    switch (type) {
      case MoveType.NORMAL -> {
        removePiece(movingPiece, from);
        if (capturedPiece != Piece.NONE) {
          removePiece(capturedPiece, to);
        }
        placePiece(movingPiece, to);
        // Double push : positionner EP inconditionnellement (convention §5.1 / X-FEN).
        if (movingType == PieceType.PAWN && Math.abs(to - from) == 16) {
          epSquare = (from + to) >>> 1;
          zobristHash ^= Zobrist.enPassantFile(Square.file(epSquare));
        }
      }
      case MoveType.PROMOTION -> {
        removePiece(movingPiece, from);
        if (capturedPiece != Piece.NONE) {
          removePiece(capturedPiece, to);
        }
        int promotedPiece = Piece.make(us, promo + PieceType.KNIGHT);
        placePiece(promotedPiece, to);
      }
      case MoveType.EN_PASSANT -> {
        removePiece(movingPiece, from);
        placePiece(movingPiece, to);
        int capturedPawnSq = (us == Color.WHITE) ? to - 8 : to + 8;
        removePiece(Piece.make(them, PieceType.PAWN), capturedPawnSq);
      }
      case MoveType.CASTLING -> {
        int king = Piece.make(us, PieceType.KING);
        removePiece(king, from);
        placePiece(king, to);
        int rookFrom;
        int rookTo;
        switch (to) {
          case Square.G1 -> {
            rookFrom = Square.H1;
            rookTo = Square.F1;
          }
          case Square.C1 -> {
            rookFrom = Square.A1;
            rookTo = Square.D1;
          }
          case Square.G8 -> {
            rookFrom = Square.H8;
            rookTo = Square.F8;
          }
          case Square.C8 -> {
            rookFrom = Square.A8;
            rookTo = Square.D8;
          }
          default ->
              throw new IllegalStateException(
                  "CASTLING vers une case invalide : " + Square.toAlgebraic(to));
        }
        int rook = Piece.make(us, PieceType.ROOK);
        removePiece(rook, rookFrom);
        placePiece(rook, rookTo);
      }
      default -> throw new IllegalStateException("MoveType invalide : " + type);
    }

    // 6. Mise à jour des droits de roque (XOR différentiel sur les bits qui ont changé).
    int oldRights = castlingRights;
    castlingRights &= CASTLING_RIGHTS_MASK[from] & CASTLING_RIGHTS_MASK[to];
    int xorRights = oldRights ^ castlingRights;
    while (xorRights != 0) {
      int bit = xorRights & -xorRights;
      xorRights &= xorRights - 1;
      zobristHash ^= Zobrist.castling(bit);
    }

    // 7. Côté au trait.
    sideToMove = them;
    zobristHash ^= Zobrist.sideBlack();

    // 8. Fullmove number : incrémenté après le coup des noirs.
    if (sideToMove == Color.WHITE) {
      fullmoveNumber += 1;
    }

    // 9. Reconstruction des occupancyBB.
    long w = 0L;
    long b = 0L;
    for (int i = 0; i < 6; i++) {
      w |= pieceBB[i];
      b |= pieceBB[6 + i];
    }
    occupancyBB[Color.WHITE] = w;
    occupancyBB[Color.BLACK] = b;
    occupancyBB[2] = w | b;

    assert assertInvariantsAfterApply();
  }

  /**
   * Helper {@code applyMove} : retire une pièce d'une case et XORe le hash Zobrist en conséquence.
   */
  private void removePiece(int piece, int square) {
    pieceBB[piece] &= ~(1L << square);
    zobristHash ^= Zobrist.pieceSquare(piece, square);
  }

  /**
   * Helper {@code applyMove} : place une pièce sur une case et XORe le hash Zobrist en conséquence.
   */
  private void placePiece(int piece, int square) {
    pieceBB[piece] |= 1L << square;
    zobristHash ^= Zobrist.pieceSquare(piece, square);
  }

  /**
   * Vérifie les invariants {@code I-Pos-1} à {@code I-Pos-5} de §4.2.1, utilisé en assertion après
   * {@link #applyMove}. Activable via {@code -ea}, no-op en production.
   */
  private boolean assertInvariantsAfterApply() {
    long w = 0L;
    long b = 0L;
    for (int i = 0; i < 6; i++) {
      w |= pieceBB[i];
      b |= pieceBB[6 + i];
    }
    if (occupancyBB[Color.WHITE] != w
        || occupancyBB[Color.BLACK] != b
        || occupancyBB[2] != (w | b)) {
      throw new AssertionError("I-Pos-1/I-Pos-3 violé après applyMove");
    }
    if ((w & b) != 0L) {
      throw new AssertionError("I-Pos-2 violé après applyMove");
    }
    if (Long.bitCount(pieceBB[Piece.WHITE_KING]) != 1
        || Long.bitCount(pieceBB[Piece.BLACK_KING]) != 1) {
      throw new AssertionError("I-Pos-4 violé après applyMove");
    }
    long expected = Zobrist.computeFull(this);
    if (zobristHash != expected) {
      throw new AssertionError(
          String.format(
              "I-Pos-5 violé après applyMove : incremental=0x%016X vs computeFull=0x%016X",
              zobristHash, expected));
    }
    return true;
  }

  // ---------------------------------------------------------------------------------------------
  // Requêtes sur la position
  // ---------------------------------------------------------------------------------------------

  /**
   * Indique si le côté au trait est en échec (son roi est attaqué par l'adversaire).
   *
   * @return {@code true} si le roi du côté au trait est sous attaque
   */
  public boolean isInCheck() {
    int kingPiece = Piece.make(sideToMove, PieceType.KING);
    long kingBB = pieceBB[kingPiece];
    if (kingBB == 0L) {
      return false;
    }
    int kingSq = Long.numberOfTrailingZeros(kingBB);
    return isSquareAttacked(kingSq, Color.opponent(sideToMove));
  }

  /**
   * Calcule un bitboard des pièces du côté donné qui attaquent la case donnée.
   *
   * <p>Implémentation par OR de toutes les attaques inverses :
   *
   * <ul>
   *   <li>Pions : {@code pawnAttacks(opponent(byColor), square) & pieceBB[byColor PAWN]} (un pion
   *       de {@code byColor} attaque {@code square} ssi un pion de la couleur opposée placé sur
   *       {@code square} attaquerait sa case ; symétrie de l'attaque diagonale).
   *   <li>Cavaliers, roi : intersection directe avec les tables non-sliders.
   *   <li>Fous, dames sur diagonales : {@code bishopAttacks(square, occupancy)}.
   *   <li>Tours, dames sur lignes droites : {@code rookAttacks(square, occupancy)}.
   * </ul>
   *
   * @param square case cible {@code 0..63}
   * @param byColor couleur des attaquants ({@link Color#WHITE} ou {@link Color#BLACK})
   * @return bitboard des pièces de {@code byColor} qui attaquent {@code square}
   */
  public long attackersOf(int square, int byColor) {
    long occ = occupancyBB[2];
    int opp = Color.opponent(byColor);
    long attackers = 0L;
    attackers |= Bitboards.pawnAttacks(opp, square) & pieceBB[Piece.make(byColor, PieceType.PAWN)];
    attackers |= Bitboards.knightAttacks(square) & pieceBB[Piece.make(byColor, PieceType.KNIGHT)];
    attackers |= Bitboards.kingAttacks(square) & pieceBB[Piece.make(byColor, PieceType.KING)];
    long bishopsQueens =
        pieceBB[Piece.make(byColor, PieceType.BISHOP)]
            | pieceBB[Piece.make(byColor, PieceType.QUEEN)];
    attackers |= Bitboards.bishopAttacks(square, occ) & bishopsQueens;
    long rooksQueens =
        pieceBB[Piece.make(byColor, PieceType.ROOK)]
            | pieceBB[Piece.make(byColor, PieceType.QUEEN)];
    attackers |= Bitboards.rookAttacks(square, occ) & rooksQueens;
    return attackers;
  }

  /**
   * Indique si une case est attaquée par au moins une pièce du côté donné.
   *
   * <p>Implémentation : équivalent fonctionnel à {@code attackersOf(square, byColor) != 0L}, sans
   * optimisation short-circuit en phase 2 (pourra être affiné en phase 5 lors de la migration
   * fully-legal de {@link MoveGen}).
   *
   * @param square case cible {@code 0..63}
   * @param byColor couleur des attaquants ({@link Color#WHITE} ou {@link Color#BLACK})
   * @return {@code true} si au moins une pièce de {@code byColor} attaque {@code square}
   */
  public boolean isSquareAttacked(int square, int byColor) {
    return attackersOf(square, byColor) != 0L;
  }

  // ---------------------------------------------------------------------------------------------
  // Helper interne : pieceAt (package-private)
  // ---------------------------------------------------------------------------------------------

  /**
   * Retourne l'index de pièce occupant {@code square} ou {@link Piece#NONE} si la case est vide.
   *
   * <p>Utilisé par {@link Move#fromUci(String, Position)} pour résoudre le type d'un coup, par les
   * sérialisations FEN/ASCII, et par {@code applyMove} (phase 6). Itération linéaire sur les 12
   * bitboards : O(12) par appel ; les hot paths peuvent éviter ce coût en s'appuyant directement
   * sur les bitboards.
   *
   * @param square index de case {@code 0..63}
   * @return index de pièce {@code 0..11} ou {@link Piece#NONE}
   */
  int pieceAt(int square) {
    long sqBB = 1L << square;
    if ((occupancyBB[2] & sqBB) == 0L) {
      return Piece.NONE;
    }
    for (int p = 0; p < Piece.NB_PIECES; p++) {
      if ((pieceBB[p] & sqBB) != 0L) {
        return p;
      }
    }
    return Piece.NONE;
  }

  // ---------------------------------------------------------------------------------------------
  // Méthodes de debug — NE PAS UTILISER en hot path (allouantes)
  // ---------------------------------------------------------------------------------------------

  /**
   * Sérialise la position en notation FEN canonique en délégant à {@link Fen#write(Position)}.
   * {@link Fen} est la source de vérité du format ; {@link Position} ne duplique pas cette
   * connaissance (cf. SPEC §5.7.3 et §13.1).
   *
   * @return FEN à 6 champs séparés par des espaces simples
   */
  public String toFen() {
    return Fen.write(this);
  }

  /**
   * Retourne une représentation ASCII de l'échiquier sur 17 lignes (cf. SPEC §13.2). Format
   * normatif : majuscules pour les blancs, minuscules pour les noirs, espace pour case vide ; rangs
   * 1-8 dans la marge gauche, fichiers a-h dans la marge basse ; pas de couleur ANSI.
   *
   * @return représentation lisible de l'échiquier
   */
  public String toAsciiBoard() {
    String separator = "  +---+---+---+---+---+---+---+---+";
    StringBuilder sb = new StringBuilder(17 * 38);
    sb.append(separator).append('\n');
    for (int rank = 7; rank >= 0; rank--) {
      sb.append(rank + 1).append(' ');
      for (int file = 0; file < 8; file++) {
        int sq = rank * 8 + file;
        int p = pieceAt(sq);
        char c = (p == Piece.NONE) ? ' ' : PieceType.toChar(Piece.typeOf(p), Piece.colorOf(p));
        sb.append("| ").append(c).append(' ');
      }
      sb.append("|\n");
      sb.append(separator).append('\n');
    }
    sb.append("    a   b   c   d   e   f   g   h\n");
    return sb.toString();
  }

  /**
   * Imprime un dump complet de la position sur le flux fourni (cf. SPEC §13.3). Format : FEN +
   * board ASCII + bloc « State » (compteurs + Zobrist hash) + bloc « Bitboards » (12 bitboards de
   * pièces et 3 occupancy en hexadécimal sur 16 chiffres).
   *
   * @param out flux destinataire
   */
  public void dump(PrintStream out) {
    out.println("Position:");
    out.print("  FEN: ");
    out.println(toFen());
    out.println();
    out.print(toAsciiBoard());
    out.println();
    out.println("State:");
    out.printf("  Side to move    : %s%n", sideToMove == Color.WHITE ? "WHITE" : "BLACK");
    out.printf("  Castling rights : %s (0x%X)%n", castlingRightsString(), castlingRights);
    out.printf(
        "  EP square       : %s (%d)%n",
        epSquare == Square.NONE ? "-" : Square.toAlgebraic(epSquare), epSquare);
    out.printf("  Halfmove clock  : %d%n", halfmoveClock);
    out.printf("  Fullmove number : %d%n", fullmoveNumber);
    out.printf("  Zobrist hash    : 0x%016X%n", zobristHash);
    out.println();
    out.println("Bitboards:");
    String[] names = {
      "WHITE_PAWN  ", "WHITE_KNIGHT", "WHITE_BISHOP", "WHITE_ROOK  ", "WHITE_QUEEN ",
      "WHITE_KING  ", "BLACK_PAWN  ", "BLACK_KNIGHT", "BLACK_BISHOP", "BLACK_ROOK  ",
      "BLACK_QUEEN ", "BLACK_KING  "
    };
    for (int p = 0; p < Piece.NB_PIECES; p++) {
      out.printf("  %s : 0x%016X%n", names[p], pieceBB[p]);
    }
    out.printf("  OCC_WHITE    : 0x%016X%n", occupancyBB[Color.WHITE]);
    out.printf("  OCC_BLACK    : 0x%016X%n", occupancyBB[Color.BLACK]);
    out.printf("  OCC_BOTH     : 0x%016X%n", occupancyBB[2]);
  }

  private String castlingRightsString() {
    if (castlingRights == Castling.NONE) {
      return "-";
    }
    StringBuilder sb = new StringBuilder(4);
    if ((castlingRights & Castling.WHITE_KINGSIDE) != 0) {
      sb.append('K');
    }
    if ((castlingRights & Castling.WHITE_QUEENSIDE) != 0) {
      sb.append('Q');
    }
    if ((castlingRights & Castling.BLACK_KINGSIDE) != 0) {
      sb.append('k');
    }
    if ((castlingRights & Castling.BLACK_QUEENSIDE) != 0) {
      sb.append('q');
    }
    return sb.toString();
  }

  /** Identique à {@link #toFen()}. Conforme à §4.2.1 du SPEC. */
  @Override
  public String toString() {
    return toFen();
  }
}
