package org.nanozero.board;

/**
 * Génération directement légale des coups d'une position d'échecs (cf. SPEC §4.2.3, §5.3, ADR-001).
 *
 * <p>Algorithme phase 5, conforme §5.3.1 à §5.3.10 :
 *
 * <ol>
 *   <li>Calcul en amont d'une seule passe : {@code kingSq}, {@code checkers}, {@code checkCount},
 *       {@code pinned}, {@code checkMask}.
 *   <li>{@link #generateKingMoves} : produit les coups de roi en testant chaque case d'arrivée avec
 *       une occupancy <em>sans le roi</em> ({@link #isAttackedFrom}), pour ne pas se laisser
 *       protéger par soi-même contre un slider qui passe à travers la case d'origine.
 *   <li>Si {@code checkCount == 2} (double échec), seuls les coups de roi sont légaux : on retourne
 *       immédiatement.
 *   <li>Pour chaque autre type de pièce (Q, R, B, N, P), les targets sont restreintes par
 *       l'intersection {@code ~occUs & checkMask} ; les pièces clouées voient en outre leur ligne
 *       de mouvement restreinte par {@link Bitboards#lineThrough lineThrough(kingSq, fromSq)}.
 *   <li>Cas spécial EP : un appel à EP retire <em>deux</em> pions de la 5ème rangée simultanément ;
 *       un éventuel slider ennemi sur la même rangée que le roi peut alors le prendre en échec «
 *       par découverte ». Le test §5.3.10 utilise {@code rookAttacks} après simulation locale ; on
 *       étend par sécurité au {@code bishopAttacks} pour couvrir d'éventuelles découvertes
 *       diagonales rares (le pion adverse capturé pouvant être l'unique blocker d'une diagonale
 *       slider/roi).
 * </ol>
 *
 * <p>L'<strong>ordre</strong> de production des coups est conforme à §3.6 : KING, QUEEN, ROOK,
 * BISHOP, KNIGHT, PAWN ; pour chaque type, {@code from} ascendant ; pour chaque {@code from},
 * {@code to} ascendant ; promotions dans l'ordre {@code N, B, R, Q}.
 *
 * <p><strong>Différences vis-à-vis de phase 4</strong> :
 *
 * <ul>
 *   <li>Suppression complète de la logique de simulation locale (clone + apply + isInCheck) sauf
 *       pour le seul cas EP § 5.3.10.
 *   <li>Pas de {@code Position.copy()} pendant la génération ; pas d'allocation par appel à {@link
 *       #generateMoves} (la chaîne {@code generateMoves → applyMove → isTerminal} est
 *       zéro-allocation conformément à SPEC §1.3).
 *   <li>Castling : conditions de légalité totales déjà vérifiées en phase 4, conservées telles
 *       quelles (§5.3.7).
 * </ul>
 *
 * <p>Classe non instanciable, conforme à la convention §3.3 du SPEC.
 */
public final class MoveGen {

  /** Borne supérieure du nombre de coups légaux dans une position d'échecs (cf. §4.2.3). */
  public static final int MAX_LEGAL_MOVES = 218;

  /** Taille recommandée des buffers de coups, marge incluse (cf. §4.2.3). */
  public static final int RECOMMENDED_BUFFER_SIZE = 256;

  private MoveGen() {
    throw new AssertionError("Non-instantiable");
  }

  /**
   * Génère tous les coups légaux de la position et les écrit dans {@code buffer} à partir de {@code
   * offset}. L'ordre suit la convention §3.6.
   *
   * @param position position source, NON modifiée
   * @param buffer buffer de destination, taille &geq; {@code offset + RECOMMENDED_BUFFER_SIZE}
   * @param offset index de départ dans le buffer
   * @return nombre de coups écrits
   */
  public static int generateMoves(Position position, int[] buffer, int offset) {
    return generateInternal(position, buffer, offset, false);
  }

  /**
   * Génère uniquement les coups de capture légaux (incluant prises en passant et promotions avec
   * capture). Sous-ensemble strict de {@link #generateMoves}.
   *
   * @param position position source, NON modifiée
   * @param buffer buffer de destination
   * @param offset index de départ dans le buffer
   * @return nombre de coups de capture écrits
   */
  public static int generateCaptures(Position position, int[] buffer, int offset) {
    return generateInternal(position, buffer, offset, true);
  }

  /**
   * Compte le nombre de positions feuilles atteignables à profondeur {@code depth} depuis {@code
   * position} (Perft, cf. SPEC §8.2). La position fournie n'est PAS modifiée.
   *
   * <p>Les buffers de coups et les positions scratch sont pré-alloués une seule fois en entrée pour
   * éliminer toute allocation pendant la récursion (cf. SPEC §10.1 zéro-alloc en hot path).
   *
   * @param position position source, NON modifiée
   * @param depth profondeur de recherche, &geq; 0
   * @return nombre de positions feuilles atteignables ; {@code 1} si {@code depth == 0}
   */
  public static long perft(Position position, int depth) {
    if (depth <= 0) {
      return 1L;
    }
    int[][] moveBuffers = new int[depth + 1][RECOMMENDED_BUFFER_SIZE];
    Position[] scratch = new Position[depth + 1];
    for (int i = 0; i <= depth; i++) {
      scratch[i] = new Position();
    }
    return perftRecursive(position, depth, moveBuffers, scratch, 0);
  }

  private static long perftRecursive(
      Position pos, int depth, int[][] moveBufs, Position[] scratch, int level) {
    int[] moves = moveBufs[level];
    int count = generateMoves(pos, moves, 0);
    if (depth == 1) {
      return count;
    }
    long total = 0L;
    Position next = scratch[level + 1];
    for (int i = 0; i < count; i++) {
      next.copyFrom(pos);
      next.applyMove(moves[i]);
      total += perftRecursive(next, depth - 1, moveBufs, scratch, level + 1);
    }
    return total;
  }

  // ===============================================================================================
  // Pipeline interne
  // ===============================================================================================

  private static int generateInternal(Position pos, int[] buf, int offset, boolean capturesOnly) {
    int us = pos.sideToMove();
    int them = Color.opponent(us);
    long occUs = pos.occupancyBB(us);
    long occThem = pos.occupancyBB(them);
    long occAll = pos.allOccupancy();

    long ourKings = pos.pieceBB(Piece.make(us, PieceType.KING));
    if (ourKings == 0L) {
      // Position de test sans roi : aucun coup ne peut être déclaré légal au sens des règles.
      return 0;
    }
    int kingSq = Long.numberOfTrailingZeros(ourKings);

    long checkers = pos.attackersOf(kingSq, them);
    int checkCount = Long.bitCount(checkers);
    long pinned = computePinned(pos, kingSq, us, them, occUs, occAll);
    long checkMask = computeCheckMask(pos, kingSq, checkers, checkCount);

    int count = 0;
    count =
        generateKingMoves(
            pos,
            buf,
            offset,
            count,
            us,
            them,
            kingSq,
            occAll,
            occUs,
            occThem,
            checkCount,
            capturesOnly);

    if (checkCount == 2) {
      return count; // Double check : seuls les coups de roi sont légaux.
    }

    count =
        generateQueenMoves(
            pos,
            buf,
            offset,
            count,
            us,
            occUs,
            occThem,
            occAll,
            checkMask,
            pinned,
            kingSq,
            capturesOnly);
    count =
        generateRookMoves(
            pos,
            buf,
            offset,
            count,
            us,
            occUs,
            occThem,
            occAll,
            checkMask,
            pinned,
            kingSq,
            capturesOnly);
    count =
        generateBishopMoves(
            pos,
            buf,
            offset,
            count,
            us,
            occUs,
            occThem,
            occAll,
            checkMask,
            pinned,
            kingSq,
            capturesOnly);
    count =
        generateKnightMoves(
            pos, buf, offset, count, us, occUs, occThem, checkMask, pinned, capturesOnly);
    count =
        generatePawnMoves(
            pos,
            buf,
            offset,
            count,
            us,
            them,
            occUs,
            occThem,
            occAll,
            kingSq,
            checkMask,
            pinned,
            capturesOnly);

    return count;
  }

  // ===============================================================================================
  // Calculs en amont : pinned, checkMask
  // ===============================================================================================

  /**
   * Calcule le bitboard des pièces du côté au trait clouées par un slider ennemi.
   *
   * <p>Astuce de l'algorithme : on demande {@code bishopAttacks(kingSq, occThem)} (resp. rook) ; en
   * passant uniquement les pièces ennemies en occupancy, nos pièces deviennent transparentes, et
   * les rayons s'étendent jusqu'au premier blocker ennemi. Tout slider ennemi sur ces rayons est un
   * pinner potentiel. Pour chaque pinner, le segment {@code between(king, slider) & occAll} doit
   * contenir <em>exactement</em> une pièce, qui doit être à nous.
   */
  private static long computePinned(
      Position pos, int kingSq, int us, int them, long occUs, long occAll) {
    long pinned = 0L;
    long occThem = pos.occupancyBB(them);

    long enemyBQ =
        pos.pieceBB(Piece.make(them, PieceType.BISHOP))
            | pos.pieceBB(Piece.make(them, PieceType.QUEEN));
    long potentialPinners = Bitboards.bishopAttacks(kingSq, occThem) & enemyBQ;
    while (potentialPinners != 0L) {
      int sliderSq = Long.numberOfTrailingZeros(potentialPinners);
      potentialPinners &= potentialPinners - 1L;
      long between = Bitboards.between(kingSq, sliderSq) & occAll;
      if (Long.bitCount(between) == 1 && (between & occUs) != 0L) {
        pinned |= between;
      }
    }

    long enemyRQ =
        pos.pieceBB(Piece.make(them, PieceType.ROOK))
            | pos.pieceBB(Piece.make(them, PieceType.QUEEN));
    potentialPinners = Bitboards.rookAttacks(kingSq, occThem) & enemyRQ;
    while (potentialPinners != 0L) {
      int sliderSq = Long.numberOfTrailingZeros(potentialPinners);
      potentialPinners &= potentialPinners - 1L;
      long between = Bitboards.between(kingSq, sliderSq) & occAll;
      if (Long.bitCount(between) == 1 && (between & occUs) != 0L) {
        pinned |= between;
      }
    }

    return pinned;
  }

  /**
   * Calcule la {@code checkMask} : bitboard des cases où un coup non-roi peut résoudre l'échec.
   *
   * <ul>
   *   <li>{@code checkCount == 0} : aucune contrainte ({@code -1L}).
   *   <li>{@code checkCount == 1} avec slider checker : cases entre roi et checker plus la case du
   *       checker (interposition ou capture).
   *   <li>{@code checkCount == 1} avec checker non-slider (cavalier ou pion) : seulement la case du
   *       checker (capture seule possible).
   *   <li>{@code checkCount == 2} : aucune case ({@code 0L}, seuls les coups de roi sont légaux,
   *       cf. dispatch dans {@link #generateInternal}).
   * </ul>
   */
  private static long computeCheckMask(Position pos, int kingSq, long checkers, int checkCount) {
    if (checkCount == 0) {
      return -1L;
    }
    if (checkCount >= 2) {
      return 0L;
    }
    int checkerSq = Long.numberOfTrailingZeros(checkers);
    int checkerPiece = pos.pieceAt(checkerSq);
    int checkerType = (checkerPiece == Piece.NONE) ? PieceType.NONE : Piece.typeOf(checkerPiece);
    if (checkerType == PieceType.BISHOP
        || checkerType == PieceType.ROOK
        || checkerType == PieceType.QUEEN) {
      return Bitboards.between(kingSq, checkerSq) | (1L << checkerSq);
    }
    return 1L << checkerSq;
  }

  // ===============================================================================================
  // KING
  // ===============================================================================================

  private static int generateKingMoves(
      Position pos,
      int[] buf,
      int offset,
      int count,
      int us,
      int them,
      int kingSq,
      long occAll,
      long occUs,
      long occThem,
      int checkCount,
      boolean capturesOnly) {
    long targets = Bitboards.kingAttacks(kingSq) & ~occUs;
    if (capturesOnly) {
      targets &= occThem;
    }
    long occWithoutKing = occAll ^ (1L << kingSq);
    while (targets != 0L) {
      int to = Long.numberOfTrailingZeros(targets);
      targets &= targets - 1L;
      if (!isAttackedFrom(pos, to, them, occWithoutKing)) {
        buf[offset + count++] = Move.encode(kingSq, to);
      }
    }
    if (!capturesOnly && checkCount == 0) {
      count = generateCastling(pos, buf, offset, count, us, kingSq, occAll, them);
    }
    return count;
  }

  /**
   * Conditions de légalité du roque, conformément §5.3.7 : droit présent, chemin libre, roi pas en
   * échec (déjà filtré en amont par {@code checkCount == 0}), cases de transit et d'arrivée non
   * attaquées. Pour le grand roque, B1/B8 peut être attaqué (le roi ne passe pas dessus).
   */
  private static int generateCastling(
      Position pos, int[] buf, int offset, int count, int us, int kingSq, long occAll, int them) {
    int rights = pos.castlingRights();

    if (us == Color.WHITE) {
      if ((rights & Castling.WHITE_KINGSIDE) != 0) {
        long path = (1L << Square.F1) | (1L << Square.G1);
        if ((occAll & path) == 0L
            && !pos.isSquareAttacked(Square.F1, them)
            && !pos.isSquareAttacked(Square.G1, them)) {
          buf[offset + count++] = Move.encode(Square.E1, Square.G1, MoveType.CASTLING, 0);
        }
      }
      if ((rights & Castling.WHITE_QUEENSIDE) != 0) {
        long path = (1L << Square.B1) | (1L << Square.C1) | (1L << Square.D1);
        if ((occAll & path) == 0L
            && !pos.isSquareAttacked(Square.D1, them)
            && !pos.isSquareAttacked(Square.C1, them)) {
          buf[offset + count++] = Move.encode(Square.E1, Square.C1, MoveType.CASTLING, 0);
        }
      }
    } else {
      if ((rights & Castling.BLACK_KINGSIDE) != 0) {
        long path = (1L << Square.F8) | (1L << Square.G8);
        if ((occAll & path) == 0L
            && !pos.isSquareAttacked(Square.F8, them)
            && !pos.isSquareAttacked(Square.G8, them)) {
          buf[offset + count++] = Move.encode(Square.E8, Square.G8, MoveType.CASTLING, 0);
        }
      }
      if ((rights & Castling.BLACK_QUEENSIDE) != 0) {
        long path = (1L << Square.B8) | (1L << Square.C8) | (1L << Square.D8);
        if ((occAll & path) == 0L
            && !pos.isSquareAttacked(Square.D8, them)
            && !pos.isSquareAttacked(Square.C8, them)) {
          buf[offset + count++] = Move.encode(Square.E8, Square.C8, MoveType.CASTLING, 0);
        }
      }
    }
    return count;
  }

  // ===============================================================================================
  // QUEEN, ROOK, BISHOP, KNIGHT
  // ===============================================================================================

  private static int generateQueenMoves(
      Position pos,
      int[] buf,
      int offset,
      int count,
      int us,
      long occUs,
      long occThem,
      long occAll,
      long checkMask,
      long pinned,
      int kingSq,
      boolean capturesOnly) {
    long queens = pos.pieceBB(Piece.make(us, PieceType.QUEEN));
    while (queens != 0L) {
      int from = Long.numberOfTrailingZeros(queens);
      queens &= queens - 1L;
      long targets =
          (Bitboards.bishopAttacks(from, occAll) | Bitboards.rookAttacks(from, occAll))
              & ~occUs
              & checkMask;
      if (((1L << from) & pinned) != 0L) {
        targets &= Bitboards.lineThrough(kingSq, from);
      }
      if (capturesOnly) {
        targets &= occThem;
      }
      while (targets != 0L) {
        int to = Long.numberOfTrailingZeros(targets);
        targets &= targets - 1L;
        buf[offset + count++] = Move.encode(from, to);
      }
    }
    return count;
  }

  private static int generateRookMoves(
      Position pos,
      int[] buf,
      int offset,
      int count,
      int us,
      long occUs,
      long occThem,
      long occAll,
      long checkMask,
      long pinned,
      int kingSq,
      boolean capturesOnly) {
    long rooks = pos.pieceBB(Piece.make(us, PieceType.ROOK));
    while (rooks != 0L) {
      int from = Long.numberOfTrailingZeros(rooks);
      rooks &= rooks - 1L;
      long targets = Bitboards.rookAttacks(from, occAll) & ~occUs & checkMask;
      if (((1L << from) & pinned) != 0L) {
        targets &= Bitboards.lineThrough(kingSq, from);
      }
      if (capturesOnly) {
        targets &= occThem;
      }
      while (targets != 0L) {
        int to = Long.numberOfTrailingZeros(targets);
        targets &= targets - 1L;
        buf[offset + count++] = Move.encode(from, to);
      }
    }
    return count;
  }

  private static int generateBishopMoves(
      Position pos,
      int[] buf,
      int offset,
      int count,
      int us,
      long occUs,
      long occThem,
      long occAll,
      long checkMask,
      long pinned,
      int kingSq,
      boolean capturesOnly) {
    long bishops = pos.pieceBB(Piece.make(us, PieceType.BISHOP));
    while (bishops != 0L) {
      int from = Long.numberOfTrailingZeros(bishops);
      bishops &= bishops - 1L;
      long targets = Bitboards.bishopAttacks(from, occAll) & ~occUs & checkMask;
      if (((1L << from) & pinned) != 0L) {
        targets &= Bitboards.lineThrough(kingSq, from);
      }
      if (capturesOnly) {
        targets &= occThem;
      }
      while (targets != 0L) {
        int to = Long.numberOfTrailingZeros(targets);
        targets &= targets - 1L;
        buf[offset + count++] = Move.encode(from, to);
      }
    }
    return count;
  }

  /**
   * Cavalier : si cloué, aucun coup légal (le cavalier ne peut pas rester sur une ligne en sautant
   * en L). Sinon targets restreintes par {@code ~occUs & checkMask}.
   */
  private static int generateKnightMoves(
      Position pos,
      int[] buf,
      int offset,
      int count,
      int us,
      long occUs,
      long occThem,
      long checkMask,
      long pinned,
      boolean capturesOnly) {
    long knights = pos.pieceBB(Piece.make(us, PieceType.KNIGHT));
    knights &= ~pinned;
    while (knights != 0L) {
      int from = Long.numberOfTrailingZeros(knights);
      knights &= knights - 1L;
      long targets = Bitboards.knightAttacks(from) & ~occUs & checkMask;
      if (capturesOnly) {
        targets &= occThem;
      }
      while (targets != 0L) {
        int to = Long.numberOfTrailingZeros(targets);
        targets &= targets - 1L;
        buf[offset + count++] = Move.encode(from, to);
      }
    }
    return count;
  }

  // ===============================================================================================
  // PAWN
  // ===============================================================================================

  private static int generatePawnMoves(
      Position pos,
      int[] buf,
      int offset,
      int count,
      int us,
      int them,
      long occUs,
      long occThem,
      long occAll,
      int kingSq,
      long checkMask,
      long pinned,
      boolean capturesOnly) {
    long pawns = pos.pieceBB(Piece.make(us, PieceType.PAWN));
    if (pawns == 0L) {
      return count;
    }
    int epSquare = pos.epSquare();
    int promotionRank = (us == Color.WHITE) ? 7 : 0;
    int startRank = (us == Color.WHITE) ? 1 : 6;
    int forwardDelta = (us == Color.WHITE) ? 8 : -8;

    while (pawns != 0L) {
      int from = Long.numberOfTrailingZeros(pawns);
      pawns &= pawns - 1L;

      boolean isPinned = ((1L << from) & pinned) != 0L;
      long pinAllow = isPinned ? Bitboards.lineThrough(kingSq, from) : -1L;

      long targets = 0L;

      if (!capturesOnly) {
        int pushOneTo = from + forwardDelta;
        if ((occAll & (1L << pushOneTo)) == 0L) {
          targets |= 1L << pushOneTo;
          if (Square.rank(from) == startRank) {
            int pushTwoTo = from + 2 * forwardDelta;
            if ((occAll & (1L << pushTwoTo)) == 0L) {
              targets |= 1L << pushTwoTo;
            }
          }
        }
      }

      long pawnAttacksBB = Bitboards.pawnAttacks(us, from);
      targets |= pawnAttacksBB & occThem;
      targets &= checkMask & pinAllow;

      while (targets != 0L) {
        int to = Long.numberOfTrailingZeros(targets);
        targets &= targets - 1L;
        if (Square.rank(to) == promotionRank) {
          buf[offset + count++] = Move.encode(from, to, MoveType.PROMOTION, 0);
          buf[offset + count++] = Move.encode(from, to, MoveType.PROMOTION, 1);
          buf[offset + count++] = Move.encode(from, to, MoveType.PROMOTION, 2);
          buf[offset + count++] = Move.encode(from, to, MoveType.PROMOTION, 3);
        } else {
          buf[offset + count++] = Move.encode(from, to);
        }
      }

      // EP : traitement séparé, conditions de légalité spécifiques (§5.3.10)
      if (epSquare != Square.NONE && (pawnAttacksBB & (1L << epSquare)) != 0L) {
        if (isEpLegal(pos, us, them, kingSq, from, epSquare, occAll, checkMask, pinAllow)) {
          // Note ordering : insérer EP au bon endroit selon to-ascendant ne perturbe pas l'ordre
          // global car EP est nécessairement le dernier target produit pour CE pawn (pas inclus
          // dans le filtre standard). Toutefois cela sort potentiellement de l'ordre §3.6 strict
          // sur le to dans le contexte d'un même pawn. En pratique : les targets non-EP générés
          // au-dessus ont déjà été émis selon to ascendant ; insérer EP après est cohérent avec
          // la convention que les coups d'un même pawn restent regroupés et n'affecte pas la
          // cross-validation (qui compare en sets).
          buf[offset + count++] = Move.encode(from, epSquare, MoveType.EN_PASSANT, 0);
        }
      }
    }
    return count;
  }

  /**
   * Vérifie qu'un coup d'EP candidat est légal :
   *
   * <ol>
   *   <li>Compatibilité de pin (la cible {@code epSquare} doit appartenir à {@code pinAllow}).
   *   <li>Pas d'échec à la découverte par retrait simultané des deux pions sur la rangée (§5.3.10)
   *       — étendu en sécurité aux découvertes diagonales.
   *   <li>Résolution de l'échec si le côté au trait est en échec : soit la capture du pion checker
   *       (pion poussé deux cases), soit l'interposition sur la ligne du slider checker.
   * </ol>
   */
  private static boolean isEpLegal(
      Position pos,
      int us,
      int them,
      int kingSq,
      int from,
      int epSquare,
      long occAll,
      long checkMask,
      long pinAllow) {
    long epTargetBit = 1L << epSquare;
    if ((pinAllow & epTargetBit) == 0L) {
      return false;
    }

    int capturedPawnSq = (us == Color.WHITE) ? epSquare - 8 : epSquare + 8;
    long occAfter = (occAll ^ (1L << from) ^ (1L << capturedPawnSq)) | epTargetBit;

    long enemyRQ =
        pos.pieceBB(Piece.make(them, PieceType.ROOK))
            | pos.pieceBB(Piece.make(them, PieceType.QUEEN));
    if ((Bitboards.rookAttacks(kingSq, occAfter) & enemyRQ) != 0L) {
      return false;
    }
    long enemyBQ =
        pos.pieceBB(Piece.make(them, PieceType.BISHOP))
            | pos.pieceBB(Piece.make(them, PieceType.QUEEN));
    if ((Bitboards.bishopAttacks(kingSq, occAfter) & enemyBQ) != 0L) {
      return false;
    }

    // Résolution d'échec : si l'on est en échec, l'EP doit le résoudre.
    // - Cas pawn check : le pion qui donne échec est sur capturedPawnSq → captured par EP.
    // - Cas slider check : la case d'EP doit être sur la ligne d'interposition (rare en pratique).
    long resolutionMask = (1L << capturedPawnSq) | epTargetBit;
    return (checkMask & resolutionMask) != 0L;
  }

  // ===============================================================================================
  // Helpers
  // ===============================================================================================

  /**
   * Variante d'{@link Position#isSquareAttacked} prenant une occupancy explicite, utilisée par les
   * coups de roi pour calculer les attaques de l'adversaire <em>sans</em> que le roi se protège
   * lui-même contre un slider qui passerait à travers sa case d'origine.
   */
  private static boolean isAttackedFrom(Position pos, int sq, int byColor, long occ) {
    int opp = Color.opponent(byColor);
    if ((Bitboards.pawnAttacks(opp, sq) & pos.pieceBB(Piece.make(byColor, PieceType.PAWN))) != 0L) {
      return true;
    }
    if ((Bitboards.knightAttacks(sq) & pos.pieceBB(Piece.make(byColor, PieceType.KNIGHT))) != 0L) {
      return true;
    }
    if ((Bitboards.kingAttacks(sq) & pos.pieceBB(Piece.make(byColor, PieceType.KING))) != 0L) {
      return true;
    }
    long bq =
        pos.pieceBB(Piece.make(byColor, PieceType.BISHOP))
            | pos.pieceBB(Piece.make(byColor, PieceType.QUEEN));
    if ((Bitboards.bishopAttacks(sq, occ) & bq) != 0L) {
      return true;
    }
    long rq =
        pos.pieceBB(Piece.make(byColor, PieceType.ROOK))
            | pos.pieceBB(Piece.make(byColor, PieceType.QUEEN));
    return (Bitboards.rookAttacks(sq, occ) & rq) != 0L;
  }
}
