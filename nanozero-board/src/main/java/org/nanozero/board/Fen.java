package org.nanozero.board;

/**
 * Parser et writer du format FEN (Forsyth-Edwards Notation), conforme à SPEC §5.7.
 *
 * <p>Le format FEN supporté est la variante standard à 6 champs séparés par des espaces simples
 * (cf. §5.7.1). Le module ne supporte PAS Chess960/Shredder-FEN/X-FEN (cf. ADR-010).
 *
 * <p>L'opération de parsing applique les 10 règles de validation de §5.7.2 et lève {@link
 * IllegalArgumentException} avec un message explicite en cas de violation.
 *
 * <p>Round-trip garanti : pour toute {@link Position} {@code p} construite via {@code
 * Fen.parse(s)}, {@code Fen.parse(Fen.write(p))} reproduit une position bit-identique sur tous les
 * bitboards et tous les compteurs (le hash Zobrist est traité séparément, cf. phase 6).
 *
 * <p>Classe non instanciable, conforme à la convention §3.3 du SPEC.
 */
public final class Fen {

  private Fen() {
    throw new AssertionError("Non-instantiable");
  }

  /** FEN de la position de départ standard. */
  public static final String STARTPOS = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

  // ---------------------------------------------------------------------------------------------
  // API publique
  // ---------------------------------------------------------------------------------------------

  /**
   * Parse une FEN et remplit la {@link Position} fournie.
   *
   * <p>L'état préexistant de {@code out} est intégralement réécrit : tous les bitboards sont remis
   * à zéro avant le placement des pièces, et tous les compteurs / drapeaux sont assignés aux
   * valeurs décodées de la FEN. Le hash Zobrist est forcé à 0 (le calcul incrémental sera
   * implémenté en phase 6 ; aucun consommateur ne doit dépendre de la valeur du hash en phase 3).
   *
   * @param fen chaîne FEN à parser
   * @param out instance {@link Position} cible
   * @throws IllegalArgumentException si la FEN est mal formée ou viole l'une des 10 règles de
   *     validation §5.7.2 (structure, comptage des rois, pions sur rangs interdits, side, format et
   *     cohérence des droits de roque, EP, halfmove, fullmove, échec illégal du non-au-trait)
   */
  public static void parse(String fen, Position out) {
    if (fen == null) {
      throw new IllegalArgumentException("FEN null");
    }
    String[] parts = fen.split(" ", -1);
    if (parts.length != 6) {
      throw new IllegalArgumentException(
          "FEN doit avoir 6 champs séparés par des espaces, trouvé " + parts.length + " : " + fen);
    }
    resetPosition(out);
    parsePiecePlacement(parts[0], out);
    validateKingsCount(out);
    validateNoPawnsOnEdgeRanks(out);
    out.sideToMove = parseSide(parts[1]);
    out.castlingRights = parseCastlingRights(parts[2], out);
    out.epSquare = parseEpSquare(parts[3], out);
    out.halfmoveClock = parseHalfmoveClock(parts[4]);
    out.fullmoveNumber = parseFullmoveNumber(parts[5]);
    validateNonSideToMoveNotInCheck(out);
    out.zobristHash = Zobrist.computeFull(out);
  }

  /**
   * Variante allouante : construit une nouvelle {@link Position} à partir d'une FEN.
   *
   * @param fen chaîne FEN à parser
   * @return nouvelle position
   * @throws IllegalArgumentException cf. {@link #parse(String, Position)}
   */
  public static Position parse(String fen) {
    Position p = new Position();
    parse(fen, p);
    return p;
  }

  /**
   * Sérialise une {@link Position} en FEN canonique (cf. §5.7.3).
   *
   * <p>Conventions d'écriture : droits de roque dans l'ordre {@code KQkq} (omission des absents,
   * {@code -} si tous absents), EP square en algébrique ou {@code -}, pas de zéros en tête sur les
   * compteurs, espace simple entre champs, pas d'espace de fin.
   *
   * @param position position à sérialiser
   * @return FEN canonique
   */
  public static String write(Position position) {
    StringBuilder sb = new StringBuilder(80);
    writePiecePlacement(position, sb);
    sb.append(' ');
    sb.append(position.sideToMove == Color.WHITE ? 'w' : 'b');
    sb.append(' ');
    writeCastlingRights(position.castlingRights, sb);
    sb.append(' ');
    if (position.epSquare == Square.NONE) {
      sb.append('-');
    } else {
      sb.append(Square.toAlgebraic(position.epSquare));
    }
    sb.append(' ');
    sb.append(position.halfmoveClock);
    sb.append(' ');
    sb.append(position.fullmoveNumber);
    return sb.toString();
  }

  // ---------------------------------------------------------------------------------------------
  // Helpers de write
  // ---------------------------------------------------------------------------------------------

  private static void writePiecePlacement(Position p, StringBuilder sb) {
    for (int rank = 7; rank >= 0; rank--) {
      int empty = 0;
      for (int file = 0; file < 8; file++) {
        int sq = rank * 8 + file;
        int piece = p.pieceAt(sq);
        if (piece == Piece.NONE) {
          empty++;
        } else {
          if (empty > 0) {
            sb.append((char) ('0' + empty));
            empty = 0;
          }
          sb.append(PieceType.toChar(Piece.typeOf(piece), Piece.colorOf(piece)));
        }
      }
      if (empty > 0) {
        sb.append((char) ('0' + empty));
      }
      if (rank > 0) {
        sb.append('/');
      }
    }
  }

  private static void writeCastlingRights(int rights, StringBuilder sb) {
    if (rights == Castling.NONE) {
      sb.append('-');
      return;
    }
    if ((rights & Castling.WHITE_KINGSIDE) != 0) {
      sb.append('K');
    }
    if ((rights & Castling.WHITE_QUEENSIDE) != 0) {
      sb.append('Q');
    }
    if ((rights & Castling.BLACK_KINGSIDE) != 0) {
      sb.append('k');
    }
    if ((rights & Castling.BLACK_QUEENSIDE) != 0) {
      sb.append('q');
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Helpers de parse — règle 1 : structure
  // ---------------------------------------------------------------------------------------------

  private static void resetPosition(Position out) {
    for (int i = 0; i < Piece.NB_PIECES; i++) {
      out.pieceBB[i] = 0L;
    }
    out.occupancyBB[Color.WHITE] = 0L;
    out.occupancyBB[Color.BLACK] = 0L;
    out.occupancyBB[2] = 0L;
    out.sideToMove = Color.WHITE;
    out.castlingRights = Castling.NONE;
    out.epSquare = Square.NONE;
    out.halfmoveClock = 0;
    out.fullmoveNumber = 1;
    out.zobristHash = 0L;
  }

  // ---------------------------------------------------------------------------------------------
  // Règle 2 : position
  // ---------------------------------------------------------------------------------------------

  private static void parsePiecePlacement(String posField, Position out) {
    String[] ranks = posField.split("/", -1);
    if (ranks.length != 8) {
      throw new IllegalArgumentException(
          "Le champ position doit avoir 8 rangs, trouvé " + ranks.length + " : " + posField);
    }
    for (int rIdx = 0; rIdx < 8; rIdx++) {
      // ranks[0] est le rang 8 (haut), ranks[7] le rang 1 (bas).
      int rank = 7 - rIdx;
      String rankStr = ranks[rIdx];
      int file = 0;
      for (int i = 0; i < rankStr.length(); i++) {
        char c = rankStr.charAt(i);
        if (c >= '1' && c <= '8') {
          file += c - '0';
        } else {
          int piece = pieceFromChar(c, rankStr);
          if (file >= 8) {
            throw new IllegalArgumentException(
                "Rang " + (rank + 1) + " déborde 8 cases : " + rankStr);
          }
          int sq = rank * 8 + file;
          out.pieceBB[piece] |= 1L << sq;
          file++;
        }
      }
      if (file != 8) {
        throw new IllegalArgumentException(
            "Rang " + (rank + 1) + " ne totalise pas 8 cases (vu " + file + ") : " + rankStr);
      }
    }
    rebuildOccupancy(out);
  }

  private static int pieceFromChar(char c, String context) {
    return switch (c) {
      case 'P' -> Piece.WHITE_PAWN;
      case 'N' -> Piece.WHITE_KNIGHT;
      case 'B' -> Piece.WHITE_BISHOP;
      case 'R' -> Piece.WHITE_ROOK;
      case 'Q' -> Piece.WHITE_QUEEN;
      case 'K' -> Piece.WHITE_KING;
      case 'p' -> Piece.BLACK_PAWN;
      case 'n' -> Piece.BLACK_KNIGHT;
      case 'b' -> Piece.BLACK_BISHOP;
      case 'r' -> Piece.BLACK_ROOK;
      case 'q' -> Piece.BLACK_QUEEN;
      case 'k' -> Piece.BLACK_KING;
      default ->
          throw new IllegalArgumentException(
              "Caractère de pièce invalide '" + c + "' dans : " + context);
    };
  }

  private static void rebuildOccupancy(Position out) {
    long w = 0L;
    long b = 0L;
    for (int i = 0; i < 6; i++) {
      w |= out.pieceBB[i];
      b |= out.pieceBB[6 + i];
    }
    out.occupancyBB[Color.WHITE] = w;
    out.occupancyBB[Color.BLACK] = b;
    out.occupancyBB[2] = w | b;
  }

  // ---------------------------------------------------------------------------------------------
  // Règle 3 : comptage des rois
  // ---------------------------------------------------------------------------------------------

  private static void validateKingsCount(Position out) {
    int whiteKings = Long.bitCount(out.pieceBB[Piece.WHITE_KING]);
    int blackKings = Long.bitCount(out.pieceBB[Piece.BLACK_KING]);
    if (whiteKings != 1) {
      throw new IllegalArgumentException(
          "FEN invalide : " + whiteKings + " rois blancs au lieu de 1");
    }
    if (blackKings != 1) {
      throw new IllegalArgumentException(
          "FEN invalide : " + blackKings + " rois noirs au lieu de 1");
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Règle 4 : pions sur rangs interdits
  // ---------------------------------------------------------------------------------------------

  private static void validateNoPawnsOnEdgeRanks(Position out) {
    long pawns = out.pieceBB[Piece.WHITE_PAWN] | out.pieceBB[Piece.BLACK_PAWN];
    long edgeRanks = Bitboards.rankBB(0) | Bitboards.rankBB(7);
    if ((pawns & edgeRanks) != 0L) {
      throw new IllegalArgumentException(
          "FEN invalide : pion sur rang 1 ou 8 (interdit par les règles d'échecs)");
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Règle 5 : side
  // ---------------------------------------------------------------------------------------------

  private static int parseSide(String s) {
    if (s.length() != 1) {
      throw new IllegalArgumentException("Side invalide (1 caractère attendu) : " + s);
    }
    char c = s.charAt(0);
    if (c == 'w') {
      return Color.WHITE;
    }
    if (c == 'b') {
      return Color.BLACK;
    }
    throw new IllegalArgumentException("Side invalide ('w' ou 'b' attendu) : " + s);
  }

  // ---------------------------------------------------------------------------------------------
  // Règle 6 : castling rights — format ordonné + cohérence positions roi/tour
  // ---------------------------------------------------------------------------------------------

  private static int parseCastlingRights(String s, Position out) {
    if (s.equals("-")) {
      return Castling.NONE;
    }
    if (s.isEmpty()) {
      throw new IllegalArgumentException("Castling rights vide (utiliser '-' pour aucun droit)");
    }
    int rights = 0;
    int idx = 0;
    if (idx < s.length() && s.charAt(idx) == 'K') {
      rights |= Castling.WHITE_KINGSIDE;
      idx++;
    }
    if (idx < s.length() && s.charAt(idx) == 'Q') {
      rights |= Castling.WHITE_QUEENSIDE;
      idx++;
    }
    if (idx < s.length() && s.charAt(idx) == 'k') {
      rights |= Castling.BLACK_KINGSIDE;
      idx++;
    }
    if (idx < s.length() && s.charAt(idx) == 'q') {
      rights |= Castling.BLACK_QUEENSIDE;
      idx++;
    }
    if (idx != s.length()) {
      throw new IllegalArgumentException(
          "Castling rights invalide (sous-ensemble ordonné de 'KQkq' ou '-' attendu) : " + s);
    }
    validateCastlingConsistency(rights, out);
    return rights;
  }

  private static void validateCastlingConsistency(int rights, Position out) {
    if ((rights & Castling.WHITE_KINGSIDE) != 0) {
      requirePieceAt(out, Piece.WHITE_KING, Square.E1, "WK castling sans roi blanc en E1");
      requirePieceAt(out, Piece.WHITE_ROOK, Square.H1, "WK castling sans tour blanche en H1");
    }
    if ((rights & Castling.WHITE_QUEENSIDE) != 0) {
      requirePieceAt(out, Piece.WHITE_KING, Square.E1, "WQ castling sans roi blanc en E1");
      requirePieceAt(out, Piece.WHITE_ROOK, Square.A1, "WQ castling sans tour blanche en A1");
    }
    if ((rights & Castling.BLACK_KINGSIDE) != 0) {
      requirePieceAt(out, Piece.BLACK_KING, Square.E8, "BK castling sans roi noir en E8");
      requirePieceAt(out, Piece.BLACK_ROOK, Square.H8, "BK castling sans tour noire en H8");
    }
    if ((rights & Castling.BLACK_QUEENSIDE) != 0) {
      requirePieceAt(out, Piece.BLACK_KING, Square.E8, "BQ castling sans roi noir en E8");
      requirePieceAt(out, Piece.BLACK_ROOK, Square.A8, "BQ castling sans tour noire en A8");
    }
  }

  private static void requirePieceAt(Position out, int piece, int square, String message) {
    if ((out.pieceBB[piece] & (1L << square)) == 0L) {
      throw new IllegalArgumentException(message);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Règle 7 : EP square — format + rang correct + pion adverse adjacent
  // ---------------------------------------------------------------------------------------------

  private static int parseEpSquare(String s, Position out) {
    if (s.equals("-")) {
      return Square.NONE;
    }
    int ep;
    try {
      ep = Square.fromAlgebraic(s);
    } catch (IllegalArgumentException e) {
      throw new IllegalArgumentException("EP square invalide : " + s, e);
    }
    int rank = Square.rank(ep);
    // Si BLACK to move, c'est WHITE qui a poussé : EP sur rang 3 (rank index 2).
    // Si WHITE to move, c'est BLACK qui a poussé : EP sur rang 6 (rank index 5).
    int expectedRank = (out.sideToMove == Color.WHITE) ? 5 : 2;
    if (rank != expectedRank) {
      throw new IllegalArgumentException(
          "EP square sur rang invalide pour side="
              + (out.sideToMove == Color.WHITE ? "w" : "b")
              + " : "
              + s);
    }
    int adjacentRank = (out.sideToMove == Color.WHITE) ? 4 : 3;
    int adjacentSq = Square.make(Square.file(ep), adjacentRank);
    int expectedPawn = (out.sideToMove == Color.WHITE) ? Piece.BLACK_PAWN : Piece.WHITE_PAWN;
    if ((out.pieceBB[expectedPawn] & (1L << adjacentSq)) == 0L) {
      throw new IllegalArgumentException(
          "EP square " + s + " sans pion adverse adjacent en " + Square.toAlgebraic(adjacentSq));
    }
    return ep;
  }

  // ---------------------------------------------------------------------------------------------
  // Règles 8 & 9 : halfmove clock & fullmove number
  // ---------------------------------------------------------------------------------------------

  private static int parseHalfmoveClock(String s) {
    int hm = parseNonNegativeInt(s, "halfmove clock");
    if (hm > 100) {
      throw new IllegalArgumentException("halfmove clock > 100 (règle 50 coups dépassée) : " + s);
    }
    return hm;
  }

  private static int parseFullmoveNumber(String s) {
    int fm = parseNonNegativeInt(s, "fullmove number");
    if (fm < 1) {
      throw new IllegalArgumentException("fullmove number < 1 : " + s);
    }
    return fm;
  }

  private static int parseNonNegativeInt(String s, String fieldName) {
    if (s.isEmpty()) {
      throw new IllegalArgumentException(fieldName + " vide");
    }
    int v;
    try {
      v = Integer.parseInt(s);
    } catch (NumberFormatException e) {
      throw new IllegalArgumentException(fieldName + " non-entier : " + s, e);
    }
    if (v < 0) {
      throw new IllegalArgumentException(fieldName + " négatif : " + s);
    }
    return v;
  }

  // ---------------------------------------------------------------------------------------------
  // Règle 10 : le côté NON au trait ne doit pas être en échec
  // ---------------------------------------------------------------------------------------------

  private static void validateNonSideToMoveNotInCheck(Position out) {
    int notSide = Color.opponent(out.sideToMove);
    long notSideKing = out.pieceBB[Piece.make(notSide, PieceType.KING)];
    int notSideKingSq = Long.numberOfTrailingZeros(notSideKing);
    if (out.isSquareAttacked(notSideKingSq, out.sideToMove)) {
      throw new IllegalArgumentException(
          "Position illégale : le côté NON au trait ("
              + (notSide == Color.WHITE ? "WHITE" : "BLACK")
              + ") est en échec — c'est à lui de jouer, pas à "
              + (out.sideToMove == Color.WHITE ? "WHITE" : "BLACK"));
    }
  }
}
