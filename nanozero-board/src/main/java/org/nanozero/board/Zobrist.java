package org.nanozero.board;

/**
 * Hashing Zobrist incrémental d'une position d'échecs (cf. SPEC §5.6, ADR-011).
 *
 * <p>Les 781 constantes sont générées au static init via le PRNG xorshift64* (Marsaglia 2003,
 * variante « starred »), seedé à {@value #SEED} et utilisant le multiplicateur final {@value
 * #XORSHIFT64_MULT}. L'ordre de génération est figé conformément à §5.6.2 :
 *
 * <ol>
 *   <li>Indices 0–767 : {@code pieceSquare[piece * 64 + square]} pour piece 0..11 et square 0..63.
 *   <li>Indices 768–771 : {@code castling[bit]} pour les 4 droits individuels (WK, WQ, BK, BQ).
 *   <li>Indices 772–779 : {@code enPassantFile[file]} pour les fichiers a..h (0..7).
 *   <li>Indice 780 : {@code sideBlack} (XORée si le côté au trait est noir).
 * </ol>
 *
 * <p>L'ordre de génération ET la seed sont figés normativement : toute modification invalide tous
 * les hashs précédemment calculés (incl. replay buffers d'entraînement). À traiter comme une
 * constante immuable du projet.
 *
 * <p>Le format n'est PAS compatible Polyglot (cf. ADR-011). NanoZero étant un moteur tabula rasa,
 * la compatibilité Polyglot n'apporte aucune valeur fonctionnelle.
 *
 * <p>Classe non instanciable, conforme à la convention §3.3 du SPEC.
 */
public final class Zobrist {

  /** Seed du PRNG xorshift64* (or doré, valeur arbitraire mais figée). */
  public static final long SEED = 0x9E3779B97F4A7C15L;

  /** Nombre total de constantes Zobrist : 768 + 4 + 8 + 1 = 781. */
  public static final int NB_CONSTANTS = 781;

  /** Multiplicateur final du xorshift64* (cf. SPEC §5.6.1). */
  private static final long XORSHIFT64_MULT = 0x2545F4914F6CDD1DL;

  // ---------------------------------------------------------------------------------------------
  // Tables statiques
  // ---------------------------------------------------------------------------------------------

  /** {@code PIECE_SQUARE[piece][square]} : 12 × 64 constantes. */
  private static final long[][] PIECE_SQUARE = new long[Piece.NB_PIECES][Square.NB_SQUARES];

  /** {@code CASTLING[bit_index]} : 4 constantes, indexées par {@code log2(bit_value)}. */
  private static final long[] CASTLING = new long[4];

  /** {@code EN_PASSANT_FILE[file]} : 8 constantes (a..h). */
  private static final long[] EN_PASSANT_FILE = new long[8];

  /** Constante XORée si le côté au trait est noir. */
  private static final long SIDE_BLACK_CONST;

  // ---------------------------------------------------------------------------------------------
  // Initialisation statique : génération séquentielle des 781 constantes
  // ---------------------------------------------------------------------------------------------

  static {
    long state = SEED;
    for (int piece = 0; piece < Piece.NB_PIECES; piece++) {
      for (int square = 0; square < Square.NB_SQUARES; square++) {
        state = xorshift64Star(state);
        PIECE_SQUARE[piece][square] = state;
      }
    }
    for (int i = 0; i < 4; i++) {
      state = xorshift64Star(state);
      CASTLING[i] = state;
    }
    for (int file = 0; file < 8; file++) {
      state = xorshift64Star(state);
      EN_PASSANT_FILE[file] = state;
    }
    state = xorshift64Star(state);
    SIDE_BLACK_CONST = state;
  }

  private Zobrist() {
    throw new AssertionError("Non-instantiable");
  }

  /**
   * Force le chargement et l'initialisation de la table Zobrist (déclenche le {@code <clinit>}).
   * Permet d'amortir le coût de génération des 781 constantes avant un benchmark.
   */
  public static void warmup() {
    // Le simple fait de référencer un membre statique provoque le chargement et l'initialisation
    // de la classe par la JVM si cela n'a pas déjà eu lieu.
  }

  // ---------------------------------------------------------------------------------------------
  // API publique
  // ---------------------------------------------------------------------------------------------

  /**
   * Constante Zobrist pour une pièce sur une case.
   *
   * @param piece index pièce {@code 0..11} (cf. {@link Piece})
   * @param square case {@code 0..63}
   * @return constante 64-bit
   */
  public static long pieceSquare(int piece, int square) {
    return PIECE_SQUARE[piece][square];
  }

  /**
   * Constante Zobrist pour un droit de roque individuel.
   *
   * @param castlingBit valeur de bit (par exemple {@link Castling#WHITE_KINGSIDE} = {@code 0x1}),
   *     PAS un index ; convertie en index via {@link Integer#numberOfTrailingZeros(int)}.
   * @return constante 64-bit
   */
  public static long castling(int castlingBit) {
    return CASTLING[Integer.numberOfTrailingZeros(castlingBit)];
  }

  /**
   * Constante Zobrist pour un fichier d'EP (a..h, file 0..7).
   *
   * @param file fichier {@code 0..7}
   * @return constante 64-bit
   */
  public static long enPassantFile(int file) {
    return EN_PASSANT_FILE[file];
  }

  /** Constante Zobrist pour le côté noir au trait (XORée si BLACK to move). */
  public static long sideBlack() {
    return SIDE_BLACK_CONST;
  }

  /**
   * Recalcule intégralement le hash d'une position depuis ses bitboards.
   *
   * <p>Utilisé pour validation d'invariant {@code I-Pos-5 : zobristHash == Zobrist.computeFull(p)}
   * après chaque {@code applyMove}, et pour initialiser le hash dans {@link Fen#parse}. Coûteux
   * (parcourt les 12 bitboards bit-par-bit) ; ne PAS appeler en hot path.
   *
   * @param position position source
   * @return hash 64-bit
   */
  public static long computeFull(Position position) {
    long h = 0L;
    for (int piece = 0; piece < Piece.NB_PIECES; piece++) {
      long bb = position.pieceBB(piece);
      while (bb != 0L) {
        int sq = Long.numberOfTrailingZeros(bb);
        bb &= bb - 1L;
        h ^= PIECE_SQUARE[piece][sq];
      }
    }
    int rights = position.castlingRights();
    while (rights != 0) {
      int bit = rights & -rights;
      rights &= rights - 1;
      h ^= CASTLING[Integer.numberOfTrailingZeros(bit)];
    }
    int ep = position.epSquare();
    if (ep != Square.NONE) {
      h ^= EN_PASSANT_FILE[Square.file(ep)];
    }
    if (position.sideToMove() == Color.BLACK) {
      h ^= SIDE_BLACK_CONST;
    }
    return h;
  }

  // ---------------------------------------------------------------------------------------------
  // PRNG interne
  // ---------------------------------------------------------------------------------------------

  /**
   * Itération xorshift64* conforme à SPEC §5.6.1 : applique la transformation xorshift sur {@code
   * state}, multiplie le résultat par {@link #XORSHIFT64_MULT} et retourne la nouvelle valeur (qui
   * devient le nouvel état pour l'appel suivant).
   */
  private static long xorshift64Star(long state) {
    long s = state;
    s ^= s << 13;
    s ^= s >>> 7;
    s ^= s << 17;
    return s * XORSHIFT64_MULT;
  }
}
