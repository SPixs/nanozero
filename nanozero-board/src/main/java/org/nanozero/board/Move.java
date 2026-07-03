package org.nanozero.board;

/**
 * Encodage et manipulation d'un coup d'échecs sur 16 bits dans un {@code int} (cf. SPEC §3.4,
 * ADR-004).
 *
 * <p>Format figé bit-par-bit :
 *
 * <pre>
 *   Bit:   15 14 | 13 12 | 11 10  9  8  7  6 |  5  4  3  2  1  0
 *          [TYPE]  [PROMO] [        TO       ] [        FROM     ]
 * </pre>
 *
 * <ul>
 *   <li>bits 0-5 ({@link #FROM_MASK}) : case de départ {@code 0..63}
 *   <li>bits 6-11 : case d'arrivée {@code 0..63}
 *   <li>bits 12-13 : pièce de promotion ({@code 0=KNIGHT, 1=BISHOP, 2=ROOK, 3=QUEEN}). Pertinent
 *       UNIQUEMENT si {@code type == PROMOTION} ; doit être à zéro pour les autres types.
 *   <li>bits 14-15 : type de coup ({@link MoveType#NORMAL}, {@link MoveType#PROMOTION}, {@link
 *       MoveType#EN_PASSANT}, {@link MoveType#CASTLING})
 * </ul>
 *
 * <p>Invariant : {@code (move >>> 16) == 0}. {@link #NULL} (= {@code 0}) est la sentinelle pour «
 * coup nul » ; aucun coup légal ne peut avoir {@code from == to == 0} (un roi en a1 ne peut
 * légalement pas bouger en a1), ce qui garantit la non-collision avec {@link #NULL}.
 *
 * <p>Classe non instanciable, conforme à la convention §3.3 du SPEC.
 */
public final class Move {

  private Move() {
    throw new AssertionError("Non-instantiable");
  }

  /** Coup nul, distinct de tout coup légal. Utilisé comme sentinelle (cf. ADR-004). */
  public static final int NULL = 0;

  /** Largeur en bits totale du format encodé. */
  public static final int BIT_WIDTH = 16;

  // Constantes internes : positions et masques de chaque champ dans le mot 16 bits.
  private static final int FROM_SHIFT = 0;
  private static final int TO_SHIFT = 6;
  private static final int PROMO_SHIFT = 12;
  private static final int TYPE_SHIFT = 14;

  private static final int FROM_MASK = 0x3F;
  private static final int TO_MASK = 0x3F;
  private static final int PROMO_MASK = 0x3;
  private static final int TYPE_MASK = 0x3;

  // Mapping bidirectionnel entre la valeur PROMO 2-bit (0..3) et le PieceType (KNIGHT..QUEEN).
  // L'encodage SPEC §3.4 stipule : 0=KNIGHT, 1=BISHOP, 2=ROOK, 3=QUEEN.
  /** Convertit une valeur de promotion 2-bit en index {@link PieceType}. */
  public static int promoToPieceType(int promo) {
    return promo + PieceType.KNIGHT;
  }

  /**
   * Convertit un index {@link PieceType} (KNIGHT..QUEEN) en valeur de promotion 2-bit.
   *
   * @throws IllegalArgumentException si {@code pieceType} n'est pas un type de promotion valide
   */
  public static int pieceTypeToPromo(int pieceType) {
    if (pieceType < PieceType.KNIGHT || pieceType > PieceType.QUEEN) {
      throw new IllegalArgumentException("Type de promotion invalide : " + pieceType);
    }
    return pieceType - PieceType.KNIGHT;
  }

  // -----------------------------------------------------------------------------------------
  // Encodage
  // -----------------------------------------------------------------------------------------

  /**
   * Encode un coup normal (sans promotion ni cas spécial). Équivalent à {@code encode(from, to,
   * MoveType.NORMAL, 0)}.
   *
   * @param from case de départ {@code 0..63}
   * @param to case d'arrivée {@code 0..63}
   * @return coup encodé, avec {@code (result >>> 16) == 0}
   */
  public static int encode(int from, int to) {
    return (from & FROM_MASK) | ((to & TO_MASK) << TO_SHIFT);
  }

  /**
   * Encode un coup avec type et pièce de promotion explicites.
   *
   * <p>Le champ {@code promo} est forcé à 0 si {@code type != PROMOTION}, conformément au SPEC §3.4
   * qui exige cet invariant pour la canonicité.
   *
   * @param from case de départ {@code 0..63}
   * @param to case d'arrivée {@code 0..63}
   * @param type un {@link MoveType}
   * @param promo valeur de promotion {@code 0..3} ({@code 0=N, 1=B, 2=R, 3=Q}) ; ignoré et forcé à
   *     0 si {@code type != MoveType.PROMOTION}
   * @return coup encodé, avec {@code (result >>> 16) == 0}
   */
  public static int encode(int from, int to, int type, int promo) {
    int promoBits = (type == MoveType.PROMOTION) ? (promo & PROMO_MASK) : 0;
    return (from & FROM_MASK)
        | ((to & TO_MASK) << TO_SHIFT)
        | (promoBits << PROMO_SHIFT)
        | ((type & TYPE_MASK) << TYPE_SHIFT);
  }

  // -----------------------------------------------------------------------------------------
  // Décodage
  // -----------------------------------------------------------------------------------------

  /** Retourne la case de départ encodée dans {@code move}. */
  public static int from(int move) {
    return (move >>> FROM_SHIFT) & FROM_MASK;
  }

  /** Retourne la case d'arrivée encodée dans {@code move}. */
  public static int to(int move) {
    return (move >>> TO_SHIFT) & TO_MASK;
  }

  /**
   * Retourne le type de coup encodé dans {@code move} ({@link MoveType#NORMAL} .. {@link
   * MoveType#CASTLING}).
   */
  public static int type(int move) {
    return (move >>> TYPE_SHIFT) & TYPE_MASK;
  }

  /**
   * Retourne la valeur de promotion encodée dans {@code move} ({@code 0..3}). N'a de sens que si
   * {@code type(move) == MoveType.PROMOTION}.
   */
  public static int promo(int move) {
    return (move >>> PROMO_SHIFT) & PROMO_MASK;
  }

  // -----------------------------------------------------------------------------------------
  // Sérialisation UCI
  // -----------------------------------------------------------------------------------------

  /**
   * Convertit un coup encodé en notation UCI.
   *
   * <p>Format de sortie :
   *
   * <ul>
   *   <li>Coup standard, capture, EP, roque : {@code "e2e4"}, {@code "e1g1"}, {@code "e5d6"} (4
   *       caractères, pas de marqueur spécial).
   *   <li>Promotion : {@code "e7e8q"} (5 caractères, lettre minuscule {@code n/b/r/q}).
   * </ul>
   *
   * @param move coup encodé
   * @return chaîne UCI (4 ou 5 caractères)
   */
  public static String toUci(int move) {
    int from = from(move);
    int to = to(move);
    int type = type(move);
    String fromAlg = Square.toAlgebraic(from);
    String toAlg = Square.toAlgebraic(to);
    if (type != MoveType.PROMOTION) {
      return fromAlg + toAlg;
    }
    int p = promo(move);
    char promoChar =
        switch (p) {
          case 0 -> 'n';
          case 1 -> 'b';
          case 2 -> 'r';
          case 3 -> 'q';
          default -> throw new IllegalStateException("Valeur de promotion invalide : " + p);
        };
    return fromAlg + toAlg + promoChar;
  }

  /**
   * Décode un coup UCI dans le contexte d'une position donnée.
   *
   * <p>Le contexte est nécessaire pour distinguer {@link MoveType#NORMAL}, {@link
   * MoveType#EN_PASSANT} et {@link MoveType#CASTLING}, qui partagent le même format 4 caractères en
   * UCI :
   *
   * <ul>
   *   <li>{@link MoveType#CASTLING} : la pièce sur {@code from} est un roi et la distance en
   *       fichier entre {@code from} et {@code to} est exactement de 2 cases.
   *   <li>{@link MoveType#EN_PASSANT} : la pièce sur {@code from} est un pion et {@code to} égale
   *       {@link Position#epSquare()}.
   *   <li>{@link MoveType#NORMAL} : tous les autres cas.
   * </ul>
   *
   * <p>Une chaîne de 5 caractères avec un suffixe valide ({@code n/b/r/q}) est traitée comme {@link
   * MoveType#PROMOTION} indépendamment du contexte.
   *
   * @param uci chaîne UCI à décoder ({@code "e2e4"} ou {@code "e7e8q"})
   * @param position position de référence pour résoudre le type
   * @return coup encodé selon le format {@link Move}
   * @throws IllegalArgumentException si {@code uci} est null, de longueur invalide, ou contient des
   *     caractères hors plages autorisées
   */
  public static int fromUci(String uci, Position position) {
    if (uci == null) {
      throw new IllegalArgumentException("UCI null");
    }
    int len = uci.length();
    if (len != 4 && len != 5) {
      throw new IllegalArgumentException("Longueur UCI invalide : " + uci);
    }
    int from = Square.fromAlgebraic(uci.substring(0, 2));
    int to = Square.fromAlgebraic(uci.substring(2, 4));
    if (len == 5) {
      char promoChar = uci.charAt(4);
      int promo =
          switch (promoChar) {
            case 'n' -> 0;
            case 'b' -> 1;
            case 'r' -> 2;
            case 'q' -> 3;
            default ->
                throw new IllegalArgumentException(
                    "Lettre de promotion UCI invalide : " + promoChar);
          };
      return encode(from, to, MoveType.PROMOTION, promo);
    }
    int piece = position.pieceAt(from);
    int pieceType = (piece == Piece.NONE) ? PieceType.NONE : Piece.typeOf(piece);
    int fromFile = Square.file(from);
    int toFile = Square.file(to);
    int type;
    if (pieceType == PieceType.KING && Math.abs(fromFile - toFile) == 2) {
      type = MoveType.CASTLING;
    } else if (pieceType == PieceType.PAWN && to == position.epSquare()) {
      type = MoveType.EN_PASSANT;
    } else {
      type = MoveType.NORMAL;
    }
    return encode(from, to, type, 0);
  }
}
