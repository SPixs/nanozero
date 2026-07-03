package org.nanozero.board;

/**
 * Constantes et utilitaires de manipulation des types de pièces.
 *
 * <p>Convention figée : {@code PAWN = 0}, {@code KNIGHT = 1}, {@code BISHOP = 2}, {@code ROOK = 3},
 * {@code QUEEN = 4}, {@code KING = 5} (cf. SPEC §3.3, ADR-007). La constante {@link #NONE} ({@code
 * 6}) sert de sentinelle pour « aucun type de pièce » dans les contextes où une valeur primitive
 * doit représenter l'absence de pièce.
 *
 * <p>Classe non instanciable, conforme à la convention §3.3 du SPEC.
 */
public final class PieceType {

  private PieceType() {
    throw new AssertionError("Non-instantiable");
  }

  public static final int PAWN = 0;
  public static final int KNIGHT = 1;
  public static final int BISHOP = 2;
  public static final int ROOK = 3;
  public static final int QUEEN = 4;
  public static final int KING = 5;

  /** Nombre de types de pièces distincts. */
  public static final int NB_PIECE_TYPES = 6;

  /** Sentinelle pour « aucun type de pièce ». */
  public static final int NONE = 6;

  /**
   * Convertit un type de pièce et une couleur en notation algébrique mono-caractère.
   *
   * <p>Convention : majuscules pour les blancs ({@code P, N, B, R, Q, K}), minuscules pour les
   * noirs ({@code p, n, b, r, q, k}). Ce mapping est utilisé par la sérialisation FEN (§5.7) et
   * l'affichage ASCII de l'échiquier (§13.2).
   *
   * @param pieceType {@link #PAWN} à {@link #KING}
   * @param color {@link Color#WHITE} ou {@link Color#BLACK}
   * @return le caractère identifiant la pièce
   * @throws IllegalArgumentException si {@code pieceType} est hors de {@code [0, 5]} ou {@code
   *     color} hors {@code [0, 1]}
   */
  public static char toChar(int pieceType, int color) {
    char base;
    switch (pieceType) {
      case PAWN -> base = 'p';
      case KNIGHT -> base = 'n';
      case BISHOP -> base = 'b';
      case ROOK -> base = 'r';
      case QUEEN -> base = 'q';
      case KING -> base = 'k';
      default -> throw new IllegalArgumentException("PieceType invalide : " + pieceType);
    }
    if (color == Color.WHITE) {
      return Character.toUpperCase(base);
    }
    if (color == Color.BLACK) {
      return base;
    }
    throw new IllegalArgumentException("Color invalide : " + color);
  }
}
