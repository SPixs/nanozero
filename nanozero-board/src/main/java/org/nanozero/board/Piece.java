package org.nanozero.board;

/**
 * Constantes et utilitaires de manipulation des pièces (couleur + type combinés).
 *
 * <p>Indexation figée : {@code piece = color * 6 + pieceType}, soit indices {@code 0..5} pour les
 * pièces blanches et {@code 6..11} pour les noires (cf. SPEC §3.2, ADR-009). Ce layout favorise
 * l'extraction de plans NN AlphaZero par tranches contiguës de 6 (un côté à la fois) et l'itération
 * par couleur.
 *
 * <p>La constante {@link #NONE} ({@code 12}) sert de sentinelle pour « aucune pièce » dans les
 * contextes où une valeur primitive doit représenter l'absence de pièce sur une case (par exemple
 * {@code Position.pieceAt(square)}).
 *
 * <p>Classe non instanciable, conforme à la convention §3.3 du SPEC.
 */
public final class Piece {

  private Piece() {
    throw new AssertionError("Non-instantiable");
  }

  public static final int WHITE_PAWN = 0;
  public static final int WHITE_KNIGHT = 1;
  public static final int WHITE_BISHOP = 2;
  public static final int WHITE_ROOK = 3;
  public static final int WHITE_QUEEN = 4;
  public static final int WHITE_KING = 5;

  public static final int BLACK_PAWN = 6;
  public static final int BLACK_KNIGHT = 7;
  public static final int BLACK_BISHOP = 8;
  public static final int BLACK_ROOK = 9;
  public static final int BLACK_QUEEN = 10;
  public static final int BLACK_KING = 11;

  /** Nombre total de pièces distinctes (6 types × 2 couleurs). */
  public static final int NB_PIECES = 12;

  /** Sentinelle pour « aucune pièce ». */
  public static final int NONE = 12;

  /**
   * Retourne la couleur d'une pièce ({@link Color#WHITE} ou {@link Color#BLACK}).
   *
   * @param piece index dans {@code [0, 11]}
   * @return la couleur de la pièce
   */
  public static int colorOf(int piece) {
    return piece / 6;
  }

  /**
   * Retourne le type d'une pièce (de {@link PieceType#PAWN} à {@link PieceType#KING}).
   *
   * @param piece index dans {@code [0, 11]}
   * @return le type de pièce
   */
  public static int typeOf(int piece) {
    return piece % 6;
  }

  /**
   * Construit l'index de pièce à partir d'une couleur et d'un type.
   *
   * @param color {@link Color#WHITE} ou {@link Color#BLACK}
   * @param pieceType {@link PieceType#PAWN} à {@link PieceType#KING}
   * @return index de pièce dans {@code [0, 11]}
   */
  public static int make(int color, int pieceType) {
    return color * 6 + pieceType;
  }
}
