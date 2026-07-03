package org.nanozero.board;

/**
 * Constantes et utilitaires de manipulation des couleurs.
 *
 * <p>Convention figée : {@code WHITE = 0}, {@code BLACK = 1} (cf. SPEC §3.3, ADR-007). Cette
 * convention permet la formule {@code opponent(c) = c ^ 1} et l'indexation des bitboards de pièces
 * {@code pieceBB[color * 6 + pieceType]} (cf. ADR-009).
 *
 * <p>Classe non instanciable, conforme à la convention §3.3 du SPEC.
 */
public final class Color {

  private Color() {
    throw new AssertionError("Non-instantiable");
  }

  /** Blanc (côté qui joue le premier coup). */
  public static final int WHITE = 0;

  /** Noir. */
  public static final int BLACK = 1;

  /** Nombre de couleurs (constante de cardinalité, utile pour les bornes de boucle). */
  public static final int NB_COLORS = 2;

  /**
   * Retourne la couleur adverse via XOR sur le bit 0.
   *
   * @param color {@link #WHITE} ou {@link #BLACK}
   * @return la couleur adverse
   */
  public static int opponent(int color) {
    return color ^ 1;
  }
}
