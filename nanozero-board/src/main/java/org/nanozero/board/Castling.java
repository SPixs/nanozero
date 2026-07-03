package org.nanozero.board;

/**
 * Constantes et utilitaires de manipulation des droits de roque (cf. SPEC §3.5).
 *
 * <p>Encodage figé sur 4 bits :
 *
 * <pre>
 *   Bit:  3  2  1  0
 *        [BQ][BK][WQ][WK]
 * </pre>
 *
 * <p>{@code WK} = petit roque blanc (kingside), {@code WQ} = grand roque blanc (queenside), {@code
 * BK} = petit roque noir, {@code BQ} = grand roque noir. Les agrégats {@link #WHITE_BOTH}, {@link
 * #BLACK_BOTH}, {@link #ALL} et {@link #NONE} permettent les manipulations de bitmask usuelles.
 *
 * <p>Chess960 / Fischer Random NE SONT PAS supportés (cf. ADR-010), c'est pourquoi les droits
 * tiennent sur 4 bits seulement (pas d'encodage par fichier de tour).
 *
 * <p>Classe non instanciable, conforme à la convention §3.3 du SPEC.
 */
public final class Castling {

  private Castling() {
    throw new AssertionError("Non-instantiable");
  }

  public static final int WHITE_KINGSIDE = 0x1;
  public static final int WHITE_QUEENSIDE = 0x2;
  public static final int BLACK_KINGSIDE = 0x4;
  public static final int BLACK_QUEENSIDE = 0x8;

  public static final int WHITE_BOTH = WHITE_KINGSIDE | WHITE_QUEENSIDE;
  public static final int BLACK_BOTH = BLACK_KINGSIDE | BLACK_QUEENSIDE;

  /** Tous les droits de roque actifs (équivaut à {@code 0xF}). */
  public static final int ALL = WHITE_BOTH | BLACK_BOTH;

  /** Aucun droit de roque actif. */
  public static final int NONE = 0x0;

  /**
   * Indique si un droit individuel est actif dans un bitmask.
   *
   * @param rights bitmask sur 4 bits
   * @param right valeur de bit (par exemple {@link #WHITE_KINGSIDE}, pas un index)
   * @return {@code true} si le bit est actif dans {@code rights}
   */
  public static boolean has(int rights, int right) {
    return (rights & right) != 0;
  }

  /**
   * Retire un droit individuel d'un bitmask et retourne le résultat.
   *
   * @param rights bitmask source
   * @param right valeur de bit à retirer
   * @return nouveau bitmask sans le droit
   */
  public static int remove(int rights, int right) {
    return rights & ~right;
  }
}
