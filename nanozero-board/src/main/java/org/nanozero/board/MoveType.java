package org.nanozero.board;

/**
 * Constantes de type de coup, encodées sur 2 bits dans le format {@link Move} (cf. SPEC §3.4,
 * ADR-004).
 *
 * <p>Convention figée :
 *
 * <ul>
 *   <li>{@code NORMAL = 0} : tout coup ne nécessitant pas de cas spécial (avance de pion, capture
 *       hors EP, déplacement de pièce, push de pion deux cases sans EP).
 *   <li>{@code PROMOTION = 1} : coup arrivant sur la 8e (blanc) ou 1ère (noir) rangée avec
 *       transformation du pion. Le champ {@code promo} de {@code Move} doit être renseigné.
 *   <li>{@code EN_PASSANT = 2} : prise en passant.
 *   <li>{@code CASTLING = 3} : roque (petit ou grand).
 * </ul>
 *
 * <p>Classe non instanciable, conforme à la convention §3.3 du SPEC.
 */
public final class MoveType {

  private MoveType() {
    throw new AssertionError("Non-instantiable");
  }

  public static final int NORMAL = 0;
  public static final int PROMOTION = 1;
  public static final int EN_PASSANT = 2;
  public static final int CASTLING = 3;

  /** Nombre de types distincts de coup. */
  public static final int NB_TYPES = 4;
}
