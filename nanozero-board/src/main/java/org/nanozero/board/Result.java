package org.nanozero.board;

/**
 * Résultat d'une partie d'échecs. Unique enum public du module ; n'apparaît jamais en hot path
 * (uniquement à la fin d'une partie ou pour interroger {@link GameState#getResult()}).
 *
 * <p>Conforme à SPEC §4.2.2 et §6.5.
 */
public enum Result {
  /** La partie est encore en cours. */
  IN_PROGRESS,

  /** Le côté blanc a gagné (mat infligé aux noirs). */
  WIN_WHITE,

  /** Le côté noir a gagné (mat infligé aux blancs). */
  WIN_BLACK,

  /** Partie nulle (pat, répétition, 50 coups, matériel insuffisant). */
  DRAW
}
