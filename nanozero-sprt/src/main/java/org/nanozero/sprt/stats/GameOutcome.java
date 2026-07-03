package org.nanozero.sprt.stats;

/**
 * Résultat d'une game du point de vue du challenger.
 *
 * <p>WIN = challenger gagne, LOSS = baseline gagne, DRAW = nul.
 *
 * <p>Cf. SPEC-sprt §4.3.
 */
public enum GameOutcome {
  WIN,
  LOSS,
  DRAW
}
