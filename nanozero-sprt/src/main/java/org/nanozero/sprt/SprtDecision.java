package org.nanozero.sprt;

/**
 * Décision finale d'un SPRT après convergence ou cap atteint.
 *
 * <p>Cf. SPEC-sprt §4.2.
 *
 * <ul>
 *   <li><strong>H1_ACCEPTED</strong> : LLR cumulé ≥ borne supérieure (challenger supérieur,
 *       promote).
 *   <li><strong>H0_ACCEPTED</strong> : LLR cumulé ≤ borne inférieure (challenger pas mieux,
 *       reject).
 *   <li><strong>MAX_GAMES</strong> : cap maxGames atteint sans franchir aucune borne (inconclusive
 *       ; convention fastchess : ne pas promote).
 *   <li><strong>CONTINUE</strong> : SPRT en cours, attendre plus de games (non-final).
 * </ul>
 */
public enum SprtDecision {
  H1_ACCEPTED,
  H0_ACCEPTED,
  MAX_GAMES,
  CONTINUE
}
