package org.nanozero.sprt.stats;

/**
 * Modèle statistique pour le calcul du LLR SPRT.
 *
 * <p>Cf. SPEC-sprt §4 et fastchess {@code SPRT::getLLR}.
 *
 * <ul>
 *   <li><strong>NORMALIZED</strong> (défaut) : <em>normalized Elo</em> (BergPan 2021,
 *       cantate.be/Fishtest). Plus robuste pour les engines NN qui produisent un nombre élevé de
 *       draws. C'est le modèle utilisé par fastchess en défaut.
 *   <li><strong>LOGISTIC</strong> : Bradley-Terry classique sur score (0, 0.5, 1.0). Score Elo
 *       logistique standard.
 *   <li><strong>BAYESIAN</strong> : BayesElo avec draw_elo estimé empiriquement depuis le ratio
 *       wins/losses observé. Requiert au moins 1 win + 1 loss.
 * </ul>
 */
public enum SprtModel {
  NORMALIZED,
  LOGISTIC,
  BAYESIAN
}
