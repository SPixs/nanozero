package org.nanozero.sprt.stats;

/**
 * Bornes SPRT pré-calculées depuis (alpha, beta, elo0, elo1, model).
 *
 * <p>Cf. SPEC-sprt §4.2.
 *
 * <p>Les bornes Wald sont :
 *
 * <ul>
 *   <li>{@code upper = log((1 - beta) / alpha)} — au-dessus, on accepte H1.
 *   <li>{@code lower = log(beta / (1 - alpha))} — en dessous, on accepte H0.
 * </ul>
 *
 * <p>Avec les valeurs standard alpha=beta=0.05 → {@code upper ≈ 2.944}, {@code lower ≈ -2.944}.
 *
 * @param alpha probabilité d'erreur Type I (reject H0 quand H0 vrai), 0 &lt; alpha &lt; 1.
 * @param beta probabilité d'erreur Type II (accept H0 quand H1 vrai), 0 &lt; beta &lt; 1, alpha +
 *     beta &lt; 1.
 * @param elo0 hypothèse nulle (typique 0 = engines équivalents).
 * @param elo1 hypothèse alternative (typique 5 = challenger ≥ 5 Elo).
 * @param model modèle statistique (cf. {@link SprtModel}).
 * @param upper borne supérieure LLR (calculée).
 * @param lower borne inférieure LLR (calculée).
 */
public record SprtBounds(
    double alpha,
    double beta,
    double elo0,
    double elo1,
    SprtModel model,
    double upper,
    double lower) {

  /**
   * Construit les bornes SPRT après validation des paramètres.
   *
   * @param alpha 0 &lt; alpha &lt; 1
   * @param beta 0 &lt; beta &lt; 1, alpha + beta &lt; 1
   * @param elo0 hypothèse nulle (Elo difference, peut être 0 ou négatif)
   * @param elo1 hypothèse alternative, doit être strictement &gt; elo0
   * @param model modèle statistique non null
   * @return bornes calculées
   * @throws IllegalArgumentException si paramètres invalides
   */
  public static SprtBounds of(
      double alpha, double beta, double elo0, double elo1, SprtModel model) {
    if (model == null) {
      throw new IllegalArgumentException("model must not be null");
    }
    if (alpha <= 0.0 || alpha >= 1.0) {
      throw new IllegalArgumentException("alpha must be in (0, 1), got " + alpha);
    }
    if (beta <= 0.0 || beta >= 1.0) {
      throw new IllegalArgumentException("beta must be in (0, 1), got " + beta);
    }
    if (alpha + beta >= 1.0) {
      throw new IllegalArgumentException("alpha + beta must be < 1, got " + (alpha + beta));
    }
    if (elo0 >= elo1) {
      throw new IllegalArgumentException(
          "elo0 must be < elo1, got elo0=" + elo0 + ", elo1=" + elo1);
    }
    double upper = Math.log((1.0 - beta) / alpha);
    double lower = Math.log(beta / (1.0 - alpha));
    return new SprtBounds(alpha, beta, elo0, elo1, model, upper, lower);
  }

  /** Standard SPRT alpha=0.05 beta=0.05 model=normalized. */
  public static SprtBounds standard(double elo0, double elo1) {
    return of(0.05, 0.05, elo0, elo1, SprtModel.NORMALIZED);
  }
}
