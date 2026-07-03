package org.nanozero.sprt.internal;

import java.util.function.DoubleUnaryOperator;

/**
 * Maximum Likelihood Estimation pour distribution discrète sous contrainte de moyenne ou
 * normalisation Elo (BergPan 2021).
 *
 * <p>Port direct des lambdas {@code mle} dans {@code
 * fastchess/app/src/matchmaking/sprt/sprt.cpp::getLLR_logistic} et {@code getLLR_normalized}.
 *
 * <p>Références :
 *
 * <ul>
 *   <li>Michel Van den Bergh, «Comparing the approximations for the generalized log likelihood
 *       ratio of a multinomial distribution», cantate.be/Fishtest/comparing_approximations.pdf
 *   <li>Michel Van den Bergh, «Comments on normalized Elo»,
 *       cantate.be/Fishtest/normalized_elo_practical.pdf
 * </ul>
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-sprt} module.
 */
public final class MleSolver {

  private static final double THETA_EPSILON_LOGISTIC = 1e-3;
  private static final double THETA_EPSILON_NORMALIZED = 1e-7;
  private static final double MLE_EPSILON_NORMALIZED = 1e-4;
  private static final int MLE_MAX_ITERATIONS_NORMALIZED = 10;

  private MleSolver() {
    throw new AssertionError("Utility class — do not instantiate");
  }

  /**
   * MLE «logistic / bayesian» — distribution avec contrainte de moyenne {@code mean = s}.
   *
   * <p>Implémente l'équation 1.3 de [BergPan, comparing_approximations] : on cherche {@code theta}
   * tel que {@code sum(phat_i * (a_i - s) / (1 + theta * (a_i - s))) = 0}. Puis {@code p_i = phat_i
   * / (1 + theta * (a_i - s))}.
   *
   * @param scores valeurs discrètes (ex: {0.0, 0.5, 1.0} pour trinomial L/D/W)
   * @param empirical probabilités empiriques observées (somme = 1)
   * @param s moyenne cible
   * @return distribution MLE de longueur N
   */
  public static double[] mleLogistic(double[] scores, double[] empirical, double s) {
    int n = scores.length;
    double minTheta = -1.0 / (scores[n - 1] - s);
    double maxTheta = -1.0 / (scores[0] - s);
    DoubleUnaryOperator f =
        x -> {
          double result = 0.0;
          for (int i = 0; i < n; i++) {
            double ai = scores[i];
            double phatI = empirical[i];
            result += phatI * (ai - s) / (1.0 + x * (ai - s));
          }
          return result;
        };
    double theta =
        ItpSolver.solve(
            f,
            minTheta,
            maxTheta,
            Double.POSITIVE_INFINITY,
            Double.NEGATIVE_INFINITY,
            ItpSolver.DEFAULT_K1,
            ItpSolver.DEFAULT_K2,
            ItpSolver.DEFAULT_N0,
            THETA_EPSILON_LOGISTIC);
    double[] p = new double[n];
    for (int i = 0; i < n; i++) {
      double ai = scores[i];
      double phatI = empirical[i];
      p[i] = phatI / (1.0 + theta * (ai - s));
    }
    return p;
  }

  /**
   * MLE «normalized» — distribution avec contrainte de t = (mu - mu_ref) / sigma.
   *
   * <p>Implémente la section 4.1 de [BergPan, normalized_elo_practical]. Itératif : à chaque step,
   * recalcule phi en fonction des estimates courants de (mu, var), puis trouve theta via ITP,
   * jusqu'à convergence (max_diff &lt; {@link #MLE_EPSILON_NORMALIZED}).
   *
   * @param scores valeurs discrètes (ex: {0.0, 0.5, 1.0} pour trinomial)
   * @param empirical probabilités empiriques observées (somme = 1)
   * @param muRef moyenne référence (typiquement 0.5)
   * @param tStar contrainte sur t (= elo / (800 / ln(10)))
   * @return distribution MLE de longueur N
   */
  public static double[] mleNormalized(
      double[] scores, double[] empirical, double muRef, double tStar) {
    int n = scores.length;
    double[] p = new double[n];
    double initial = 1.0 / n;
    for (int i = 0; i < n; i++) {
      p[i] = initial;
    }

    for (int iter = 0; iter < MLE_MAX_ITERATIONS_NORMALIZED; iter++) {
      double mu = mean(scores, p);
      double var = variance(scores, p, mu);
      double sigma = Math.sqrt(var);
      double[] phi = new double[n];
      for (int i = 0; i < n; i++) {
        double ai = scores[i];
        double aiMinusMuOverSigma = (ai - mu) / sigma;
        phi[i] = ai - muRef - 0.5 * tStar * sigma * (1.0 + aiMinusMuOverSigma * aiMinusMuOverSigma);
      }

      double u = phi[0];
      double v = phi[0];
      for (int i = 1; i < n; i++) {
        if (phi[i] < u) u = phi[i];
        if (phi[i] > v) v = phi[i];
      }
      double minTheta = -1.0 / v;
      double maxTheta = -1.0 / u;

      final double[] phiFinal = phi;
      final double[] empFinal = empirical;
      DoubleUnaryOperator f =
          x -> {
            double result = 0.0;
            for (int i = 0; i < n; i++) {
              double phatI = empFinal[i];
              result += phatI * phiFinal[i] / (1.0 + x * phiFinal[i]);
            }
            return result;
          };
      double theta =
          ItpSolver.solve(
              f,
              minTheta,
              maxTheta,
              Double.POSITIVE_INFINITY,
              Double.NEGATIVE_INFINITY,
              ItpSolver.DEFAULT_K1,
              ItpSolver.DEFAULT_K2,
              ItpSolver.DEFAULT_N0,
              THETA_EPSILON_NORMALIZED);

      double maxDiff = 0.0;
      for (int i = 0; i < n; i++) {
        double phatI = empirical[i];
        double newPi = phatI / (1.0 + theta * phi[i]);
        double diff = Math.abs(newPi - p[i]);
        if (diff > maxDiff) maxDiff = diff;
        p[i] = newPi;
      }
      if (maxDiff < MLE_EPSILON_NORMALIZED) {
        break;
      }
    }
    return p;
  }

  /** Moyenne pondérée par probabilités. */
  static double mean(double[] x, double[] p) {
    double result = 0.0;
    for (int i = 0; i < x.length; i++) {
      result += x[i] * p[i];
    }
    return result;
  }

  /** Variance pondérée par probabilités. */
  static double variance(double[] x, double[] p, double mu) {
    double result = 0.0;
    for (int i = 0; i < x.length; i++) {
      double d = x[i] - mu;
      result += p[i] * d * d;
    }
    return result;
  }
}
