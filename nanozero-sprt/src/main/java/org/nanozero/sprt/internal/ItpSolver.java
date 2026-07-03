package org.nanozero.sprt.internal;

import java.util.function.DoubleUnaryOperator;

/**
 * Implémentation du root-finder ITP (Interpolate Truncate Project) — Oliveira &amp; Takahashi
 * 2020/2021 «An Enhancement of the Bisection Method Average Performance Preserving Minmax
 * Optimality», ACM Trans. Math. Softw. 47, 1, Article 5.
 *
 * <p>Port direct de {@code fastchess/app/src/matchmaking/sprt/sprt.cpp::itp}.
 *
 * <p>Algorithme : combine regula falsi (interpolation) avec bissection (sécurisé) + truncation +
 * projection. Convergence super-linéaire en pratique vs bissection pure, tout en conservant la
 * garantie minmax optimal de la bissection.
 *
 * <p>Hypothèse : {@code f(a)} et {@code f(b)} ont des signes opposés (root existe entre a et b).
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-sprt} module.
 */
public final class ItpSolver {

  /** Constantes par défaut hérités de fastchess (Oliveira-Takahashi optimal-ish). */
  public static final double DEFAULT_K1 = 0.1;

  public static final double DEFAULT_K2 = 2.0;
  public static final double DEFAULT_N0 = 0.99;

  private ItpSolver() {
    throw new AssertionError("Utility class — do not instantiate");
  }

  /**
   * Trouve la racine de {@code f} dans l'intervalle {@code [a, b]} via ITP.
   *
   * <p>{@code f_a} et {@code f_b} sont les valeurs pré-calculées (économise 2 évaluations à
   * l'initialisation). Si l'un est {@code Infinity}, l'algo le tolère et procède via bissection
   * initiale.
   *
   * @param f fonction à zéroter, non null
   * @param a borne basse de l'intervalle initial
   * @param b borne haute
   * @param fA {@code f(a)} (peut être {@code Double.POSITIVE_INFINITY} pour borne pure)
   * @param fB {@code f(b)} (peut être {@code Double.NEGATIVE_INFINITY} pour borne pure)
   * @param k1 paramètre truncation (défaut {@link #DEFAULT_K1})
   * @param k2 paramètre truncation (défaut {@link #DEFAULT_K2})
   * @param n0 paramètre projection (défaut {@link #DEFAULT_N0})
   * @param epsilon précision de la racine (l'algo s'arrête quand {@code |b - a| ≤ 2 epsilon})
   * @return point milieu de l'intervalle final
   */
  public static double solve(
      DoubleUnaryOperator f,
      double a,
      double b,
      double fA,
      double fB,
      double k1,
      double k2,
      double n0,
      double epsilon) {
    if (fA > 0.0) {
      // Swap pour garantir fA < 0 < fB.
      double tmpA = a;
      a = b;
      b = tmpA;
      double tmpFa = fA;
      fA = fB;
      fB = tmpFa;
    }
    // À ce stade : fA <= 0 < fB.

    double nHalf = Math.ceil(Math.log(Math.abs(b - a) / (2.0 * epsilon)) / Math.log(2.0));
    double nMax = nHalf + n0;
    int i = 0;
    while (Math.abs(b - a) > 2.0 * epsilon) {
      double xHalf = (a + b) / 2.0;
      double r = epsilon * Math.pow(2.0, nMax - i) - (b - a) / 2.0;
      double delta = k1 * Math.pow(b - a, k2);

      // Regula falsi.
      double xF = (fB * a - fA * b) / (fB - fA);

      double diff = xHalf - xF;
      double sigma = diff >= 0 ? 1.0 : -1.0;
      double xT = delta <= Math.abs(diff) ? xF + sigma * delta : xHalf;

      double xItp = Math.abs(xT - xHalf) <= r ? xT : xHalf - sigma * r;

      double fItp = f.applyAsDouble(xItp);
      if (fItp == 0.0) {
        a = xItp;
        b = xItp;
      } else if (fItp < 0.0) {
        a = xItp;
        fA = fItp;
      } else {
        b = xItp;
        fB = fItp;
      }
      i++;
    }
    return (a + b) / 2.0;
  }
}
