package org.nanozero.engine.internal;

import java.util.Random;

/**
 * Marsaglia-Tsang sampler for the Dirichlet distribution.
 *
 * <p>Used in v1.1.0+ to inject Dirichlet noise at the MCTS root during self-play training (cf.
 * ADR-012 and SPEC-engine §5.2). Disabled by default in production inference ({@code
 * EngineConfig.defaults()} yields {@code epsilon=0}).
 *
 * <p>Sampling algorithm:
 *
 * <ol>
 *   <li>Sample {@code n} independent values from {@code Gamma(alpha, 1)} using Marsaglia-Tsang
 *       (2000).
 *   <li>Normalize the vector so the sum equals 1.0.
 * </ol>
 *
 * <p>Determinism: the caller must pass the same {@link Random} instance (with same seed) across
 * runs to obtain bit-for-bit identical samples. Instantiating a new Random per call breaks
 * determinism.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
final class DirichletSampler {

  private DirichletSampler() {
    // utility class, no instantiation
  }

  /**
   * Sample a Dirichlet distribution of dimension {@code n} with concentration {@code alpha}.
   *
   * @param alpha concentration parameter (must be {@code > 0})
   * @param n dimension of the output vector (must be {@code > 0})
   * @param rng Random instance (must be non-null; pass the same seeded instance for determinism)
   * @return array of length {@code n} with sum approximately equal to 1.0 and all values {@code >=
   *     0}
   * @throws IllegalArgumentException if {@code alpha <= 0} or {@code n <= 0}
   * @throws NullPointerException if {@code rng} is null
   */
  static float[] sample(float alpha, int n, Random rng) {
    if (alpha <= 0f) {
      throw new IllegalArgumentException("alpha must be > 0, got " + alpha);
    }
    if (n <= 0) {
      throw new IllegalArgumentException("n must be > 0, got " + n);
    }
    if (rng == null) {
      throw new NullPointerException("rng must be non-null");
    }

    float[] samples = new float[n];
    float sum = 0f;
    for (int i = 0; i < n; i++) {
      samples[i] = (float) sampleGamma(alpha, rng);
      sum += samples[i];
    }

    if (sum > 0f) {
      for (int i = 0; i < n; i++) {
        samples[i] /= sum;
      }
    } else {
      // Extremely rare fallback (alpha > 0 should yield positive samples).
      // Uniform distribution preserves invariant sum=1.
      float uniform = 1f / n;
      for (int i = 0; i < n; i++) {
        samples[i] = uniform;
      }
    }

    return samples;
  }

  /**
   * Sample from {@code Gamma(alpha, 1)} using the Marsaglia-Tsang method (2000).
   *
   * <p>For {@code alpha < 1}, uses the boost technique: {@code Gamma(alpha) = Gamma(alpha + 1) *
   * U^(1/alpha)}.
   *
   * @param alpha shape parameter (must be {@code > 0})
   * @param rng Random instance
   * @return sample from {@code Gamma(alpha, 1)}
   */
  private static double sampleGamma(double alpha, Random rng) {
    if (alpha < 1.0) {
      // Boost: Gamma(alpha) = Gamma(alpha+1) * U^(1/alpha)
      double g = sampleGamma(alpha + 1.0, rng);
      double u = rng.nextDouble();
      return g * Math.pow(u, 1.0 / alpha);
    }

    double d = alpha - 1.0 / 3.0;
    double c = 1.0 / Math.sqrt(9.0 * d);

    while (true) {
      double x = rng.nextGaussian();
      double v = Math.pow(1.0 + c * x, 3);
      if (v <= 0) {
        continue;
      }
      double u = rng.nextDouble();
      double xSq = x * x;

      // Squeeze step (fast acceptance)
      if (u < 1.0 - 0.0331 * xSq * xSq) {
        return d * v;
      }

      // Full log test (slow acceptance)
      if (Math.log(u) < 0.5 * xSq + d * (1.0 - v + Math.log(v))) {
        return d * v;
      }
      // Otherwise: reject and loop
    }
  }
}
