package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Tests unitaires de {@link DirichletSampler} (cf. ADR-012, SPEC-engine §5.2). Propriétés
 * statistiques, déterminisme, edge cases et validations.
 */
class DirichletSamplerTest {

  @Test
  @DisplayName("sample : somme = 1.0 sur configurations alpha × n variées")
  void sample_sumEqualsOne() {
    Random rng = new Random(42L);
    float[] result = DirichletSampler.sample(0.3f, 20, rng);
    float sum = 0f;
    for (float v : result) {
      sum += v;
    }
    assertThat(sum).isCloseTo(1.0f, within(1e-5f));

    for (float alpha : new float[] {0.1f, 1.0f, 5.0f}) {
      for (int n : new int[] {2, 5, 50, 100}) {
        float[] r = DirichletSampler.sample(alpha, n, new Random(0L));
        float s = 0f;
        for (float v : r) {
          s += v;
        }
        assertThat(s).isCloseTo(1.0f, within(1e-4f));
      }
    }
  }

  @Test
  @DisplayName("sample : toutes les valeurs ≥ 0 sur 1000 trials")
  void sample_allValuesNonNegative() {
    Random rng = new Random(123L);
    for (int trial = 0; trial < 1000; trial++) {
      float[] r = DirichletSampler.sample(0.3f, 20, rng);
      for (float v : r) {
        assertThat(v).isGreaterThanOrEqualTo(0f);
      }
    }
  }

  @Test
  @DisplayName("sample : déterminisme avec seed fixée (résultats bit-à-bit identiques)")
  void sample_determinismWithFixedSeed() {
    Random rng1 = new Random(42L);
    Random rng2 = new Random(42L);
    float[] r1 = DirichletSampler.sample(0.3f, 20, rng1);
    float[] r2 = DirichletSampler.sample(0.3f, 20, rng2);
    assertThat(r1).isEqualTo(r2);
  }

  @Test
  @DisplayName("sample : seeds différentes → résultats différents")
  void sample_differentSeedsDiffer() {
    Random rng1 = new Random(1L);
    Random rng2 = new Random(2L);
    float[] r1 = DirichletSampler.sample(0.3f, 20, rng1);
    float[] r2 = DirichletSampler.sample(0.3f, 20, rng2);
    assertThat(r1).isNotEqualTo(r2);
  }

  @Test
  @DisplayName("sample : alpha=1.0 → distribution approximativement uniforme (mean ~ 1/n)")
  void sample_distributionApproximatesUniformWhenAlphaEqualsOne() {
    int n = 10;
    int trials = 10_000;
    double[] means = new double[n];
    Random rng = new Random(7L);
    for (int t = 0; t < trials; t++) {
      float[] s = DirichletSampler.sample(1.0f, n, rng);
      for (int i = 0; i < n; i++) {
        means[i] += s[i];
      }
    }
    for (int i = 0; i < n; i++) {
      means[i] /= trials;
      assertThat(means[i]).isCloseTo(0.1, within(0.01));
    }
  }

  @Test
  @DisplayName("sample : alpha très petit (0.01) → distribution peaky")
  void sample_concentrationAlphaSmallProducesPeaky() {
    Random rng = new Random(11L);
    int peakyCount = 0;
    int trials = 1000;
    for (int t = 0; t < trials; t++) {
      float[] s = DirichletSampler.sample(0.01f, 10, rng);
      float max = 0f;
      for (float v : s) {
        max = Math.max(max, v);
      }
      if (max > 0.9f) {
        peakyCount++;
      }
    }
    assertThat(peakyCount).isGreaterThan(trials / 2);
  }

  @Test
  @DisplayName("sample : alpha grand (100.0) → distribution presque uniforme (variance faible)")
  void sample_concentrationAlphaLargeProducesNearUniform() {
    Random rng = new Random(13L);
    int n = 10;
    float[] s = DirichletSampler.sample(100.0f, n, rng);
    float max = 0f;
    float min = 1f;
    for (float v : s) {
      max = Math.max(max, v);
      min = Math.min(min, v);
    }
    assertThat(max - min).isLessThan(0.1f);
  }

  @Test
  @DisplayName("sample : alpha = 0 → IAE")
  void sample_invalidAlphaZero() {
    Random rng = new Random();
    assertThatThrownBy(() -> DirichletSampler.sample(0.0f, 5, rng))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("alpha");
  }

  @Test
  @DisplayName("sample : alpha négatif → IAE")
  void sample_invalidAlphaNegative() {
    Random rng = new Random();
    assertThatThrownBy(() -> DirichletSampler.sample(-1.0f, 5, rng))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("alpha");
  }

  @Test
  @DisplayName("sample : n = 0 → IAE")
  void sample_invalidNZero() {
    Random rng = new Random();
    assertThatThrownBy(() -> DirichletSampler.sample(0.3f, 0, rng))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("n");
  }

  @Test
  @DisplayName("sample : n négatif → IAE")
  void sample_invalidNNegative() {
    Random rng = new Random();
    assertThatThrownBy(() -> DirichletSampler.sample(0.3f, -1, rng))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("n");
  }

  @Test
  @DisplayName("sample : rng null → NPE")
  void sample_nullRng() {
    assertThatThrownBy(() -> DirichletSampler.sample(0.3f, 5, null))
        .isInstanceOf(NullPointerException.class);
  }

  @Test
  @DisplayName("sample : n=1 trivial → [1.0]")
  void sample_n1Trivial() {
    Random rng = new Random(17L);
    float[] r = DirichletSampler.sample(0.3f, 1, rng);
    assertThat(r).hasSize(1);
    assertThat(r[0]).isCloseTo(1.0f, within(1e-6f));
  }

  @Test
  @DisplayName("sample : grand n (1000) reste stable numériquement (somme = 1)")
  void sample_largeNNumericalStability() {
    Random rng = new Random(19L);
    float[] r = DirichletSampler.sample(0.3f, 1000, rng);
    assertThat(r).hasSize(1000);
    float sum = 0f;
    for (float v : r) {
      sum += v;
      assertThat(v).isGreaterThanOrEqualTo(0f);
    }
    assertThat(sum).isCloseTo(1.0f, within(1e-4f));
  }

  @Test
  @DisplayName(
      "sample : alpha exactement = 1.0 → branche directe Marsaglia-Tsang (pas de boost, max>=min)")
  void sample_alphaEqualsOneTakesDirectBranch() {
    Random rng = new Random(23L);
    float[] r = DirichletSampler.sample(1.0f, 5, rng);
    assertThat(r).hasSize(5);
    float sum = 0f;
    for (float v : r) {
      sum += v;
    }
    assertThat(sum).isCloseTo(1.0f, within(1e-5f));
  }

  /**
   * Couvre la branche {@code v <= 0 → continue} dans la boucle Marsaglia-Tsang. Avec {@code
   * alpha=1.0}, on a {@code d=2/3} et {@code c=1/sqrt(6) ≈ 0.408}. La branche est atteinte dès que
   * {@code x < -1/c ≈ -2.45} (alors {@code v = (1 + c*x)^3 <= 0}). On force une première valeur
   * très négative, puis on rend la main au {@link Random} standard pour permettre l'acceptation
   * finale.
   */
  @Test
  @DisplayName("sample : couvre la branche v ≤ 0 (continue) via Random custom")
  void sample_coversVNonPositiveBranch() {
    Random rng =
        new Random(31L) {
          private int gaussianCallCount = 0;

          @Override
          public synchronized double nextGaussian() {
            gaussianCallCount++;
            if (gaussianCallCount == 1) {
              return -10.0;
            }
            return super.nextGaussian();
          }
        };
    float[] r = DirichletSampler.sample(1.0f, 5, rng);
    assertThat(r).hasSize(5);
    float sum = 0f;
    for (float v : r) {
      sum += v;
      assertThat(v).isGreaterThanOrEqualTo(0f);
    }
    assertThat(sum).isCloseTo(1.0f, within(1e-5f));
  }

  /**
   * Couvre la branche fallback uniforme ({@code sum == 0}) du sampler. Pour {@code alpha < 1}, la
   * boost technique calcule {@code gamma = sampleGamma(alpha+1) * U^(1/alpha)}. Avec {@code U=0},
   * le résultat est 0. En forçant {@code nextDouble() → 0.0} systématiquement, tous les samples
   * sont à 0 ; la sum est 0 et le fallback uniforme est emprunté.
   *
   * <p><strong>Note d'implémentation</strong> : on override aussi {@code nextGaussian()} pour
   * retourner une valeur fixe. {@link Random#nextGaussian()} appelle {@link Random#nextDouble()} en
   * interne (Box-Muller) et boucle tant que {@code s == 0 || s >= 1}. Avec {@code nextDouble = 0}
   * systématique, cette boucle interne serait infinie.
   */
  @Test
  @DisplayName("sample : couvre le fallback uniforme (sum == 0) via Random custom")
  void sample_coversUniformFallback() {
    Random rng =
        new Random(37L) {
          @Override
          public double nextDouble() {
            return 0.0; // force boost U=0 ⇒ gamma=0 ⇒ sample=0
          }

          @Override
          public synchronized double nextGaussian() {
            return 0.0; // bypass internal nextDouble (infinite loop in Random.nextGaussian)
          }
        };
    int n = 5;
    float[] r = DirichletSampler.sample(0.3f, n, rng);
    assertThat(r).hasSize(n);
    float expected = 1f / n;
    for (float v : r) {
      assertThat(v).isEqualTo(expected);
    }
    float sum = 0f;
    for (float v : r) {
      sum += v;
    }
    assertThat(sum).isCloseTo(1.0f, within(1e-6f));
  }
}
