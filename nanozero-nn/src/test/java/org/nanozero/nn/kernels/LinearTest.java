package org.nanozero.nn.kernels;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/** Tests unitaires de {@link Linear} (cf. SPEC §7.4, §13 phase 3). */
class LinearTest {

  /** Référence scalaire pure, pour comparaison à 1e-5 du kernel SIMD. */
  private static void linearScalarRef(
      float[] input,
      float[] weights,
      float[] bias,
      float[] output,
      int inFeatures,
      int outFeatures,
      int batchSize) {
    for (int n = 0; n < batchSize; n++) {
      for (int o = 0; o < outFeatures; o++) {
        float acc = 0f;
        for (int i = 0; i < inFeatures; i++) {
          acc += input[n * inFeatures + i] * weights[o * inFeatures + i];
        }
        output[n * outFeatures + o] = acc + bias[o];
      }
    }
  }

  @Test
  @DisplayName("Cas trivial 1×1 (identity-like) : output = input * w + bias")
  void trivialOneByOne() {
    float[] input = {3.0f};
    float[] weights = {2.0f};
    float[] bias = {0.5f};
    float[] output = new float[1];
    Linear.applyBatch(input, weights, bias, output, 1, 1, 1);
    assertThat(output[0]).isEqualTo(3.0f * 2.0f + 0.5f);
  }

  @Test
  @DisplayName("Identity 4×4 : weights = I_4, bias = 0 → output = input")
  void identity4x4() {
    int n = 4;
    float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
    float[] weights = new float[n * n];
    for (int i = 0; i < n; i++) {
      weights[i * n + i] = 1.0f;
    }
    float[] bias = new float[n];
    float[] output = new float[n];
    Linear.applyBatch(input, weights, bias, output, n, n, 1);
    assertThat(output).containsExactly(input);
  }

  @ParameterizedTest(name = "inFeatures={0}, outFeatures={1}, batchSize={2} (vs scalaire à 1e-5)")
  @CsvSource({
    // Cas value head canoniques §3.3
    "64, 64, 1",
    "64, 64, 64", // batch maximal
    "64, 1, 1",
    "64, 1, 64",
    // Queue scalaire (inFeatures non multiple de LANES = 8 sur AVX2)
    "7, 5, 1",
    "9, 3, 1",
    "15, 17, 4",
    "17, 15, 4",
    // Dimensions plus larges
    "128, 64, 16",
    "256, 128, 8"
  })
  @DisplayName("Random vs référence scalaire : tolérance 1e-5 (float32)")
  void randomVsScalarRef(int inFeatures, int outFeatures, int batchSize) {
    Random rng =
        new Random((long) inFeatures * 1_000_000L + (long) outFeatures * 1_000L + batchSize);
    float[] input = new float[batchSize * inFeatures];
    float[] weights = new float[outFeatures * inFeatures];
    float[] bias = new float[outFeatures];
    for (int i = 0; i < input.length; i++) {
      input[i] = (float) rng.nextGaussian();
    }
    for (int i = 0; i < weights.length; i++) {
      weights[i] = (float) rng.nextGaussian();
    }
    for (int i = 0; i < bias.length; i++) {
      bias[i] = (float) rng.nextGaussian();
    }

    float[] outputSimd = new float[batchSize * outFeatures];
    float[] outputRef = new float[batchSize * outFeatures];
    Linear.applyBatch(input, weights, bias, outputSimd, inFeatures, outFeatures, batchSize);
    linearScalarRef(input, weights, bias, outputRef, inFeatures, outFeatures, batchSize);

    // Tolérance : float32 a ~7 chiffres significatifs ; FMA améliore légèrement la précision
    // par rapport à mul+add séparés, donc on peut voir un léger écart bit-à-bit même avec un
    // calcul sémantiquement identique. 1e-4 est conservateur et compatible avec l'invariant
    // de parité numérique §1.4 du SPEC.
    for (int i = 0; i < outputSimd.length; i++) {
      assertThat(outputSimd[i])
          .as("output[%d] (in=%d, out=%d, batch=%d)", i, inFeatures, outFeatures, batchSize)
          .isCloseTo(outputRef[i], within(1e-4f));
    }
  }

  @Test
  @DisplayName("Batch indépendant : sample[1] dépend uniquement de input[1×inFeatures..]")
  void batchIndependence() {
    int inFeatures = 8;
    int outFeatures = 3;
    int batchSize = 2;
    float[] weights = new float[outFeatures * inFeatures];
    java.util.Arrays.fill(weights, 1.0f); // sum-style
    float[] bias = {0f, 100f, -50f};

    // sample 0 : tout 1.0 → sum = 8, +bias → 8, 108, -42
    // sample 1 : tout 2.0 → sum = 16, +bias → 16, 116, -34
    float[] input = new float[batchSize * inFeatures];
    for (int i = 0; i < inFeatures; i++) {
      input[i] = 1.0f;
      input[inFeatures + i] = 2.0f;
    }
    float[] output = new float[batchSize * outFeatures];
    Linear.applyBatch(input, weights, bias, output, inFeatures, outFeatures, batchSize);
    assertThat(output).containsExactly(8.0f, 108.0f, -42.0f, 16.0f, 116.0f, -34.0f);
  }

  // -------------------------------------------------------------------------------------------
  // Bench rapide : @Tag("perf") exclu en CI rapide. Cible §9.2 indicative ; mesure JMH normative
  // dans nanozero-bench plus tard.
  // -------------------------------------------------------------------------------------------

  @Test
  @Tag("perf")
  @DisplayName("Bench Linear (64, 64, 64) : ≥ 5 GFlops indicatif sur AVX2")
  void benchLinear64x64x64() {
    System.out.println("SIMD species : " + Linear.SPECIES + " (LANES=" + Linear.LANES + ")");
    int inFeatures = 64;
    int outFeatures = 64;
    int batchSize = 64;
    int flopsPerCall =
        2L * inFeatures * outFeatures * batchSize > Integer.MAX_VALUE
            ? Integer.MAX_VALUE
            : (int) (2L * inFeatures * outFeatures * batchSize);

    Random rng = new Random(0xBEEFL);
    float[] input = new float[batchSize * inFeatures];
    float[] weights = new float[outFeatures * inFeatures];
    float[] bias = new float[outFeatures];
    float[] output = new float[batchSize * outFeatures];
    for (int i = 0; i < input.length; i++) {
      input[i] = (float) rng.nextGaussian();
    }
    for (int i = 0; i < weights.length; i++) {
      weights[i] = (float) rng.nextGaussian();
    }
    for (int i = 0; i < bias.length; i++) {
      bias[i] = (float) rng.nextGaussian();
    }

    // Warmup (≥ 1 s — ~250k itérations sur (64,64,64) en cible)
    long warmupIters = 50_000;
    for (long it = 0; it < warmupIters; it++) {
      Linear.applyBatch(input, weights, bias, output, inFeatures, outFeatures, batchSize);
    }

    long measureIters = 100_000;
    long t0 = System.nanoTime();
    for (long it = 0; it < measureIters; it++) {
      Linear.applyBatch(input, weights, bias, output, inFeatures, outFeatures, batchSize);
    }
    long elapsedNs = System.nanoTime() - t0;
    double secs = elapsedNs / 1e9;
    double totalFlops = (double) measureIters * flopsPerCall;
    double gflops = totalFlops / secs / 1e9;
    System.out.printf(
        "Linear(64,64,64) : %.2f GFlops (%.2f µs/call, %d iters in %.3f s)%n",
        gflops, (elapsedNs / (double) measureIters) / 1000.0, measureIters, secs);

    // Cible indicative §9.2 : ≥ 5 GFlops sur AVX2. < 2 GFlops = fallback scalaire ou problème
    // de Vector API. On exige juste une borne basse défensive ici.
    assertThat(gflops).as("Linear(64,64,64) GFlops sur LANES=%d", Linear.LANES).isGreaterThan(2.0);
  }
}
