package org.nanozero.nn;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.nio.file.Path;
import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Tests du chemin batché par paliers de {@link NetworkOnnx} (amendement v1.6.0, ADR-013).
 *
 * <p>Le chemin batché est testé en CI via le seam CPU EP ({@link
 * NetworkOnnx#loadCpuBatchedForTests}) : la logique paliers/padding/extraction WDL est identique au
 * mode CUDA, seul l'execution provider diffère. La validation CUDA EP réelle (perf + parité) est
 * faite hors CI sur machine GPU (cf. PERF-GPU-BATCH-phase0.md).
 *
 * <p>Fixture : {@code /onnx/tiny_wdl.onnx} — modèle minimal au contrat I/O deployment v1.5.0 (board
 * [N,119,8,8] dynamique → policy_logits [N,4672] + value [N,3] WDL), généré par {@code
 * docs/python/generate_test_onnx.py} (seed 42).
 */
final class NetworkOnnxBatchedTest {

  private static final float TOLERANCE = 1e-5f;

  private static Path fixture;
  private static NetworkOnnx loopNet;
  private static NetworkOnnx batchedNet;

  @BeforeAll
  static void load() throws Exception {
    var url = NetworkOnnxBatchedTest.class.getResource("/onnx/tiny_wdl.onnx");
    assertThat(url).isNotNull();
    fixture = Path.of(url.toURI());
    loopNet = NetworkOnnx.load(fixture);
    batchedNet = NetworkOnnx.loadCpuBatchedForTests(fixture);
  }

  /** Planes pseudo-aléatoires déterministes pour {@code n} positions, à la capacité demandée. */
  private static float[] randomPlanes(int capacity, long seed) {
    Random rng = new Random(seed);
    float[] planes = new float[capacity * NetworkOnnx.INPUT_FLOATS_PER_POS];
    for (int i = 0; i < planes.length; i++) {
      planes[i] = rng.nextInt(4) == 0 ? 1.0f : 0.0f;
    }
    return planes;
  }

  @Nested
  final class BucketFor {

    @ParameterizedTest
    @CsvSource({
      "1, 1",
      "2, 4",
      "4, 4",
      "5, 8",
      "8, 8",
      "9, 16",
      "16, 16",
      "17, 32",
      "33, 64",
      "64, 64",
      "65, 128",
      "100, 128",
      "129, 256",
      "256, 256"
    })
    void mapsToSmallestBucketAboveOrEqual(int batchSize, int expectedBucket) {
      assertThat(NetworkOnnx.bucketFor(batchSize)).isEqualTo(expectedBucket);
    }

    @ParameterizedTest
    @ValueSource(ints = {0, -1, 257, 1024})
    void rejectsOutOfRange(int batchSize) {
      assertThatThrownBy(() -> NetworkOnnx.bucketFor(batchSize))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Nested
  final class MaxBatchCapacity {

    @Test
    void loopNetworkKeepsHistoricalContract() {
      assertThat(loopNet.maxBatch()).isEqualTo(Network.MAX_BATCH);
    }

    @Test
    void batchedNetworkExposesCudaCapacity() {
      assertThat(batchedNet.maxBatch()).isEqualTo(NetworkOnnx.CUDA_MAX_BATCH);
    }

    @Test
    void nnOutputDefaultCapacityIsMaxBatch() {
      assertThat(new NNOutput().capacity()).isEqualTo(Network.MAX_BATCH);
    }

    @Test
    void nnOutputExplicitCapacity() {
      NNOutput out = new NNOutput(NetworkOnnx.CUDA_MAX_BATCH);
      assertThat(out.capacity()).isEqualTo(256);
      // index 100 valide à capacité 256 (rejeté à capacité 64)
      assertThat(out.valueOf(100)).isEqualTo(0.0f);
      assertThatThrownBy(() -> new NNOutput().valueOf(100))
          .isInstanceOf(IndexOutOfBoundsException.class);
    }

    @Test
    void nnOutputRejectsInvalidCapacity() {
      assertThatThrownBy(() -> new NNOutput(0)).isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Nested
  final class Validation {

    @Test
    void rejectsBatchSizeAboveInstanceCapacity() {
      float[] planes = randomPlanes(256, 1);
      NNOutput out = new NNOutput(256);
      assertThatThrownBy(() -> loopNet.forward(planes, 65, out))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> batchedNet.forward(planes, 257, out))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsZeroBatchSize() {
      float[] planes = randomPlanes(1, 2);
      NNOutput out = new NNOutput();
      assertThatThrownBy(() -> batchedNet.forward(planes, 0, out))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsUndersizedOutput() {
      float[] planes = randomPlanes(8, 3);
      assertThatThrownBy(() -> batchedNet.forward(planes, 8, new NNOutput(4)))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsUndersizedPlanes() {
      float[] planes = new float[2 * NetworkOnnx.INPUT_FLOATS_PER_POS];
      assertThatThrownBy(() -> batchedNet.forward(planes, 8, new NNOutput()))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Nested
  final class Parity {

    /**
     * Parité chemin batché vs loop K×forward(1) pour des tailles palier et non-palier (padding).
     * Mêmes positions → mêmes sorties, à tolérance numérique près.
     */
    @ParameterizedTest
    @ValueSource(ints = {1, 3, 8, 17, 64})
    void batchedMatchesLoopReference(int batchSize) {
      float[] planes = randomPlanes(64, 42);
      NNOutput expected = new NNOutput();
      NNOutput actual = new NNOutput(256);
      loopNet.forward(planes, batchSize, expected);
      batchedNet.forward(planes, batchSize, actual);
      assertOutputsMatch(expected, actual, batchSize);
    }

    /**
     * Au-delà de {@code MAX_BATCH=64} (hors contrat loop), la référence est le même réseau batché
     * appelé position par position (bucket=1).
     */
    @ParameterizedTest
    @ValueSource(ints = {65, 100, 128, 200, 256})
    void beyondHistoricalMaxBatchMatchesPerPositionReference(int batchSize) {
      float[] planes = randomPlanes(256, 7);
      NNOutput actual = new NNOutput(256);
      batchedNet.forward(planes, batchSize, actual);

      float[] single = new float[NetworkOnnx.INPUT_FLOATS_PER_POS];
      NNOutput ref = new NNOutput(1);
      for (int b = 0; b < batchSize; b += 37) { // échantillonnage (256 forwards unitaires = lent)
        System.arraycopy(
            planes,
            b * NetworkOnnx.INPUT_FLOATS_PER_POS,
            single,
            0,
            NetworkOnnx.INPUT_FLOATS_PER_POS);
        batchedNet.forward(single, 1, ref);
        assertThat(actual.valueOf(b)).isCloseTo(ref.valueOf(0), within(TOLERANCE));
        assertThat(actual.logitsOf(b))
            .usingElementComparator(floatComparator())
            .containsExactly(box(ref.logitsOf(0)));
      }
    }

    /**
     * Le buffer hôte persistant (v1.6.0 IO zéro-alloc) est partagé entre les appels : après un gros
     * batch, un batch plus petit doit re-zéroer la zone de padding devenue sale (high-water mark
     * {@code dirtyPositions}) — parité avec la référence loop préservée à chaque étape.
     */
    @Test
    void reusedHostBufferStaysCleanAcrossShrinkingBatches() {
      int[] sizes = {60, 5, 2}; // buckets 64 → 8 → 4 : padding sale re-zéroé à chaque étape
      for (int i = 0; i < sizes.length; i++) {
        int batchSize = sizes[i];
        float[] planes = randomPlanes(64, 100 + i);
        NNOutput expected = new NNOutput();
        NNOutput actual = new NNOutput(256);
        loopNet.forward(planes, batchSize, expected);
        batchedNet.forward(planes, batchSize, actual);
        assertOutputsMatch(expected, actual, batchSize);
      }
    }

    /** Le padding ne mute jamais le buffer du caller et n'est pas sensible à son contenu. */
    @Test
    void paddingNeitherMutatesCallerBufferNorReadsBeyondBatch() {
      int batchSize = 3; // bucket=4 → 1 position de padding
      float[] planes = randomPlanes(64, 11);
      // pollue la zone au-delà des 3 positions effectives avec des NaN
      java.util.Arrays.fill(
          planes, batchSize * NetworkOnnx.INPUT_FLOATS_PER_POS, planes.length, Float.NaN);
      float[] snapshot = planes.clone();

      NNOutput expected = new NNOutput();
      NNOutput actual = new NNOutput(256);
      loopNet.forward(planes, batchSize, expected);
      batchedNet.forward(planes, batchSize, actual);

      assertOutputsMatch(expected, actual, batchSize);
      assertThat(planes).containsExactly(snapshot); // buffer caller intact (NaN compris)
    }

    private static void assertOutputsMatch(NNOutput expected, NNOutput actual, int batchSize) {
      for (int b = 0; b < batchSize; b++) {
        assertThat(actual.valueOf(b)).isCloseTo(expected.valueOf(b), within(TOLERANCE));
        assertThat(actual.logitsOf(b))
            .usingElementComparator(floatComparator())
            .containsExactly(box(expected.logitsOf(b)));
      }
    }

    private static java.util.Comparator<Float> floatComparator() {
      return (a, b) -> Math.abs(a - b) <= TOLERANCE ? 0 : Float.compare(a, b);
    }

    private static Float[] box(float[] values) {
      Float[] boxed = new Float[values.length];
      for (int i = 0; i < values.length; i++) {
        boxed[i] = values[i];
      }
      return boxed;
    }
  }
}
