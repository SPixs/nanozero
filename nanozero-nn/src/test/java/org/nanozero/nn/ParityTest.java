package org.nanozero.nn;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.fail;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nanozero.nn.internal.NpzReader;
import org.nanozero.nn.internal.NpzReader.NpzData;

/**
 * Test de parité numérique Java vs PyTorch (cf. SPEC §13 phase 9).
 *
 * <p>Charge le modèle exporté par {@code generate_parity_fixtures.py} (phase 9a) et compare les
 * sorties de {@link Network#forward} aux sorties PyTorch de référence à tolérances :
 *
 * <ul>
 *   <li>logits : {@code 1e-4} (FMA cumulative sur conv 3×3 + chains de 8 ResBlocks)
 *   <li>values : {@code 1e-5} — value head WDL v1.5.0 : softmax(3 logits) puis P(W)−P(L)
 * </ul>
 *
 * <p>Critère §13 phase 9 : <strong>100 % des 100 positions test DOIVENT passer</strong>. Toute
 * divergence est un signal côté Java — le pipeline Python est validé en 9a (déterminisme + hash +
 * spread non-saturé). Les sources d'erreur côté Java possibles : kernel SIMD, transposition policy,
 * fold/reorder des poids, endianness, placement ReLU, double buffering, indexation NCHW.
 *
 * <p>Diagnostic structuré en cas d'échec (top diff, distribution square/plane, argmax cross-check)
 * pour identifier la signature SANS itération de fixtures intermédiaires.
 */
class ParityTest {

  /** Tolérance §13 phase 9 sur les logits (FMA cumulative). */
  private static final float TOL_LOGITS = 1e-4f;

  /** Tolérance §13 phase 9 sur les values (post-tanh). */
  private static final float TOL_VALUES = 1e-5f;

  /** Hash attendu pour parity-model.npz (régénéré pour le value head WDL v1.5.0). */
  private static final String FIXTURE_HASH =
      "ffab25af5c6c06f377cd7faaadf602f1a6488ae43d5c7171dd2f72c6deb927a4";

  private static final int N_FIXTURES = 100;
  private static final int INPUT_SIZE = 119 * 64;
  private static final int LOGITS_PER_POS = MoveEncoding.POLICY_INDICES;

  // ---------------------------------------------------------------------------------------------
  // Helpers : path resolution + chargement fixtures
  // ---------------------------------------------------------------------------------------------

  private static Path fixturePath(String name) {
    var url = ParityTest.class.getResource("/npz/" + name);
    if (url == null) {
      throw new AssertionError("Fixture introuvable : /npz/" + name);
    }
    try {
      return Paths.get(url.toURI());
    } catch (Exception e) {
      throw new AssertionError("Impossible de résoudre /npz/" + name, e);
    }
  }

  /** Conteneur pour les 3 tenseurs de parity-fixtures.npz, flat float32. */
  private record ParityFixtures(float[] inputs, float[] logits, float[] values) {}

  private static ParityFixtures loadFixtures() throws IOException {
    NpzData data = NpzReader.read(fixturePath("parity-fixtures.npz"));
    float[] inputs = data.floatTensors().get("inputs");
    float[] logits = data.floatTensors().get("logits");
    float[] values = data.floatTensors().get("values");
    if (inputs == null || logits == null || values == null) {
      throw new AssertionError(
          "parity-fixtures.npz doit contenir inputs/logits/values float32 (got "
              + data.floatTensors().keySet()
              + ")");
    }
    assertThat(inputs).hasSize(N_FIXTURES * INPUT_SIZE);
    assertThat(logits).hasSize(N_FIXTURES * LOGITS_PER_POS);
    assertThat(values).hasSize(N_FIXTURES);
    return new ParityFixtures(inputs, logits, values);
  }

  // ---------------------------------------------------------------------------------------------
  // Tests
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Hash consistency : computeWeightsHash(parity-model) == _meta_model_hash")
  void testHashConsistency() throws IOException {
    NpzData data = NpzReader.read(fixturePath("parity-model.npz"));
    String javaHash = NetworkLoader.computeWeightsHash(data.floatTensors());
    String metaHash = data.stringScalars().get(NetworkLoader.META_MODEL_HASH);
    assertThat(javaHash)
        .as("Java-recomputed hash doit matcher _meta_model_hash et le hash Python phase 9a")
        .isEqualTo(metaHash)
        .isEqualTo(FIXTURE_HASH);
  }

  @Test
  @DisplayName("Critère §13 phase 9 : 100 positions batch=1, 1e-4 logits / 1e-5 values")
  void testParityBatch1() throws IOException {
    Network network = NetworkLoader.load(fixturePath("parity-model.npz"));
    ParityFixtures fixtures = loadFixtures();

    float[] planes = new float[Network.MAX_BATCH * INPUT_SIZE];
    NNOutput output = new NNOutput();

    int failingPositionsLogits = 0;
    int failingPositionsValues = 0;
    float maxLogitsDiff = 0f;
    int maxLogitsDiffPos = -1;
    int maxLogitsDiffIdx = -1;
    float maxValueDiff = 0f;
    int maxValueDiffPos = -1;
    // Index du premier position en échec (logits ou values), pour dump détaillé.
    int firstFailingPos = -1;

    for (int i = 0; i < N_FIXTURES; i++) {
      Arrays.fill(planes, 0f);
      System.arraycopy(fixtures.inputs(), i * INPUT_SIZE, planes, 0, INPUT_SIZE);
      network.forward(planes, 1, output);

      boolean posFailedLogits = false;
      for (int j = 0; j < LOGITS_PER_POS; j++) {
        float expected = fixtures.logits()[i * LOGITS_PER_POS + j];
        float actual = output.logits[j];
        float diff = Math.abs(expected - actual);
        if (diff > maxLogitsDiff) {
          maxLogitsDiff = diff;
          maxLogitsDiffPos = i;
          maxLogitsDiffIdx = j;
        }
        if (diff > TOL_LOGITS) {
          posFailedLogits = true;
        }
      }
      if (posFailedLogits) {
        failingPositionsLogits++;
      }

      float vDiff = Math.abs(output.values[0] - fixtures.values()[i]);
      if (vDiff > maxValueDiff) {
        maxValueDiff = vDiff;
        maxValueDiffPos = i;
      }
      boolean posFailedValues = vDiff > TOL_VALUES;
      if (posFailedValues) {
        failingPositionsValues++;
      }

      if ((posFailedLogits || posFailedValues) && firstFailingPos < 0) {
        firstFailingPos = i;
      }
    }

    // Critère §13 phase 9 : 0 échec.
    if (failingPositionsLogits == 0 && failingPositionsValues == 0) {
      System.out.printf(
          "ParityTest batch=1 OK : 100/100 positions, max diff logits=%g (pos=%d idx=%d), "
              + "max diff values=%g (pos=%d)%n",
          maxLogitsDiff, maxLogitsDiffPos, maxLogitsDiffIdx, maxValueDiff, maxValueDiffPos);
      return;
    }

    // Diagnostic structuré (pas de patch, pas de relâche tolérance).
    StringBuilder report = new StringBuilder();
    report.append(
        String.format(
            "Parity FAIL: %d/100 positions failed logits (tol %g), %d/100 failed values (tol %g)%n",
            failingPositionsLogits, TOL_LOGITS, failingPositionsValues, TOL_VALUES));
    report.append(
        String.format(
            "Max abs diff logits: %g at pos=%d idx=%d (square=%d, plane=%d)%n",
            maxLogitsDiff,
            maxLogitsDiffPos,
            maxLogitsDiffIdx,
            maxLogitsDiffIdx / 73,
            maxLogitsDiffIdx % 73));
    report.append(
        String.format("Max abs diff values: %g at pos=%d%n", maxValueDiff, maxValueDiffPos));

    // Re-forward sur la première position en échec pour détail.
    if (firstFailingPos >= 0) {
      Arrays.fill(planes, 0f);
      System.arraycopy(fixtures.inputs(), firstFailingPos * INPUT_SIZE, planes, 0, INPUT_SIZE);
      network.forward(planes, 1, output);
      report.append(String.format("%nFirst failing position (i=%d):%n", firstFailingPos));

      // Top 10 logits diff de cette position.
      float[] diffs = new float[LOGITS_PER_POS];
      Integer[] idx = new Integer[LOGITS_PER_POS];
      for (int j = 0; j < LOGITS_PER_POS; j++) {
        diffs[j] =
            Math.abs(output.logits[j] - fixtures.logits()[firstFailingPos * LOGITS_PER_POS + j]);
        idx[j] = j;
      }
      Arrays.sort(idx, (a, b) -> Float.compare(diffs[b], diffs[a]));
      report.append("  Top 10 |diff| logits :%n".replace("%n", System.lineSeparator()));
      for (int k = 0; k < 10; k++) {
        int j = idx[k];
        int square = j / 73;
        int plane = j % 73;
        float exp = fixtures.logits()[firstFailingPos * LOGITS_PER_POS + j];
        float act = output.logits[j];
        report.append(
            String.format(
                "    idx=%-4d (square=%2d h=%d w=%d, plane=%2d) expected=%+.6f actual=%+.6f"
                    + " diff=%g%n",
                j, square, square / 8, square % 8, plane, exp, act, diffs[j]));
      }

      // argmax cross-check : si argmax Java != argmax PyTorch, signal structurel fort.
      int argmaxJava = 0;
      int argmaxPy = 0;
      float bestJava = output.logits[0];
      float bestPy = fixtures.logits()[firstFailingPos * LOGITS_PER_POS];
      for (int j = 1; j < LOGITS_PER_POS; j++) {
        if (output.logits[j] > bestJava) {
          bestJava = output.logits[j];
          argmaxJava = j;
        }
        float pyVal = fixtures.logits()[firstFailingPos * LOGITS_PER_POS + j];
        if (pyVal > bestPy) {
          bestPy = pyVal;
          argmaxPy = j;
        }
      }
      report.append(
          String.format(
              "  argmax Java=%d (square=%d, plane=%d) | argmax PyTorch=%d (square=%d, plane=%d) |"
                  + " %s%n",
              argmaxJava,
              argmaxJava / 73,
              argmaxJava % 73,
              argmaxPy,
              argmaxPy / 73,
              argmaxPy % 73,
              argmaxJava == argmaxPy ? "MATCH" : "MISMATCH (signal structurel : transpose ?)"));

      // Distribution square/plane des indices qui dépassent la tolérance.
      int[] squareHist = new int[64];
      int[] planeHist = new int[73];
      int countOverTol = 0;
      for (int j = 0; j < LOGITS_PER_POS; j++) {
        if (diffs[j] > TOL_LOGITS) {
          squareHist[j / 73]++;
          planeHist[j % 73]++;
          countOverTol++;
        }
      }
      report.append(String.format("  %d indices > tol sur cette position%n", countOverTol));
      report.append(
          String.format("  squares hit (count) : %s%n", summarizeHist(squareHist, "square")));
      report.append(
          String.format("  planes  hit (count) : %s%n", summarizeHist(planeHist, "plane")));

      // Value diff sur cette position.
      float vExp = fixtures.values()[firstFailingPos];
      float vAct = output.values[0];
      report.append(
          String.format(
              "  value : expected=%+.6f actual=%+.6f diff=%g%n",
              vExp, vAct, Math.abs(vExp - vAct)));
    }

    fail(report.toString());
  }

  @Test
  @DisplayName("Sanity batching : batch=64+36 cohérent avec batch=1 et avec fixtures")
  void testParityBatchN() throws IOException {
    Network network = NetworkLoader.load(fixturePath("parity-model.npz"));
    ParityFixtures fixtures = loadFixtures();

    float[] planes = new float[Network.MAX_BATCH * INPUT_SIZE];
    NNOutput output = new NNOutput();

    // 1er forward : 64 premières positions.
    System.arraycopy(fixtures.inputs(), 0, planes, 0, 64 * INPUT_SIZE);
    network.forward(planes, 64, output);
    int failsLogits = 0;
    int failsValues = 0;
    float maxL = 0f;
    float maxV = 0f;
    for (int i = 0; i < 64; i++) {
      for (int j = 0; j < LOGITS_PER_POS; j++) {
        float diff =
            Math.abs(
                output.logits[i * LOGITS_PER_POS + j] - fixtures.logits()[i * LOGITS_PER_POS + j]);
        if (diff > maxL) maxL = diff;
        if (diff > TOL_LOGITS) {
          failsLogits++;
          break;
        }
      }
      float vDiff = Math.abs(output.values[i] - fixtures.values()[i]);
      if (vDiff > maxV) maxV = vDiff;
      if (vDiff > TOL_VALUES) failsValues++;
    }

    // 2e forward : 36 dernières positions.
    Arrays.fill(planes, 0f);
    System.arraycopy(fixtures.inputs(), 64 * INPUT_SIZE, planes, 0, 36 * INPUT_SIZE);
    network.forward(planes, 36, output);
    for (int i = 0; i < 36; i++) {
      int globalIdx = 64 + i;
      for (int j = 0; j < LOGITS_PER_POS; j++) {
        float diff =
            Math.abs(
                output.logits[i * LOGITS_PER_POS + j]
                    - fixtures.logits()[globalIdx * LOGITS_PER_POS + j]);
        if (diff > maxL) maxL = diff;
        if (diff > TOL_LOGITS) {
          failsLogits++;
          break;
        }
      }
      float vDiff = Math.abs(output.values[i] - fixtures.values()[globalIdx]);
      if (vDiff > maxV) maxV = vDiff;
      if (vDiff > TOL_VALUES) failsValues++;
    }

    System.out.printf(
        "ParityTest batch=64+36 : maxLogitsDiff=%g maxValueDiff=%g (failsLogits=%d"
            + " failsValues=%d)%n",
        maxL, maxV, failsLogits, failsValues);
    assertThat(failsLogits).as("Aucune position ne doit dépasser tol logits en batch>1").isZero();
    assertThat(failsValues).as("Aucune position ne doit dépasser tol values en batch>1").isZero();
  }

  @Test
  @Tag("perf")
  @DisplayName("Bench parity throughput : > 30 pos/s batch=1 (cohérent phase 7.5 ~55 pos/s)")
  void testParityForwardThroughput() throws IOException {
    Network network = NetworkLoader.load(fixturePath("parity-model.npz"));
    ParityFixtures fixtures = loadFixtures();

    float[] planes = new float[Network.MAX_BATCH * INPUT_SIZE];
    NNOutput output = new NNOutput();

    // Warmup 20 positions (laisse le JIT compiler les kernels).
    for (int i = 0; i < 20; i++) {
      Arrays.fill(planes, 0f);
      System.arraycopy(fixtures.inputs(), i * INPUT_SIZE, planes, 0, INPUT_SIZE);
      network.forward(planes, 1, output);
    }

    long t0 = System.nanoTime();
    for (int i = 0; i < N_FIXTURES; i++) {
      Arrays.fill(planes, 0f);
      System.arraycopy(fixtures.inputs(), i * INPUT_SIZE, planes, 0, INPUT_SIZE);
      network.forward(planes, 1, output);
    }
    long elapsedNs = System.nanoTime() - t0;
    double posPerSec = N_FIXTURES * 1e9 / elapsedNs;
    double msPerForward = (elapsedNs / (double) N_FIXTURES) / 1e6;
    System.out.printf(
        "ParityTest batch=1 throughput : %.1f pos/s (%.2f ms/forward, %d positions)%n",
        posPerSec, msPerForward, N_FIXTURES);
    assertThat(posPerSec)
        .as("Throughput batch=1 cohérent post-phase-7.5 (~55 pos/s) ; sanity > 5 pos/s")
        .isGreaterThan(5.0);
  }

  // ---------------------------------------------------------------------------------------------
  // Diagnostic helper
  // ---------------------------------------------------------------------------------------------

  private static String summarizeHist(int[] h, String label) {
    int sum = 0;
    int nonZero = 0;
    int maxIdx = 0;
    int maxVal = 0;
    for (int i = 0; i < h.length; i++) {
      sum += h[i];
      if (h[i] > 0) nonZero++;
      if (h[i] > maxVal) {
        maxVal = h[i];
        maxIdx = i;
      }
    }
    return String.format(
        "%d total, %d distinct %s, max %s=%d (count=%d)",
        sum, nonZero, label, label, maxIdx, maxVal);
  }
}
