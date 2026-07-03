package org.nanozero.nn;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.nanozero.board.GameState;
import org.nanozero.nn.internal.WeightsLayout;

/**
 * Tests d'intégration de {@link Network} (cf. SPEC §13 phase 7).
 *
 * <p>Critère phase 7 PERMISSIF : forward exécute sans crash, dimensions correctes, pas de NaN.
 * Parité numérique stricte vs PyTorch en phase 9.
 */
class NetworkTest {

  // ---------------------------------------------------------------------------------------------
  // Helpers : factory de Network avec poids random reproductibles
  // ---------------------------------------------------------------------------------------------

  /** Génère un float[length] aux valeurs N(0, sigma) reproductibles via Random(seed). */
  private static float[] randomGaussian(Random rng, int length, double sigma) {
    float[] out = new float[length];
    for (int i = 0; i < length; i++) {
      out[i] = (float) (rng.nextGaussian() * sigma);
    }
    return out;
  }

  /**
   * Construit un {@link Network} aux dimensions canoniques §3.3 avec poids random N(0, 0.02) et
   * biais à zéro. PRNG seed stable pour reproductibilité bit-à-bit.
   */
  private static Network testRandomNetwork(long seed) {
    Random rng = new Random(seed);
    double sigma = 0.02;

    // Input conv 119 → 96, kernel 3×3
    float[] inputConvW =
        WeightsLayout.reorderConv3x3(randomGaussian(rng, 96 * 119 * 9, sigma), 119, 96);
    float[] inputConvB = new float[96];

    // 8 blocs résiduels
    float[][] block1W = new float[8][];
    float[][] block1B = new float[8][];
    float[][] block2W = new float[8][];
    float[][] block2B = new float[8][];
    for (int i = 0; i < 8; i++) {
      block1W[i] = WeightsLayout.reorderConv3x3(randomGaussian(rng, 96 * 96 * 9, sigma), 96, 96);
      block1B[i] = new float[96];
      block2W[i] = WeightsLayout.reorderConv3x3(randomGaussian(rng, 96 * 96 * 9, sigma), 96, 96);
      block2B[i] = new float[96];
    }

    // Policy head 96 → 73 (NON reorderé, Conv2D1x1)
    float[] policyW = randomGaussian(rng, 73 * 96, sigma);
    float[] policyB = new float[73];

    // Value head conv 1×1 96 → 1
    float[] valueConvW = randomGaussian(rng, 1 * 96, sigma);
    float[] valueConvB = new float[1];

    // Value head FC 64 → 64
    float[] fc1W = randomGaussian(rng, 64 * 64, sigma);
    float[] fc1B = new float[64];

    // Value head FC 64 → 3 (WDL v1.5.0)
    float[] fc2W = randomGaussian(rng, 3 * 64, sigma);
    float[] fc2B = new float[3];

    NetworkMetadata meta =
        new NetworkMetadata("resnet8x96-v1", "0".repeat(64), 0, "test", "alphazero-119");
    return new NetworkVectorApi(
        meta,
        inputConvW,
        inputConvB,
        block1W,
        block1B,
        block2W,
        block2B,
        policyW,
        policyB,
        valueConvW,
        valueConvB,
        fc1W,
        fc1B,
        fc2W,
        fc2B);
  }

  /** Génère des planes aléatoires (uniformes [0,1] via nextFloat) pour batch=MAX_BATCH. */
  private static float[] randomPlanes(Random rng) {
    float[] planes = new float[Network.MAX_BATCH * 119 * 64];
    for (int i = 0; i < planes.length; i++) {
      planes[i] = rng.nextFloat();
    }
    return planes;
  }

  // ---------------------------------------------------------------------------------------------
  // Test 1 : forward exécute sans crash + dimensions + pas de NaN/Inf, value ∈ [-1, 1]
  // ---------------------------------------------------------------------------------------------

  @ParameterizedTest(name = "batchSize={0}")
  @ValueSource(ints = {1, 8, 32, 64})
  @DisplayName("forward random no-crash : pas de NaN/Inf, value ∈ [-1, 1] (post-tanh)")
  void forwardNoCrashRandom(int batchSize) {
    Network net = testRandomNetwork(0xC0FFEEBEEFL);
    NNOutput out = new NNOutput();
    Random rng = new Random(0xBADCAFEEL ^ batchSize);
    float[] planes = randomPlanes(rng);
    net.forward(planes, batchSize, out);
    // Fail-fast scan plutôt que ~300k assertThat (chacun allouant un message ; 300k × overhead =
    // ~30s sur batch=64). Une seule assertion finale agrège.
    int nonFiniteLogits = 0;
    for (int i = 0; i < batchSize * MoveEncoding.POLICY_INDICES; i++) {
      if (!Float.isFinite(out.logits[i])) {
        nonFiniteLogits++;
      }
    }
    int outOfRangeValues = 0;
    for (int n = 0; n < batchSize; n++) {
      float v = out.values[n];
      if (!Float.isFinite(v) || v < -1.0f || v > 1.0f) {
        outOfRangeValues++;
      }
    }
    assertThat(nonFiniteLogits).as("logits non-finis (batchSize=%d)", batchSize).isZero();
    assertThat(outOfRangeValues)
        .as("values non-finies ou hors [-1, 1] (batchSize=%d)", batchSize)
        .isZero();
  }

  // ---------------------------------------------------------------------------------------------
  // Test 2 : forward sur planes = 0, le réseau produit des outputs déterminés par les biais
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("forward planes=0 : pas de NaN, logits diversifiés (biais distincts)")
  void forwardZeroPlanes() {
    // Bias non-zero pour diversifier les logits, sigma faible pour éviter saturation tanh.
    Network net = testRandomNetworkWithBias(0xCAFEFACEL);
    NNOutput out = new NNOutput();
    float[] planes = new float[Network.MAX_BATCH * 119 * 64]; // tout 0
    net.forward(planes, 8, out);
    int nonFinite = 0;
    for (int i = 0; i < 8 * MoveEncoding.POLICY_INDICES; i++) {
      if (!Float.isFinite(out.logits[i])) {
        nonFinite++;
      }
    }
    for (int n = 0; n < 8; n++) {
      if (!Float.isFinite(out.values[n])) {
        nonFinite++;
      }
    }
    assertThat(nonFinite).isZero();
    // Les logits du sample 0 ne sont pas tous égaux (biais conv + fc apportent contributions
    // distinctes par plane).
    float ref = out.logits[0];
    boolean diverges = false;
    for (int li = 1; li < MoveEncoding.POLICY_INDICES; li++) {
      if (out.logits[li] != ref) {
        diverges = true;
        break;
      }
    }
    assertThat(diverges).as("logits ne doivent pas être tous égaux").isTrue();
  }

  /** Variante de testRandomNetwork avec biais non-zéro pour diversifier. */
  private static Network testRandomNetworkWithBias(long seed) {
    Random rng = new Random(seed);
    double sigma = 0.02;
    float[] inputConvW =
        WeightsLayout.reorderConv3x3(randomGaussian(rng, 96 * 119 * 9, sigma), 119, 96);
    float[] inputConvB = randomGaussian(rng, 96, sigma);
    float[][] block1W = new float[8][];
    float[][] block1B = new float[8][];
    float[][] block2W = new float[8][];
    float[][] block2B = new float[8][];
    for (int i = 0; i < 8; i++) {
      block1W[i] = WeightsLayout.reorderConv3x3(randomGaussian(rng, 96 * 96 * 9, sigma), 96, 96);
      block1B[i] = randomGaussian(rng, 96, sigma);
      block2W[i] = WeightsLayout.reorderConv3x3(randomGaussian(rng, 96 * 96 * 9, sigma), 96, 96);
      block2B[i] = randomGaussian(rng, 96, sigma);
    }
    float[] policyW = randomGaussian(rng, 73 * 96, sigma);
    float[] policyB = randomGaussian(rng, 73, sigma);
    float[] valueConvW = randomGaussian(rng, 1 * 96, sigma);
    float[] valueConvB = randomGaussian(rng, 1, sigma);
    float[] fc1W = randomGaussian(rng, 64 * 64, sigma);
    float[] fc1B = randomGaussian(rng, 64, sigma);
    float[] fc2W = randomGaussian(rng, 3 * 64, sigma); // WDL v1.5.0
    float[] fc2B = randomGaussian(rng, 3, sigma);
    NetworkMetadata meta =
        new NetworkMetadata("resnet8x96-v1", "0".repeat(64), 0, "test", "alphazero-119");
    return new NetworkVectorApi(
        meta,
        inputConvW,
        inputConvB,
        block1W,
        block1B,
        block2W,
        block2B,
        policyW,
        policyB,
        valueConvW,
        valueConvB,
        fc1W,
        fc1B,
        fc2W,
        fc2B);
  }

  // ---------------------------------------------------------------------------------------------
  // Test 3 : déterminisme bit-à-bit (deux appels successifs identiques)
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("forward quasi-déterministe : deux appels avec mêmes inputs → outputs ~identiques")
  void forwardDeterminism() {
    Network net = testRandomNetwork(0xDE7EA1L);
    NNOutput out1 = new NNOutput();
    NNOutput out2 = new NNOutput();
    Random rng = new Random(0xC0DEC0L);
    float[] planes = randomPlanes(rng);
    net.forward(planes, 32, out1);
    net.forward(planes, 32, out2);
    // Tolérance ~1e-5 : Math.tanh n'est pas garanti bit-exact entre appels (cf. javadoc Math :
    // "the Math methods are documented to NOT have this [reproducibility across runs] property").
    // Le JIT peut aussi recompiler les kernels SIMD entre les deux appels avec un ordre FMA
    // légèrement différent. Diff observée typique : ~1e-7 par valeur, largement sous la cible
    // §1.4 et §8.3 amendée. Fail-fast scan, une seule assertion finale.
    int divergent = 0;
    for (int i = 0; i < out1.logits.length; i++) {
      if (Math.abs(out1.logits[i] - out2.logits[i]) >= 1e-5f) {
        divergent++;
      }
    }
    for (int i = 0; i < out1.values.length; i++) {
      if (Math.abs(out1.values[i] - out2.values[i]) >= 1e-5f) {
        divergent++;
      }
    }
    assertThat(divergent).as("éléments divergents > 1e-5 entre 2 forwards identiques").isZero();
  }

  // ---------------------------------------------------------------------------------------------
  // Test 4 : forwardSingle ≡ forward(planes, 1, ...) sur position de départ
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("forwardSingle ≡ forward(planes, 1, ...) sur position de départ")
  void forwardSingleEquivalence() {
    Network net = testRandomNetwork(0x515ABEL);
    GameState gs = new GameState(); // startpos
    NNSingleResult result = net.forwardSingle(gs);

    float[] planes = new float[Network.MAX_BATCH * 119 * 64];
    gs.toPlanes(planes, 0, net.planeEncoder());
    NNOutput out = new NNOutput();
    net.forward(planes, 1, out);

    assertThat(result.logits()).hasSize(MoveEncoding.POLICY_INDICES);
    // Comparaison globale plutôt que 4672 assertThat individuels.
    float[] expectedLogits = new float[MoveEncoding.POLICY_INDICES];
    System.arraycopy(out.logits, 0, expectedLogits, 0, MoveEncoding.POLICY_INDICES);
    // Tolérance 1e-5 : forwardSingle utilise un scratch ThreadLocal différent de forward, le JIT
    // peut prendre un chemin légèrement divergent (cf. forwardDeterminism).
    int divergent = 0;
    for (int i = 0; i < MoveEncoding.POLICY_INDICES; i++) {
      if (Math.abs(result.logits()[i] - expectedLogits[i]) >= 1e-5f) {
        divergent++;
      }
    }
    assertThat(divergent).as("logits divergents forwardSingle vs forward(planes, 1)").isZero();
    assertThat(Math.abs(result.value() - out.values[0])).isLessThan(1e-5f);
  }

  // ---------------------------------------------------------------------------------------------
  // Test 5 : validation IAE
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("Validation arguments forward")
  class Validation {
    private final Network net = testRandomNetwork(0x12345L);
    private final NNOutput validOutput = new NNOutput();
    private final float[] validPlanes = new float[Network.MAX_BATCH * 119 * 64];

    @Test
    @DisplayName("planes null → IAE")
    void nullPlanes() {
      assertThatThrownBy(() -> net.forward(null, 1, validOutput))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("planes longueur invalide → IAE")
    void wrongPlanesLength() {
      assertThatThrownBy(() -> net.forward(new float[100], 1, validOutput))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @ParameterizedTest(name = "batchSize={0}")
    @ValueSource(ints = {-1, 0, 65, 100})
    @DisplayName("batchSize hors plage → IAE")
    void batchSizeOutOfRange(int badBatch) {
      assertThatThrownBy(() -> net.forward(validPlanes, badBatch, validOutput))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("output null → IAE")
    void nullOutput() {
      assertThatThrownBy(() -> net.forward(validPlanes, 1, null))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Test 6 : I-Net-2 thread safety (4 threads, NNOutput par thread, scratch ThreadLocal)
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("I-Net-2 : 4 threads × 5 forwards concurrents (batch=4), aucune corruption")
  void threadSafetyINet2() throws InterruptedException {
    Network net = testRandomNetwork(0xACADE3L);
    int threads = 4;
    int iters = 5; // batch=4, ~0.55s/forward → 5 iters × 4 threads parallèles ≈ 3 s wall clock
    CountDownLatch start = new CountDownLatch(1);
    CountDownLatch done = new CountDownLatch(threads);
    AtomicInteger anomalies = new AtomicInteger(0);

    for (int t = 0; t < threads; t++) {
      final int tid = t;
      Thread worker =
          new Thread(
              () -> {
                try {
                  start.await();
                  Random rng = new Random(0xDEADL + tid);
                  NNOutput out = new NNOutput();
                  for (int it = 0; it < iters; it++) {
                    float[] planes = randomPlanes(rng);
                    net.forward(planes, 4, out);
                    for (int n = 0; n < 4; n++) {
                      float v = out.values[n];
                      if (Float.isNaN(v) || Float.isInfinite(v) || v < -1.0f || v > 1.0f) {
                        anomalies.incrementAndGet();
                      }
                    }
                  }
                } catch (InterruptedException e) {
                  Thread.currentThread().interrupt();
                } finally {
                  done.countDown();
                }
              });
      worker.start();
    }
    start.countDown();
    done.await();
    assertThat(anomalies.get()).as("NaN/Inf/value hors [-1,1] détectées").isZero();
  }

  // ---------------------------------------------------------------------------------------------
  // Test 7 (CRUCIAL) : transposition NCHW → format §3.5.1 isolée et signature paramétrique
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName(
      "NetworkVectorApi.transposeNCHWtoLogits : signature paramétrique testable en isolation")
  class TransposeSignature {

    @Test
    @DisplayName("Signature dst[n*4672 + square*73 + plane] == src[n*73*64 + plane*64 + square]")
    void signaturePerElement() {
      // Pour chaque (n, plane, square), src[plane*64 + square] = plane * 1000 + square.
      // Vérifier dst[square*73 + plane] == plane * 1000 + square exactement.
      int batchSize = 3;
      float[] src = new float[batchSize * NetworkVectorApi.POLICY_PLANES * NetworkVectorApi.HW];
      for (int n = 0; n < batchSize; n++) {
        for (int plane = 0; plane < NetworkVectorApi.POLICY_PLANES; plane++) {
          for (int square = 0; square < NetworkVectorApi.HW; square++) {
            src[
                    n * NetworkVectorApi.POLICY_PLANES * NetworkVectorApi.HW
                        + plane * NetworkVectorApi.HW
                        + square] =
                n * 1_000_000f + plane * 1000f + square;
          }
        }
      }
      float[] dst = new float[batchSize * MoveEncoding.POLICY_INDICES];
      NetworkVectorApi.transposeNCHWtoLogits(src, dst, batchSize);

      for (int n = 0; n < batchSize; n++) {
        for (int square = 0; square < NetworkVectorApi.HW; square++) {
          for (int plane = 0; plane < NetworkVectorApi.POLICY_PLANES; plane++) {
            float expected = n * 1_000_000f + plane * 1000f + square;
            float actual = dst[n * MoveEncoding.POLICY_INDICES + square * 73 + plane];
            assertThat(actual)
                .as(
                    "dst[n=%d, square=%d, plane=%d] doit avoir signature %f",
                    n, square, plane, expected)
                .isEqualTo(expected);
          }
        }
      }
    }

    @Test
    @DisplayName("Tous les éléments écrits, aucune valeur sentinelle restante")
    void writesAllElements() {
      int batchSize = 2;
      float[] src = new float[batchSize * NetworkVectorApi.POLICY_PLANES * NetworkVectorApi.HW];
      Arrays.fill(src, 7.0f); // valeur reconnaissable
      float[] dst = new float[batchSize * MoveEncoding.POLICY_INDICES];
      Arrays.fill(dst, -99.0f); // sentinelle
      NetworkVectorApi.transposeNCHWtoLogits(src, dst, batchSize);
      for (int i = 0; i < dst.length; i++) {
        assertThat(dst[i]).as("dst[%d] doit avoir été écrit", i).isEqualTo(7.0f);
      }
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Test optionnel : bench @Tag("perf") indicatif
  // ---------------------------------------------------------------------------------------------

  @Test
  @Tag("perf")
  @DisplayName("Bench Network.forward(batch=64) — cible §9.1 ≥ 7000 pos/s indicatif")
  void benchForwardBatch64() {
    Network net = testRandomNetwork(0xBE1077L);
    NNOutput out = new NNOutput();
    Random rng = new Random(0xBEEFFEEDL);
    float[] planes = randomPlanes(rng);

    int warmupIters = 20;
    for (int i = 0; i < warmupIters; i++) {
      net.forward(planes, 64, out);
    }

    int measureIters = 50;
    long t0 = System.nanoTime();
    for (int i = 0; i < measureIters; i++) {
      net.forward(planes, 64, out);
    }
    long elapsedNs = System.nanoTime() - t0;
    double secs = elapsedNs / 1e9;
    double posPerSec = (double) measureIters * 64 / secs;
    double msPerForward = (elapsedNs / (double) measureIters) / 1_000_000.0;
    System.out.printf(
        "Network.forward(64) : %.0f pos/s (%.2f ms/forward, %d iters in %.3f s)%n",
        posPerSec, msPerForward, measureIters, secs);

    // Historique sur la machine de référence (AVX2, LANES=8) :
    //   - phase 7   (§7.2.3 gather/scatter par tuple) : ~7 pos/s (~1100 ms/forward)
    //   - phase 7.5 (§7.2.3-alt accVec en registre)   : ~55 pos/s (~1100 ms/forward batch=64)
    // Soit ~8× après le swap du pattern Conv2D3x3 (cf. queue d'amendements SPEC entrée 6
    // anticipée). Reste à ~127× sous la cible §9.1 ≥ 7000 pos/s (cible identifiée comme
    // physiquement non atteignable single-thread sur AVX2 — recalibrage prévu en queue
    // d'amendements). Phase 9 (parité PyTorch sur 100 positions) devient tractable : ~2 s/iter
    // au lieu de ~15 min.
    //
    // Borne défensive très permissive (> 1 pos/s = pas de blocage complet) ; le signal réel est
    // dans le print du throughput.
    assertThat(posPerSec)
        .as("Network.forward batch=64 throughput (post-swap phase 7.5 ; cible §9.1 à recalibrer)")
        .isGreaterThan(1.0);
  }

  // ---------------------------------------------------------------------------------------------
  // Phase 11 — bench informatif scaling par batchSize (scope A : profile + doc, pas d'optim)
  // ---------------------------------------------------------------------------------------------

  @Test
  @Tag("perf")
  @DisplayName("Scaling forward par batchSize : informatif (cf. docs/perf/PERF-MEASUREMENTS-nn.md)")
  void testForwardScalingByBatchSize() {
    Network net = testRandomNetwork(0xBA7C45CAL);
    NNOutput out = new NNOutput();
    Random rng = new Random(0xBEEFFEEDL);
    float[] planes = randomPlanes(rng);

    // Warmup : 50 forwards batch=64 pour amortir le JIT et le ThreadLocal scratch lazy.
    for (int i = 0; i < 50; i++) {
      net.forward(planes, 64, out);
    }

    int[] batchSizes = {1, 8, 32, 64};
    System.out.println("Forward scaling by batchSize (machine de référence + JaCoCo OFF requis) :");
    for (int bs : batchSizes) {
      // Plus d'iterations pour batch=1 (forward plus court individuellement, sample plus large
      // pour stabiliser la mesure ; pour batch=64 on amortit déjà avec 50 iters).
      int iters = (bs == 1) ? 200 : Math.max(50, 1000 / bs);
      long t0 = System.nanoTime();
      for (int i = 0; i < iters; i++) {
        net.forward(planes, bs, out);
      }
      long elapsedNs = System.nanoTime() - t0;
      double avgMs = elapsedNs / 1_000_000.0 / iters;
      double posPerSec = bs * iters * 1e9 / elapsedNs;
      System.out.printf(
          "  batch=%2d : %7.2f ms/forward, %7.1f pos/s (%d iter)%n", bs, avgMs, posPerSec, iters);
    }
    // Pas d'assertion de cible : phase 11 SCOPE A est informatif. Un signal réel d'anomalie
    // serait géré par benchForwardBatch64 ci-dessus (borne défensive > 1 pos/s).
  }
}
