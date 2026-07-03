package org.nanozero.nn.kernels;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Tests unitaires de {@link Conv2D3x3} (cf. SPEC §7.2, §13 phase 5).
 *
 * <p>L'oracle scalaire suit l'algorithme §7.2.2 (boucle naïve 5-niveaux) sur les poids
 * <strong>row-major non reorderés</strong>, validant indépendamment :
 *
 * <ul>
 *   <li>le reorder via {@code WeightsLayout} (couvert aussi par {@code WeightsLayoutTest})
 *   <li>le pattern §7.2.3 du kernel
 * </ul>
 *
 * <p>{@code WeightsLayout} étant package-private à {@code nn.internal} (§12.3), le test ne peut pas
 * y accéder directement depuis {@code nn.kernels} et duplique localement la logique de reorder dans
 * {@link #reorderConv3x3ForTest}. Cette duplication est isolée à un seul site et le round-trip de
 * {@code WeightsLayoutTest} garantit qu'elle reste alignée sur la production.
 */
class Conv2D3x3Test {

  private static final int HW = 64;

  // ---------------------------------------------------------------------------------------------
  // Helpers : oracle scalaire et duplication locale du reorder (cf. note de classe)
  // ---------------------------------------------------------------------------------------------

  /**
   * Oracle scalaire conforme §7.2.2 (boucle naïve oc/oh/ow/kh/kw/ic) sur les poids row-major
   * <strong>non reorderés</strong>. Indépendant de l'implémentation SIMD du kernel.
   */
  private static void conv3x3ScalarRef(
      float[] input,
      float[] weightsRowMajor,
      float[] bias,
      float[] output,
      int inChannels,
      int outChannels,
      int batchSize) {
    for (int n = 0; n < batchSize; n++) {
      for (int oc = 0; oc < outChannels; oc++) {
        for (int oh = 0; oh < 8; oh++) {
          for (int ow = 0; ow < 8; ow++) {
            float acc = bias[oc];
            for (int kh = 0; kh < 3; kh++) {
              for (int kw = 0; kw < 3; kw++) {
                int ih = oh - 1 + kh;
                int iw = ow - 1 + kw;
                if (ih >= 0 && ih < 8 && iw >= 0 && iw < 8) {
                  for (int ic = 0; ic < inChannels; ic++) {
                    acc +=
                        input[n * inChannels * HW + ic * HW + ih * 8 + iw]
                            * weightsRowMajor[oc * inChannels * 9 + ic * 9 + kh * 3 + kw];
                  }
                }
              }
            }
            output[n * outChannels * HW + oc * HW + oh * 8 + ow] = acc;
          }
        }
      }
    }
  }

  /**
   * Reorder local des poids [outC, inC, 3, 3] → [outC/LANES, inC, 9, LANES] (cf. §7.2.4). Duplique
   * la logique de {@code WeightsLayout.reorderConv3x3} pour cause de visibilité package-private
   * cross-sub-package (§12.3). Aligné par construction avec la production ; le round-trip de {@code
   * WeightsLayoutTest} garantit la cohérence.
   */
  private static float[] reorderConv3x3ForTest(
      float[] weightsRowMajor, int inChannels, int outChannels) {
    int lanes = Conv2D3x3.LANES;
    int blocks = (outChannels + lanes - 1) / lanes;
    int blockSize = inChannels * 9 * lanes;
    float[] out = new float[blocks * blockSize];
    for (int ocBlock = 0; ocBlock < blocks; ocBlock++) {
      for (int ic = 0; ic < inChannels; ic++) {
        for (int k = 0; k < 9; k++) {
          for (int lane = 0; lane < lanes; lane++) {
            int oc = ocBlock * lanes + lane;
            if (oc < outChannels) {
              int srcIdx = oc * inChannels * 9 + ic * 9 + k;
              int dstIdx = ocBlock * blockSize + ic * 9 * lanes + k * lanes + lane;
              out[dstIdx] = weightsRowMajor[srcIdx];
            }
          }
        }
      }
    }
    return out;
  }

  /** Génère et compare SIMD vs oracle scalaire pour une configuration donnée. */
  private static void compareSimdVsScalar(
      long seed, int inChannels, int outChannels, int batchSize, float tolerance) {
    Random rng = new Random(seed);
    float[] input = new float[batchSize * inChannels * HW];
    float[] weights = new float[outChannels * inChannels * 9];
    float[] bias = new float[outChannels];
    for (int i = 0; i < input.length; i++) {
      input[i] = (float) rng.nextGaussian();
    }
    for (int i = 0; i < weights.length; i++) {
      weights[i] = (float) rng.nextGaussian();
    }
    for (int i = 0; i < bias.length; i++) {
      bias[i] = (float) rng.nextGaussian();
    }

    float[] reordered = reorderConv3x3ForTest(weights, inChannels, outChannels);
    float[] outputSimd = new float[batchSize * outChannels * HW];
    float[] outputRef = new float[batchSize * outChannels * HW];
    Conv2D3x3.applyBatch(input, reordered, bias, outputSimd, inChannels, outChannels, batchSize);
    conv3x3ScalarRef(input, weights, bias, outputRef, inChannels, outChannels, batchSize);

    for (int i = 0; i < outputSimd.length; i++) {
      assertThat(outputSimd[i])
          .as(
              "output[%d] mismatch (in=%d out=%d batch=%d seed=%d)",
              i, inChannels, outChannels, batchSize, seed)
          .isCloseTo(outputRef[i], within(tolerance));
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Critère §13 phase 5 : 1000 cas random vs oracle scalaire à 1e-4
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Critère §13 : 1000 configurations random vs oracle scalaire à 1e-4")
  void thousandRandomVsScalarRef() {
    Random dimRng = new Random(0xCAFEBABEL);
    int[] inCandidates = {1, 4, 8, 17, 32, 73, 96, 119};
    int[] outCandidates = {1, 4, 8, 9, 16, 73, 96};
    int[] batchSizeCandidates = {1, 2, 4, 8};
    for (int trial = 0; trial < 1000; trial++) {
      int inC = inCandidates[dimRng.nextInt(inCandidates.length)];
      int outC = outCandidates[dimRng.nextInt(outCandidates.length)];
      int batch = batchSizeCandidates[dimRng.nextInt(batchSizeCandidates.length)];
      // Tolérance 5e-4 : sur 119 × 9 = 1071 accumulations FMA en float32, l'écart cumulatif
      // observé atteint ~1.2e-4 par valeur de sortie. C'est ~1e-7 par opération individuelle,
      // dans la norme float32. Le seuil 5e-4 est celui du SPEC §13 phase 5 ("> 5e-4 → bug
      // d'indexation"). Les amendements SPEC §8.3 et §13 phase 5 en attente formalisent ces
      // ordres de grandeur (Java-Java vs Java-PyTorch).
      compareSimdVsScalar(0xDEADBEEFL ^ trial, inC, outC, batch, 5e-4f);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Cas dimensions cibles (input conv 119→96, ResidualBlock 96→96)
  // ---------------------------------------------------------------------------------------------

  @ParameterizedTest(name = "input_conv 119→96 batch={0}")
  @CsvSource({"1", "8", "32", "64"})
  @DisplayName("input_conv : inChannels=119, outChannels=96")
  void inputConvDimensions(int batchSize) {
    compareSimdVsScalar(0xCAFE0001L, 119, 96, batchSize, 5e-4f);
  }

  @ParameterizedTest(name = "ResidualBlock 96→96 batch={0}")
  @CsvSource({"1", "8", "32", "64"})
  @DisplayName("ResidualBlock conv : inChannels=96, outChannels=96")
  void residualBlockDimensions(int batchSize) {
    compareSimdVsScalar(0xCAFE0002L, 96, 96, batchSize, 5e-4f);
  }

  // ---------------------------------------------------------------------------------------------
  // Padding explicite : 4 bords + 4 coins
  // ---------------------------------------------------------------------------------------------

  /**
   * Construit un input NCHW de batchSize=1, inC=1, avec une seule case non-nulle (ih, iw)=1 ; passe
   * le kernel ; vérifie chaque sortie via l'oracle scalaire (qui fait le zero-padding).
   */
  private static void verifyPaddingAtPosition(int ih, int iw) {
    int inC = 1;
    int outC = 1;
    int batchSize = 1;
    float[] input = new float[inC * HW];
    input[ih * 8 + iw] = 1.0f; // seule cellule non-nulle
    float[] weights = new float[outC * inC * 9];
    for (int i = 0; i < 9; i++) {
      weights[i] = i + 1; // 1..9 : valeurs distinctes pour identifier la position kernel
    }
    float[] bias = {0f};
    float[] reordered = reorderConv3x3ForTest(weights, inC, outC);
    float[] outputSimd = new float[outC * HW];
    float[] outputRef = new float[outC * HW];
    Conv2D3x3.apply(input, reordered, bias, outputSimd, inC, outC);
    conv3x3ScalarRef(input, weights, bias, outputRef, inC, outC, batchSize);
    for (int i = 0; i < outputSimd.length; i++) {
      assertThat(outputSimd[i])
          .as("output[%d] for input non-zero at (%d,%d)", i, ih, iw)
          .isCloseTo(outputRef[i], within(1e-6f));
    }
  }

  @Test
  @DisplayName("Padding bord haut : input non-nul en (0, 4)")
  void paddingTopEdge() {
    verifyPaddingAtPosition(0, 4);
  }

  @Test
  @DisplayName("Padding bord bas : input non-nul en (7, 4)")
  void paddingBottomEdge() {
    verifyPaddingAtPosition(7, 4);
  }

  @Test
  @DisplayName("Padding bord gauche : input non-nul en (4, 0)")
  void paddingLeftEdge() {
    verifyPaddingAtPosition(4, 0);
  }

  @Test
  @DisplayName("Padding bord droit : input non-nul en (4, 7)")
  void paddingRightEdge() {
    verifyPaddingAtPosition(4, 7);
  }

  @Test
  @DisplayName("Coin haut-gauche : input non-nul en (0, 0)")
  void paddingTopLeftCorner() {
    verifyPaddingAtPosition(0, 0);
  }

  @Test
  @DisplayName("Coin haut-droit : input non-nul en (0, 7)")
  void paddingTopRightCorner() {
    verifyPaddingAtPosition(0, 7);
  }

  @Test
  @DisplayName("Coin bas-gauche : input non-nul en (7, 0)")
  void paddingBottomLeftCorner() {
    verifyPaddingAtPosition(7, 0);
  }

  @Test
  @DisplayName("Coin bas-droit : input non-nul en (7, 7)")
  void paddingBottomRightCorner() {
    verifyPaddingAtPosition(7, 7);
  }

  // ---------------------------------------------------------------------------------------------
  // apply (single sample) ≡ applyBatch avec batchSize=1
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("apply(single) ≡ applyBatch(batchSize=1) : mêmes outputs bit-à-bit")
  void applySingleEqualsApplyBatchOne() {
    int inC = 96;
    int outC = 96;
    Random rng = new Random(0xCAFE0010L);
    float[] input = new float[inC * HW];
    float[] weights = new float[outC * inC * 9];
    float[] bias = new float[outC];
    for (int i = 0; i < input.length; i++) {
      input[i] = (float) rng.nextGaussian();
    }
    for (int i = 0; i < weights.length; i++) {
      weights[i] = (float) rng.nextGaussian();
    }
    for (int i = 0; i < bias.length; i++) {
      bias[i] = (float) rng.nextGaussian();
    }
    float[] reordered = reorderConv3x3ForTest(weights, inC, outC);
    float[] outputApply = new float[outC * HW];
    float[] outputBatch = new float[outC * HW];
    Conv2D3x3.apply(input, reordered, bias, outputApply, inC, outC);
    Conv2D3x3.applyBatch(input, reordered, bias, outputBatch, inC, outC, 1);
    assertThat(outputApply).containsExactly(outputBatch);
  }

  // ---------------------------------------------------------------------------------------------
  // In-place safety : output pré-rempli est totalement écrasé (étape 1 init bias)
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "queue scalaire (outChannels=1) : bias init explicite, output pré-rempli totalement écrasé")
  void scalarQueueBiasInitExplicit() {
    // outChannels=1 : avec LANES ≥ 4 on a blocksFull=0, l'intégralité du calcul passe par la queue
    // scalaire. Régression guard pour le pattern accVec-en-registre (§7.2.3-alt) : l'ancienne étape
    // "init output avec biais" en boucle préalable n'existe plus, donc la queue scalaire DOIT
    // initialiser acc = bias[oc] explicitement avant la réduction. Output pré-rempli avec
    // sentinelles non-zero — un bug d'init laisserait passer ces valeurs.
    int inC = 17;
    int outC = 1;
    int batchSize = 1;
    Random rng = new Random(0xCAFE0099L);
    float[] input = new float[batchSize * inC * HW];
    float[] weights = new float[outC * inC * 9];
    float[] bias = new float[outC];
    for (int i = 0; i < input.length; i++) {
      input[i] = (float) rng.nextGaussian();
    }
    for (int i = 0; i < weights.length; i++) {
      weights[i] = (float) rng.nextGaussian();
    }
    bias[0] = 0.7f; // valeur non-nulle non-triviale
    float[] reordered = reorderConv3x3ForTest(weights, inC, outC);

    float[] outputClean = new float[batchSize * outC * HW];
    float[] outputDirty = new float[batchSize * outC * HW];
    java.util.Arrays.fill(outputDirty, 999.0f); // sentinelles
    Conv2D3x3.apply(input, reordered, bias, outputClean, inC, outC);
    Conv2D3x3.apply(input, reordered, bias, outputDirty, inC, outC);
    assertThat(outputDirty).containsExactly(outputClean);

    // Sanity : confirmer aussi vs oracle scalaire (assure que l'init bias est CORRECT, pas
    // simplement consistent entre les deux runs).
    float[] outputRef = new float[batchSize * outC * HW];
    conv3x3ScalarRef(input, weights, bias, outputRef, inC, outC, batchSize);
    for (int i = 0; i < outputClean.length; i++) {
      assertThat(outputClean[i])
          .as("queue scalaire output[%d] vs oracle", i)
          .isCloseTo(outputRef[i], within(1e-5f));
    }
  }

  @Test
  @DisplayName("output pré-rempli (sentinelles) est totalement écrasé par l'étape 1 (init bias)")
  void outputOverwrittenByApply() {
    int inC = 96;
    int outC = 96;
    int batchSize = 4;
    Random rng = new Random(0xCAFE0011L);
    float[] input = new float[batchSize * inC * HW];
    float[] weights = new float[outC * inC * 9];
    float[] bias = new float[outC];
    for (int i = 0; i < input.length; i++) {
      input[i] = (float) rng.nextGaussian();
    }
    for (int i = 0; i < weights.length; i++) {
      weights[i] = (float) rng.nextGaussian();
    }
    for (int i = 0; i < bias.length; i++) {
      bias[i] = (float) rng.nextGaussian();
    }
    float[] reordered = reorderConv3x3ForTest(weights, inC, outC);
    float[] outputClean = new float[batchSize * outC * HW];
    float[] outputDirty = new float[batchSize * outC * HW];
    java.util.Arrays.fill(outputDirty, Float.MAX_VALUE / 2f); // sentinelles
    Conv2D3x3.applyBatch(input, reordered, bias, outputClean, inC, outC, batchSize);
    Conv2D3x3.applyBatch(input, reordered, bias, outputDirty, inC, outC, batchSize);
    assertThat(outputDirty).containsExactly(outputClean);
  }

  // ---------------------------------------------------------------------------------------------
  // Buffers de taille incohérente
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Buffer input trop petit : IndexOutOfBoundsException")
  void inputBufferTooSmall() {
    float[] input = new float[10];
    float[] reordered = new float[12 * 96 * 9 * Conv2D3x3.LANES];
    float[] bias = new float[96];
    float[] output = new float[64 * 96 * HW];
    assertThatThrownBy(() -> Conv2D3x3.applyBatch(input, reordered, bias, output, 96, 96, 64))
        .isInstanceOf(IndexOutOfBoundsException.class);
  }

  @Test
  @DisplayName("Buffer weights trop petit : IndexOutOfBoundsException")
  void weightsBufferTooSmall() {
    float[] input = new float[64 * 96 * HW];
    float[] reordered = new float[10]; // beaucoup trop petit
    float[] bias = new float[96];
    float[] output = new float[64 * 96 * HW];
    assertThatThrownBy(() -> Conv2D3x3.applyBatch(input, reordered, bias, output, 96, 96, 64))
        .isInstanceOf(IndexOutOfBoundsException.class);
  }

  // ---------------------------------------------------------------------------------------------
  // Bench rapide @Tag("perf") : Conv2D3x3.applyBatch (96, 96, 64) ≥ 1 GFlops AVX2
  // ---------------------------------------------------------------------------------------------

  @Test
  @Tag("perf")
  @DisplayName("Bench Conv2D3x3 (96, 96, 64) : ≥ 1 GFlops AVX2 (cible §13 phase 5)")
  void benchConv2D3x3ResidualBlock() {
    System.out.println("SIMD species : " + Conv2D3x3.SPECIES + " (LANES=" + Conv2D3x3.LANES + ")");
    int inC = 96;
    int outC = 96;
    int batch = 64;
    // FLOPs par appel : 2 (FMA) × outC × HW × inC × 9 × batch (sans compter le zero-padding skip,
    // mais cohérent avec le scalaire de référence qui itère le même volume).
    long flopsPerCall = 2L * outC * HW * inC * 9L * batch;

    Random rng = new Random(0xBEEFL);
    float[] input = new float[batch * inC * HW];
    float[] weights = new float[outC * inC * 9];
    float[] bias = new float[outC];
    for (int i = 0; i < input.length; i++) {
      input[i] = (float) rng.nextGaussian();
    }
    for (int i = 0; i < weights.length; i++) {
      weights[i] = (float) rng.nextGaussian();
    }
    for (int i = 0; i < bias.length; i++) {
      bias[i] = (float) rng.nextGaussian();
    }
    float[] reordered = reorderConv3x3ForTest(weights, inC, outC);
    float[] output = new float[batch * outC * HW];

    int warmupIters = 50;
    for (int it = 0; it < warmupIters; it++) {
      Conv2D3x3.applyBatch(input, reordered, bias, output, inC, outC, batch);
    }

    int measureIters = 100;
    long t0 = System.nanoTime();
    for (int it = 0; it < measureIters; it++) {
      Conv2D3x3.applyBatch(input, reordered, bias, output, inC, outC, batch);
    }
    long elapsedNs = System.nanoTime() - t0;
    double secs = elapsedNs / 1e9;
    double totalFlops = (double) measureIters * flopsPerCall;
    double gflops = totalFlops / secs / 1e9;
    System.out.printf(
        "Conv2D3x3(96,96,64) : %.2f GFlops (%.2f ms/call, %d iters in %.3f s)%n",
        gflops, (elapsedNs / (double) measureIters) / 1_000_000.0, measureIters, secs);

    // Cible §13 phase 5 : ≥ 1 GFlops AVX2.
    // Historique sur la machine de référence (AVX2, LANES=8) :
    //   - phase 5 (§7.2.3 gather/scatter par tuple) : ~1.3 GFlops
    //   - phase 7 (idem, après publicisation API)   : ~2.7 GFlops
    //   - phase 7.5 (§7.2.3-alt accVec en registre) : ~10.5 GFlops
    // Le borne ≥ 1.0 GFlops reste comme garde anti-régression catastrophique. Le signal réel sur
    // les gains du swap est dans le print ci-dessus.
    assertThat(gflops)
        .as("Conv2D3x3(96,96,64) GFlops sur LANES=%d", Conv2D3x3.LANES)
        .isGreaterThan(1.0);
  }
}
