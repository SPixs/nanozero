package org.nanozero.nn.kernels;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Convolution 3×3 NCHW avec padding 1, kernel principal du module {@code nanozero-nn} (cf. SPEC
 * §5.6.1, §7.2). Exécuté 17 fois par forward (1 input conv + 8 blocs × 2 conv) ; chemin chaud.
 *
 * <p>Les poids DOIVENT être pré-réordonnés au layout {@code [outC/LANES, inC, 9, LANES]} via {@code
 * WeightsLayout.reorderConv3x3} (cf. §7.2.4). Ce reorder rend le chargement vectoriel {@code wVec =
 * weights[oc..oc+LANES-1, ic, kh, kw]} contigu en mémoire — sans cela, le kernel tomberait sur du
 * gather/scatter à 10× le coût (cf. les ~2.7 GFlops mesurés en phase 4).
 *
 * <p>Algorithme conforme §7.2.3 variante "accVec persistant en registre" (cf. queue d'amendements
 * SPEC entrée 6, anticipée pré-phase 8) :
 *
 * <ol>
 *   <li>Boucle externe sur {@code (oc_block, oh, ow)} ; {@code accVec} initialisé en registre via
 *       chargement contigu de {@code bias[oc_block * LANES..oc_block * LANES + LANES]}.
 *   <li>Boucle interne sur {@code (ic, k)} ; pour chaque position valide on émet une FMA {@code
 *       accVec = inVec.fma(wVec, accVec)}. {@code accVec} reste en registre pendant toute la
 *       réduction sur {@code inChannels × 9} contributions.
 *   <li>Un seul scatter via {@code OUTPUT_INDEX_MAP} par tuple {@code (oc_block, oh, ow)} en fin de
 *       réduction. Le pattern précédent émettait {@code inChannels × 9} couples gather+scatter par
 *       sortie, dominés par le coût ~10 cycles de chaque scattered access sur AVX2 ; ce coût est
 *       quasi éliminé.
 * </ol>
 *
 * <p>L'ordre d'accumulation par élément de sortie est {@code (ic outer, k inner)}, identique à la
 * variante précédente. Les opérandes et la séquence de FMA sont bit-pour-bit identiques, à
 * tolérance vectorisation près.
 *
 * <p>Queue scalaire pour {@code outChannels} non-multiples de {@code LANES} : la dernière tranche
 * partielle est traitée channel-par-channel, en lisant le poids via sa lane dans le bloc partiel
 * réordonné. <strong>Le biais y est initialisé explicitement</strong> en début de réduction
 * (l'output buffer n'est plus pré-rempli en amont par une étape "init bias").
 *
 * <p><strong>Zéro allocation hot path</strong> : {@link #OUTPUT_INDEX_MAP} en {@code static final}.
 * Aucune allocation par appel.
 *
 * <p><strong>Visibilité</strong> : SPEC §12.4 prescrit "package-private". Sans {@code
 * module-info.java}, l'accès cross-package depuis {@link org.nanozero.nn.Network} (root package)
 * impose de publier la classe et les méthodes {@code apply*}. Internal API uniquement — non exposée
 * à l'extérieur du module. Note pour amendement §12.4 conjointement avec §12.3.
 */
public final class Conv2D3x3 {

  /** Species SIMD préférée. DOIT correspondre à celle de {@code WeightsLayout}. */
  static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  static final int LANES = SPECIES.length();

  /** Taille spatiale H × W (8 × 8 fixé par §3.3). */
  private static final int HW = 64;

  /** Largeur spatiale (8). */
  private static final int W = 8;

  /** Hauteur spatiale (8). */
  private static final int H = 8;

  /** Nombre de positions du noyau Conv 3×3. */
  private static final int K = 9;

  /**
   * IndexMap statique pour gather/scatter de la sortie : lane {@code k} → offset {@code k * HW}.
   * Stride NCHW entre output channels consécutifs toujours {@code HW = 64}, indépendant des
   * dimensions du tenseur.
   */
  private static final int[] OUTPUT_INDEX_MAP = new int[LANES];

  static {
    for (int k = 0; k < LANES; k++) {
      OUTPUT_INDEX_MAP[k] = k * HW;
    }
  }

  private Conv2D3x3() {
    throw new AssertionError("Non-instantiable");
  }

  /**
   * Convolution 3×3 sur un seul échantillon ({@code batchSize = 1}) (cf. §5.6.1).
   *
   * @param input buffer {@code [inChannels × 64]} aplati NCHW (1, inC, 8, 8)
   * @param weightsReordered poids pré-réordonnés via {@code WeightsLayout.reorderConv3x3}, layout
   *     {@code [outC/LANES, inC, 9, LANES]}
   * @param bias buffer {@code [outChannels]}
   * @param output buffer {@code [outChannels × 64]} aplati NCHW (rempli en place)
   * @param inChannels nombre de channels d'entrée
   * @param outChannels nombre de channels de sortie
   */
  public static void apply(
      float[] input,
      float[] weightsReordered,
      float[] bias,
      float[] output,
      int inChannels,
      int outChannels) {
    applyImpl(input, 0, weightsReordered, bias, output, 0, inChannels, outChannels);
  }

  /**
   * Convolution 3×3 batched (cf. §5.6.1, §7.2.5 option A).
   *
   * @param input buffer {@code [batchSize × inChannels × 64]} aplati NCHW
   * @param weightsReordered poids pré-réordonnés
   * @param bias buffer {@code [outChannels]}
   * @param output buffer {@code [batchSize × outChannels × 64]} aplati NCHW (rempli en place)
   * @param inChannels nombre de channels d'entrée
   * @param outChannels nombre de channels de sortie
   * @param batchSize nombre de samples
   */
  public static void applyBatch(
      float[] input,
      float[] weightsReordered,
      float[] bias,
      float[] output,
      int inChannels,
      int outChannels,
      int batchSize) {
    int inStride = inChannels * HW;
    int outStride = outChannels * HW;
    for (int n = 0; n < batchSize; n++) {
      applyImpl(
          input,
          n * inStride,
          weightsReordered,
          bias,
          output,
          n * outStride,
          inChannels,
          outChannels);
    }
  }

  /**
   * Cœur de l'algorithme §7.2.3 variante "accVec persistant en registre", factorisé pour {@link
   * #apply} et {@link #applyBatch} avec offsets de base dans {@code input} et {@code output}.
   *
   * <p>Boucle externe {@code (oc_block, oh, ow)} ; {@code accVec} en registre pendant toute la
   * réduction {@code (ic, k)} ; un seul scatter par tuple. Queue scalaire identique mais avec init
   * bias explicite (l'output buffer n'est plus pré-rempli).
   */
  private static void applyImpl(
      float[] input,
      int inputBase,
      float[] weightsReordered,
      float[] bias,
      float[] output,
      int outputBase,
      int inChannels,
      int outChannels) {
    int blocksFull = outChannels / LANES;
    int outBound = blocksFull * LANES;
    int blockStride = inChannels * K * LANES;
    int icKLanes = K * LANES;

    // Boucle SIMD principale : LANES output channels par bloc, accVec persistant en registre.
    for (int ocBlock = 0; ocBlock < blocksFull; ocBlock++) {
      int wBlockBase = ocBlock * blockStride;
      int outBlockBase = outputBase + ocBlock * LANES * HW;
      int biasBase = ocBlock * LANES;
      for (int oh = 0; oh < H; oh++) {
        for (int ow = 0; ow < W; ow++) {
          // Init accVec avec le bias (LANES floats contigus).
          FloatVector accVec = FloatVector.fromArray(SPECIES, bias, biasBase);
          // Réduction sur (ic, k).
          for (int ic = 0; ic < inChannels; ic++) {
            int inputChanBase = inputBase + ic * HW;
            int wIcBase = wBlockBase + ic * icKLanes;
            for (int k = 0; k < K; k++) {
              int kh = k / 3;
              int ih = oh + kh - 1;
              if (ih < 0 || ih >= H) {
                continue;
              }
              int kw = k - 3 * kh;
              int iw = ow + kw - 1;
              if (iw < 0 || iw >= W) {
                continue;
              }
              float inScalar = input[inputChanBase + ih * W + iw];
              FloatVector wVec =
                  FloatVector.fromArray(SPECIES, weightsReordered, wIcBase + k * LANES);
              FloatVector inVec = FloatVector.broadcast(SPECIES, inScalar);
              accVec = inVec.fma(wVec, accVec);
            }
          }
          // Un seul scatter par (ocBlock, oh, ow).
          accVec.intoArray(output, outBlockBase + oh * W + ow, OUTPUT_INDEX_MAP, 0);
        }
      }
    }

    // Queue scalaire : output channels au-delà du dernier bloc complet (outBound..outChannels).
    // Output buffer dimensionné strictement à outChannels × HW (cf. §5.6.1) ; le scatter SIMD
    // déborderait hors du buffer pour le bloc partiel. On lit les poids via leur lane individuelle
    // dans le buffer réordonné, et on initialise le biais explicitement (l'étape "init bias"
    // préalable n'existe plus dans ce pattern).
    int wPartialBase = blocksFull * blockStride;
    for (int oc = outBound; oc < outChannels; oc++) {
      int laneInBlock = oc - outBound;
      int outChanBase = outputBase + oc * HW;
      float biasVal = bias[oc];
      for (int oh = 0; oh < H; oh++) {
        for (int ow = 0; ow < W; ow++) {
          float acc = biasVal;
          for (int ic = 0; ic < inChannels; ic++) {
            int inputChanBase = inputBase + ic * HW;
            int wIcBase = wPartialBase + ic * icKLanes;
            for (int k = 0; k < K; k++) {
              int kh = k / 3;
              int ih = oh + kh - 1;
              if (ih < 0 || ih >= H) {
                continue;
              }
              int kw = k - 3 * kh;
              int iw = ow + kw - 1;
              if (iw < 0 || iw >= W) {
                continue;
              }
              acc +=
                  input[inputChanBase + ih * W + iw]
                      * weightsReordered[wIcBase + k * LANES + laneInBlock];
            }
          }
          output[outChanBase + oh * W + ow] = acc;
        }
      }
    }
  }
}
