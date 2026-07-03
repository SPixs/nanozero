package org.nanozero.nn.kernels;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Convolution 1×1 NCHW = GEMM par pixel (cf. SPEC §5.6.2, §7.3).
 *
 * <p>Pour chaque position spatiale {@code (h, w)} d'un tenseur {@code [N, inC, 8, 8]}, calcule une
 * transformation linéaire indépendante {@code outC × inC} produisant {@code [N, outC, 8, 8]}. Pas
 * de padding, pas de voisinage, pas de noyau spatial — équivalent fonctionnel à {@code
 * Linear.applyBatch} appliqué pixel par pixel.
 *
 * <p>Cas d'usage NanoZero :
 *
 * <ul>
 *   <li>{@code policy_head} : 96 → 73 channels (queue scalaire active : 73 % 8 = 1)
 *   <li>{@code value_head} : 96 → 1 channel (queue scalaire domine, 0 itérations SIMD sur oc)
 * </ul>
 *
 * <p>Algorithme : pour chaque {@code (n, pos)} et chaque bloc SIMD de {@code LANES} output
 * channels, on accumule via FMA sur les input channels avec :
 *
 * <ul>
 *   <li>{@code inVec} : broadcast scalaire de {@code input[n, ic, h, w]}
 *   <li>{@code wVec} : <strong>gather</strong> des poids {@code weights[oc..oc+LANES-1, ic]}
 *       (espacés en mémoire de {@code inChannels} positions car layout row-major {@code
 *       [outC][inC]})
 *   <li>{@code accVec = inVec.fma(wVec, accVec)}
 * </ul>
 *
 * <p>L'écriture finale dans {@code output} est elle aussi <strong>scattered</strong> : les {@code
 * LANES} channels consécutifs sont espacés de {@code HW = 64} positions en mémoire (layout NCHW).
 * On utilise {@link FloatVector#intoArray(float[], int, int[], int)} avec un indexMap pré-calculé.
 *
 * <p>Note de perf : les opérations gather/scatter AVX2 sont notablement plus lentes que des loads
 * contigus (~10 cycles vs ~1 pour {@code vgatherdps}). Une optimisation possible est de réordonner
 * les poids au chargement vers un layout {@code [outC/LANES][inC][LANES]} qui rend les loads
 * contigus (cf. SPEC §7.3 fin de section). À traiter en phase 8 si le bench n'atteint pas la cible.
 *
 * <p><strong>Visibilité</strong> : publique pour permettre l'accès cross-package depuis {@link
 * org.nanozero.nn.Network}. Internal API uniquement (cf. §12.4).
 */
public final class Conv2D1x1 {

  /** Species SIMD préférée. */
  static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  static final int LANES = SPECIES.length();

  /** Taille spatiale H × W (8 × 8 fixé par §3.3). */
  private static final int HW = 64;

  /**
   * IndexMap statique pour le scatter de la sortie : lane {@code k} → offset {@code k * HW}. Ne
   * dépend ni de {@code inChannels} ni de {@code outChannels} (le stride entre channels NCHW
   * consécutifs est toujours {@code HW = 64}). Pré-calculé une fois à l'init pour préserver la
   * règle « zéro allocation en hot path » (cf. AGENTS.md, SPEC §1.3).
   */
  private static final int[] OUTPUT_INDEX_MAP = new int[LANES];

  static {
    for (int k = 0; k < LANES; k++) {
      OUTPUT_INDEX_MAP[k] = k * HW;
    }
  }

  private Conv2D1x1() {
    throw new AssertionError("Non-instantiable");
  }

  /**
   * Convolution 1×1 batched NCHW.
   *
   * @param input buffer {@code [batchSize, inChannels, 8, 8]} aplati ({@code length = batchSize *
   *     inChannels * 64})
   * @param weights buffer {@code [outChannels, inChannels]} row-major ({@code length = outChannels
   *     * inChannels})
   * @param bias buffer {@code [outChannels]}
   * @param output buffer {@code [batchSize, outChannels, 8, 8]} aplati (rempli en place)
   * @param inChannels nombre de channels d'entrée
   * @param outChannels nombre de channels de sortie
   * @param batchSize nombre de samples
   */
  public static void applyBatch(
      float[] input,
      float[] weights,
      float[] bias,
      float[] output,
      int inChannels,
      int outChannels,
      int batchSize) {
    // IndexMap pour gather des poids : lane k → weights[(oc+k) * inChannels + ic]
    // base = oc*inChannels + ic, indexMap[k] = k * inChannels. Dépend de inChannels donc local
    // (sera éliminé en phase 8 par le reorder des poids vers un layout contigu, cf. §7.3).
    int[] weightIndexMap = new int[LANES];
    for (int k = 0; k < LANES; k++) {
      weightIndexMap[k] = k * inChannels;
    }

    int outBound = (outChannels / LANES) * LANES;

    for (int n = 0; n < batchSize; n++) {
      int inputBase = n * inChannels * HW;
      int outputBase = n * outChannels * HW;

      for (int pos = 0; pos < HW; pos++) {
        int oc = 0;
        // Boucle SIMD principale : LANES output channels par itération
        for (; oc < outBound; oc += LANES) {
          FloatVector accVec = FloatVector.fromArray(SPECIES, bias, oc);
          for (int ic = 0; ic < inChannels; ic++) {
            float inScalar = input[inputBase + ic * HW + pos];
            int wBase = oc * inChannels + ic;
            FloatVector wVec = FloatVector.fromArray(SPECIES, weights, wBase, weightIndexMap, 0);
            FloatVector inVec = FloatVector.broadcast(SPECIES, inScalar);
            accVec = inVec.fma(wVec, accVec);
          }
          int outIdx = outputBase + oc * HW + pos;
          accVec.intoArray(output, outIdx, OUTPUT_INDEX_MAP, 0);
        }
        // Queue scalaire pour outChannels non multiple de LANES (par ex. 73 → 72 SIMD + 1 scalaire,
        // 1 → 0 SIMD + 1 scalaire pour value_head).
        for (; oc < outChannels; oc++) {
          float acc = bias[oc];
          int weightBase = oc * inChannels;
          for (int ic = 0; ic < inChannels; ic++) {
            acc += input[inputBase + ic * HW + pos] * weights[weightBase + ic];
          }
          output[outputBase + oc * HW + pos] = acc;
        }
      }
    }
  }
}
