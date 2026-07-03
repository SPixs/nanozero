package org.nanozero.nn.kernels;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/**
 * Couche fully-connected (GEMV avec accumulation FMA, cf. SPEC §5.6.3, §7.4, §7.1.3).
 *
 * <p>Pour chaque sample du batch et chaque output feature {@code o}, calcule :
 *
 * <pre>
 *   output[n*outFeatures + o] = sum_{i=0..inFeatures-1} input[n*inFeatures + i] * weights[o*inFeatures + i] + bias[o]
 * </pre>
 *
 * <p>Layout {@code weights} row-major {@code [outFeatures][inFeatures]} : pour un output feature
 * fixé, les coefficients d'entrée sont contigus en mémoire — chargement vectoriel optimal.
 *
 * <p>Pattern d'accumulation : un accumulateur SIMD {@link FloatVector} maintenu dans un registre
 * pendant la traversée des input features, mis à jour par {@link
 * FloatVector#fma(jdk.incubator.vector.Vector, jdk.incubator.vector.Vector)} (multiply-add
 * fusionné, une seule instruction CPU sur AVX2+/NEON, meilleure précision numérique). En fin de
 * boucle SIMD, les lanes de l'accumulateur sont réduites par {@link VectorOperators#ADD}, puis la
 * queue scalaire ajoute les éléments restants ({@code inFeatures % LANES}).
 *
 * <p><strong>Visibilité</strong> : publique pour permettre l'accès cross-package depuis {@link
 * org.nanozero.nn.Network}. Internal API uniquement (cf. §12.4).
 */
public final class Linear {

  /** Species SIMD préférée. */
  static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  static final int LANES = SPECIES.length();

  private Linear() {
    throw new AssertionError("Non-instantiable");
  }

  /**
   * Applique la couche linéaire sur un batch de {@code batchSize} samples.
   *
   * @param input buffer {@code [batchSize × inFeatures]} row-major
   * @param weights buffer {@code [outFeatures × inFeatures]} row-major
   * @param bias buffer {@code [outFeatures]}
   * @param output buffer {@code [batchSize × outFeatures]} row-major (rempli en place)
   * @param inFeatures nombre de features d'entrée
   * @param outFeatures nombre de features de sortie
   * @param batchSize nombre de samples à traiter
   */
  public static void applyBatch(
      float[] input,
      float[] weights,
      float[] bias,
      float[] output,
      int inFeatures,
      int outFeatures,
      int batchSize) {
    int upperBound = SPECIES.loopBound(inFeatures);
    for (int n = 0; n < batchSize; n++) {
      int inputBase = n * inFeatures;
      int outputBase = n * outFeatures;
      for (int o = 0; o < outFeatures; o++) {
        int weightBase = o * inFeatures;
        FloatVector accVec = FloatVector.zero(SPECIES);
        int i = 0;
        for (; i < upperBound; i += LANES) {
          FloatVector inVec = FloatVector.fromArray(SPECIES, input, inputBase + i);
          FloatVector wVec = FloatVector.fromArray(SPECIES, weights, weightBase + i);
          accVec = inVec.fma(wVec, accVec);
        }
        float acc = accVec.reduceLanes(VectorOperators.ADD);
        for (; i < inFeatures; i++) {
          acc += input[inputBase + i] * weights[weightBase + i];
        }
        output[outputBase + o] = acc + bias[o];
      }
    }
  }
}
