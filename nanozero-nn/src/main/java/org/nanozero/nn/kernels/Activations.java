package org.nanozero.nn.kernels;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Kernels d'activations en place (cf. SPEC §5.6.4, §7.5, §7.6).
 *
 * <ul>
 *   <li>{@link #reluInPlace(float[], int)} : implémentation Vector API ({@code .max(zero)}) avec
 *       queue scalaire pour {@code length} non multiple de {@link #LANES}.
 *   <li>{@link #tanhInPlace(float[], int)} : implémentation scalaire pure via {@link Math#tanh}.
 *       Pas d'optimisation prématurée (cf. §7.6 : seulement {@code MAX_BATCH = 64} appels par
 *       forward, ~3 µs total — négligeable face aux conv).
 * </ul>
 *
 * <p><strong>Visibilité</strong> : publique pour permettre l'accès cross-package depuis {@link
 * org.nanozero.nn.Network}. Internal API uniquement (cf. §12.4).
 */
public final class Activations {

  /** Species SIMD préférée par la JVM (8 lanes en AVX2, 16 en AVX-512, 4 en NEON). */
  static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  /** Nombre de lanes du species choisi. */
  static final int LANES = SPECIES.length();

  private Activations() {
    throw new AssertionError("Non-instantiable");
  }

  /**
   * Applique ReLU en place sur les {@code length} premiers éléments de {@code buf} : {@code buf[i]
   * = max(buf[i], 0)}.
   *
   * <p>Pattern obligatoire (cf. SPEC §7.1.1) : boucle SIMD principale via {@link
   * FloatVector#max(jdk.incubator.vector.Vector)} contre un vecteur zéro pré-alloué + queue
   * scalaire pour les {@code length % LANES} éléments restants.
   *
   * @param buf buffer modifié en place
   * @param length nombre d'éléments à traiter (doit être {@code <= buf.length})
   */
  public static void reluInPlace(float[] buf, int length) {
    int i = 0;
    int upperBound = SPECIES.loopBound(length);
    FloatVector zero = FloatVector.zero(SPECIES);
    for (; i < upperBound; i += LANES) {
      FloatVector v = FloatVector.fromArray(SPECIES, buf, i);
      v.max(zero).intoArray(buf, i);
    }
    for (; i < length; i++) {
      if (buf[i] < 0.0f) {
        buf[i] = 0.0f;
      }
    }
  }

  /**
   * Applique tanh en place sur les {@code length} premiers éléments de {@code buf}, via {@link
   * Math#tanh(double)}. Implémentation scalaire pure, justifiée en §7.6 (tanh n'a pas d'instruction
   * CPU dédiée et n'est appelé que sur la sortie du value head, ~64 floats par forward).
   *
   * @param buf buffer modifié en place
   * @param length nombre d'éléments à traiter
   */
  public static void tanhInPlace(float[] buf, int length) {
    for (int i = 0; i < length; i++) {
      buf[i] = (float) Math.tanh(buf[i]);
    }
  }
}
