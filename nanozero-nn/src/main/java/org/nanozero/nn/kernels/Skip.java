package org.nanozero.nn.kernels;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Skip connection in-place (cf. SPEC §5.6.5, §7.7).
 *
 * <p>{@link #addInPlace(float[], float[], int)} : {@code dst[i] += src[i]} sur les {@code length}
 * premiers éléments, via boucle SIMD principale + queue scalaire.
 *
 * <p><strong>Visibilité</strong> : publique pour permettre l'accès cross-package depuis {@link
 * org.nanozero.nn.Network}. Internal API uniquement (cf. §12.4).
 */
public final class Skip {

  /** Species SIMD préférée. */
  static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  static final int LANES = SPECIES.length();

  private Skip() {
    throw new AssertionError("Non-instantiable");
  }

  /**
   * Ajoute en place {@code src} dans {@code dst} sur les {@code length} premiers éléments. {@code
   * dst} est muté ; {@code src} est lu seulement.
   *
   * @param dst buffer destination (muté en place)
   * @param src buffer source (lu seulement)
   * @param length nombre d'éléments à traiter
   */
  public static void addInPlace(float[] dst, float[] src, int length) {
    int i = 0;
    int upperBound = SPECIES.loopBound(length);
    for (; i < upperBound; i += LANES) {
      FloatVector d = FloatVector.fromArray(SPECIES, dst, i);
      FloatVector s = FloatVector.fromArray(SPECIES, src, i);
      d.add(s).intoArray(dst, i);
    }
    for (; i < length; i++) {
      dst[i] += src[i];
    }
  }
}
