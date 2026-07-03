package org.nanozero.nn.internal;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

/**
 * Réorganise les tenseurs de poids depuis le layout {@code state_dict()} PyTorch natif vers les
 * layouts SIMD-friendly attendus par les kernels du module {@code nanozero-nn} (cf. SPEC §7.2.4).
 *
 * <p>Conv 3×3 row-major PyTorch est en {@code [outC, inC, 3, 3]}. Pour permettre un chargement
 * vectoriel contigu de {@code wVec = weights[oc..oc+LANES-1, ic, kh, kw]} dans la boucle hot du
 * kernel (cf. §7.2.3), on transpose vers {@code [outC/LANES, inC, 9, LANES]}. Cette opération est
 * effectuée une seule fois au chargement par {@code NetworkLoader}, transparente pour le runtime
 * d'inférence.
 *
 * <p><strong>Visibilité</strong> : SPEC §12.3 prescrit "package-private" pour le sub-package {@code
 * internal/}. Java ne permet pas d'accès cross-package package-private au sein d'un même module
 * sans {@code module-info.java}. Cette classe est donc publique (class + méthodes statiques) à des
 * fins d'usage interne uniquement par {@link org.nanozero.nn.Network} (constructeur de test phase
 * 7) et {@code NetworkLoader} (phase 8). Le caractère "internal" est conservé par le nom du
 * sub-package et par cette note. Les utilisateurs externes du module ne doivent PAS dépendre de
 * cette API ; aucune garantie de stabilité au-delà de ce qui est testé par {@code
 * WeightsLayoutTest}.
 *
 * <p>Note pour amendement SPEC §12.3 : la prescription "package-private" est aspirationnelle sans
 * {@code module-info.java}. À reformuler en "interne au module, non exporté" quand le module
 * descriptor sera ajouté (probable phase 7+).
 */
public final class WeightsLayout {

  /** Species SIMD préférée — DOIT correspondre à celle utilisée par {@code Conv2D3x3}. */
  private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

  /** Nombre de lanes SIMD (8 sur AVX2, 16 sur AVX-512, 4 sur NEON). */
  private static final int LANES = SPECIES.length();

  /** Nombre de positions du noyau Conv 3×3 (3 × 3). */
  private static final int K = 9;

  private WeightsLayout() {
    throw new AssertionError("Non-instantiable");
  }

  /**
   * Réordonne les poids Conv 3×3 du layout row-major PyTorch {@code [outC, inC, 3, 3]} vers le
   * layout SIMD-friendly {@code [outC/LANES, inC, 9, LANES]} (cf. SPEC §7.2.4).
   *
   * <p>Si {@code outChannels} n'est pas un multiple de {@link #LANES}, le dernier bloc est complété
   * avec des zéros (padding queue) — l'accumulation FMA contre ces poids est neutre pour les lanes
   * qui correspondent à des output channels inexistants.
   *
   * @param weightsRowMajor buffer {@code [outChannels × inChannels × 9]} aplati row-major
   * @param inChannels nombre de channels d'entrée
   * @param outChannels nombre de channels de sortie
   * @return nouveau buffer de taille {@code ceil(outChannels / LANES) × LANES × inChannels × 9},
   *     layout {@code [outC/LANES, inC, 9, LANES]} aplati. {@code weightsRowMajor} est lu
   *     seulement, jamais modifié.
   */
  public static float[] reorderConv3x3(float[] weightsRowMajor, int inChannels, int outChannels) {
    int blocks = (outChannels + LANES - 1) / LANES;
    int blockSize = inChannels * K * LANES;
    float[] out = new float[blocks * blockSize];
    for (int ocBlock = 0; ocBlock < blocks; ocBlock++) {
      for (int ic = 0; ic < inChannels; ic++) {
        for (int k = 0; k < K; k++) {
          for (int lane = 0; lane < LANES; lane++) {
            int oc = ocBlock * LANES + lane;
            if (oc < outChannels) {
              int srcIdx = oc * inChannels * K + ic * K + k;
              int dstIdx = ocBlock * blockSize + ic * K * LANES + k * LANES + lane;
              out[dstIdx] = weightsRowMajor[srcIdx];
            }
            // sinon : padding queue, déjà 0 (valeur par défaut Java)
          }
        }
      }
    }
    return out;
  }

  /**
   * Inverse de {@link #reorderConv3x3} : recompose le layout row-major {@code [outC, inC, 9]} à
   * partir du layout reorderé. Utile pour les tests round-trip ; n'est PAS appelé en hot path.
   *
   * @param reordered buffer issu de {@link #reorderConv3x3}
   * @param inChannels nombre de channels d'entrée
   * @param outChannels nombre de channels de sortie
   * @return buffer {@code [outChannels × inChannels × 9]} row-major
   */
  public static float[] inverseReorderConv3x3(float[] reordered, int inChannels, int outChannels) {
    int blocks = (outChannels + LANES - 1) / LANES;
    int blockSize = inChannels * K * LANES;
    float[] out = new float[outChannels * inChannels * K];
    for (int ocBlock = 0; ocBlock < blocks; ocBlock++) {
      for (int ic = 0; ic < inChannels; ic++) {
        for (int k = 0; k < K; k++) {
          for (int lane = 0; lane < LANES; lane++) {
            int oc = ocBlock * LANES + lane;
            if (oc < outChannels) {
              int srcIdx = ocBlock * blockSize + ic * K * LANES + k * LANES + lane;
              int dstIdx = oc * inChannels * K + ic * K + k;
              out[dstIdx] = reordered[srcIdx];
            }
          }
        }
      }
    }
    return out;
  }

  /** Retourne {@code LANES} ; utile aux tests pour calculer la taille attendue du buffer. */
  public static int lanes() {
    return LANES;
  }
}
