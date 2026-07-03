package org.nanozero.board;

/**
 * Foncteur d'encodage d'un bitboard en un plan {@code float[]} 8×8 (cf. SPEC §7.6).
 *
 * <p>Cette interface est le point d'extension permettant à un module aval (typiquement {@code
 * nanozero-nn}) de substituer une implémentation vectorisée (Vector API SIMD) à l'implémentation
 * scalaire de référence fournie par {@link GameState}, sans introduire dans le module {@code
 * nanozero-board} de dépendance sur {@code jdk.incubator.vector}.
 *
 * <p>Contrat : pour le bitboard {@code bb} et la case {@code sq ∈ [0..63]}, l'implémentation DOIT
 * écrire dans {@code dest[planeOffset + sq]} la valeur {@code 1.0f} si le bit {@code sq} est à 1
 * dans {@code bb}, {@code 0.0f} sinon. La convention de bit suit ADR-008 ({@code bit 0 = a1},
 * {@code bit 7 = h1}, ..., {@code bit 63 = h8}).
 *
 * <p>Cohérence référentielle : le résultat DOIT être identique bit-à-bit à l'implémentation
 * scalaire de référence (§7.6) pour TOUTE valeur de {@code bb}. Les implémentations vectorisées
 * peuvent traiter plusieurs cases en parallèle, mais ne peuvent ni élargir ni rétrécir le
 * vocabulaire de sortie ({@code 0.0f} ou {@code 1.0f} stricts).
 */
@FunctionalInterface
public interface BitboardPlaneEncoder {

  /**
   * Écrit dans {@code dest} les 64 floats correspondant au bitboard fourni.
   *
   * @param bitboard bitboard source (long 64 bits, convention LSB = a1)
   * @param dest tableau destination, de taille {@code >= planeOffset + 64}
   * @param planeOffset index de la première case du plan dans {@code dest}
   */
  void encode(long bitboard, float[] dest, int planeOffset);
}
