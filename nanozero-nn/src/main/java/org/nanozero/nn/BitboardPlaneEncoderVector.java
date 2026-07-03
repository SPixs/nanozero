package org.nanozero.nn;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorMask;
import jdk.incubator.vector.VectorSpecies;
import org.nanozero.board.BitboardPlaneEncoder;

/**
 * Implémentation Vector API du contrat {@link BitboardPlaneEncoder} exposé par {@code
 * nanozero-board} (cf. SPEC-nn §5.4, §7.8).
 *
 * <p>Pour chaque rangée de l'échiquier (8 cases consécutives, alignées sur un octet du bitboard),
 * on construit un {@link VectorMask} depuis les 8 bits correspondants via {@link
 * VectorMask#fromLong(VectorSpecies, long)}, puis on écrit en mémoire {@code 1.0f} ou {@code 0.0f}
 * selon le mask via {@link FloatVector#blend(float, VectorMask)}. Une rangée = une instruction SIMD
 * 256-bit (8 lanes float32 = 256 bits) ; 8 itérations couvrent les 64 cases.
 *
 * <p><strong>Singleton stateless</strong>, thread-safe par construction. À utiliser via la
 * référence statique {@link #INSTANCE} ; le constructeur est privé.
 *
 * <p><strong>Invariant {@code I-BPE-1}</strong> : pour tout bitboard, le résultat est strictement
 * identique à l'implémentation scalaire de référence ({@link
 * org.nanozero.board.GameState#scalarEncode}, cf. SPEC-board §7.6) — pas de tolérance, pas de
 * différence d'arrondi (les valeurs sortantes sont {@code 0.0f} ou {@code 1.0f} exact, jamais
 * d'opération arithmétique intermédiaire). Validé sur 100 000 bitboards aléatoires (cf. {@code
 * BitboardPlaneEncoderVectorTest}).
 *
 * <p>Choix de {@link FloatVector#SPECIES_256} (et non {@code SPECIES_PREFERRED}) : 8 lanes
 * correspondent exactement à une rangée d'échiquier (8 cases) — boucle simple, pas de queue
 * scalaire. Sur AVX-512, on n'exploite pas les 16 lanes ; optimisation possible en phase 11
 * (encoder 2 rangées par itération avec {@link FloatVector#SPECIES_512}).
 */
public final class BitboardPlaneEncoderVector implements BitboardPlaneEncoder {

  /** Singleton stateless, thread-safe. */
  public static final BitboardPlaneEncoderVector INSTANCE = new BitboardPlaneEncoderVector();

  private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_256;

  /** Vecteur SIMD constant rempli de {@code 1.0f} sur les 8 lanes (valeur "bit set"). */
  private static final FloatVector ONE = FloatVector.broadcast(SPECIES, 1.0f);

  /** Vecteur SIMD constant rempli de {@code 0.0f} sur les 8 lanes (valeur "bit clear"). */
  private static final FloatVector ZERO = FloatVector.zero(SPECIES);

  private BitboardPlaneEncoderVector() {
    // Singleton : ne pas instancier hors d'INSTANCE.
  }

  /**
   * Encode un bitboard 64 bits en 64 floats {@code 0.0f}/{@code 1.0f} dans {@code dest} à partir de
   * {@code planeOffset}. La case d'index {@code sq ∈ [0..63]} (convention LSB = a1, cf. ADR-008
   * board) est écrite en {@code dest[planeOffset + sq]}.
   *
   * @param bitboard bitboard 64 bits source
   * @param dest tableau destination (longueur {@code >= planeOffset + 64})
   * @param planeOffset offset dans {@code dest} où commence la plane
   */
  @Override
  public void encode(long bitboard, float[] dest, int planeOffset) {
    for (int row = 0; row < 8; row++) {
      long rowBits = (bitboard >>> (row * 8)) & 0xFFL;
      VectorMask<Float> mask = VectorMask.fromLong(SPECIES, rowBits);
      ZERO.blend(ONE, mask).intoArray(dest, planeOffset + row * 8);
    }
  }
}
