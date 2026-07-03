package org.nanozero.nn;

import static org.assertj.core.api.Assertions.assertThat;

import java.lang.reflect.Constructor;
import java.lang.reflect.Modifier;
import java.util.Arrays;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Tests unitaires de {@link BitboardPlaneEncoderVector} (cf. SPEC-nn §13 phase 6).
 *
 * <p>Critère de complétion : 0 divergence sur 100 000 bitboards aléatoires vs implémentation
 * scalaire de référence (§7.6 SPEC-board, dupliquée inline ici pour indépendance du test).
 */
class BitboardPlaneEncoderVectorTest {

  /** Seed PRNG fixe pour reproductibilité bit-à-bit (cf. convention seedée des autres tests). */
  private static final long SEED = 0xCAFEBABEL;

  /**
   * Implémentation scalaire de référence dupliquée depuis SPEC-board §7.6 (5 lignes).
   * Volontairement locale au test pour ne dépendre d'aucun symbole non public de board ; le test
   * vérifie ainsi l'invariant {@code I-BPE-1} contre une référence indépendante.
   */
  private static void scalarEncode(long bb, float[] dest, int planeOffset) {
    for (int sq = 0; sq < 64; sq++) {
      dest[planeOffset + sq] = ((bb >>> sq) & 1L) == 0L ? 0.0f : 1.0f;
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Critère §13 phase 6 : 100 000 bitboards aléatoires, 0 divergence
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "Critère §13 phase 6 : 100 000 bitboards aléatoires, 0 divergence vs scalaire (I-BPE-1)")
  void crossValidationVsScalarReference() {
    Random rng = new Random(SEED);
    float[] expected = new float[64];
    float[] actual = new float[64];
    for (int trial = 0; trial < 100_000; trial++) {
      long bb = rng.nextLong();
      // Réinitialise les buffers pour ne pas masquer un bug d'écriture partielle.
      Arrays.fill(expected, -1.0f);
      Arrays.fill(actual, -1.0f);
      scalarEncode(bb, expected, 0);
      BitboardPlaneEncoderVector.INSTANCE.encode(bb, actual, 0);
      // Égalité STRICTE (0.0f/1.0f exact, pas de tolérance).
      assertThat(actual).as("trial #%d, bitboard=0x%016X", trial, bb).isEqualTo(expected);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Edge cases : bitboard = 0, -1, single-bit pour chacune des 64 cases
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("Edge cases : 0, -1, single-bit pour chaque case")
  class EdgeCases {

    @Test
    @DisplayName("bitboard = 0L : toutes les 64 cases à 0.0f")
    void zeroBitboard() {
      float[] dest = new float[64];
      Arrays.fill(dest, -1.0f);
      BitboardPlaneEncoderVector.INSTANCE.encode(0L, dest, 0);
      for (int i = 0; i < 64; i++) {
        assertThat(dest[i]).as("dest[%d]", i).isEqualTo(0.0f);
      }
    }

    @Test
    @DisplayName("bitboard = -1L (tous bits set) : toutes les 64 cases à 1.0f")
    void allBitsBitboard() {
      float[] dest = new float[64];
      BitboardPlaneEncoderVector.INSTANCE.encode(-1L, dest, 0);
      for (int i = 0; i < 64; i++) {
        assertThat(dest[i]).as("dest[%d]", i).isEqualTo(1.0f);
      }
    }

    @Test
    @DisplayName("bitboard = 1L (LSB = a1) : dest[0] = 1.0f, reste = 0.0f")
    void lsbOnly() {
      float[] dest = new float[64];
      BitboardPlaneEncoderVector.INSTANCE.encode(1L, dest, 0);
      assertThat(dest[0]).isEqualTo(1.0f);
      for (int i = 1; i < 64; i++) {
        assertThat(dest[i]).as("dest[%d]", i).isEqualTo(0.0f);
      }
    }

    @Test
    @DisplayName("bitboard = 1L << 63 (MSB = h8) : dest[63] = 1.0f, reste = 0.0f")
    void msbOnly() {
      float[] dest = new float[64];
      BitboardPlaneEncoderVector.INSTANCE.encode(1L << 63, dest, 0);
      assertThat(dest[63]).isEqualTo(1.0f);
      for (int i = 0; i < 63; i++) {
        assertThat(dest[i]).as("dest[%d]", i).isEqualTo(0.0f);
      }
    }

    @ParameterizedTest(name = "single-bit set au square {0}")
    @ValueSource(
        ints = {
          0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
          16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
          32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
          48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63
        })
    @DisplayName("Single-bit pour chacune des 64 cases : dest[sq] = 1.0f, autres = 0.0f")
    void singleBitForEachSquare(int sq) {
      float[] dest = new float[64];
      BitboardPlaneEncoderVector.INSTANCE.encode(1L << sq, dest, 0);
      for (int i = 0; i < 64; i++) {
        float expected = (i == sq) ? 1.0f : 0.0f;
        assertThat(dest[i]).as("sq=%d : dest[%d]", sq, i).isEqualTo(expected);
      }
    }
  }

  // ---------------------------------------------------------------------------------------------
  // planeOffset : l'encoder n'écrit qu'en [planeOffset, planeOffset + 64)
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("planeOffset : seuls dest[offset..offset+64) sont touchés (sentinelles intactes)")
  void planeOffsetBoundaries() {
    int planeOffset = 128;
    int totalSize = 64 * 4; // 4 plans dans le buffer
    float sentinel = -7.5f;
    float[] dest = new float[totalSize];
    Arrays.fill(dest, sentinel);

    long bb = 0xDEADBEEFCAFEBABEL;
    BitboardPlaneEncoderVector.INSTANCE.encode(bb, dest, planeOffset);

    // Avant la plane : sentinelles intactes
    for (int i = 0; i < planeOffset; i++) {
      assertThat(dest[i]).as("dest[%d] avant plane", i).isEqualTo(sentinel);
    }
    // Dans la plane : valeurs encodées attendues
    float[] expectedPlane = new float[64];
    scalarEncode(bb, expectedPlane, 0);
    for (int i = 0; i < 64; i++) {
      assertThat(dest[planeOffset + i])
          .as("dest[%d] dans plane", planeOffset + i)
          .isEqualTo(expectedPlane[i]);
    }
    // Après la plane : sentinelles intactes
    for (int i = planeOffset + 64; i < totalSize; i++) {
      assertThat(dest[i]).as("dest[%d] après plane", i).isEqualTo(sentinel);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Singleton : INSTANCE non null, constructeur privé unique, identité stable
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("INSTANCE non-null + identité stable")
  void instanceIsStableAndNonNull() {
    assertThat(BitboardPlaneEncoderVector.INSTANCE).isNotNull();
    assertThat(BitboardPlaneEncoderVector.INSTANCE).isSameAs(BitboardPlaneEncoderVector.INSTANCE);
  }

  @Test
  @DisplayName("Constructeur privé unique (pas d'instanciation côté caller)")
  void singlePrivateConstructor() {
    Constructor<?>[] ctors = BitboardPlaneEncoderVector.class.getDeclaredConstructors();
    assertThat(ctors).hasSize(1);
    assertThat(Modifier.isPrivate(ctors[0].getModifiers())).isTrue();
  }

  // ---------------------------------------------------------------------------------------------
  // Thread safety sanity : 4 threads encodant simultanément, aucune corruption
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Thread safety sanity : 4 threads × 1000 bitboards, aucune divergence vs scalaire")
  void threadSafetySanity() throws InterruptedException {
    int threads = 4;
    int iters = 1000;
    CountDownLatch start = new CountDownLatch(1);
    CountDownLatch done = new CountDownLatch(threads);
    AtomicInteger failures = new AtomicInteger(0);

    for (int t = 0; t < threads; t++) {
      final int threadId = t;
      Thread worker =
          new Thread(
              () -> {
                try {
                  start.await();
                  Random rng = new Random(0xBEEF0000L + threadId);
                  float[] expected = new float[64];
                  float[] actual = new float[64];
                  for (int it = 0; it < iters; it++) {
                    long bb = rng.nextLong();
                    scalarEncode(bb, expected, 0);
                    BitboardPlaneEncoderVector.INSTANCE.encode(bb, actual, 0);
                    if (!Arrays.equals(expected, actual)) {
                      failures.incrementAndGet();
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
    assertThat(failures.get()).as("divergences détectées sur threads concurrents").isZero();
  }
}
