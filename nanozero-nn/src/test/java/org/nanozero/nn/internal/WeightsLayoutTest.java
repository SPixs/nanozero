package org.nanozero.nn.internal;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * Tests unitaires de {@link WeightsLayout} (cf. SPEC §7.2.4, §13 phase 5).
 *
 * <p>Les bugs de reorder sont silencieux et corrompent toutes les sorties Conv 3×3 — round-trip
 * exhaustif obligatoire avant tout test de Conv2D3x3.
 */
class WeightsLayoutTest {

  // -----------------------------------------------------------------------------------------
  // Round-trip exhaustif sur 100 cas (inC, outC) variés
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Round-trip 100 cas : inverseReorder(reorder(W)) == W modulo padding queue")
  void roundTripHundredCases() {
    Random rng = new Random(0xCAFEBABEL);
    int[] inCandidates = {1, 4, 8, 9, 17, 32, 73, 96, 119};
    int[] outCandidates = {1, 4, 7, 8, 9, 16, 17, 73, 96, 119};

    for (int trial = 0; trial < 100; trial++) {
      int inC = inCandidates[rng.nextInt(inCandidates.length)];
      int outC = outCandidates[rng.nextInt(outCandidates.length)];
      float[] original = new float[outC * inC * 9];
      for (int i = 0; i < original.length; i++) {
        original[i] = (float) rng.nextGaussian();
      }
      float[] reordered = WeightsLayout.reorderConv3x3(original, inC, outC);
      float[] roundTrip = WeightsLayout.inverseReorderConv3x3(reordered, inC, outC);
      assertThat(roundTrip)
          .as("trial #%d (inC=%d, outC=%d) round-trip identité", trial, inC, outC)
          .containsExactly(original);
    }
  }

  // -----------------------------------------------------------------------------------------
  // Cas dégénérés : outC = 1, LANES, LANES+1, 96
  // -----------------------------------------------------------------------------------------

  @ParameterizedTest(name = "outC={0}")
  @ValueSource(ints = {1, 8, 9, 96})
  @DisplayName("Cas dégénérés : outC = 1, LANES, LANES+1, 96 — round-trip")
  void degenerateOutChannels(int outC) {
    int inC = 16;
    Random rng = new Random((long) outC + 0x42);
    float[] original = new float[outC * inC * 9];
    for (int i = 0; i < original.length; i++) {
      original[i] = (float) rng.nextGaussian();
    }
    float[] reordered = WeightsLayout.reorderConv3x3(original, inC, outC);
    float[] roundTrip = WeightsLayout.inverseReorderConv3x3(reordered, inC, outC);
    assertThat(roundTrip).containsExactly(original);
  }

  // -----------------------------------------------------------------------------------------
  // Taille du buffer reorderé : ceil(outC / LANES) * LANES * inC * 9
  // -----------------------------------------------------------------------------------------

  @ParameterizedTest(name = "inC={0}, outC={1} → expected size {2}")
  @CsvSource({
    "96, 96, 82944", // 12 × 8 × 96 × 9, outC multiple de LANES, pas de padding
    "119, 96, 102816", // 12 × 8 × 119 × 9
    "96, 73, 69120", // 10 × 8 × 96 × 9 (outC=73 → 10 blocs avec padding)
    "96, 1, 6912", // 1 × 8 × 96 × 9 (outC=1 → 1 bloc, 7 lanes paddées)
    "1, 1, 72" // 1 × 8 × 1 × 9 (cas minimal, padding 7 lanes)
  })
  @DisplayName("Taille buffer reorderé conforme : ceil(outC/LANES) * LANES * inC * 9")
  void reorderedBufferSize(int inC, int outC, int expectedSize) {
    float[] weights = new float[outC * inC * 9];
    float[] reordered = WeightsLayout.reorderConv3x3(weights, inC, outC);
    assertThat(reordered).hasSize(expectedSize);
  }

  // -----------------------------------------------------------------------------------------
  // Padding queue à zéro : valeurs au-delà de outChannels strictement nulles
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Padding queue : lanes au-delà de outChannels strictement à zéro")
  void paddingQueueIsZero() {
    int inC = 4;
    int outC = 1; // 1 bloc, lane 0 utilisé, lanes 1..LANES-1 paddées
    int lanes = WeightsLayout.lanes();
    Random rng = new Random(0x42);
    float[] original = new float[outC * inC * 9];
    for (int i = 0; i < original.length; i++) {
      // Valeurs non-nulles pour distinguer des zéros de padding
      original[i] = 1.0f + (float) rng.nextGaussian();
    }
    float[] reordered = WeightsLayout.reorderConv3x3(original, inC, outC);
    int blockSize = inC * 9 * lanes;
    assertThat(reordered).hasSize(blockSize);
    // Pour chaque (ic, k) du seul bloc, vérifier que les lanes >= outC sont à zéro.
    for (int ic = 0; ic < inC; ic++) {
      for (int k = 0; k < 9; k++) {
        int base = ic * 9 * lanes + k * lanes;
        // Lane 0 = oc=0 < outC : doit contenir la valeur originale
        assertThat(reordered[base])
            .as("lane 0 (oc=0) ic=%d k=%d", ic, k)
            .isEqualTo(original[ic * 9 + k]);
        // Lanes 1..lanes-1 (oc=1..lanes-1, tous >= outC=1) : padding zéro
        for (int lane = 1; lane < lanes; lane++) {
          assertThat(reordered[base + lane])
              .as("lane %d (oc=%d >= outC) ic=%d k=%d", lane, lane, ic, k)
              .isZero();
        }
      }
    }
  }

  // -----------------------------------------------------------------------------------------
  // Méthode pure : weightsRowMajor n'est pas muté
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("reorderConv3x3 ne mute pas weightsRowMajor")
  void inputNotMutated() {
    int inC = 8;
    int outC = 9;
    Random rng = new Random(0x123);
    float[] original = new float[outC * inC * 9];
    for (int i = 0; i < original.length; i++) {
      original[i] = (float) rng.nextGaussian();
    }
    float[] copy = original.clone();
    WeightsLayout.reorderConv3x3(original, inC, outC);
    assertThat(original).containsExactly(copy);
  }
}
