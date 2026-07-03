package org.nanozero.nn.kernels;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Tests unitaires de {@link Activations} (cf. SPEC §7.5, §7.6, §13 phase 3). */
class ActivationsTest {

  @Nested
  @DisplayName("reluInPlace : SIMD principale + queue scalaire")
  class ReLU {

    @Test
    @DisplayName("Mix positif/négatif/zéro : attendu max(x, 0)")
    void mixedSigns() {
      float[] buf = {-3.0f, -0.001f, 0.0f, 0.001f, 1.0f, 5.0f, -100.0f, 42.0f};
      Activations.reluInPlace(buf, buf.length);
      assertThat(buf).containsExactly(0.0f, 0.0f, 0.0f, 0.001f, 1.0f, 5.0f, 0.0f, 42.0f);
    }

    @ParameterizedTest(name = "length={0} (queue scalaire = {0} % LANES)")
    @ValueSource(
        ints = {1, 2, 3, 7, 8, 9, 15, 16, 17, 24, 32, 33, 64, 96, 119, 128, 256, 1024, 7616})
    @DisplayName("Longueurs variées : SIMD bound + queue scalaire correctes")
    void variousLengths(int length) {
      float[] buf = new float[length];
      // Pattern alterné : pair → +i, impair → -i
      for (int i = 0; i < length; i++) {
        buf[i] = (i % 2 == 0) ? (float) i : -(float) i;
      }
      Activations.reluInPlace(buf, length);
      for (int i = 0; i < length; i++) {
        float expected = (i % 2 == 0) ? (float) i : 0.0f;
        assertThat(buf[i]).as("buf[%d]", i).isEqualTo(expected);
      }
    }

    @Test
    @DisplayName("In-place : seuls les `length` premiers éléments sont touchés")
    void onlyLengthTouched() {
      float[] buf = new float[Activations.LANES * 2 + 3];
      java.util.Arrays.fill(buf, -1.0f);
      int lenToTouch = Activations.LANES + 1; // toucher LANES+1 éléments
      Activations.reluInPlace(buf, lenToTouch);
      for (int i = 0; i < lenToTouch; i++) {
        assertThat(buf[i]).as("buf[%d]", i).isEqualTo(0.0f);
      }
      for (int i = lenToTouch; i < buf.length; i++) {
        assertThat(buf[i]).as("buf[%d] (au-delà de length)", i).isEqualTo(-1.0f);
      }
    }

    @Test
    @DisplayName("Length = 0 : no-op")
    void zeroLength() {
      float[] buf = {-1.0f, -2.0f, -3.0f};
      Activations.reluInPlace(buf, 0);
      assertThat(buf).containsExactly(-1.0f, -2.0f, -3.0f);
    }
  }

  @Nested
  @DisplayName("tanhInPlace : Math.tanh scalaire")
  class Tanh {

    @Test
    @DisplayName("Valeurs symétriques (-1, 0, +1, ±large) vs Math.tanh")
    void symmetricValues() {
      float[] buf = {-10.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 10.0f};
      float[] expected = new float[buf.length];
      for (int i = 0; i < buf.length; i++) {
        expected[i] = (float) Math.tanh(buf[i]);
      }
      Activations.tanhInPlace(buf, buf.length);
      for (int i = 0; i < buf.length; i++) {
        assertThat(buf[i]).as("tanh(%f)", expected[i]).isCloseTo(expected[i], within(1e-7f));
      }
    }

    @Test
    @DisplayName("tanh(0) = 0, tanh(±∞) → ±1, tanh(±large) ≈ ±1")
    void edgeValues() {
      float[] buf = {0.0f, 100.0f, -100.0f};
      Activations.tanhInPlace(buf, buf.length);
      assertThat(buf[0]).isEqualTo(0.0f);
      assertThat(buf[1]).isCloseTo(1.0f, within(1e-7f));
      assertThat(buf[2]).isCloseTo(-1.0f, within(1e-7f));
    }

    @Test
    @DisplayName("In-place + length partielle")
    void inPlacePartial() {
      float[] buf = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f};
      Activations.tanhInPlace(buf, 3);
      assertThat(buf[0]).isCloseTo(0.0f, within(1e-7f));
      assertThat(buf[1]).isCloseTo((float) Math.tanh(1.0), within(1e-7f));
      assertThat(buf[2]).isCloseTo((float) Math.tanh(2.0), within(1e-7f));
      // Au-delà : non touché
      assertThat(buf[3]).isEqualTo(3.0f);
      assertThat(buf[4]).isEqualTo(4.0f);
    }
  }

  @Test
  @DisplayName("LANES > 1 attendu : SIMD réel actif (pas de fallback scalaire)")
  void simdRealNotFallback() {
    // Si Activations.LANES == 1, c'est un fallback scalaire — Vector API ne s'est pas chargée
    // correctement. Sur AVX2 desktop attendu LANES = 8, sur AVX-512 LANES = 16, sur NEON ARM
    // LANES = 4. Quel que soit le matériel, > 1.
    assertThat(Activations.LANES).isGreaterThan(1);
  }
}
