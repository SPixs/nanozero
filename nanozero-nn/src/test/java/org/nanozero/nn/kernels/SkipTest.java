package org.nanozero.nn.kernels;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Tests unitaires de {@link Skip} (cf. SPEC §7.7, §13 phase 3). */
class SkipTest {

  @Test
  @DisplayName("addInPlace : valeurs simples, dst muté, src non muté")
  void simpleAddition() {
    float[] dst = {1.0f, 2.0f, 3.0f, 4.0f};
    float[] src = {10.0f, 20.0f, 30.0f, 40.0f};
    float[] srcCopy = src.clone();
    Skip.addInPlace(dst, src, dst.length);
    assertThat(dst).containsExactly(11.0f, 22.0f, 33.0f, 44.0f);
    assertThat(src).containsExactly(srcCopy); // src non muté
  }

  @ParameterizedTest(name = "length={0}")
  @ValueSource(ints = {1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 96, 119, 128, 7616})
  @DisplayName("Longueurs variées : SIMD bound + queue scalaire identiques au scalaire pur")
  void variousLengths(int length) {
    Random rng = new Random(0xCAFE0042L);
    float[] dst = new float[length];
    float[] src = new float[length];
    float[] expected = new float[length];
    for (int i = 0; i < length; i++) {
      dst[i] = (float) rng.nextGaussian();
      src[i] = (float) rng.nextGaussian();
      expected[i] = dst[i] + src[i];
    }
    Skip.addInPlace(dst, src, length);
    for (int i = 0; i < length; i++) {
      assertThat(dst[i]).as("dst[%d]", i).isEqualTo(expected[i]);
    }
  }

  @Test
  @DisplayName("Length = 0 : no-op (ni dst ni src touchés)")
  void zeroLength() {
    float[] dst = {1.0f, 2.0f};
    float[] src = {10.0f, 20.0f};
    Skip.addInPlace(dst, src, 0);
    assertThat(dst).containsExactly(1.0f, 2.0f);
    assertThat(src).containsExactly(10.0f, 20.0f);
  }

  @Test
  @DisplayName("Au-delà de length : dst non touché")
  void onlyLengthTouched() {
    float[] dst = new float[Skip.LANES * 2 + 3];
    float[] src = new float[Skip.LANES * 2 + 3];
    java.util.Arrays.fill(dst, 1.0f);
    java.util.Arrays.fill(src, 5.0f);
    int lenToTouch = Skip.LANES + 1;
    Skip.addInPlace(dst, src, lenToTouch);
    for (int i = 0; i < lenToTouch; i++) {
      assertThat(dst[i]).isEqualTo(6.0f);
    }
    for (int i = lenToTouch; i < dst.length; i++) {
      assertThat(dst[i]).as("dst[%d] au-delà de length", i).isEqualTo(1.0f);
    }
  }
}
