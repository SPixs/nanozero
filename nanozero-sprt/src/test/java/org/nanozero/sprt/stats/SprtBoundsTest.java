package org.nanozero.sprt.stats;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

class SprtBoundsTest {

  private static final double TOL = 1e-9;

  @Test
  void standardBoundsAlphaBetaPoint05() {
    SprtBounds b = SprtBounds.standard(0.0, 5.0);
    // upper = log((1 - 0.05) / 0.05) = log(19) ≈ 2.94443898
    assertThat(b.upper()).isCloseTo(Math.log(19.0), org.assertj.core.data.Offset.offset(TOL));
    // lower = log(0.05 / (1 - 0.05)) = log(1/19) ≈ -2.94443898
    assertThat(b.lower()).isCloseTo(-Math.log(19.0), org.assertj.core.data.Offset.offset(TOL));
    assertThat(b.elo0()).isEqualTo(0.0);
    assertThat(b.elo1()).isEqualTo(5.0);
    assertThat(b.alpha()).isEqualTo(0.05);
    assertThat(b.beta()).isEqualTo(0.05);
    assertThat(b.model()).isEqualTo(SprtModel.NORMALIZED);
  }

  @Test
  void customAlphaBetaBoundsAsymmetric() {
    SprtBounds b = SprtBounds.of(0.01, 0.10, 0.0, 5.0, SprtModel.LOGISTIC);
    // upper = log((1 - 0.10) / 0.01) = log(90)
    assertThat(b.upper()).isCloseTo(Math.log(90.0), org.assertj.core.data.Offset.offset(TOL));
    // lower = log(0.10 / (1 - 0.01)) = log(0.10 / 0.99)
    assertThat(b.lower())
        .isCloseTo(Math.log(0.10 / 0.99), org.assertj.core.data.Offset.offset(TOL));
  }

  @Test
  void invalidAlphaThrows() {
    assertThatThrownBy(() -> SprtBounds.of(0.0, 0.05, 0.0, 5.0, SprtModel.NORMALIZED))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("alpha");
    assertThatThrownBy(() -> SprtBounds.of(1.0, 0.05, 0.0, 5.0, SprtModel.NORMALIZED))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("alpha");
  }

  @Test
  void invalidBetaThrows() {
    assertThatThrownBy(() -> SprtBounds.of(0.05, 0.0, 0.0, 5.0, SprtModel.NORMALIZED))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("beta");
    assertThatThrownBy(() -> SprtBounds.of(0.05, 1.0, 0.0, 5.0, SprtModel.NORMALIZED))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("beta");
  }

  @Test
  void alphaPlusBetaTooHighThrows() {
    assertThatThrownBy(() -> SprtBounds.of(0.5, 0.6, 0.0, 5.0, SprtModel.NORMALIZED))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("alpha + beta");
  }

  @Test
  void elo0GreaterEqualElo1Throws() {
    assertThatThrownBy(() -> SprtBounds.of(0.05, 0.05, 5.0, 5.0, SprtModel.NORMALIZED))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("elo0");
    assertThatThrownBy(() -> SprtBounds.of(0.05, 0.05, 10.0, 5.0, SprtModel.NORMALIZED))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("elo0");
  }

  @Test
  void nullModelThrows() {
    assertThatThrownBy(() -> SprtBounds.of(0.05, 0.05, 0.0, 5.0, null))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("model");
  }

  @Test
  void negativeElo0Ok() {
    SprtBounds b = SprtBounds.of(0.05, 0.05, -5.0, 5.0, SprtModel.NORMALIZED);
    assertThat(b.elo0()).isEqualTo(-5.0);
    assertThat(b.elo1()).isEqualTo(5.0);
  }
}
