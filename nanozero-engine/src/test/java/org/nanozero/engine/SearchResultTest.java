package org.nanozero.engine;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class SearchResultTest {

  @Test
  @DisplayName("Construction valide : tous champs cohérents")
  void validConstruction() {
    SearchResult r =
        new SearchResult(
            0x1234,
            new int[] {0x1234, 0x5678},
            0.42f,
            1000,
            123_456_789L,
            new int[] {500, 300, 200},
            new int[] {0x1234, 0x5678, 0x9ABC},
            true);
    assertThat(r.bestMove()).isEqualTo(0x1234);
    assertThat(r.principalVariation()).containsExactly(0x1234, 0x5678);
    assertThat(r.value()).isEqualTo(0.42f);
    assertThat(r.simulationsCount()).isEqualTo(1000);
    assertThat(r.elapsedNanos()).isEqualTo(123_456_789L);
    assertThat(r.childVisits()).containsExactly(500, 300, 200);
    assertThat(r.childMoves()).containsExactly(0x1234, 0x5678, 0x9ABC);
    assertThat(r.terminated()).isTrue();
  }

  @Test
  @DisplayName("Cas simulationsCount=0 : value=NaN, PV vide, childVisits/Moves vides → OK")
  void zeroSimulationsCaseValid() {
    SearchResult r =
        new SearchResult(0, new int[0], Float.NaN, 0, 0L, new int[0], new int[0], false);
    assertThat(r.simulationsCount()).isZero();
    assertThat(Float.isNaN(r.value())).isTrue();
    assertThat(r.principalVariation()).isEmpty();
    assertThat(r.childVisits()).isEmpty();
    assertThat(r.childMoves()).isEmpty();
  }

  @Test
  @DisplayName("principalVariation null → NPE")
  void principalVariationNullRejected() {
    assertThatThrownBy(
            () -> new SearchResult(0, null, 0f, 1, 0L, new int[] {1}, new int[] {1}, true))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("principalVariation");
  }

  @Test
  @DisplayName("childVisits null → NPE")
  void childVisitsNullRejected() {
    assertThatThrownBy(() -> new SearchResult(0, new int[0], 0f, 1, 0L, null, new int[0], true))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("childVisits");
  }

  @Test
  @DisplayName("childMoves null → NPE")
  void childMovesNullRejected() {
    assertThatThrownBy(() -> new SearchResult(0, new int[0], 0f, 1, 0L, new int[0], null, true))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("childMoves");
  }

  @Test
  @DisplayName("childVisits.length != childMoves.length → IAE")
  void mismatchedChildArrayLengthsRejected() {
    assertThatThrownBy(
            () ->
                new SearchResult(
                    0, new int[0], 0f, 1, 0L, new int[] {1, 2, 3}, new int[] {1, 2}, true))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("childVisits.length")
        .hasMessageContaining("childMoves.length");
  }

  @Test
  @DisplayName("value > 1 → IAE")
  void valueAboveOneRejected() {
    assertThatThrownBy(
            () -> new SearchResult(0, new int[0], 1.5f, 1, 0L, new int[0], new int[0], true))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("value")
        .hasMessageContaining("[-1, 1]");
  }

  @Test
  @DisplayName("value < -1 → IAE")
  void valueBelowMinusOneRejected() {
    assertThatThrownBy(
            () -> new SearchResult(0, new int[0], -1.5f, 1, 0L, new int[0], new int[0], true))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("value");
  }

  @Test
  @DisplayName("value = NaN accepté (cas simulationsCount=0, I-Result-4)")
  void valueNaNAccepted() {
    new SearchResult(0, new int[0], Float.NaN, 0, 0L, new int[0], new int[0], false);
  }

  @Test
  @DisplayName("value aux bornes -1 et +1 acceptées (intervalle fermé)")
  void valueBoundariesAccepted() {
    new SearchResult(0, new int[0], -1.0f, 1, 0L, new int[0], new int[0], true);
    new SearchResult(0, new int[0], 1.0f, 1, 0L, new int[0], new int[0], true);
  }

  @Test
  @DisplayName("simulationsCount négatif → IAE")
  void simulationsCountNegativeRejected() {
    assertThatThrownBy(
            () -> new SearchResult(0, new int[0], 0f, -1, 0L, new int[0], new int[0], true))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("simulationsCount");
  }

  @Test
  @DisplayName("elapsedNanos négatif → IAE")
  void elapsedNanosNegativeRejected() {
    assertThatThrownBy(
            () -> new SearchResult(0, new int[0], 0f, 1, -1L, new int[0], new int[0], true))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("elapsedNanos");
  }

  @Test
  @DisplayName("valueAsCentipawns : value=0.5f → 50, value=-0.7f → -70")
  void valueAsCentipawnsTypical() {
    SearchResult positive =
        new SearchResult(0, new int[0], 0.5f, 1, 0L, new int[0], new int[0], true);
    SearchResult negative =
        new SearchResult(0, new int[0], -0.7f, 1, 0L, new int[0], new int[0], true);
    assertThat(positive.valueAsCentipawns()).isEqualTo(50);
    assertThat(negative.valueAsCentipawns()).isEqualTo(-70);
  }

  @Test
  @DisplayName("valueAsCentipawns : value=NaN → 0 (Math.round(NaN) == 0)")
  void valueAsCentipawnsNaN() {
    SearchResult r =
        new SearchResult(0, new int[0], Float.NaN, 0, 0L, new int[0], new int[0], false);
    assertThat(r.valueAsCentipawns()).isZero();
  }

  @Test
  @DisplayName("valueAsCentipawns : value=1.0f → 100 et value=-1.0f → -100 (bornes)")
  void valueAsCentipawnsBoundaries() {
    SearchResult max = new SearchResult(0, new int[0], 1.0f, 1, 0L, new int[0], new int[0], true);
    SearchResult min = new SearchResult(0, new int[0], -1.0f, 1, 0L, new int[0], new int[0], true);
    assertThat(max.valueAsCentipawns()).isEqualTo(100);
    assertThat(min.valueAsCentipawns()).isEqualTo(-100);
  }
}
