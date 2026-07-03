package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Tests unitaires de {@link Square}. */
class SquareTest {

  @Test
  @DisplayName("Constants A1..H8 cohérentes avec la formule make(file, rank)")
  void constantsCoherentWithMakeFormula() {
    assertThat(Square.A1).isEqualTo(0);
    assertThat(Square.H1).isEqualTo(7);
    assertThat(Square.A8).isEqualTo(56);
    assertThat(Square.H8).isEqualTo(63);
    assertThat(Square.E4).isEqualTo(Square.make(4, 3));
    assertThat(Square.D5).isEqualTo(Square.make(3, 4));
  }

  @Test
  @DisplayName("NB_SQUARES vaut 64 et NONE vaut -1")
  void cardinalConstants() {
    assertThat(Square.NB_SQUARES).isEqualTo(64);
    assertThat(Square.NONE).isEqualTo(-1);
  }

  @ParameterizedTest(name = "square {0}")
  @ValueSource(ints = {0, 1, 7, 8, 27, 28, 32, 56, 63})
  @DisplayName("file et rank cohérents avec make sur cases représentatives")
  void fileRankMakeRoundTrip(int square) {
    int f = Square.file(square);
    int r = Square.rank(square);
    assertThat(f).isBetween(0, 7);
    assertThat(r).isBetween(0, 7);
    assertThat(Square.make(f, r)).isEqualTo(square);
  }

  @Test
  @DisplayName("Round-trip exhaustif sur toutes les cases [0,63]")
  void roundTripAllSquares() {
    for (int sq = 0; sq < 64; sq++) {
      int f = Square.file(sq);
      int r = Square.rank(sq);
      assertThat(Square.make(f, r)).as("make(file=%d,rank=%d)", f, r).isEqualTo(sq);
      String alg = Square.toAlgebraic(sq);
      assertThat(alg).hasSize(2);
      assertThat(Square.fromAlgebraic(alg)).as("fromAlgebraic(%s)", alg).isEqualTo(sq);
    }
  }

  @Test
  @DisplayName("toAlgebraic sur cases connues")
  void toAlgebraicKnownCases() {
    assertThat(Square.toAlgebraic(Square.A1)).isEqualTo("a1");
    assertThat(Square.toAlgebraic(Square.E4)).isEqualTo("e4");
    assertThat(Square.toAlgebraic(Square.H8)).isEqualTo("h8");
    assertThat(Square.toAlgebraic(Square.B7)).isEqualTo("b7");
  }

  @Test
  @DisplayName("fromAlgebraic sur entrées connues")
  void fromAlgebraicKnownCases() {
    assertThat(Square.fromAlgebraic("a1")).isEqualTo(Square.A1);
    assertThat(Square.fromAlgebraic("e4")).isEqualTo(Square.E4);
    assertThat(Square.fromAlgebraic("h8")).isEqualTo(Square.H8);
    assertThat(Square.fromAlgebraic("c3")).isEqualTo(Square.C3);
  }

  @Test
  @DisplayName("toAlgebraic rejette les indices hors plage")
  void toAlgebraicRejectsOutOfRange() {
    assertThatThrownBy(() -> Square.toAlgebraic(-1)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Square.toAlgebraic(64)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Square.toAlgebraic(Integer.MIN_VALUE))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("fromAlgebraic rejette les entrées invalides")
  void fromAlgebraicRejectsInvalidInputs() {
    assertThatThrownBy(() -> Square.fromAlgebraic(null))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Square.fromAlgebraic("")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Square.fromAlgebraic("a"))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Square.fromAlgebraic("a12"))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Square.fromAlgebraic("i1"))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Square.fromAlgebraic("a9"))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Square.fromAlgebraic("A1"))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Square.fromAlgebraic("a0"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("Constructeur privé non instanciable")
  void constructorIsPrivate() throws Exception {
    var ctor = Square.class.getDeclaredConstructor();
    ctor.setAccessible(true);
    assertThatThrownBy(ctor::newInstance).hasCauseInstanceOf(AssertionError.class);
  }
}
