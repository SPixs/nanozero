package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Tests unitaires de {@link Color}. */
class ColorTest {

  @Test
  @DisplayName("WHITE=0, BLACK=1, NB_COLORS=2")
  void constants() {
    assertThat(Color.WHITE).isZero();
    assertThat(Color.BLACK).isEqualTo(1);
    assertThat(Color.NB_COLORS).isEqualTo(2);
  }

  @Test
  @DisplayName("opponent : involution sur 2 cycles")
  void opponentInvolution() {
    assertThat(Color.opponent(Color.WHITE)).isEqualTo(Color.BLACK);
    assertThat(Color.opponent(Color.BLACK)).isEqualTo(Color.WHITE);
    assertThat(Color.opponent(Color.opponent(Color.WHITE))).isEqualTo(Color.WHITE);
    assertThat(Color.opponent(Color.opponent(Color.BLACK))).isEqualTo(Color.BLACK);
  }

  @Test
  @DisplayName("Constructeur privé non instanciable")
  void constructorIsPrivate() throws Exception {
    var ctor = Color.class.getDeclaredConstructor();
    ctor.setAccessible(true);
    assertThatThrownBy(ctor::newInstance).hasCauseInstanceOf(AssertionError.class);
  }
}
