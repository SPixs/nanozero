package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Tests unitaires de {@link MoveType}. */
class MoveTypeTest {

  @Test
  @DisplayName("Constantes : NORMAL=0, PROMOTION=1, EN_PASSANT=2, CASTLING=3, NB_TYPES=4")
  void constants() {
    assertThat(MoveType.NORMAL).isZero();
    assertThat(MoveType.PROMOTION).isEqualTo(1);
    assertThat(MoveType.EN_PASSANT).isEqualTo(2);
    assertThat(MoveType.CASTLING).isEqualTo(3);
    assertThat(MoveType.NB_TYPES).isEqualTo(4);
  }

  @Test
  @DisplayName("Constructeur privé non instanciable")
  void constructorIsPrivate() throws Exception {
    var ctor = MoveType.class.getDeclaredConstructor();
    ctor.setAccessible(true);
    assertThatThrownBy(ctor::newInstance).hasCauseInstanceOf(AssertionError.class);
  }
}
