package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Tests unitaires de {@link Castling}. */
class CastlingTest {

  @Test
  @DisplayName("Bit positions WK=0x1, WQ=0x2, BK=0x4, BQ=0x8")
  void individualBits() {
    assertThat(Castling.WHITE_KINGSIDE).isEqualTo(0x1);
    assertThat(Castling.WHITE_QUEENSIDE).isEqualTo(0x2);
    assertThat(Castling.BLACK_KINGSIDE).isEqualTo(0x4);
    assertThat(Castling.BLACK_QUEENSIDE).isEqualTo(0x8);
  }

  @Test
  @DisplayName("Agrégats WHITE_BOTH=0x3, BLACK_BOTH=0xC, ALL=0xF, NONE=0x0")
  void aggregates() {
    assertThat(Castling.WHITE_BOTH).isEqualTo(0x3);
    assertThat(Castling.BLACK_BOTH).isEqualTo(0xC);
    assertThat(Castling.ALL).isEqualTo(0xF);
    assertThat(Castling.NONE).isZero();
  }

  @Test
  @DisplayName("has(rights, right) : retourne true ssi le bit est actif")
  void hasIndividualBits() {
    int rights = Castling.WHITE_KINGSIDE | Castling.BLACK_QUEENSIDE;
    assertThat(Castling.has(rights, Castling.WHITE_KINGSIDE)).isTrue();
    assertThat(Castling.has(rights, Castling.BLACK_QUEENSIDE)).isTrue();
    assertThat(Castling.has(rights, Castling.WHITE_QUEENSIDE)).isFalse();
    assertThat(Castling.has(rights, Castling.BLACK_KINGSIDE)).isFalse();
    assertThat(Castling.has(Castling.NONE, Castling.WHITE_KINGSIDE)).isFalse();
    assertThat(Castling.has(Castling.ALL, Castling.WHITE_KINGSIDE)).isTrue();
    assertThat(Castling.has(Castling.ALL, Castling.BLACK_QUEENSIDE)).isTrue();
  }

  @Test
  @DisplayName("remove(rights, right) : efface le bit ciblé sans toucher aux autres")
  void removeBits() {
    assertThat(Castling.remove(Castling.ALL, Castling.WHITE_KINGSIDE))
        .isEqualTo(Castling.WHITE_QUEENSIDE | Castling.BLACK_BOTH);
    assertThat(Castling.remove(Castling.ALL, Castling.WHITE_BOTH)).isEqualTo(Castling.BLACK_BOTH);
    assertThat(Castling.remove(Castling.NONE, Castling.WHITE_KINGSIDE)).isEqualTo(Castling.NONE);
    assertThat(Castling.remove(Castling.WHITE_KINGSIDE, Castling.WHITE_KINGSIDE))
        .isEqualTo(Castling.NONE);
  }

  @Test
  @DisplayName("Constructeur privé non instanciable")
  void constructorIsPrivate() throws Exception {
    var ctor = Castling.class.getDeclaredConstructor();
    ctor.setAccessible(true);
    assertThatThrownBy(ctor::newInstance).hasCauseInstanceOf(AssertionError.class);
  }
}
