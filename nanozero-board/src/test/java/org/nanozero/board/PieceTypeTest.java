package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Tests unitaires de {@link PieceType}. */
class PieceTypeTest {

  @Test
  @DisplayName("Constantes 0..5 + NONE=6 + NB_PIECE_TYPES=6")
  void constants() {
    assertThat(PieceType.PAWN).isZero();
    assertThat(PieceType.KNIGHT).isEqualTo(1);
    assertThat(PieceType.BISHOP).isEqualTo(2);
    assertThat(PieceType.ROOK).isEqualTo(3);
    assertThat(PieceType.QUEEN).isEqualTo(4);
    assertThat(PieceType.KING).isEqualTo(5);
    assertThat(PieceType.NONE).isEqualTo(6);
    assertThat(PieceType.NB_PIECE_TYPES).isEqualTo(6);
  }

  @Test
  @DisplayName("toChar : majuscules pour blancs, minuscules pour noirs")
  void toCharBaseMapping() {
    assertThat(PieceType.toChar(PieceType.PAWN, Color.WHITE)).isEqualTo('P');
    assertThat(PieceType.toChar(PieceType.KNIGHT, Color.WHITE)).isEqualTo('N');
    assertThat(PieceType.toChar(PieceType.BISHOP, Color.WHITE)).isEqualTo('B');
    assertThat(PieceType.toChar(PieceType.ROOK, Color.WHITE)).isEqualTo('R');
    assertThat(PieceType.toChar(PieceType.QUEEN, Color.WHITE)).isEqualTo('Q');
    assertThat(PieceType.toChar(PieceType.KING, Color.WHITE)).isEqualTo('K');

    assertThat(PieceType.toChar(PieceType.PAWN, Color.BLACK)).isEqualTo('p');
    assertThat(PieceType.toChar(PieceType.KNIGHT, Color.BLACK)).isEqualTo('n');
    assertThat(PieceType.toChar(PieceType.BISHOP, Color.BLACK)).isEqualTo('b');
    assertThat(PieceType.toChar(PieceType.ROOK, Color.BLACK)).isEqualTo('r');
    assertThat(PieceType.toChar(PieceType.QUEEN, Color.BLACK)).isEqualTo('q');
    assertThat(PieceType.toChar(PieceType.KING, Color.BLACK)).isEqualTo('k');
  }

  @Test
  @DisplayName("toChar rejette les types et couleurs hors plages")
  void toCharRejectsInvalid() {
    assertThatThrownBy(() -> PieceType.toChar(PieceType.NONE, Color.WHITE))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> PieceType.toChar(-1, Color.WHITE))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> PieceType.toChar(PieceType.PAWN, 2))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> PieceType.toChar(PieceType.PAWN, -1))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("Constructeur privé non instanciable")
  void constructorIsPrivate() throws Exception {
    var ctor = PieceType.class.getDeclaredConstructor();
    ctor.setAccessible(true);
    assertThatThrownBy(ctor::newInstance).hasCauseInstanceOf(AssertionError.class);
  }
}
