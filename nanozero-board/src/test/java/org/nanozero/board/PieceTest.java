package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Tests unitaires de {@link Piece}. */
class PieceTest {

  @Test
  @DisplayName("Constantes 0..11 + NONE=12 + NB_PIECES=12")
  void constants() {
    assertThat(Piece.WHITE_PAWN).isZero();
    assertThat(Piece.WHITE_KNIGHT).isEqualTo(1);
    assertThat(Piece.WHITE_BISHOP).isEqualTo(2);
    assertThat(Piece.WHITE_ROOK).isEqualTo(3);
    assertThat(Piece.WHITE_QUEEN).isEqualTo(4);
    assertThat(Piece.WHITE_KING).isEqualTo(5);
    assertThat(Piece.BLACK_PAWN).isEqualTo(6);
    assertThat(Piece.BLACK_KNIGHT).isEqualTo(7);
    assertThat(Piece.BLACK_BISHOP).isEqualTo(8);
    assertThat(Piece.BLACK_ROOK).isEqualTo(9);
    assertThat(Piece.BLACK_QUEEN).isEqualTo(10);
    assertThat(Piece.BLACK_KING).isEqualTo(11);
    assertThat(Piece.NB_PIECES).isEqualTo(12);
    assertThat(Piece.NONE).isEqualTo(12);
  }

  @Test
  @DisplayName("Layout d'indexation §3.2 : color × 6 + pieceType")
  void indexLayout() {
    assertThat(Piece.make(Color.WHITE, PieceType.PAWN)).isEqualTo(Piece.WHITE_PAWN);
    assertThat(Piece.make(Color.WHITE, PieceType.KING)).isEqualTo(Piece.WHITE_KING);
    assertThat(Piece.make(Color.BLACK, PieceType.PAWN)).isEqualTo(Piece.BLACK_PAWN);
    assertThat(Piece.make(Color.BLACK, PieceType.KING)).isEqualTo(Piece.BLACK_KING);
  }

  @Test
  @DisplayName("Round-trip exhaustif make(colorOf(p), typeOf(p)) == p sur les 12 pièces")
  void roundTripAllPieces() {
    for (int p = 0; p < Piece.NB_PIECES; p++) {
      int c = Piece.colorOf(p);
      int t = Piece.typeOf(p);
      assertThat(c).as("colorOf(%d)", p).isBetween(Color.WHITE, Color.BLACK);
      assertThat(t).as("typeOf(%d)", p).isBetween(PieceType.PAWN, PieceType.KING);
      assertThat(Piece.make(c, t)).as("make(%d,%d)", c, t).isEqualTo(p);
    }
  }

  @Test
  @DisplayName("colorOf et typeOf : valeurs canoniques")
  void colorOfTypeOfKnownValues() {
    assertThat(Piece.colorOf(Piece.WHITE_PAWN)).isEqualTo(Color.WHITE);
    assertThat(Piece.colorOf(Piece.WHITE_KING)).isEqualTo(Color.WHITE);
    assertThat(Piece.colorOf(Piece.BLACK_PAWN)).isEqualTo(Color.BLACK);
    assertThat(Piece.colorOf(Piece.BLACK_KING)).isEqualTo(Color.BLACK);
    assertThat(Piece.typeOf(Piece.WHITE_PAWN)).isEqualTo(PieceType.PAWN);
    assertThat(Piece.typeOf(Piece.WHITE_KNIGHT)).isEqualTo(PieceType.KNIGHT);
    assertThat(Piece.typeOf(Piece.BLACK_BISHOP)).isEqualTo(PieceType.BISHOP);
    assertThat(Piece.typeOf(Piece.BLACK_QUEEN)).isEqualTo(PieceType.QUEEN);
  }

  @Test
  @DisplayName("Constructeur privé non instanciable")
  void constructorIsPrivate() throws Exception {
    var ctor = Piece.class.getDeclaredConstructor();
    ctor.setAccessible(true);
    assertThatThrownBy(ctor::newInstance).hasCauseInstanceOf(AssertionError.class);
  }
}
