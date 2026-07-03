package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/** Tests unitaires de {@link Move}. */
class MoveTest {

  @Test
  @DisplayName("Move.NULL vaut 0")
  void nullMove() {
    assertThat(Move.NULL).isZero();
  }

  @Test
  @DisplayName("Round-trip exhaustif 64×64×4×4 = 65 536 combinaisons (encode/decode + invariant)")
  void roundTripExhaustive() {
    for (int from = 0; from < 64; from++) {
      for (int to = 0; to < 64; to++) {
        for (int type = 0; type < 4; type++) {
          for (int promo = 0; promo < 4; promo++) {
            int move = Move.encode(from, to, type, promo);
            assertThat(move >>> 16)
                .as(
                    "invariant 16 bits hauts à zéro pour from=%d to=%d type=%d promo=%d",
                    from, to, type, promo)
                .isZero();
            assertThat(Move.from(move)).isEqualTo(from);
            assertThat(Move.to(move)).isEqualTo(to);
            assertThat(Move.type(move)).isEqualTo(type);
            int expectedPromo = (type == MoveType.PROMOTION) ? promo : 0;
            assertThat(Move.promo(move))
                .as("promo canonique pour type=%d, promo passé=%d", type, promo)
                .isEqualTo(expectedPromo);
          }
        }
      }
    }
  }

  @Test
  @DisplayName("encode(from, to) équivaut à encode(from, to, NORMAL, 0)")
  void encodeShortFormEqualsLongForm() {
    for (int from = 0; from < 64; from++) {
      for (int to = 0; to < 64; to++) {
        int shortForm = Move.encode(from, to);
        int longForm = Move.encode(from, to, MoveType.NORMAL, 0);
        assertThat(shortForm).isEqualTo(longForm);
        assertThat(Move.type(shortForm)).isEqualTo(MoveType.NORMAL);
        assertThat(Move.promo(shortForm)).isZero();
      }
    }
  }

  @Test
  @DisplayName("promoToPieceType / pieceTypeToPromo : round-trip sur N/B/R/Q")
  void promoMappingRoundTrip() {
    for (int promo = 0; promo < 4; promo++) {
      int pieceType = Move.promoToPieceType(promo);
      assertThat(pieceType).isBetween(PieceType.KNIGHT, PieceType.QUEEN);
      assertThat(Move.pieceTypeToPromo(pieceType)).isEqualTo(promo);
    }
  }

  @Test
  @DisplayName("pieceTypeToPromo rejette les types non promouvants")
  void pieceTypeToPromoRejectsInvalid() {
    assertThatThrownBy(() -> Move.pieceTypeToPromo(PieceType.PAWN))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Move.pieceTypeToPromo(PieceType.KING))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Move.pieceTypeToPromo(PieceType.NONE))
        .isInstanceOf(IllegalArgumentException.class);
  }

  // -----------------------------------------------------------------------------------------
  // Sérialisation UCI
  // -----------------------------------------------------------------------------------------

  @ParameterizedTest(name = "{0} -> {1}")
  @CsvSource({
    "0, e2e4",
    "1, e7e8n",
    "2, e7e8b",
    "3, e7e8r",
    "4, e7e8q",
    "5, e1g1",
    "6, e5d6",
    "7, a1h8",
    "8, h8a1"
  })
  @DisplayName("toUci : valeurs canoniques sur cas représentatifs")
  void toUciKnownCases(int caseId, String expected) {
    int move =
        switch (caseId) {
          case 0 -> Move.encode(Square.E2, Square.E4);
          case 1 -> Move.encode(Square.E7, Square.E8, MoveType.PROMOTION, 0);
          case 2 -> Move.encode(Square.E7, Square.E8, MoveType.PROMOTION, 1);
          case 3 -> Move.encode(Square.E7, Square.E8, MoveType.PROMOTION, 2);
          case 4 -> Move.encode(Square.E7, Square.E8, MoveType.PROMOTION, 3);
          case 5 -> Move.encode(Square.E1, Square.G1, MoveType.CASTLING, 0);
          case 6 -> Move.encode(Square.E5, Square.D6, MoveType.EN_PASSANT, 0);
          case 7 -> Move.encode(Square.A1, Square.H8);
          case 8 -> Move.encode(Square.H8, Square.A1);
          default -> throw new IllegalStateException();
        };
    assertThat(Move.toUci(move)).isEqualTo(expected);
  }

  @Test
  @DisplayName("toUci : round-trip avec fromUci pour PROMOTION sur position vide")
  void toUciFromUciPromotionRoundTrip() {
    Position empty = new Position();
    for (int promo = 0; promo < 4; promo++) {
      int original = Move.encode(Square.E7, Square.E8, MoveType.PROMOTION, promo);
      String uci = Move.toUci(original);
      int decoded = Move.fromUci(uci, empty);
      assertThat(decoded).isEqualTo(original);
    }
  }

  @Test
  @DisplayName("fromUci : NORMAL sur position vide (rien sur from)")
  void fromUciNormalOnEmptyPosition() {
    Position empty = new Position();
    int m = Move.fromUci("e2e4", empty);
    assertThat(Move.from(m)).isEqualTo(Square.E2);
    assertThat(Move.to(m)).isEqualTo(Square.E4);
    assertThat(Move.type(m)).isEqualTo(MoveType.NORMAL);
    assertThat(Move.promo(m)).isZero();
  }

  @Test
  @DisplayName("fromUci : CASTLING détecté quand un roi est en E1 et UCI = e1g1")
  void fromUciCastlingWhiteKingside() {
    Position pos = new Position();
    pos.pieceBB[Piece.WHITE_KING] = 1L << Square.E1;
    pos.occupancyBB[Color.WHITE] = pos.pieceBB[Piece.WHITE_KING];
    pos.occupancyBB[2] = pos.occupancyBB[Color.WHITE];

    int m = Move.fromUci("e1g1", pos);
    assertThat(Move.type(m)).isEqualTo(MoveType.CASTLING);
    assertThat(Move.from(m)).isEqualTo(Square.E1);
    assertThat(Move.to(m)).isEqualTo(Square.G1);
  }

  @Test
  @DisplayName("fromUci : CASTLING noir (e8c8) avec roi noir en E8")
  void fromUciCastlingBlackQueenside() {
    Position pos = new Position();
    pos.pieceBB[Piece.BLACK_KING] = 1L << Square.E8;
    pos.occupancyBB[Color.BLACK] = pos.pieceBB[Piece.BLACK_KING];
    pos.occupancyBB[2] = pos.occupancyBB[Color.BLACK];

    int m = Move.fromUci("e8c8", pos);
    assertThat(Move.type(m)).isEqualTo(MoveType.CASTLING);
  }

  @Test
  @DisplayName("fromUci : EN_PASSANT détecté quand pawn sur from et to == epSquare")
  void fromUciEnPassantWhite() {
    Position pos = new Position();
    pos.pieceBB[Piece.WHITE_PAWN] = 1L << Square.E5;
    pos.occupancyBB[Color.WHITE] = pos.pieceBB[Piece.WHITE_PAWN];
    pos.occupancyBB[2] = pos.occupancyBB[Color.WHITE];
    pos.epSquare = Square.D6;

    int m = Move.fromUci("e5d6", pos);
    assertThat(Move.type(m)).isEqualTo(MoveType.EN_PASSANT);
  }

  @Test
  @DisplayName("fromUci : pawn move vers epSquare seulement = EN_PASSANT, sinon NORMAL")
  void fromUciNormalPawnMove() {
    Position pos = new Position();
    pos.pieceBB[Piece.WHITE_PAWN] = 1L << Square.E2;
    pos.occupancyBB[Color.WHITE] = pos.pieceBB[Piece.WHITE_PAWN];
    pos.occupancyBB[2] = pos.occupancyBB[Color.WHITE];
    // pas d'epSquare -> reste -1
    int m = Move.fromUci("e2e4", pos);
    assertThat(Move.type(m)).isEqualTo(MoveType.NORMAL);
  }

  @Test
  @DisplayName("fromUci : king qui bouge d'une seule case n'est PAS du castling")
  void fromUciKingSingleStepIsNormal() {
    Position pos = new Position();
    pos.pieceBB[Piece.WHITE_KING] = 1L << Square.E1;
    pos.occupancyBB[Color.WHITE] = pos.pieceBB[Piece.WHITE_KING];
    pos.occupancyBB[2] = pos.occupancyBB[Color.WHITE];

    int m = Move.fromUci("e1f1", pos);
    assertThat(Move.type(m)).isEqualTo(MoveType.NORMAL);
  }

  @Test
  @DisplayName("fromUci : longueur 5 avec promo -> PROMOTION même sans pion sur from")
  void fromUciPromotionByLength() {
    Position empty = new Position();
    int m = Move.fromUci("e7e8q", empty);
    assertThat(Move.type(m)).isEqualTo(MoveType.PROMOTION);
    assertThat(Move.promo(m)).isEqualTo(3);
  }

  @Test
  @DisplayName("fromUci : rejette null, longueur invalide, lettre de promo invalide")
  void fromUciRejectsInvalid() {
    Position empty = new Position();
    assertThatThrownBy(() -> Move.fromUci(null, empty))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Move.fromUci("", empty)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Move.fromUci("e2e", empty))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Move.fromUci("e2e4q5", empty))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Move.fromUci("e7e8x", empty))
        .isInstanceOf(IllegalArgumentException.class);
    // K majuscule rejetée (UCI normatif est en minuscule pour les pieces de promo)
    assertThatThrownBy(() -> Move.fromUci("e7e8Q", empty))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("Constructeur privé non instanciable")
  void constructorIsPrivate() throws Exception {
    var ctor = Move.class.getDeclaredConstructor();
    ctor.setAccessible(true);
    assertThatThrownBy(ctor::newInstance).hasCauseInstanceOf(AssertionError.class);
  }
}
