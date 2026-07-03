package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.board.Move;
import org.nanozero.board.MoveGen;
import org.nanozero.board.MoveType;
import org.nanozero.board.Square;

/**
 * Tests unitaires de {@link UciMoveCodec} (cf. SPEC §5.4, ADR-005, §12 phase 2).
 *
 * <p>Couvre :
 *
 * <ul>
 *   <li>Round-trip exhaustif sur tous les coups légaux de la position de départ.
 *   <li>Round-trip 1000 positions aléatoires (seed fixe pour déterminisme).
 *   <li>Cas particuliers encode : push simple, promotion 4 pièces, roque kingside / queenside ×
 *       blanc / noir, en passant.
 *   <li>Cas particuliers decode : push, promotion lower/upper case, roque, en passant, longueur
 *       invalide, square invalide, char promotion invalide, coup illégal, args null.
 * </ul>
 */
class UciMoveCodecTest {

  // -------------------------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------------------------

  /**
   * Cherche dans {@code state.legalMoves} le premier coup avec {@code from}/{@code to} donnés. Pour
   * les promotions, utilise {@link #findPromotion}.
   */
  private static int findMove(GameState state, int from, int to) {
    int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = state.generateMoves(buf, 0);
    for (int i = 0; i < n; i++) {
      if (Move.from(buf[i]) == from
          && Move.to(buf[i]) == to
          && Move.type(buf[i]) != MoveType.PROMOTION) {
        return buf[i];
      }
    }
    throw new AssertionError(
        "no legal non-promotion move from "
            + Square.toAlgebraic(from)
            + " to "
            + Square.toAlgebraic(to));
  }

  private static int findPromotion(GameState state, int from, int to, int promoBits) {
    int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = state.generateMoves(buf, 0);
    for (int i = 0; i < n; i++) {
      if (Move.from(buf[i]) == from
          && Move.to(buf[i]) == to
          && Move.type(buf[i]) == MoveType.PROMOTION
          && Move.promo(buf[i]) == promoBits) {
        return buf[i];
      }
    }
    throw new AssertionError("no promotion move matching from/to/promo");
  }

  // -------------------------------------------------------------------------------------------
  // Round-trip exhaustif
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Round-trip startpos : 20 coups légaux, 0 divergence")
  void roundTripStartposLegalMoves() {
    var state = new GameState();
    int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = state.generateMoves(buf, 0);
    assertThat(n).isEqualTo(20);
    for (int i = 0; i < n; i++) {
      int m = buf[i];
      String s = UciMoveCodec.encode(m);
      int decoded = UciMoveCodec.decode(s, state);
      assertThat(decoded).as("round-trip on move %s", s).isEqualTo(m);
    }
  }

  @Test
  @DisplayName("Round-trip 1000 positions aléatoires (seed fixe), 0 divergence")
  void roundTrip1000RandomPositions() {
    Random rng = new Random(42L);
    int totalRoundTrips = 0;
    for (int iter = 0; iter < 1000; iter++) {
      var state = new GameState();
      int depth = rng.nextInt(40) + 1;
      int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
      for (int d = 0; d < depth; d++) {
        if (state.isTerminal()) {
          break;
        }
        int n = state.generateMoves(buf, 0);
        if (n == 0) {
          break;
        }
        state.applyMove(buf[rng.nextInt(n)]);
      }
      if (state.isTerminal()) {
        continue;
      }
      int n = state.generateMoves(buf, 0);
      for (int i = 0; i < n; i++) {
        int m = buf[i];
        String s = UciMoveCodec.encode(m);
        int decoded = UciMoveCodec.decode(s, state);
        assertThat(decoded).as("iter=%d move=%s", iter, s).isEqualTo(m);
        totalRoundTrips++;
      }
    }
    assertThat(totalRoundTrips).as("total round-trips effectués").isGreaterThan(10_000);
  }

  // -------------------------------------------------------------------------------------------
  // encode : cas particuliers
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("encode push simple : e2e4 sur startpos")
  void encodeSimplePush() {
    var state = new GameState();
    int e2 = Square.fromAlgebraic("e2");
    int e4 = Square.fromAlgebraic("e4");
    int m = findMove(state, e2, e4);
    assertThat(UciMoveCodec.encode(m)).isEqualTo("e2e4");
  }

  @Test
  @DisplayName("encode promotion 4 pièces : e7e8q/r/b/n")
  void encodePromotionAllPieces() {
    var state = new GameState("k7/4P3/8/8/8/8/8/7K w - - 0 1");
    int e7 = Square.fromAlgebraic("e7");
    int e8 = Square.fromAlgebraic("e8");
    assertThat(UciMoveCodec.encode(findPromotion(state, e7, e8, 0))).isEqualTo("e7e8n");
    assertThat(UciMoveCodec.encode(findPromotion(state, e7, e8, 1))).isEqualTo("e7e8b");
    assertThat(UciMoveCodec.encode(findPromotion(state, e7, e8, 2))).isEqualTo("e7e8r");
    assertThat(UciMoveCodec.encode(findPromotion(state, e7, e8, 3))).isEqualTo("e7e8q");
  }

  @Test
  @DisplayName("encode castling kingside blanc : e1g1")
  void encodeCastlingKingsideWhite() {
    var state = new GameState("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    int e1 = Square.fromAlgebraic("e1");
    int g1 = Square.fromAlgebraic("g1");
    int m = findMove(state, e1, g1);
    assertThat(Move.type(m)).isEqualTo(MoveType.CASTLING);
    assertThat(UciMoveCodec.encode(m)).isEqualTo("e1g1");
  }

  @Test
  @DisplayName("encode castling queenside blanc : e1c1")
  void encodeCastlingQueensideWhite() {
    var state = new GameState("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    int e1 = Square.fromAlgebraic("e1");
    int c1 = Square.fromAlgebraic("c1");
    int m = findMove(state, e1, c1);
    assertThat(Move.type(m)).isEqualTo(MoveType.CASTLING);
    assertThat(UciMoveCodec.encode(m)).isEqualTo("e1c1");
  }

  @Test
  @DisplayName("encode castling kingside noir : e8g8")
  void encodeCastlingKingsideBlack() {
    var state = new GameState("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1");
    int e8 = Square.fromAlgebraic("e8");
    int g8 = Square.fromAlgebraic("g8");
    int m = findMove(state, e8, g8);
    assertThat(Move.type(m)).isEqualTo(MoveType.CASTLING);
    assertThat(UciMoveCodec.encode(m)).isEqualTo("e8g8");
  }

  @Test
  @DisplayName("encode castling queenside noir : e8c8")
  void encodeCastlingQueensideBlack() {
    var state = new GameState("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1");
    int e8 = Square.fromAlgebraic("e8");
    int c8 = Square.fromAlgebraic("c8");
    int m = findMove(state, e8, c8);
    assertThat(Move.type(m)).isEqualTo(MoveType.CASTLING);
    assertThat(UciMoveCodec.encode(m)).isEqualTo("e8c8");
  }

  @Test
  @DisplayName("encode en passant : pas de marqueur spécial, juste from+to")
  void encodeEnPassant() {
    // After 1.e4 d5 2.e5 f5 — black pawn just moved f7-f5, ep target = f6.
    // White e5 can capture e5xf6 en passant.
    var state = new GameState("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
    int e5 = Square.fromAlgebraic("e5");
    int f6 = Square.fromAlgebraic("f6");
    int m = findMove(state, e5, f6);
    assertThat(Move.type(m)).isEqualTo(MoveType.EN_PASSANT);
    assertThat(UciMoveCodec.encode(m)).isEqualTo("e5f6");
  }

  // -------------------------------------------------------------------------------------------
  // decode : cas particuliers
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("decode push simple : e2e4 sur startpos")
  void decodeSimplePush() {
    var state = new GameState();
    int decoded = UciMoveCodec.decode("e2e4", state);
    assertThat(Move.from(decoded)).isEqualTo(Square.fromAlgebraic("e2"));
    assertThat(Move.to(decoded)).isEqualTo(Square.fromAlgebraic("e4"));
    assertThat(Move.type(decoded)).isEqualTo(MoveType.NORMAL);
    // Round-trip
    assertThat(UciMoveCodec.encode(decoded)).isEqualTo("e2e4");
  }

  @Test
  @DisplayName("decode promotion lowercase : e7e8q produit Move PROMOTION queen")
  void decodePromotionLowercase() {
    var state = new GameState("k7/4P3/8/8/8/8/8/7K w - - 0 1");
    int decoded = UciMoveCodec.decode("e7e8q", state);
    assertThat(Move.type(decoded)).isEqualTo(MoveType.PROMOTION);
    assertThat(Move.promo(decoded)).isEqualTo(3); // queen
    assertThat(UciMoveCodec.encode(decoded)).isEqualTo("e7e8q");
  }

  @Test
  @DisplayName("decode promotion uppercase : tolérance UCI sur la casse, output minuscule strict")
  void decodePromotionUppercaseTolerated() {
    var state = new GameState("k7/4P3/8/8/8/8/8/7K w - - 0 1");
    int decodedQ = UciMoveCodec.decode("e7e8Q", state);
    int decodedR = UciMoveCodec.decode("e7e8R", state);
    int decodedB = UciMoveCodec.decode("e7e8B", state);
    int decodedN = UciMoveCodec.decode("e7e8N", state);
    assertThat(UciMoveCodec.encode(decodedQ)).isEqualTo("e7e8q");
    assertThat(UciMoveCodec.encode(decodedR)).isEqualTo("e7e8r");
    assertThat(UciMoveCodec.encode(decodedB)).isEqualTo("e7e8b");
    assertThat(UciMoveCodec.encode(decodedN)).isEqualTo("e7e8n");
  }

  @Test
  @DisplayName("decode castling : e1g1 produit Move CASTLING (pas push)")
  void decodeCastling() {
    var state = new GameState("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1");
    int decoded = UciMoveCodec.decode("e1g1", state);
    assertThat(Move.type(decoded)).isEqualTo(MoveType.CASTLING);
  }

  @Test
  @DisplayName("decode en passant : e5f6 produit Move EN_PASSANT (pas capture standard)")
  void decodeEnPassant() {
    var state = new GameState("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3");
    int decoded = UciMoveCodec.decode("e5f6", state);
    assertThat(Move.type(decoded)).isEqualTo(MoveType.EN_PASSANT);
  }

  // -------------------------------------------------------------------------------------------
  // decode : validations
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("decode null uciMove → NPE")
  void decodeNullStringThrowsNpe() {
    var state = new GameState();
    assertThatThrownBy(() -> UciMoveCodec.decode(null, state))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("uciMove");
  }

  @Test
  @DisplayName("decode null state → NPE")
  void decodeNullStateThrowsNpe() {
    assertThatThrownBy(() -> UciMoveCodec.decode("e2e4", null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("state");
  }

  @Test
  @DisplayName("decode longueur invalide → IAE")
  void decodeInvalidLengthThrowsIae() {
    var state = new GameState();
    assertThatThrownBy(() -> UciMoveCodec.decode("", state))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("invalid UCI move length");
    assertThatThrownBy(() -> UciMoveCodec.decode("e2", state))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("length");
    assertThatThrownBy(() -> UciMoveCodec.decode("e2e4q5", state))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("length");
  }

  @Test
  @DisplayName("decode square invalide → IAE avec contexte UCI complet")
  void decodeInvalidSquareThrowsIae() {
    var state = new GameState();
    assertThatThrownBy(() -> UciMoveCodec.decode("z9z9", state))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("invalid square");
    assertThatThrownBy(() -> UciMoveCodec.decode("a0b1", state))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("invalid square");
  }

  @Test
  @DisplayName("decode promotion char invalide → IAE")
  void decodeInvalidPromotionCharThrowsIae() {
    var state = new GameState("k7/4P3/8/8/8/8/8/7K w - - 0 1");
    assertThatThrownBy(() -> UciMoveCodec.decode("e7e8k", state))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("invalid UCI promotion char");
    assertThatThrownBy(() -> UciMoveCodec.decode("e7e8x", state))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("invalid UCI promotion char");
  }

  @Test
  @DisplayName("decode coup illégal dans la position → IAE")
  void decodeIllegalMoveThrowsIae() {
    var state = new GameState();
    // Black not to move from startpos
    assertThatThrownBy(() -> UciMoveCodec.decode("e7e5", state))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("not found among legal moves");
    // Path blocked
    assertThatThrownBy(() -> UciMoveCodec.decode("a1a8", state))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("not found among legal moves");
  }

  @Test
  @DisplayName("decode promotion 4 chars sur position promotion-ready → IAE (manque char)")
  void decodePromotionMissingCharThrowsIae() {
    var state = new GameState("k7/4P3/8/8/8/8/8/7K w - - 0 1");
    // "e7e8" sans char promo : legalMoves de cette position n'a QUE des promotions
    // (le pion DOIT promouvoir), donc aucun match en 4 chars → IAE.
    assertThatThrownBy(() -> UciMoveCodec.decode("e7e8", state))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("not found among legal moves");
  }
}
