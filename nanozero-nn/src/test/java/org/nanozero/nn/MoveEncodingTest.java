package org.nanozero.nn;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.nanozero.board.Color;
import org.nanozero.board.Fen;
import org.nanozero.board.GameState;
import org.nanozero.board.Move;
import org.nanozero.board.MoveGen;
import org.nanozero.board.MoveType;
import org.nanozero.board.PieceType;
import org.nanozero.board.Position;
import org.nanozero.board.Square;

/**
 * Tests unitaires de {@link MoveEncoding}. Critère de complétion §13 phase 1 : round-trip exhaustif
 * sur 10 000 positions (invariant {@code I-ME-1}) + cas spéciaux EP, castling, sous-promotions.
 */
class MoveEncodingTest {

  // -----------------------------------------------------------------------------------------
  // Constantes et tables figées
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Constantes : POLICY_INDICES = 4672, POLICY_PLANES = 73")
  void policyConstants() {
    assertThat(MoveEncoding.POLICY_INDICES).isEqualTo(4672);
    assertThat(MoveEncoding.POLICY_PLANES).isEqualTo(73);
  }

  @Test
  @DisplayName("QUEEN_DELTAS : ordre normatif §5.5.2 (N, NE, E, SE, S, SW, W, NW)")
  void queenDeltasOrder() {
    int[][] expected = {
      {+1, 0}, {+1, +1}, {0, +1}, {-1, +1},
      {-1, 0}, {-1, -1}, {0, -1}, {+1, -1}
    };
    assertThat(MoveEncoding.QUEEN_DELTAS).isDeepEqualTo(expected);
  }

  @Test
  @DisplayName("KNIGHT_DELTAS : ordre normatif §5.5.2 (8 sauts L-shape)")
  void knightDeltasOrder() {
    int[][] expected = {
      {+2, +1}, {+1, +2}, {-1, +2}, {-2, +1},
      {-2, -1}, {-1, -2}, {+1, -2}, {+2, -1}
    };
    assertThat(MoveEncoding.KNIGHT_DELTAS).isDeepEqualTo(expected);
  }

  @Test
  @DisplayName("UNDERPROMO_FILE_DELTAS : capture-left / push / capture-right")
  void underpromoFileDeltas() {
    assertThat(MoveEncoding.UNDERPROMO_FILE_DELTAS).containsExactly(-1, 0, +1);
  }

  @Test
  @DisplayName("UNDERPROMO_PIECES : KNIGHT, BISHOP, ROOK (queen exclue)")
  void underpromoPieces() {
    assertThat(MoveEncoding.UNDERPROMO_PIECES)
        .containsExactly(PieceType.KNIGHT, PieceType.BISHOP, PieceType.ROOK);
  }

  // -----------------------------------------------------------------------------------------
  // Critère §13 phase 1 : round-trip exhaustif sur 10 000 positions
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "I-ME-1 : decode(encode(m, side)) == m pour TOUS les coups légaux de 10 000 positions"
          + " (random play seedé)")
  void roundTripExhaustive10kPositions() {
    Random rng = new Random(0xC0FFEEBEEFL);
    int[] moveBuffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int positionsTested = 0;
    int gameCount = 0;
    long totalRoundTrips = 0;

    while (positionsTested < 10_000) {
      GameState gs = new GameState();
      int maxPly = 1 + rng.nextInt(70);
      for (int ply = 0; ply < maxPly && positionsTested < 10_000; ply++) {
        int side = gs.currentPosition().sideToMove();
        int n = gs.generateMoves(moveBuffer, 0);
        if (n == 0) {
          break;
        }
        for (int i = 0; i < n; i++) {
          int original = moveBuffer[i];
          int idx = MoveEncoding.encode(original, side);
          assertThat(idx)
              .as(
                  "encode index hors plage : pos #%d move=%s fen=%s",
                  positionsTested, Move.toUci(original), gs.toFen())
              .isBetween(0, MoveEncoding.POLICY_INDICES - 1);
          int decoded = MoveEncoding.decode(idx, gs.currentPosition());
          assertThat(decoded)
              .as(
                  "round-trip divergence : pos #%d move=%s (orig=0x%04X, decoded=0x%04X) fen=%s",
                  positionsTested, Move.toUci(original), original, decoded, gs.toFen())
              .isEqualTo(original);
          totalRoundTrips++;
        }
        positionsTested++;
        // Avancer le jeu d'un coup pour générer la prochaine position.
        gs.applyMove(moveBuffer[rng.nextInt(n)]);
      }
      gameCount++;
    }
    assertThat(positionsTested).isEqualTo(10_000);
    assertThat(totalRoundTrips).isPositive();
  }

  // -----------------------------------------------------------------------------------------
  // Cas spéciaux : EN_PASSANT
  // -----------------------------------------------------------------------------------------

  @Nested
  @DisplayName("EN_PASSANT : encode/decode avec préservation du type")
  class EnPassantCases {

    @Test
    @DisplayName("White EP : e5xd6 sur position issue de 1.e4 d5 2.e5 d5-d4? non — préparation EP")
    void whiteEpCapture() {
      // Position : pion blanc e5, pion noir d5 vient de pousser, EP square d6, white to move.
      GameState gs = new GameState("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1");
      int move = Move.encode(Square.E5, Square.D6, MoveType.EN_PASSANT, 0);
      int idx = MoveEncoding.encode(move, Color.WHITE);
      int decoded = MoveEncoding.decode(idx, gs.currentPosition());
      assertThat(Move.type(decoded)).isEqualTo(MoveType.EN_PASSANT);
      assertThat(decoded).isEqualTo(move);
    }

    @Test
    @DisplayName("Black EP : exd3 sur position symétrique, black to move")
    void blackEpCapture() {
      // Position : pion noir e4, pion blanc d4 vient de pousser, EP square d3, black to move.
      GameState gs = new GameState("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1");
      int move = Move.encode(Square.E4, Square.D3, MoveType.EN_PASSANT, 0);
      int idx = MoveEncoding.encode(move, Color.BLACK);
      int decoded = MoveEncoding.decode(idx, gs.currentPosition());
      assertThat(Move.type(decoded)).isEqualTo(MoveType.EN_PASSANT);
      assertThat(decoded).isEqualTo(move);
    }
  }

  // -----------------------------------------------------------------------------------------
  // Cas spéciaux : CASTLING
  // -----------------------------------------------------------------------------------------

  @Nested
  @DisplayName("CASTLING : O-O et O-O-O des deux couleurs")
  class CastlingCases {

    @Test
    @DisplayName("White O-O : e1g1, type CASTLING")
    void whiteKingsideCastle() {
      GameState gs = new GameState("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
      int move = Move.encode(Square.E1, Square.G1, MoveType.CASTLING, 0);
      int decoded =
          MoveEncoding.decode(MoveEncoding.encode(move, Color.WHITE), gs.currentPosition());
      assertThat(decoded).isEqualTo(move);
      assertThat(Move.type(decoded)).isEqualTo(MoveType.CASTLING);
    }

    @Test
    @DisplayName("White O-O-O : e1c1, type CASTLING")
    void whiteQueensideCastle() {
      GameState gs = new GameState("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
      int move = Move.encode(Square.E1, Square.C1, MoveType.CASTLING, 0);
      int decoded =
          MoveEncoding.decode(MoveEncoding.encode(move, Color.WHITE), gs.currentPosition());
      assertThat(decoded).isEqualTo(move);
      assertThat(Move.type(decoded)).isEqualTo(MoveType.CASTLING);
    }

    @Test
    @DisplayName("Black O-O : e8g8, type CASTLING")
    void blackKingsideCastle() {
      GameState gs = new GameState("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
      int move = Move.encode(Square.E8, Square.G8, MoveType.CASTLING, 0);
      int decoded =
          MoveEncoding.decode(MoveEncoding.encode(move, Color.BLACK), gs.currentPosition());
      assertThat(decoded).isEqualTo(move);
      assertThat(Move.type(decoded)).isEqualTo(MoveType.CASTLING);
    }

    @Test
    @DisplayName("Black O-O-O : e8c8, type CASTLING")
    void blackQueensideCastle() {
      GameState gs = new GameState("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1");
      int move = Move.encode(Square.E8, Square.C8, MoveType.CASTLING, 0);
      int decoded =
          MoveEncoding.decode(MoveEncoding.encode(move, Color.BLACK), gs.currentPosition());
      assertThat(decoded).isEqualTo(move);
      assertThat(Move.type(decoded)).isEqualTo(MoveType.CASTLING);
    }
  }

  // -----------------------------------------------------------------------------------------
  // Cas spéciaux : promotions (queen + 3 underpromotions, 3 directions)
  // -----------------------------------------------------------------------------------------

  @Nested
  @DisplayName("Promotions : queen via queen-style, N/B/R via underpromotion (planes 64-72)")
  class PromotionCases {

    @Test
    @DisplayName("White e7-e8 push : 4 promotions (Q via queen-style, N/B/R via underpromo)")
    void whitePushPromotionsAllPieces() {
      GameState gs = new GameState("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1");
      int[] promos = {3, 0, 1, 2}; // Q, N, B, R
      for (int promo : promos) {
        int move = Move.encode(Square.E7, Square.E8, MoveType.PROMOTION, promo);
        int idx = MoveEncoding.encode(move, Color.WHITE);
        int decoded = MoveEncoding.decode(idx, gs.currentPosition());
        assertThat(decoded)
            .as("promo=%d round-trip : encoded=%d decoded=0x%04X", promo, idx, decoded)
            .isEqualTo(move);
      }
    }

    @Test
    @DisplayName("White e7xd8=R promotion-with-capture (capture-left, underpromo plane 70)")
    void whiteCaptureLeftRookPromotion() {
      GameState gs = new GameState("3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1");
      int move = Move.encode(Square.E7, Square.D8, MoveType.PROMOTION, 2); // 2 = ROOK
      int idx = MoveEncoding.encode(move, Color.WHITE);
      // plane attendu : 64 + directionIndex(=0 pour capture-left) * 3 + pieceIndex(=2 pour ROOK) =
      // 66
      assertThat(idx % MoveEncoding.POLICY_PLANES).isEqualTo(66);
      int decoded = MoveEncoding.decode(idx, gs.currentPosition());
      assertThat(decoded).isEqualTo(move);
    }

    @Test
    @DisplayName("White e7xf8=B promotion-with-capture (capture-right, underpromo plane 71)")
    void whiteCaptureRightBishopPromotion() {
      GameState gs = new GameState("4kn2/4P3/8/8/8/8/8/4K3 w - - 0 1");
      int move = Move.encode(Square.E7, Square.F8, MoveType.PROMOTION, 1); // 1 = BISHOP
      int idx = MoveEncoding.encode(move, Color.WHITE);
      // plane attendu : 64 + 2 (capture-right) * 3 + 1 (BISHOP) = 71
      assertThat(idx % MoveEncoding.POLICY_PLANES).isEqualTo(71);
      int decoded = MoveEncoding.decode(idx, gs.currentPosition());
      assertThat(decoded).isEqualTo(move);
    }

    @Test
    @DisplayName("Black e2-e1=N push (BLACK side, perspective flip)")
    void blackPushKnightPromotion() {
      GameState gs = new GameState("4k3/8/8/8/8/8/4p3/4K3 b - - 0 1");
      int move = Move.encode(Square.E2, Square.E1, MoveType.PROMOTION, 0); // KNIGHT
      int idx = MoveEncoding.encode(move, Color.BLACK);
      // En perspective P1 (after flip), from = e7, to = e8, deltaFile = 0, push = directionIndex 1.
      // plane attendu : 64 + 1 * 3 + 0 (KNIGHT) = 67
      assertThat(idx % MoveEncoding.POLICY_PLANES).isEqualTo(67);
      int decoded = MoveEncoding.decode(idx, gs.currentPosition());
      assertThat(decoded).isEqualTo(move);
    }

    @Test
    @DisplayName("Black exd1=B capture-promotion : perspective flip + capture-right en P1")
    void blackCaptureBishopPromotion() {
      // Black pawn e2 captures white piece on d1 with bishop promotion.
      GameState gs = new GameState("4k3/8/8/8/8/8/4p3/3RK3 b - - 0 1");
      int move = Move.encode(Square.E2, Square.D1, MoveType.PROMOTION, 1); // BISHOP
      int idx = MoveEncoding.encode(move, Color.BLACK);
      // En perspective P1 : from e2 ^ 56 = e7 (file 4, rank 6). to d1 ^ 56 = d8 (file 3, rank 7).
      // deltaFile P1 = 3 - 4 = -1 → directionIndex = 0 (capture-left en P1).
      // pieceIndex = 1 (BISHOP). plane = 64 + 0 * 3 + 1 = 65.
      assertThat(idx % MoveEncoding.POLICY_PLANES).isEqualTo(65);
      int decoded = MoveEncoding.decode(idx, gs.currentPosition());
      assertThat(decoded).isEqualTo(move);
    }
  }

  // -----------------------------------------------------------------------------------------
  // Cas spéciaux : knight moves (planes 56-63)
  // -----------------------------------------------------------------------------------------

  @Nested
  @DisplayName("Knight moves : 8 deltas testés sur startpos")
  class KnightMoveCases {

    @Test
    @DisplayName("White Nb1-c3 : queen-style? non, knight-shape (delta +2,+1) → plane 56")
    void whiteKnightB1c3PlaneIndex() {
      GameState gs = new GameState();
      int move = Move.encode(Square.B1, Square.C3);
      int idx = MoveEncoding.encode(move, Color.WHITE);
      // delta = (+2, +1) → KNIGHT_DELTAS[0] → plane 56
      assertThat(idx % MoveEncoding.POLICY_PLANES).isEqualTo(56);
      int decoded = MoveEncoding.decode(idx, gs.currentPosition());
      assertThat(decoded).isEqualTo(move);
    }

    @Test
    @DisplayName("White Ng1-f3 : delta +2,-1 → plane 56+7 = 63")
    void whiteKnightG1F3PlaneIndex() {
      GameState gs = new GameState();
      int move = Move.encode(Square.G1, Square.F3);
      int idx = MoveEncoding.encode(move, Color.WHITE);
      // delta = (+2, -1) → KNIGHT_DELTAS[7] → plane 63
      assertThat(idx % MoveEncoding.POLICY_PLANES).isEqualTo(63);
      int decoded = MoveEncoding.decode(idx, gs.currentPosition());
      assertThat(decoded).isEqualTo(move);
    }
  }

  // -----------------------------------------------------------------------------------------
  // Inversion de perspective : encode WHITE startpos vs BLACK position miroir
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Inversion perspective : white e2-e4 et black e7-e5 produisent la MÊME planeIndex")
  void perspectiveFlipSymmetry() {
    GameState whiteStart = new GameState();
    int whiteE2E4 = Move.encode(Square.E2, Square.E4);
    int whiteIdx = MoveEncoding.encode(whiteE2E4, Color.WHITE);

    GameState blackStart =
        new GameState("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    int blackE7E5 = Move.encode(Square.E7, Square.E5);
    int blackIdx = MoveEncoding.encode(blackE7E5, Color.BLACK);

    // Après flip, from = e7^56 = e2. plane identique (deux cases en avant N, distance 2, plane 1).
    assertThat(blackIdx).isEqualTo(whiteIdx);

    // Round-trips dans leurs positions respectives donnent les coups originaux.
    assertThat(MoveEncoding.decode(whiteIdx, whiteStart.currentPosition())).isEqualTo(whiteE2E4);
    assertThat(MoveEncoding.decode(blackIdx, blackStart.currentPosition())).isEqualTo(blackE7E5);
  }

  // -----------------------------------------------------------------------------------------
  // decodePolicy : softmax masqué + invariant I-ME-2 (somme = 1)
  // -----------------------------------------------------------------------------------------

  @Nested
  @DisplayName("decodePolicy : softmax masqué et invariant I-ME-2")
  class DecodePolicyCases {

    @Test
    @DisplayName("I-ME-2 : sum(priors[0..n-1]) == 1.0 (modulo 1e-6)")
    void sumEqualsOneOnStartpos() {
      GameState gs = new GameState();
      int[] legalMoves = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
      int n = gs.generateMoves(legalMoves, 0);
      assertThat(n).isEqualTo(20);

      Random rng = new Random(0xDEADBEEFL);
      float[] logits = new float[MoveEncoding.POLICY_INDICES];
      for (int i = 0; i < logits.length; i++) {
        logits[i] = (float) (rng.nextGaussian() * 2.0);
      }

      float[] priors = new float[n];
      MoveEncoding.decodePolicy(logits, legalMoves, n, Color.WHITE, priors);

      double sum = 0;
      for (int i = 0; i < n; i++) {
        assertThat(priors[i]).isGreaterThanOrEqualTo(0f);
        sum += priors[i];
      }
      assertThat((float) sum).isCloseTo(1.0f, within(1e-6f));
    }

    @Test
    @DisplayName("Logits égaux → distribution uniforme 1/n")
    void uniformLogitsProduceUniformDistribution() {
      GameState gs = new GameState();
      int[] legalMoves = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
      int n = gs.generateMoves(legalMoves, 0);

      float[] logits = new float[MoveEncoding.POLICY_INDICES]; // tous à 0
      float[] priors = new float[n];
      MoveEncoding.decodePolicy(logits, legalMoves, n, Color.WHITE, priors);

      float expected = 1.0f / n;
      for (int i = 0; i < n; i++) {
        assertThat(priors[i]).as("prior %d uniforme", i).isCloseTo(expected, within(1e-6f));
      }
    }

    @Test
    @DisplayName("Logits extrêmes (un grand, autres zéros) → masse concentrée sur le grand")
    void extremeLogitConcentratesMass() {
      GameState gs = new GameState();
      int[] legalMoves = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
      int n = gs.generateMoves(legalMoves, 0);

      float[] logits = new float[MoveEncoding.POLICY_INDICES];
      // Mettre un logit énorme sur le coup #0
      int idx0 = MoveEncoding.encode(legalMoves[0], Color.WHITE);
      logits[idx0] = 50f;

      float[] priors = new float[n];
      MoveEncoding.decodePolicy(logits, legalMoves, n, Color.WHITE, priors);

      assertThat(priors[0]).isGreaterThan(0.999f);
      double sum = 0;
      for (int i = 0; i < n; i++) {
        sum += priors[i];
      }
      assertThat((float) sum).isCloseTo(1.0f, within(1e-6f));
    }

    @Test
    @DisplayName(
        "Critère §13 phase 10 : invariant I-ME-2 sur 1000 positions random play (seed"
            + " déterministe)")
    void invariantSumEqualsOneOn1000RandomPositions() {
      // Génère 1000 positions diverses par random play depuis startpos. Si on tombe sur une
      // position terminale (aucun coup légal), on redémarre depuis startpos.
      Random walk = new Random(0xC0FFEEFEEDL);
      Random logitRng = new Random(0xDEADBEEFCAFEL);
      int[] legalMoves = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
      float[] logits = new float[MoveEncoding.POLICY_INDICES];
      float[] priors = new float[MoveGen.RECOMMENDED_BUFFER_SIZE];

      GameState gs = new GameState();
      int collected = 0;
      int safety = 0;
      while (collected < 1000) {
        if (++safety > 200_000) {
          throw new AssertionError(
              "random walk n'a pas généré 1000 positions non-terminales en 200k pas");
        }
        int n = gs.generateMoves(legalMoves, 0);
        if (n == 0) {
          // Position terminale : redémarre depuis startpos.
          gs = new GameState();
          continue;
        }

        // Logits gaussiens N(0, 2) reproductibles.
        for (int i = 0; i < logits.length; i++) {
          logits[i] = (float) (logitRng.nextGaussian() * 2.0);
        }

        int sideToMove = gs.currentPosition().sideToMove();
        MoveEncoding.decodePolicy(logits, legalMoves, n, sideToMove, priors);

        double sum = 0;
        for (int i = 0; i < n; i++) {
          assertThat(priors[i])
              .as("position %d : prior[%d] >= 0 (n=%d)", collected, i, n)
              .isGreaterThanOrEqualTo(0f);
          sum += priors[i];
        }
        assertThat((float) sum)
            .as("position %d : sum priors == 1.0 ± 1e-6 (n=%d)", collected, n)
            .isCloseTo(1.0f, within(1e-6f));

        // Avance d'un coup random pour la position suivante.
        int chosen = legalMoves[walk.nextInt(n)];
        gs.applyMove(chosen);
        collected++;
      }
      assertThat(collected).isEqualTo(1000);
    }

    @Test
    @DisplayName("Single legal move : prior unique == 1.0f exact")
    void singleLegalMoveYieldsExactlyOne() {
      // FEN forgée : roi blanc a8 en échec par tour noire a1 ; roi noir a6 contrôle a7/b7 ;
      // donc seule case d'évasion = b8. Probé : MoveGen produit exactement 1 coup légal (a8b8).
      // Si la FEN cesse un jour de donner 1 coup unique, le sanity assertEquals ci-dessous lève
      // un message explicite.
      GameState gs = new GameState("K7/8/k7/8/8/8/8/r7 w - - 0 1");
      int[] legalMoves = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
      int n = gs.generateMoves(legalMoves, 0);
      // Sanity : si la FEN ne donne pas exactement 1 coup légal sur cette machine, rappeler
      // le résultat pour faciliter le debug.
      assertThat(n)
          .as("FEN single-move attendue : exactement 1 coup légal, en a obtenu %d", n)
          .isEqualTo(1);

      float[] logits = new float[MoveEncoding.POLICY_INDICES];
      // Logit arbitraire ; n=1 → softmax sur 1 élément donne toujours 1.0.
      logits[MoveEncoding.encode(legalMoves[0], Color.WHITE)] = -5.7f;
      float[] priors = new float[1];
      MoveEncoding.decodePolicy(logits, legalMoves, 1, Color.WHITE, priors);
      assertThat(priors[0]).isEqualTo(1.0f);
    }

    @Test
    @DisplayName("Logits tous négatifs : softmax stable produit distribution valide")
    void allNegativeLogitsProduceValidDistribution() {
      // Subtract max gère les négatifs : exp(neg - neg) = exp(0..val raisonnable) ne déborde pas.
      GameState gs = new GameState();
      int[] legalMoves = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
      int n = gs.generateMoves(legalMoves, 0);
      Random rng = new Random(0xBADCAFEL);
      float[] logits = new float[MoveEncoding.POLICY_INDICES];
      // Tous négatifs autour de -50.
      for (int i = 0; i < logits.length; i++) {
        logits[i] = -50f + (float) rng.nextGaussian();
      }
      float[] priors = new float[n];
      MoveEncoding.decodePolicy(logits, legalMoves, n, Color.WHITE, priors);
      double sum = 0;
      for (int i = 0; i < n; i++) {
        assertThat(priors[i]).isGreaterThanOrEqualTo(0f);
        assertThat(Float.isFinite(priors[i])).isTrue();
        sum += priors[i];
      }
      assertThat((float) sum).isCloseTo(1.0f, within(1e-6f));
    }

    @Test
    @DisplayName("Monotonicité : logit_i > logit_j ⇒ prior_i > prior_j (100 positions)")
    void priorsMonotonicWithLogits() {
      Random walk = new Random(0x0BC1010A1L);
      Random logitRng = new Random(0x0BC1010B2L);
      int[] legalMoves = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
      float[] logits = new float[MoveEncoding.POLICY_INDICES];
      float[] priors = new float[MoveGen.RECOMMENDED_BUFFER_SIZE];

      GameState gs = new GameState();
      int checked = 0;
      int safety = 0;
      while (checked < 100) {
        if (++safety > 50_000) {
          throw new AssertionError("walk monotonicité : 50k pas sans 100 positions");
        }
        int n = gs.generateMoves(legalMoves, 0);
        if (n == 0) {
          gs = new GameState();
          continue;
        }
        if (n < 2) {
          // Single legal move : monotonicité vacuously true, on saute.
          gs.applyMove(legalMoves[0]);
          continue;
        }
        for (int i = 0; i < logits.length; i++) {
          logits[i] = (float) (logitRng.nextGaussian() * 2.0);
        }
        int sideToMove = gs.currentPosition().sideToMove();
        // Snapshot logits aux indices des coups légaux pour comparaison post-softmax.
        float[] inputLogits = new float[n];
        for (int i = 0; i < n; i++) {
          inputLogits[i] = logits[MoveEncoding.encode(legalMoves[i], sideToMove)];
        }
        MoveEncoding.decodePolicy(logits, legalMoves, n, sideToMove, priors);
        for (int i = 0; i < n; i++) {
          for (int j = 0; j < n; j++) {
            if (inputLogits[i] > inputLogits[j]) {
              assertThat(priors[i])
                  .as(
                      "monotonicité pos %d : logit[%d]=%f > logit[%d]=%f donc prior[%d]=%f >"
                          + " prior[%d]=%f",
                      checked, i, inputLogits[i], j, inputLogits[j], i, priors[i], j, priors[j])
                  .isGreaterThan(priors[j]);
            }
          }
        }
        gs.applyMove(legalMoves[walk.nextInt(n)]);
        checked++;
      }
      assertThat(checked).isEqualTo(100);
    }

    @Test
    @DisplayName("Zero-alloc sanity : 100 appels sur mêmes buffers, OK et invariant tenu")
    void zeroAllocOnRepeatedCalls() {
      GameState gs = new GameState();
      int[] legalMoves = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
      int n = gs.generateMoves(legalMoves, 0);
      float[] logits = new float[MoveEncoding.POLICY_INDICES];
      float[] priors = new float[n];
      Random rng = new Random(0xACA1100CL);
      for (int call = 0; call < 100; call++) {
        for (int i = 0; i < logits.length; i++) {
          logits[i] = (float) rng.nextGaussian();
        }
        MoveEncoding.decodePolicy(logits, legalMoves, n, Color.WHITE, priors);
        double sum = 0;
        for (int i = 0; i < n; i++) {
          sum += priors[i];
        }
        assertThat((float) sum).as("call %d", call).isCloseTo(1.0f, within(1e-6f));
      }
    }

    @Test
    @DisplayName("Black to move : decodePolicy applique l'inversion de perspective via encode()")
    void blackToMoveAppliesPerspectiveInversion() {
      // Position après 1.e4 : noirs au trait. 20 coups légaux noirs.
      GameState gs = new GameState();
      gs.applyMove(Move.encode(Square.E2, Square.E4, MoveType.NORMAL, 0));
      int[] legalMoves = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
      int n = gs.generateMoves(legalMoves, 0);
      assertThat(n).isEqualTo(20);

      // Vérifie que les indices encodés sont uniques (pas de collision entre 2 coups distincts).
      // C'est I-ME-1 dans le contexte de cette position : encode(m, BLACK) doit être bijectif sur
      // les coups légaux distincts.
      Set<Integer> seen = new java.util.TreeSet<>();
      for (int i = 0; i < n; i++) {
        int idx = MoveEncoding.encode(legalMoves[i], Color.BLACK);
        assertThat(idx)
            .as("idx coup %d hors [0, 4672)", i)
            .isBetween(0, MoveEncoding.POLICY_INDICES - 1);
        assertThat(seen.add(idx))
            .as("collision encode sur 2 coups noirs distincts à idx=%d", idx)
            .isTrue();
      }

      // Distribution valide via decodePolicy.
      Random rng = new Random(0xB1ACL);
      float[] logits = new float[MoveEncoding.POLICY_INDICES];
      for (int i = 0; i < logits.length; i++) {
        logits[i] = (float) rng.nextGaussian();
      }
      float[] priors = new float[n];
      MoveEncoding.decodePolicy(logits, legalMoves, n, Color.BLACK, priors);
      double sum = 0;
      for (int i = 0; i < n; i++) {
        assertThat(priors[i]).isGreaterThanOrEqualTo(0f);
        sum += priors[i];
      }
      assertThat((float) sum).isCloseTo(1.0f, within(1e-6f));
    }
  }

  // -----------------------------------------------------------------------------------------
  // Validation arguments §13 phase 10
  // -----------------------------------------------------------------------------------------

  @Nested
  @DisplayName("decodePolicy : validation des arguments")
  class DecodePolicyValidation {

    private final float[] validLogits = new float[MoveEncoding.POLICY_INDICES];
    private final int[] validLegalMoves =
        new int[] {Move.encode(Square.E2, Square.E4, MoveType.NORMAL, 0)};
    private final float[] validDest = new float[1];

    @Test
    @DisplayName("logits null → IAE 'logits must not be null'")
    void rejectNullLogits() {
      assertThatThrownBy(
              () -> MoveEncoding.decodePolicy(null, validLegalMoves, 1, Color.WHITE, validDest))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("logits");
    }

    @Test
    @DisplayName("logits.length != 4672 → IAE")
    void rejectWrongLogitsLength() {
      float[] short_ = new float[100];
      assertThatThrownBy(
              () -> MoveEncoding.decodePolicy(short_, validLegalMoves, 1, Color.WHITE, validDest))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("4672");
    }

    @Test
    @DisplayName("legalMoves null → IAE")
    void rejectNullLegalMoves() {
      assertThatThrownBy(
              () -> MoveEncoding.decodePolicy(validLogits, null, 1, Color.WHITE, validDest))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("legalMoves");
    }

    @Test
    @DisplayName("numLegalMoves == 0 → IAE explicite (terminal state)")
    void rejectZeroLegalMoves() {
      assertThatThrownBy(
              () ->
                  MoveEncoding.decodePolicy(
                      validLogits, validLegalMoves, 0, Color.WHITE, validDest))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("at least 1 legal move")
          .hasMessageContaining("terminal state");
    }

    @Test
    @DisplayName("numLegalMoves négatif → IAE")
    void rejectNegativeNumLegalMoves() {
      assertThatThrownBy(
              () ->
                  MoveEncoding.decodePolicy(
                      validLogits, validLegalMoves, -1, Color.WHITE, validDest))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("numLegalMoves > legalMoves.length → IAE")
    void rejectNumLegalMovesExceedsBuffer() {
      assertThatThrownBy(
              () ->
                  MoveEncoding.decodePolicy(
                      validLogits, validLegalMoves, 50, Color.WHITE, new float[50]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("exceeds legalMoves buffer");
    }

    @Test
    @DisplayName("dest null → IAE")
    void rejectNullDest() {
      assertThatThrownBy(
              () -> MoveEncoding.decodePolicy(validLogits, validLegalMoves, 1, Color.WHITE, null))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("dest");
    }

    @Test
    @DisplayName("dest.length < numLegalMoves → IAE")
    void rejectDestTooSmall() {
      int[] moves = new int[5];
      java.util.Arrays.fill(moves, validLegalMoves[0]);
      assertThatThrownBy(
              () -> MoveEncoding.decodePolicy(validLogits, moves, 5, Color.WHITE, new float[3]))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("dest length");
    }
  }

  // -----------------------------------------------------------------------------------------
  // Erreurs : indices hors plage, encodages invalides
  // -----------------------------------------------------------------------------------------

  @Nested
  @DisplayName("Cas d'erreur : indices hors plage et coups inencodables")
  class ErrorCases {

    @Test
    @DisplayName("decode(indice négatif) lève IllegalArgumentException")
    void decodeNegativeIndexThrows() {
      Position p = new Position();
      Fen.parse(Fen.STARTPOS, p);
      assertThatThrownBy(() -> MoveEncoding.decode(-1, p))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("decode(indice >= 4672) lève IllegalArgumentException")
    void decodeIndexTooLargeThrows() {
      Position p = new Position();
      Fen.parse(Fen.STARTPOS, p);
      assertThatThrownBy(() -> MoveEncoding.decode(4672, p))
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    @DisplayName("encode coverage : tous les indices retournés sont dans [0, 4671]")
    void allEncodedIndicesInRange() {
      // Sur 100 positions random, vérifier que chaque index encodé est valide.
      Random rng = new Random(0xBADBEEFL);
      int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
      Set<Integer> seen = new java.util.TreeSet<>();
      for (int trial = 0; trial < 100; trial++) {
        GameState gs = new GameState();
        int depth = 1 + rng.nextInt(40);
        for (int i = 0; i < depth; i++) {
          int n = gs.generateMoves(buf, 0);
          if (n == 0) {
            break;
          }
          gs.applyMove(buf[rng.nextInt(n)]);
        }
        int n = gs.generateMoves(buf, 0);
        int side = gs.currentPosition().sideToMove();
        for (int i = 0; i < n; i++) {
          int idx = MoveEncoding.encode(buf[i], side);
          assertThat(idx).isBetween(0, MoveEncoding.POLICY_INDICES - 1);
          seen.add(idx);
        }
      }
      // On doit avoir vu une diversité raisonnable d'indices distincts.
      assertThat(seen.size()).isGreaterThan(50);
    }
  }
}
