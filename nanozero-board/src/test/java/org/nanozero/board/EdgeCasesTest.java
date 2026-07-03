package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Suite de régression dédiée aux 17 cas piégeurs §8.6 du SPEC.
 *
 * <p>Cette classe constitue un tableau de bord consultable pour les contributeurs : chaque cas
 * référencé par §8.6 / §15.2 / §15.3 a au moins un test dédié, avec FEN explicite et résultat
 * attendu commenté. Les FENs sont alignées sur la version amendée du SPEC (patches A-E appliqués en
 * début phase 10).
 *
 * <p>Beaucoup de ces cas sont déjà couverts implicitement par {@code GameStateTest} (insufficient
 * material, mat/pat, 50-coups), {@code MoveGenTest} (EP discovered, pinned moves), ou la suite
 * Perft cross-validée par chesslib. {@code EdgeCasesTest} ne tente PAS d'être exhaustif sur les
 * variantes : il fixe une référence stable contre les régressions sur les cas critiques.
 *
 * <p>Catalogue documenté de toutes les FENs dans {@code src/test/resources/edge-cases.txt}.
 */
class EdgeCasesTest {

  // ===========================================================================================
  // Helpers
  // ===========================================================================================

  /** Collecte les coups légaux de la position courante au format UCI. */
  private static Set<String> collectUci(GameState gs) {
    int[] buffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = gs.generateMoves(buffer, 0);
    Set<String> out = new HashSet<>(n * 2);
    for (int i = 0; i < n; i++) {
      out.add(Move.toUci(buffer[i]));
    }
    return out;
  }

  /** Filtre les coups partant de la case algébrique donnée (par ex. "e2"). */
  private static Set<String> collectMovesFrom(GameState gs, String fromAlgebraic) {
    int fromSq = Square.fromAlgebraic(fromAlgebraic);
    int[] buffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = gs.generateMoves(buffer, 0);
    Set<String> out = new HashSet<>();
    for (int i = 0; i < n; i++) {
      if (Move.from(buffer[i]) == fromSq) {
        out.add(Move.toUci(buffer[i]));
      }
    }
    return out;
  }

  // ===========================================================================================
  // §8.6 cas 1, 9 et §15.2 — EP discovered check
  // ===========================================================================================

  @Nested
  @DisplayName("EP discovered check (§5.3.10, §15.2)")
  class EnPassantDiscoveredCheck {

    @Test
    @DisplayName(
        "EP horizontal : 8/8/8/KPp4r/8/8/8/4k3 → b5xc6 e.p. INTERDIT (tour h5 voit Ka5 après"
            + " retrait des pions)")
    void epDiscoveredCheckHorizontalRook() {
      GameState gs = new GameState("8/8/8/KPp4r/8/8/8/4k3 w - c6 0 1");
      assertThat(collectUci(gs)).doesNotContain("b5c6");
    }

    @Test
    @DisplayName(
        "EP diagonal : 7b/8/8/3Pp3/8/8/8/K6k → d5xe6 e.p. INTERDIT (fou h8 voit Ka1 après"
            + " retrait des pions)")
    void epDiscoveredCheckDiagonalBishop() {
      // Cf. SPEC §5.3.10 post-patch A : la vérification EP couvre rook ET bishop attackers.
      GameState gs = new GameState("7b/8/8/3Pp3/8/8/8/K6k w - e6 0 1");
      assertThat(collectUci(gs)).doesNotContain("d5e6");
    }

    @Test
    @DisplayName("EP légal quand aucune pièce ne se découvre : 4k3/8/8/3Pp3/8/8/8/4K3 → d5xe6 OK")
    void epLegalWhenNoDiscovery() {
      GameState gs = new GameState("4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1");
      assertThat(collectUci(gs)).contains("d5e6");
    }
  }

  // ===========================================================================================
  // §8.6 cas 2, 3, 4 et §15.3 — Pinned piece moves
  // ===========================================================================================

  @Nested
  @DisplayName("Pinned piece moves (§5.3.3, §15.3)")
  class PinnedPieceMoves {

    @Test
    @DisplayName("§15.3 ligne 1 : cavalier blanc cloué orthogonalement par tour noire → 0 coup")
    void pinnedKnightOrthogonalCannotMove() {
      GameState gs = new GameState("4k3/4r3/8/8/8/8/4N3/4K3 w - - 0 1");
      assertThat(collectMovesFrom(gs, "e2")).isEmpty();
    }

    @Test
    @DisplayName("§15.3 ligne 2 : fou blanc cloué orthogonalement par dame noire → 0 coup")
    void pinnedBishopOrthogonalCannotMove() {
      GameState gs = new GameState("4k3/8/8/8/4q3/8/4B3/4K3 w - - 0 1");
      assertThat(collectMovesFrom(gs, "e2")).isEmpty();
    }

    @Test
    @DisplayName(
        "§15.3 ligne 3 : dame blanche clouée verticalement → bouge le long du pin uniquement")
    void pinnedQueenAlongPinLine() {
      GameState gs = new GameState("4k3/8/8/4q3/8/8/4Q3/4K3 w - - 0 1");
      Set<String> queenMoves = collectMovesFrom(gs, "e2");
      // Le long de la ligne (e-file entre roi e1 et dame attaquante e5)
      assertThat(queenMoves).contains("e2e3", "e2e4", "e2e5");
      // Sortir de la ligne expose le roi
      assertThat(queenMoves).doesNotContain("e2d2", "e2f2", "e2h5", "e2a6");
    }

    @Test
    @DisplayName(
        "Pion pinné diagonalement : capture du cloueur le long du pin OK, push interdit"
            + " (8/6k1/5p2/4B3/8/8/8/7K)")
    void pinnedPawnCanCaptureAlongPinDiagonal() {
      // Position construite : pion noir f6 cloué diagonalement par fou blanc e5 (diagonale
      // e5-f6-g7). Pion peut capturer le fou en xe5 (sur la diagonale du pin), mais ne peut pas
      // pousser f6-f5 (sortie de la diagonale → roi noir g7 exposé au fou).
      GameState gs = new GameState("8/6k1/5p2/4B3/8/8/8/7K b - - 0 1");
      Set<String> pawnMoves = collectMovesFrom(gs, "f6");
      assertThat(pawnMoves).contains("f6e5");
      assertThat(pawnMoves).doesNotContain("f6f5");
    }

    @Test
    @DisplayName(
        "§15.3 ligne 5 (positif) : pas de pin → pion d5 noir bouge librement"
            + " (4k3/8/8/3pP3/8/8/8/4K3)")
    void noPinPawnMovesFreely() {
      GameState gs = new GameState("4k3/8/8/3pP3/8/8/8/4K3 b - - 0 1");
      Set<String> pawnMoves = collectMovesFrom(gs, "d5");
      assertThat(pawnMoves).contains("d5d4");
    }

    @Test
    @DisplayName(
        "§15.3 ligne 6 (post-patch B) : EP avec pin diagonal → dxc6 e.p. INTERDIT"
            + " (4k3/8/4K3/2pP4/8/1b6/8/8)")
    void epWithDiagonalPinForbidden() {
      // Fou b3 cloue le pion d5 sur la diagonale e6-d5-c4-b3 → roi blanc e6.
      GameState gs = new GameState("4k3/8/4K3/2pP4/8/1b6/8/8 w - c6 0 1");
      assertThat(collectUci(gs)).doesNotContain("d5c6");
    }
  }

  // ===========================================================================================
  // §8.6 cas 5, 6, 7 — Castling edge cases
  // ===========================================================================================

  @Nested
  @DisplayName("Castling edge cases (§5.3.7)")
  class CastlingEdgeCases {

    @Test
    @DisplayName(
        "Roque INTERDIT à travers case attaquée : O-O passe par f1 contrôlé par tour noire f2"
            + " (r3k2r/8/8/8/8/8/5r2/R3K2R)")
    void castlingForbiddenWhenIntermediateAttacked() {
      GameState gs = new GameState("r3k2r/8/8/8/8/8/5r2/R3K2R w KQkq - 0 1");
      Set<String> moves = collectUci(gs);
      assertThat(moves).doesNotContain("e1g1"); // O-O bloqué (f1 attaqué)
      // O-O-O reste légal car d1, c1 non attaqués
      assertThat(moves).contains("e1c1");
    }

    @Test
    @DisplayName(
        "Roque INTERDIT depuis échec : roi blanc en échec par tour noire e2"
            + " (r3k2r/8/8/8/8/8/4r3/R3K2R) → ni O-O ni O-O-O")
    void castlingForbiddenFromCheck() {
      GameState gs = new GameState("r3k2r/8/8/8/8/8/4r3/R3K2R w KQkq - 0 1");
      Set<String> moves = collectUci(gs);
      assertThat(moves).doesNotContain("e1g1", "e1c1");
    }

    @Test
    @DisplayName(
        "Castling rights partiels : Kk → seul O-O légal, O-O-O interdit"
            + " (r3k2r/8/8/8/8/8/8/R3K2R w Kk - 0 1)")
    void castlingForbiddenAfterRookMoved() {
      // Position canonique avec castling rights "Kk" : tours queenside considérées comme bougées
      // précédemment puis revenues, leurs droits perdus définitivement (cf. SPEC §3.5).
      GameState gs = new GameState("r3k2r/8/8/8/8/8/8/R3K2R w Kk - 0 1");
      Set<String> moves = collectUci(gs);
      assertThat(moves).contains("e1g1"); // O-O OK (droit kingside conservé)
      assertThat(moves).doesNotContain("e1c1"); // O-O-O interdit (droit queenside perdu)
    }
  }

  // ===========================================================================================
  // §8.6 cas 8 — Promotion avec capture (4 sous-promotions)
  // ===========================================================================================

  @Nested
  @DisplayName("Promotion edge cases (§3.4, §5.3.10)")
  class PromotionEdgeCases {

    @Test
    @DisplayName(
        "Promotion avec capture : 4 sous-promotions générées en e7xd8 ; push e7-e8 interdit"
            + " (3rk3/4P3/8/8/8/8/8/4K3)")
    void promotionWithCaptureGeneratesFourSubpromotions() {
      // Pion blanc e7, tour noire d8. La case e8 est occupée par le roi noir → pas de push
      // possible. Capture en d8 produit les 4 sous-promotions (Q, R, B, N).
      GameState gs = new GameState("3rk3/4P3/8/8/8/8/8/4K3 w - - 0 1");
      Set<String> pawnMoves = collectMovesFrom(gs, "e7");
      assertThat(pawnMoves).contains("e7d8q", "e7d8r", "e7d8b", "e7d8n");
      assertThat(pawnMoves).doesNotContain("e7e8q", "e7e8r", "e7e8b", "e7e8n");
    }
  }

  // ===========================================================================================
  // §8.6 cas 10, 11 — Mat / pat
  // ===========================================================================================

  @Nested
  @DisplayName("Checkmate and stalemate (§6.1)")
  class CheckmateAndStalemate {

    @Test
    @DisplayName(
        "Mat du couloir : roi blanc h1, pions f2/g2/h2, tour noire a1 mate → 0 coup, isCheckmate"
            + " true")
    void backRankMate() {
      GameState gs = new GameState("4k3/8/8/8/8/8/5PPP/r6K w - - 0 1");
      assertThat(gs.isCheckmate()).isTrue();
      assertThat(collectUci(gs)).isEmpty();
    }

    @Test
    @DisplayName("Pat construit : 7k/5Q2/6K1/8/8/8/8/8 b → roi noir h8 sans coup, isStalemate true")
    void stalematePosition() {
      GameState gs = new GameState("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
      assertThat(gs.isStalemate()).isTrue();
      assertThat(collectUci(gs)).isEmpty();
    }
  }

  // ===========================================================================================
  // §8.6 cas 12-15 — Insufficient material (matrice complète en GameStateTest, sentinelles ici)
  // ===========================================================================================

  @Nested
  @DisplayName("Insufficient material — sentinelles §8.6 (matrice complète en GameStateTest)")
  class InsufficientMaterial {

    @Test
    @DisplayName("KvK : isInsufficientMaterial true (4k3/8/8/8/8/8/8/4K3)")
    void kingVsKing() {
      assertThat(new GameState("4k3/8/8/8/8/8/8/4K3 w - - 0 1").isInsufficientMaterial()).isTrue();
    }

    @Test
    @DisplayName(
        "KBvKB même couleur (cases sombres) : isInsufficientMaterial true"
            + " (4k3/8/3b4/8/8/8/3B4/4K3)")
    void kbvKbSameColor() {
      assertThat(new GameState("4k3/8/3b4/8/8/8/3B4/4K3 w - - 0 1").isInsufficientMaterial())
          .isTrue();
    }

    @Test
    @DisplayName(
        "KBvKB couleurs différentes : isInsufficientMaterial false (4k3/3b4/8/8/8/8/3B4/4K3)")
    void kbvKbDifferentColors() {
      assertThat(new GameState("4k3/3b4/8/8/8/8/3B4/4K3 w - - 0 1").isInsufficientMaterial())
          .isFalse();
    }

    @Test
    @DisplayName("KNN vs K : isInsufficientMaterial false (4k3/8/8/8/8/2N5/4N3/4K3)")
    void knnVsK() {
      assertThat(new GameState("4k3/8/8/8/8/2N5/4N3/4K3 w - - 0 1").isInsufficientMaterial())
          .isFalse();
    }
  }

  // ===========================================================================================
  // §8.6 cas 16 — 50-move rule au seuil
  // ===========================================================================================

  @Nested
  @DisplayName("Fifty-move rule threshold (§6.3)")
  class FiftyMoveRule {

    @Test
    @DisplayName("halfmoveClock = 99 → isFiftyMoveRule false")
    void atThreshold99False() {
      assertThat(new GameState("4k3/8/8/8/8/8/8/3RK3 w - - 99 1").isFiftyMoveRule()).isFalse();
    }

    @Test
    @DisplayName("halfmoveClock = 100 → isFiftyMoveRule true")
    void atThreshold100True() {
      assertThat(new GameState("4k3/8/8/8/8/8/8/3RK3 w - - 100 1").isFiftyMoveRule()).isTrue();
    }
  }

  // ===========================================================================================
  // §8.6 cas 17 — Threefold repetition + cas avec castling rights différents
  // ===========================================================================================

  @Nested
  @DisplayName("Threefold repetition (§6.2, ADR-012)")
  class ThreefoldRepetition {

    @Test
    @DisplayName(
        "Threefold strict : 8 plies de cycle knight shuffle b1c3/b8c6/c3b1/c6b8 →"
            + " isRepetition(3) true")
    void threefoldExactKnightShuffle() {
      GameState gs = new GameState();
      String[] cycle = {"b1c3", "b8c6", "c3b1", "c6b8"};
      for (int rep = 0; rep < 2; rep++) {
        for (String uci : cycle) {
          gs.applyMove(Move.fromUci(uci, gs.currentPosition()));
        }
      }
      assertThat(gs.isRepetition(3)).isTrue();
    }

    @Test
    @DisplayName(
        "Mêmes bitboards mais castling rights ≠ → Zobrist différent → NE compte PAS comme la"
            + " même position")
    void sameBitboardsDifferentCastlingRightsDoNotMatch() {
      // FENs identiques au niveau bitboard mais castling rights différents.
      String fenAll = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";
      String fenBlackOnly = "r3k2r/8/8/8/8/8/8/R3K2R w kq - 0 1";
      GameState gsAll = new GameState(fenAll);
      GameState gsBlackOnly = new GameState(fenBlackOnly);
      // Les bitboards de pièces sont identiques, mais castlingRights diffère → Zobrist diffère
      // (cf. SPEC §5.6 : 4 constantes castling distinctes XOR-ées dans le hash).
      assertThat(gsAll.currentPosition().zobristHash())
          .isNotEqualTo(gsBlackOnly.currentPosition().zobristHash());
      // Conséquence : ces positions ne pourraient PAS être confondues par isRepetition (qui
      // compare les Zobrist hashes), conformément à la règle FIDE de répétition stricte.
      for (int p = 0; p < Piece.NB_PIECES; p++) {
        assertThat(gsAll.currentPosition().pieceBB(p))
            .as("piece %d bitboard equality", p)
            .isEqualTo(gsBlackOnly.currentPosition().pieceBB(p));
      }
    }
  }
}
