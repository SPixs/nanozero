package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Tests unitaires de {@link MoveGen}. Le critère §14 phase 4 est validé principalement par {@code
 * CrossValidationChesslibTest} (10 000 positions) ; cette classe couvre les invariants statiques
 * (counts perft d=1 sur les 6 positions de référence, ordre déterministe §3.6, cas spéciaux
 * castling/EP/promotion, échec).
 */
class MoveGenTest {

  // -----------------------------------------------------------------------------------------
  // Constantes API publique
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Constantes publiques : MAX_LEGAL_MOVES=218, RECOMMENDED_BUFFER_SIZE=256")
  void publicConstants() {
    assertThat(MoveGen.MAX_LEGAL_MOVES).isEqualTo(218);
    assertThat(MoveGen.RECOMMENDED_BUFFER_SIZE).isEqualTo(256);
  }

  // -----------------------------------------------------------------------------------------
  // Perft profondeur 1 sur les 6 positions de référence (cf. SPEC §15.1)
  // -----------------------------------------------------------------------------------------

  @ParameterizedTest(name = "{0} -> {1}")
  @CsvSource({
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1, 20",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1, 48",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1, 14",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2pP/R2Q1RK1 w kq - 0 1, 6",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8, 44",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10, 46"
  })
  @DisplayName("Perft d=1 sur les 6 positions de référence §15.1")
  void perftDepthOneReferencePositions(String fen, int expected) {
    Position p = Fen.parse(fen);
    int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int count = MoveGen.generateMoves(p, buf, 0);
    assertThat(count).isEqualTo(expected);
  }

  // -----------------------------------------------------------------------------------------
  // Ordre déterministe §3.6 sur startpos
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Ordre §3.6 sur startpos : 20 coups dans l'ordre normatif")
  void orderingOnStartpos() {
    Position p = Fen.parse(Fen.STARTPOS);
    int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int count = MoveGen.generateMoves(p, buf, 0);

    // Ordre attendu :
    //   1. KING / QUEEN / ROOK / BISHOP : aucun coup possible (toutes pièces bloquées)
    //   2. KNIGHT : par from croissant b1, g1 ; pour chaque, par to croissant
    //   3. PAWN : par from croissant a2..h2 ; pour chaque pawn, push 1 puis push 2
    String[] expected = {
      "b1a3", "b1c3", "g1f3", "g1h3", "a2a3", "a2a4", "b2b3", "b2b4", "c2c3", "c2c4", "d2d3",
      "d2d4", "e2e3", "e2e4", "f2f3", "f2f4", "g2g3", "g2g4", "h2h3", "h2h4"
    };
    assertThat(count).isEqualTo(expected.length);
    for (int i = 0; i < count; i++) {
      assertThat(Move.toUci(buf[i])).as("startpos move at index %d", i).isEqualTo(expected[i]);
    }
  }

  @Test
  @DisplayName("Ordre §3.6 : within-piece by from ascending et within-from by to ascending")
  void orderingWithinPiece() {
    // Position avec deux cavaliers en c3 et f3, pour vérifier que c3 (lower from) génère avant f3.
    Position p = Fen.parse("4k3/8/8/8/8/2N2N2/8/4K3 w - - 0 1");
    List<String> moves = generateUciMoves(p);
    int idxC3 = moves.indexOf("c3a2");
    int idxF3 = moves.indexOf("f3d2");
    assertThat(idxC3).isNotNegative();
    assertThat(idxF3).isNotNegative();
    assertThat(idxC3).as("knight c3 produces moves before knight f3").isLessThan(idxF3);
  }

  // -----------------------------------------------------------------------------------------
  // Cas spéciaux : castling
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "Castling : kingside et queenside générés quand chemin libre, droits OK, pas d'échec")
  void castlingWhenAvailable() {
    Position p = Fen.parse("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    List<Integer> castles = collectMovesByType(p, MoveType.CASTLING);
    assertThat(castles).hasSize(2);
    assertThat(castles)
        .anyMatch(m -> Move.from(m) == Square.E1 && Move.to(m) == Square.G1)
        .anyMatch(m -> Move.from(m) == Square.E1 && Move.to(m) == Square.C1);
  }

  @Test
  @DisplayName("Castling interdit si roi en échec")
  void castlingForbiddenWhenInCheck() {
    // Tour noire en e7 attaque le roi blanc en e1 via la colonne e.
    // Tous les droits white déclarés mais castling illégal car roi en échec.
    Position p = Fen.parse("3k4/4r3/8/8/8/8/8/R3K2R w KQ - 0 1");
    List<Integer> castles = collectMovesByType(p, MoveType.CASTLING);
    assertThat(castles).isEmpty();
  }

  @Test
  @DisplayName("Castling interdit si case intermédiaire attaquée (F1 attaquée pour kingside)")
  void castlingForbiddenWhenIntermediateAttacked() {
    // Tour noire en f7 attaque la colonne f → f1 attaqué.
    // Queenside : ni d1 ni c1 attaqués → autorisé.
    Position p = Fen.parse("4k3/5r2/8/8/8/8/8/R3K2R w KQ - 0 1");
    List<Integer> castles = collectMovesByType(p, MoveType.CASTLING);
    assertThat(castles)
        .as("queenside autorisé, kingside interdit (f1 attaquée par r-f7)")
        .anyMatch(m -> Move.to(m) == Square.C1)
        .noneMatch(m -> Move.to(m) == Square.G1);
  }

  @Test
  @DisplayName("Castling interdit si chemin occupé")
  void castlingForbiddenWhenPathBlocked() {
    // Cavalier blanc en b1 bloque le grand roque (b1 doit être vide)
    Position p = Fen.parse("4k3/8/8/8/8/8/8/RN2K2R w KQ - 0 1");
    List<Integer> castles = collectMovesByType(p, MoveType.CASTLING);
    assertThat(castles)
        .as("kingside autorisé, queenside bloqué par cavalier en b1")
        .anyMatch(m -> Move.to(m) == Square.G1)
        .noneMatch(m -> Move.to(m) == Square.C1);
  }

  // -----------------------------------------------------------------------------------------
  // Cas spéciaux : EP
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("EN_PASSANT : généré si epSquare et pawn adjacent")
  void enPassantGenerated() {
    // Après e7-e5 : white to move, ep=e6, white pawn at d5 peut prendre d5xe6 EP
    Position p = Fen.parse("rnbqkbnr/pppp1ppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3");
    List<Integer> eps = collectMovesByType(p, MoveType.EN_PASSANT);
    assertThat(eps).hasSize(1);
    assertThat(eps)
        .anyMatch(
            m ->
                Move.from(m) == Square.D5
                    && Move.to(m) == Square.E6
                    && Move.type(m) == MoveType.EN_PASSANT);
  }

  // -----------------------------------------------------------------------------------------
  // Cas spéciaux : promotion
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("PROMOTION : 4 coups (N, B, R, Q) dans cet ordre quand pawn atteint dernier rang")
  void promotionFourMoves() {
    // Roi noir en a8, pion blanc en e7, roi blanc en h1. e8 vide → 4 promotions (push e8).
    Position p = Fen.parse("k7/4P3/8/8/8/8/8/7K w - - 0 1");
    List<Integer> promos = collectMovesByType(p, MoveType.PROMOTION);
    assertThat(promos).hasSize(4);
    // Ordre § 3.6 : KNIGHT (0), BISHOP (1), ROOK (2), QUEEN (3).
    assertThat(Move.promo(promos.get(0))).isZero();
    assertThat(Move.promo(promos.get(1))).isEqualTo(1);
    assertThat(Move.promo(promos.get(2))).isEqualTo(2);
    assertThat(Move.promo(promos.get(3))).isEqualTo(3);
    for (int m : promos) {
      assertThat(Move.from(m)).isEqualTo(Square.E7);
      assertThat(Move.to(m)).isEqualTo(Square.E8);
    }
  }

  @Test
  @DisplayName("PROMOTION avec capture : 4 promotions par cible diagonale")
  void promotionCapture() {
    // Pion blanc en e7, pièces noires en d8 et f8 → 4 promotions × 2 captures + 4 push = 12
    Position p = Fen.parse("3r1n1k/4P3/8/8/8/8/8/7K w - - 0 1");
    List<Integer> promos = collectMovesByType(p, MoveType.PROMOTION);
    // 4 promos vers d8 (capture rook) + 4 vers e7 (push? non e7 est le pion lui-même) + 4 vers f8
    // (capture knight).
    // Wait: e7 push to e8 (empty? Pas de pièce sur e8 ici, vide), 4 promos.
    // Donc : d8 (capture) 4 + e8 (push) 4 + f8 (capture) 4 = 12 promotions.
    assertThat(promos).hasSize(12);
  }

  // -----------------------------------------------------------------------------------------
  // Filtrage de légalité (échec)
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Roi en échec : seuls les coups d'évasion sont retenus")
  void kingInCheckEvasionsOnly() {
    // Roi blanc en e1, tour noire en e7 → roi attaqué via e-file, doit s'évader.
    // Évasions possibles : e1d1, e1d2, e1f1, e1f2 (e2 attaqué par tour). Pas de pièce blanche
    // hors le roi → 4 coups légaux.
    Position p = Fen.parse("4k3/4r3/8/8/8/8/8/4K3 w - - 0 1");
    int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int count = MoveGen.generateMoves(p, buf, 0);
    assertThat(count).isEqualTo(4);
    List<String> uci = new ArrayList<>();
    for (int i = 0; i < count; i++) {
      uci.add(Move.toUci(buf[i]));
    }
    assertThat(uci).containsExactlyInAnyOrder("e1d1", "e1d2", "e1f1", "e1f2");
  }

  @Test
  @DisplayName("Pinned piece : ne peut pas quitter la ligne de pin")
  void pinnedPieceCannotLeavePinLine() {
    // Cavalier blanc en e2 cloué par tour noire en e7 sur le roi blanc en e1.
    // Le cavalier ne peut bouger nulle part (cloué orthogonalement, ne peut suivre la ligne).
    Position p = Fen.parse("4k3/4r3/8/8/8/8/4N3/4K3 w - - 0 1");
    int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int count = MoveGen.generateMoves(p, buf, 0);
    List<String> uci = new ArrayList<>();
    for (int i = 0; i < count; i++) {
      uci.add(Move.toUci(buf[i]));
    }
    // Aucun coup du cavalier (e2)
    assertThat(uci).noneMatch(s -> s.startsWith("e2"));
  }

  @Test
  @DisplayName("Pin orthogonal §15.3 : fou cloué sur file → aucun coup possible")
  void pinnedBishopOrthogonal() {
    // Fou blanc e2 cloué par dame noire e4 sur la file e du roi blanc e1.
    // Le fou ne peut bouger (clouage orthogonal sur file → mouvement diagonal interdit).
    Position p = Fen.parse("4k3/8/8/8/4q3/8/4B3/4K3 w - - 0 1");
    List<String> uci = generateUciMoves(p);
    assertThat(uci).noneMatch(s -> s.startsWith("e2"));
  }

  @Test
  @DisplayName("Pin orthogonal §15.3 : dame clouée file e → glisse uniquement sur la file")
  void pinnedQueenAlongFile() {
    // Dame blanche e2 clouée par dame noire e5 sur file e du roi blanc e1.
    // Coups autorisés : Qe2-e3, Qe2-e4, Qe2xe5 (le long de la file de pin).
    // Coups interdits : tout déplacement diagonal ou horizontal (Qe2-d2, Qe2-h5, etc.).
    Position p = Fen.parse("4k3/8/8/4q3/8/8/4Q3/4K3 w - - 0 1");
    List<String> uci = generateUciMoves(p);
    List<String> queenMoves = uci.stream().filter(s -> s.startsWith("e2")).toList();
    assertThat(queenMoves).containsExactlyInAnyOrder("e2e3", "e2e4", "e2e5");
  }

  // -----------------------------------------------------------------------------------------
  // Cas critique : EP discovered check (cf. SPEC §5.3.10, §15.2)
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("EP discovered check §15.2 : b5xc6 INTERDIT (tour h5 attaque le roi a5 après EP)")
  void epDiscoveredCheckSpec1521() {
    // Position normative §15.2 row 1 : la prise en passant b5xc6 retire b5 ET c5 de la rangée 5,
    // libérant l'attaque horizontale de la tour noire h5 sur le roi blanc a5.
    Position p = Fen.parse("8/8/8/KPp4r/8/8/8/4k3 w - c6 0 1");
    List<String> uci = generateUciMoves(p);
    assertThat(uci)
        .as("La prise EP b5c6 ne doit pas figurer dans les coups légaux : %s", uci)
        .doesNotContain("b5c6");
  }

  @Test
  @DisplayName("EP discovered check §15.2 (symétrique noir) : e4xf3 INTERDIT")
  void epDiscoveredCheckSymmetricBlack() {
    // Symétrique : pion noir e4 capture EP en f3 ; retirer e4 et f4 expose le roi noir g4 à la
    // tour blanche... non, ici on a la tour noire b4 qui attaque le roi NOIR g4. Donc c'est plutôt
    // le roi blanc qu'on regarde — non on cherche l'échec du côté qui prend l'EP (les noirs).
    // Construction symétrique : roi blanc a8, roi noir g4, tour BLANCHE b4, pion blanc f4 (vient
    // de pousser f2-f4), pion noir e4. Black à jouer ; EP=f3 ; e4xf3 retirerait e4 et f4 de la
    // rangée 4, exposant le roi noir g4 à la tour blanche b4.
    Position p = Fen.parse("K7/8/8/8/1R2pPk1/8/8/8 b - f3 0 1");
    List<String> uci = generateUciMoves(p);
    assertThat(uci)
        .as("La prise EP e4f3 ne doit pas figurer dans les coups légaux : %s", uci)
        .doesNotContain("e4f3");
  }

  @Test
  @DisplayName("EP standard sans découverte : reste légal (test de non-régression)")
  void epStandardLegal() {
    // Position simple où l'EP est légal (aucun slider ennemi sur la rangée du roi).
    Position p = Fen.parse("rnbqkbnr/pppp1ppp/8/3Pp3/8/8/PPP1PPPP/RNBQKBNR w KQkq e6 0 3");
    List<Integer> eps = collectMovesByType(p, MoveType.EN_PASSANT);
    assertThat(eps)
        .as("L'EP d5xe6 doit rester légal en position standard")
        .anyMatch(m -> Move.from(m) == Square.D5 && Move.to(m) == Square.E6);
  }

  // -----------------------------------------------------------------------------------------
  // generateCaptures
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("generateCaptures sur startpos : 0 captures (aucune pièce adverse atteignable)")
  void generateCapturesEmptyOnStartpos() {
    Position p = Fen.parse(Fen.STARTPOS);
    int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int count = MoveGen.generateCaptures(p, buf, 0);
    assertThat(count).isZero();
  }

  @Test
  @DisplayName("generateCaptures : sous-ensemble strict de generateMoves")
  void generateCapturesIsSubsetOfGenerateMoves() {
    Position p = Fen.parse("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
    int[] all = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int allCount = MoveGen.generateMoves(p, all, 0);
    int[] caps = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int capCount = MoveGen.generateCaptures(p, caps, 0);
    assertThat(capCount).isLessThanOrEqualTo(allCount);
    // Toute capture doit être dans l'ensemble complet
    for (int i = 0; i < capCount; i++) {
      int captureMove = caps[i];
      boolean found = false;
      for (int j = 0; j < allCount; j++) {
        if (all[j] == captureMove) {
          found = true;
          break;
        }
      }
      assertThat(found).as("capture %s in generateMoves", Move.toUci(captureMove)).isTrue();
    }
  }

  // -----------------------------------------------------------------------------------------
  // Helpers de test
  // -----------------------------------------------------------------------------------------

  private static List<Integer> collectMovesByType(Position p, int moveType) {
    int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int count = MoveGen.generateMoves(p, buf, 0);
    List<Integer> out = new ArrayList<>();
    for (int i = 0; i < count; i++) {
      if (Move.type(buf[i]) == moveType) {
        out.add(buf[i]);
      }
    }
    return out;
  }

  private static List<String> generateUciMoves(Position p) {
    int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int count = MoveGen.generateMoves(p, buf, 0);
    List<String> out = new ArrayList<>(count);
    for (int i = 0; i < count; i++) {
      out.add(Move.toUci(buf[i]));
    }
    return out;
  }
}
