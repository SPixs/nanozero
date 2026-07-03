package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Tests unitaires de {@link Fen}. Le critère de complétion §14 phase 3 impose un round-trip
 * Fen.parse → Fen.write identique sur 10 000 FEN générées via random play (initialement avec
 * chesslib comme oracle ; phase 12 refactorise avec {@link GameState} comme source pour supprimer
 * la dépendance chesslib), et la couverture explicite des 10 règles de validation §5.7.2 avec cas
 * positifs et négatifs.
 */
class FenTest {

  // -----------------------------------------------------------------------------------------
  // Constante STARTPOS et parse / write basiques
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("STARTPOS : FEN canonique de la position de départ")
  void startposConstant() {
    assertThat(Fen.STARTPOS).isEqualTo("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  }

  @Test
  @DisplayName("parse(STARTPOS) : bitboards corrects + état")
  void parseStartpos() {
    Position p = Fen.parse(Fen.STARTPOS);
    assertThat(p.pieceBB(Piece.WHITE_PAWN)).isEqualTo(0x000000000000FF00L);
    assertThat(p.pieceBB(Piece.BLACK_PAWN)).isEqualTo(0x00FF000000000000L);
    assertThat(p.pieceBB(Piece.WHITE_KING)).isEqualTo(1L << Square.E1);
    assertThat(p.pieceBB(Piece.BLACK_KING)).isEqualTo(1L << Square.E8);
    assertThat(p.pieceBB(Piece.WHITE_QUEEN)).isEqualTo(1L << Square.D1);
    assertThat(p.pieceBB(Piece.BLACK_QUEEN)).isEqualTo(1L << Square.D8);
    assertThat(p.occupancyBB(Color.WHITE)).isEqualTo(0x000000000000FFFFL);
    assertThat(p.occupancyBB(Color.BLACK)).isEqualTo(0xFFFF000000000000L);
    assertThat(p.allOccupancy()).isEqualTo(0xFFFF00000000FFFFL);
    assertThat(p.sideToMove()).isEqualTo(Color.WHITE);
    assertThat(p.castlingRights()).isEqualTo(Castling.ALL);
    assertThat(p.epSquare()).isEqualTo(Square.NONE);
    assertThat(p.halfmoveClock()).isZero();
    assertThat(p.fullmoveNumber()).isEqualTo(1);
  }

  @Test
  @DisplayName("parse(String, Position) : réécrit l'état préexistant intégralement")
  void parseInPlaceResetsExistingState() {
    Position p = new Position();
    // Pollue l'état avec des valeurs absurdes.
    p.pieceBB[Piece.WHITE_PAWN] = 0xFFFFFFFFFFFFFFFFL;
    p.castlingRights = Castling.ALL;
    p.epSquare = Square.E3;
    p.halfmoveClock = 42;
    p.fullmoveNumber = 99;

    Fen.parse(Fen.STARTPOS, p);
    assertThat(p.pieceBB(Piece.WHITE_PAWN)).isEqualTo(0x000000000000FF00L);
    assertThat(p.castlingRights()).isEqualTo(Castling.ALL);
    assertThat(p.epSquare()).isEqualTo(Square.NONE);
    assertThat(p.halfmoveClock()).isZero();
    assertThat(p.fullmoveNumber()).isEqualTo(1);
  }

  @Test
  @DisplayName("write(parse(STARTPOS)) == STARTPOS")
  void writeStartposRoundTrip() {
    assertThat(Fen.write(Fen.parse(Fen.STARTPOS))).isEqualTo(Fen.STARTPOS);
  }

  @Test
  @DisplayName("Position.toFen délègue à Fen.write (refactor phase 3)")
  void positionToFenDelegatesToFenWrite() {
    Position p = Fen.parse(Fen.STARTPOS);
    assertThat(p.toFen()).isEqualTo(Fen.write(p));
  }

  // -----------------------------------------------------------------------------------------
  // Round-trip 10 000 FEN via random play (seedé) — auto-sourcé en phase 12 (chesslib supprimé)
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Round-trip 10 000 FEN issues du random play : Fen.parse → Fen.write idempotent")
  void roundTripTenThousandViaRandomPlay() {
    Random rng = new Random(0xDEADBEEFL);
    int[] moveBuffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int collected = 0;
    int gameCount = 0;

    while (collected < 10_000) {
      GameState gs = new GameState();
      // Profondeur de partie limitée pour éviter halfmoveClock > 100 (rejeté par règle 8 §5.7.2).
      int maxPly = 1 + rng.nextInt(70);

      for (int ply = 0; ply < maxPly && collected < 10_000; ply++) {
        // 1) FEN source via toFen() : capture l'état courant.
        String sourceFen = gs.toFen();

        // 2) Parse via Fen.parse → re-write via toFen, vérifier idempotence.
        Position parsed = Fen.parse(sourceFen);
        String roundTripFen = Fen.write(parsed);
        assertThat(roundTripFen)
            .as(
                "FEN #%d (game %d, ply %d) round-trip : input=%s",
                collected, gameCount, ply, sourceFen)
            .isEqualTo(sourceFen);
        collected++;

        // 3) Avancer le jeu d'un coup pour générer la prochaine position.
        int n = gs.generateMoves(moveBuffer, 0);
        if (n == 0) {
          break;
        }
        gs.applyMove(moveBuffer[rng.nextInt(n)]);
      }
      gameCount++;
    }
    assertThat(collected).isEqualTo(10_000);
  }

  // -----------------------------------------------------------------------------------------
  // Règle 1 : structure (6 champs séparés par espaces simples)
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Règle 1 : 6 champs requis (positif via STARTPOS, négatifs sur structures cassées)")
  void rule1Structure() {
    // Positif : déjà couvert par parseStartpos.
    assertThatThrownBy(() -> Fen.parse(null)).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Fen.parse("")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Fen.parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Fen.parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w"))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () -> Fen.parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 extra"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  // -----------------------------------------------------------------------------------------
  // Règle 2 : position field (8 rangs / 8 cases / caractères autorisés)
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Règle 2 : 8 rangs séparés par /, 8 cases par rang, caractères PNBRQKpnbrqk12345678")
  void rule2Position() {
    // Positif : 7 rangs (manque un rang)
    assertThatThrownBy(() -> Fen.parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // 9 rangs
    assertThatThrownBy(
            () -> Fen.parse("rnbqkbnr/pppppppp/8/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // Rang totalisant 7 cases au lieu de 8
    assertThatThrownBy(() -> Fen.parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBN w KQkq - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // Rang totalisant 9 cases (un caractère de pièce en trop)
    assertThatThrownBy(() -> Fen.parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNRR w KQkq - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // Caractère invalide ('X')
    assertThatThrownBy(() -> Fen.parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNX w KQkq - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // Caractère invalide ('9')
    assertThatThrownBy(() -> Fen.parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/9NBQKBNR w KQkq - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  // -----------------------------------------------------------------------------------------
  // Règle 3 : exactement 1 roi blanc et 1 roi noir
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Règle 3 : exactement 1 roi blanc et 1 roi noir")
  void rule3KingsCount() {
    // Positif : startpos a 1+1 rois.
    assertThat(Fen.parse(Fen.STARTPOS).pieceBB(Piece.WHITE_KING)).isNotZero();
    // 0 roi blanc
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/8 w - - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // 0 roi noir
    assertThatThrownBy(() -> Fen.parse("8/8/8/8/8/8/8/4K3 w - - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // 2 rois blancs
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/K3K3 w - - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // 2 rois noirs
    assertThatThrownBy(() -> Fen.parse("k3k3/8/8/8/8/8/8/4K3 w - - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  // -----------------------------------------------------------------------------------------
  // Règle 4 : pas de pion sur les rangs 1 et 8
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Règle 4 : aucun pion sur rang 1 ou rang 8")
  void rule4PawnsOnEdgeRanks() {
    // Pion blanc sur rang 8
    assertThatThrownBy(() -> Fen.parse("P3k3/8/8/8/8/8/8/4K3 w - - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // Pion noir sur rang 8
    assertThatThrownBy(() -> Fen.parse("p3k3/8/8/8/8/8/8/4K3 w - - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // Pion blanc sur rang 1 (à coté du roi blanc en E1, sans interférence d'autre pièce)
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/P3K3 w - - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // Pion noir sur rang 1
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/p3K3 w - - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  // -----------------------------------------------------------------------------------------
  // Règle 5 : side strict
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Règle 5 : side strict 'w' ou 'b'")
  void rule5Side() {
    assertThat(Fen.parse("4k3/8/8/8/8/8/8/4K3 w - - 0 1").sideToMove()).isEqualTo(Color.WHITE);
    assertThat(Fen.parse("4k3/8/8/8/8/8/8/4K3 b - - 0 1").sideToMove()).isEqualTo(Color.BLACK);
    // Majuscule rejetée
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 W - - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // Lettre quelconque
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 z - - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // Plusieurs caractères
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 wb - - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  // -----------------------------------------------------------------------------------------
  // Règle 6 : castling rights — format ordonné + cohérence positions roi/tour
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Règle 6a : format ordonné KQkq")
  void rule6CastlingFormat() {
    // Positifs
    assertThat(Fen.parse("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").castlingRights())
        .isEqualTo(Castling.ALL);
    assertThat(Fen.parse("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1").castlingRights())
        .isEqualTo(Castling.WHITE_KINGSIDE | Castling.BLACK_QUEENSIDE);
    assertThat(Fen.parse("4k3/8/8/8/8/8/8/4K3 w - - 0 1").castlingRights())
        .isEqualTo(Castling.NONE);
    // Mauvais ordre
    assertThatThrownBy(() -> Fen.parse("r3k2r/8/8/8/8/8/8/R3K2R w QK - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Fen.parse("r3k2r/8/8/8/8/8/8/R3K2R w qK - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // Caractère invalide
    assertThatThrownBy(() -> Fen.parse("r3k2r/8/8/8/8/8/8/R3K2R w X - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // Duplication
    assertThatThrownBy(() -> Fen.parse("r3k2r/8/8/8/8/8/8/R3K2R w KKqq - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("Règle 6b : cohérence positions roi/tour pour chaque droit déclaré")
  void rule6CastlingConsistency() {
    // WK avec roi en E1 et tour en H1 → OK
    assertThat(Fen.parse("4k3/8/8/8/8/8/8/4K2R w K - 0 1").castlingRights())
        .isEqualTo(Castling.WHITE_KINGSIDE);
    // WQ avec roi en E1 et tour en A1 → OK
    assertThat(Fen.parse("4k3/8/8/8/8/8/8/R3K3 w Q - 0 1").castlingRights())
        .isEqualTo(Castling.WHITE_QUEENSIDE);
    // BK avec roi en E8 et tour en H8 → OK (king blanc fictif en E1)
    assertThat(Fen.parse("4k2r/8/8/8/8/8/8/4K3 w k - 0 1").castlingRights())
        .isEqualTo(Castling.BLACK_KINGSIDE);

    // WK déclaré sans tour en H1
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 w K - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // WK déclaré avec roi pas en E1
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/3K3R w K - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // BK déclaré sans tour en H8
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 w k - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // BQ déclaré avec roi pas en E8
    assertThatThrownBy(() -> Fen.parse("r2k4/8/8/8/8/8/8/4K3 w q - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  // -----------------------------------------------------------------------------------------
  // Règle 7 : EP square — format + rang correct + pion adverse adjacent
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Règle 7a : EP au bon rang selon side et avec pion adverse adjacent (positif)")
  void rule7EnPassantPositive() {
    // Après e2-e4 : noir au trait, EP sur e3, pion blanc sur e4
    Position p = Fen.parse("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    assertThat(p.epSquare()).isEqualTo(Square.E3);
    // Après e7-e5 : blanc au trait, EP sur e6, pion noir sur e5
    Position p2 = Fen.parse("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 2");
    assertThat(p2.epSquare()).isEqualTo(Square.E6);
  }

  @Test
  @DisplayName("Règle 7b : EP au mauvais rang pour le side")
  void rule7EpRankMismatch() {
    // BLACK to move avec EP sur rang 6 (devrait être rang 3)
    assertThatThrownBy(
            () -> Fen.parse("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR b KQkq e6 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // WHITE to move avec EP sur rang 3 (devrait être rang 6)
    assertThatThrownBy(
            () -> Fen.parse("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e3 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("Règle 7c : EP sans pion adverse sur la case adjacente")
  void rule7EpMissingAdjacentPawn() {
    // EP=e6 (rang 5) attendu pour WHITE to move, mais aucun pion noir en e5
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 w - e6 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // EP=e3 (rang 2) attendu pour BLACK to move, mais aucun pion blanc en e4
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 b - e3 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("Règle 7d : EP square non-algébrique")
  void rule7EpInvalidFormat() {
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 w - X9 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 w - e9 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 w - e 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  // -----------------------------------------------------------------------------------------
  // Règle 8 : halfmove clock ≥ 0 et ≤ 100
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Règle 8 : halfmove ≥ 0 et ≤ 100")
  void rule8Halfmove() {
    assertThat(Fen.parse("4k3/8/8/8/8/8/8/4K3 w - - 0 1").halfmoveClock()).isZero();
    assertThat(Fen.parse("4k3/8/8/8/8/8/8/4K3 w - - 100 1").halfmoveClock()).isEqualTo(100);
    // Négatif
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 w - - -1 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // > 100
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 w - - 101 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // Non-entier
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 w - - foo 1"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  // -----------------------------------------------------------------------------------------
  // Règle 9 : fullmove number ≥ 1
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Règle 9 : fullmove ≥ 1")
  void rule9Fullmove() {
    assertThat(Fen.parse("4k3/8/8/8/8/8/8/4K3 w - - 0 1").fullmoveNumber()).isEqualTo(1);
    assertThat(Fen.parse("4k3/8/8/8/8/8/8/4K3 w - - 0 100").fullmoveNumber()).isEqualTo(100);
    assertThat(Fen.parse("4k3/8/8/8/8/8/8/4K3 w - - 0 9999").fullmoveNumber()).isEqualTo(9999);
    // Zéro rejeté
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 w - - 0 0"))
        .isInstanceOf(IllegalArgumentException.class);
    // Négatif rejeté
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 w - - 0 -5"))
        .isInstanceOf(IllegalArgumentException.class);
    // Non-entier
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/8/4K3 w - - 0 bar"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  // -----------------------------------------------------------------------------------------
  // Règle 10 : le côté NON au trait ne doit pas être en échec
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Règle 10 : le côté NON au trait ne doit pas être en échec")
  void rule10NonSideToMoveNotInCheck() {
    // Positif : roi blanc seul, BLACK au trait, pas d'attaque sur le roi blanc
    assertThat(Fen.parse("4k3/8/8/8/8/8/8/4K3 b - - 0 1").sideToMove()).isEqualTo(Color.BLACK);
    // BLACK au trait mais WHITE en échec (tour noire en e2 attaque roi blanc en e1) → invalide
    assertThatThrownBy(() -> Fen.parse("4k3/8/8/8/8/8/4r3/4K3 b - - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
    // WHITE au trait mais BLACK en échec (tour blanche en e7 attaque roi noir en e8) → invalide
    assertThatThrownBy(() -> Fen.parse("4k3/4R3/8/8/8/8/8/4K3 w - - 0 1"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  // -----------------------------------------------------------------------------------------
  // Robustesse : parse alloue uniquement la nouvelle Position lorsqu'on utilise la 1-arg form
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "parse(String) alloue une nouvelle Position ; parse(String, Position) ne le fait pas")
  void parseAllocations() {
    Position p1 = Fen.parse(Fen.STARTPOS);
    Position p2 = Fen.parse(Fen.STARTPOS);
    assertThat(p1).isNotSameAs(p2);

    Position dst = new Position();
    Fen.parse(Fen.STARTPOS, dst);
    assertThat(Fen.write(dst)).isEqualTo(Fen.STARTPOS);
  }

  @Test
  @DisplayName("Constructeur privé non instanciable")
  void constructorIsPrivate() throws Exception {
    var ctor = Fen.class.getDeclaredConstructor();
    ctor.setAccessible(true);
    assertThatThrownBy(ctor::newInstance).hasCauseInstanceOf(AssertionError.class);
  }
}
