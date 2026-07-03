package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Random;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Tests unitaires de {@link Position}. */
class PositionTest {

  // -----------------------------------------------------------------------------------------
  // Constructeur et état initial
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Constructeur : tous bitboards à 0, WHITE au trait, fullmove=1, ep=-1")
  void defaultConstructorYieldsEmptyState() {
    Position p = new Position();
    for (int piece = 0; piece < Piece.NB_PIECES; piece++) {
      assertThat(p.pieceBB(piece)).isZero();
    }
    assertThat(p.occupancyBB(Color.WHITE)).isZero();
    assertThat(p.occupancyBB(Color.BLACK)).isZero();
    assertThat(p.allOccupancy()).isZero();
    assertThat(p.sideToMove()).isEqualTo(Color.WHITE);
    assertThat(p.castlingRights()).isEqualTo(Castling.NONE);
    assertThat(p.epSquare()).isEqualTo(Square.NONE);
    assertThat(p.halfmoveClock()).isZero();
    assertThat(p.fullmoveNumber()).isEqualTo(1);
    assertThat(p.zobristHash()).isZero();
  }

  @Test
  @DisplayName("Getters de tableau exposent la référence interne (pas une copie)")
  void arrayGettersReturnInternalReference() {
    Position p = new Position();
    assertThat(p.pieceBB()).isSameAs(p.pieceBB());
    assertThat(p.occupancyBB()).isSameAs(p.occupancyBB());
    assertThat(p.pieceBB()).hasSize(Piece.NB_PIECES);
    assertThat(p.occupancyBB()).hasSize(3);
  }

  // -----------------------------------------------------------------------------------------
  // copyFrom et copy
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("copyFrom : 1000 positions construites manuellement, comparaison field-by-field")
  void copyFromExhaustiveOnRandomStates() {
    Random rng = new Random(0xCAFEBABEL);
    Position src = new Position();
    Position dst = new Position();

    for (int trial = 0; trial < 1000; trial++) {
      fillRandom(src, rng);
      // Pollue dst avec un état différent pour s'assurer que copyFrom écrase tout.
      fillRandom(dst, rng);
      dst.copyFrom(src);
      assertPositionsEqual(dst, src, "trial " + trial);
    }
  }

  @Test
  @DisplayName("copyFrom : indépendance après mutation de la source (pas de partage de tableaux)")
  void copyFromYieldsIndependentArrays() {
    Position src = new Position();
    src.pieceBB[Piece.WHITE_PAWN] = 0xFF00L;
    src.occupancyBB[Color.WHITE] = 0xFF00L;
    src.occupancyBB[2] = 0xFF00L;
    src.sideToMove = Color.BLACK;

    Position dst = new Position();
    dst.copyFrom(src);

    // Mutation post-copy de la source ne doit pas affecter la destination.
    src.pieceBB[Piece.WHITE_PAWN] = 0L;
    src.occupancyBB[Color.WHITE] = 0L;
    src.occupancyBB[2] = 0L;
    src.sideToMove = Color.WHITE;

    assertThat(dst.pieceBB(Piece.WHITE_PAWN)).isEqualTo(0xFF00L);
    assertThat(dst.occupancyBB(Color.WHITE)).isEqualTo(0xFF00L);
    assertThat(dst.allOccupancy()).isEqualTo(0xFF00L);
    assertThat(dst.sideToMove()).isEqualTo(Color.BLACK);
  }

  @Test
  @DisplayName("copy() : nouvelle instance, contenu équivalent, références indépendantes")
  void copyAllocatesNewInstance() {
    Position src = startposManual();
    Position cp = src.copy();

    assertThat(cp).isNotSameAs(src);
    assertThat(cp.pieceBB()).isNotSameAs(src.pieceBB());
    assertThat(cp.occupancyBB()).isNotSameAs(src.occupancyBB());
    assertPositionsEqual(cp, src, "copy()");
  }

  // -----------------------------------------------------------------------------------------
  // isInCheck / attackersOf / isSquareAttacked
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("isInCheck : position vide → false (pas de roi)")
  void isInCheckEmptyPosition() {
    Position p = new Position();
    assertThat(p.isInCheck()).isFalse();
  }

  @Test
  @DisplayName("isInCheck : roi blanc seul, pas d'attaquant → false")
  void isInCheckWhiteKingAlone() {
    Position p = new Position();
    putPiece(p, Piece.WHITE_KING, Square.E1);
    rebuildOccupancy(p);
    assertThat(p.isInCheck()).isFalse();
  }

  @Test
  @DisplayName("isInCheck : roi blanc en E1, tour noire en E8 → true")
  void isInCheckByRookSameFile() {
    Position p = new Position();
    putPiece(p, Piece.WHITE_KING, Square.E1);
    putPiece(p, Piece.BLACK_ROOK, Square.E8);
    rebuildOccupancy(p);
    assertThat(p.isInCheck()).isTrue();
  }

  @Test
  @DisplayName("isInCheck : tour bloquée par pion ne donne pas échec")
  void isInCheckRookBlockedByPawn() {
    Position p = new Position();
    putPiece(p, Piece.WHITE_KING, Square.E1);
    putPiece(p, Piece.BLACK_ROOK, Square.E8);
    putPiece(p, Piece.WHITE_PAWN, Square.E4); // bloque la tour
    rebuildOccupancy(p);
    assertThat(p.isInCheck()).isFalse();
  }

  @Test
  @DisplayName("isInCheck : cavalier noir donne échec depuis F3 sur roi en E1")
  void isInCheckByKnight() {
    Position p = new Position();
    putPiece(p, Piece.WHITE_KING, Square.E1);
    putPiece(p, Piece.BLACK_KNIGHT, Square.F3);
    rebuildOccupancy(p);
    assertThat(p.isInCheck()).isTrue();
  }

  @Test
  @DisplayName("isInCheck : pion noir en F2 donne échec sur roi blanc en E1, mais pas en E2")
  void isInCheckByPawn() {
    Position pAttacked = new Position();
    putPiece(pAttacked, Piece.WHITE_KING, Square.E1);
    putPiece(pAttacked, Piece.BLACK_PAWN, Square.F2);
    rebuildOccupancy(pAttacked);
    assertThat(pAttacked.isInCheck()).isTrue();

    // Un pion noir en E2 n'attaque PAS E1 (les pions noirs attaquent diagonalement vers le bas).
    Position pSafe = new Position();
    putPiece(pSafe, Piece.WHITE_KING, Square.E1);
    putPiece(pSafe, Piece.BLACK_PAWN, Square.E2);
    rebuildOccupancy(pSafe);
    assertThat(pSafe.isInCheck()).isFalse();
  }

  @Test
  @DisplayName("isInCheck : applique correctement à BLACK side to move")
  void isInCheckBlackToMove() {
    Position p = new Position();
    putPiece(p, Piece.BLACK_KING, Square.E8);
    putPiece(p, Piece.WHITE_QUEEN, Square.E1);
    rebuildOccupancy(p);
    p.sideToMove = Color.BLACK;
    assertThat(p.isInCheck()).isTrue();

    // Côté blanc au trait : pas d'échec sur le roi blanc (qui n'existe pas ici, bitboard à 0)
    p.sideToMove = Color.WHITE;
    assertThat(p.isInCheck()).isFalse();
  }

  @Test
  @DisplayName("attackersOf : pion blanc en D2 attaque C3 et E3, pas D3")
  void attackersOfPawn() {
    Position p = new Position();
    putPiece(p, Piece.WHITE_PAWN, Square.D2);
    rebuildOccupancy(p);
    assertThat(p.attackersOf(Square.C3, Color.WHITE)).isEqualTo(1L << Square.D2);
    assertThat(p.attackersOf(Square.E3, Color.WHITE)).isEqualTo(1L << Square.D2);
    assertThat(p.attackersOf(Square.D3, Color.WHITE)).isZero();
    assertThat(p.attackersOf(Square.D2, Color.WHITE)).isZero();
  }

  @Test
  @DisplayName("attackersOf : aggrégation multi-pièces (cavalier + tour) sur même case")
  void attackersOfMultiplePieces() {
    Position p = new Position();
    putPiece(p, Piece.WHITE_KNIGHT, Square.D2);
    putPiece(p, Piece.WHITE_ROOK, Square.E1);
    rebuildOccupancy(p);
    long attackers = p.attackersOf(Square.E4, Color.WHITE);
    // Cavalier en D2 → E4 (knight attack), tour en E1 → E4 (rook attack)
    assertThat(attackers & (1L << Square.D2)).isNotZero();
    assertThat(attackers & (1L << Square.E1)).isNotZero();
    assertThat(Long.bitCount(attackers)).isEqualTo(2);
  }

  @Test
  @DisplayName("attackersOf : les attaquants sont filtrés par couleur")
  void attackersOfFiltersByColor() {
    Position p = new Position();
    putPiece(p, Piece.WHITE_ROOK, Square.E4);
    putPiece(p, Piece.BLACK_ROOK, Square.E8);
    rebuildOccupancy(p);

    // Sur E5 : la tour blanche en E4 attaque, la tour noire est bloquée par E5 elle-même
    assertThat(p.attackersOf(Square.E5, Color.WHITE)).isEqualTo(1L << Square.E4);
    // E5 attaqué par tour noire si on regarde depuis l'autre côté (bloqueur en E4)
    // La tour noire en E8 a un raycast bloqué par E4 (cible E5 atteinte avant E4)
    assertThat(p.attackersOf(Square.E5, Color.BLACK)).isEqualTo(1L << Square.E8);

    // Sur E3 : la tour blanche en E4 attaque (raycast vers le bas), la tour noire bloquée par E4
    assertThat(p.attackersOf(Square.E3, Color.WHITE)).isEqualTo(1L << Square.E4);
    assertThat(p.attackersOf(Square.E3, Color.BLACK)).isZero();
  }

  @Test
  @DisplayName("isSquareAttacked == (attackersOf != 0)")
  void isSquareAttackedConsistencyWithAttackersOf() {
    Position p = startposManual();
    for (int sq = 0; sq < 64; sq++) {
      for (int color = 0; color < 2; color++) {
        boolean attacked = p.isSquareAttacked(sq, color);
        long attackers = p.attackersOf(sq, color);
        assertThat(attacked).as("sq=%d color=%d", sq, color).isEqualTo(attackers != 0L);
      }
    }
  }

  // -----------------------------------------------------------------------------------------
  // pieceAt
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("pieceAt : position vide → NONE pour toutes les cases")
  void pieceAtEmptyPosition() {
    Position p = new Position();
    for (int sq = 0; sq < 64; sq++) {
      assertThat(p.pieceAt(sq)).as("pieceAt(%d)", sq).isEqualTo(Piece.NONE);
    }
  }

  @Test
  @DisplayName("pieceAt : startpos manuel renvoie la bonne pièce sur chaque case occupée")
  void pieceAtStartpos() {
    Position p = startposManual();
    assertThat(p.pieceAt(Square.A1)).isEqualTo(Piece.WHITE_ROOK);
    assertThat(p.pieceAt(Square.E1)).isEqualTo(Piece.WHITE_KING);
    assertThat(p.pieceAt(Square.D1)).isEqualTo(Piece.WHITE_QUEEN);
    assertThat(p.pieceAt(Square.E2)).isEqualTo(Piece.WHITE_PAWN);
    assertThat(p.pieceAt(Square.E7)).isEqualTo(Piece.BLACK_PAWN);
    assertThat(p.pieceAt(Square.E8)).isEqualTo(Piece.BLACK_KING);
    assertThat(p.pieceAt(Square.E4)).isEqualTo(Piece.NONE);
    assertThat(p.pieceAt(Square.D5)).isEqualTo(Piece.NONE);
  }

  // -----------------------------------------------------------------------------------------
  // Sérialisation FEN / ASCII / dump
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("toFen : startpos manuel produit la FEN canonique")
  void toFenStartpos() {
    Position p = startposManual();
    assertThat(p.toFen()).isEqualTo("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  }

  @Test
  @DisplayName("toFen : position vide w/ pas de droits → '8/8/8/8/8/8/8/8 w - - 0 1'")
  void toFenEmpty() {
    Position p = new Position();
    assertThat(p.toFen()).isEqualTo("8/8/8/8/8/8/8/8 w - - 0 1");
  }

  @Test
  @DisplayName("toFen : sérialise les 4 droits de roque dans l'ordre KQkq")
  void toFenCastlingOrder() {
    Position p = new Position();
    putPiece(p, Piece.WHITE_KING, Square.E1);
    rebuildOccupancy(p);
    p.castlingRights = Castling.ALL;
    assertThat(p.toFen()).contains(" KQkq ");

    p.castlingRights = Castling.WHITE_KINGSIDE | Castling.BLACK_QUEENSIDE;
    assertThat(p.toFen()).contains(" Kq ");

    p.castlingRights = Castling.NONE;
    assertThat(p.toFen()).contains(" - ");
  }

  @Test
  @DisplayName("toFen : sérialise EP en algébrique ou '-'")
  void toFenEpSquare() {
    Position p = new Position();
    p.epSquare = Square.E3;
    assertThat(p.toFen()).contains(" - e3 ");
    p.epSquare = Square.NONE;
    assertThat(p.toFen()).contains(" - - ");
  }

  @Test
  @DisplayName("toString == toFen")
  void toStringEqualsToFen() {
    Position p = startposManual();
    assertThat(p.toString()).isEqualTo(p.toFen());
  }

  @Test
  @DisplayName("toAsciiBoard : startpos contient les pièces dans le bon ordre")
  void toAsciiBoardStartpos() {
    Position p = startposManual();
    String ascii = p.toAsciiBoard();
    // Ligne du rang 8 : "8 | r | n | b | q | k | b | n | r |"
    assertThat(ascii).contains("8 | r | n | b | q | k | b | n | r |");
    // Ligne du rang 1 : "1 | R | N | B | Q | K | B | N | R |"
    assertThat(ascii).contains("1 | R | N | B | Q | K | B | N | R |");
    // Marge basse fichiers
    assertThat(ascii).contains("    a   b   c   d   e   f   g   h");
  }

  @Test
  @DisplayName("dump : smoke test, contient FEN, State block, Bitboards block")
  void dumpSmokeTest() {
    Position p = startposManual();
    var bos = new ByteArrayOutputStream();
    p.dump(new PrintStream(bos, true, StandardCharsets.UTF_8));
    String out = bos.toString(StandardCharsets.UTF_8);
    assertThat(out).contains("Position:");
    assertThat(out).contains("FEN: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    assertThat(out).contains("State:");
    assertThat(out).contains("Side to move    : WHITE");
    assertThat(out).contains("Bitboards:");
    assertThat(out).contains("WHITE_PAWN");
    assertThat(out).contains("OCC_BOTH");
  }

  // -----------------------------------------------------------------------------------------
  // Helpers de test
  // -----------------------------------------------------------------------------------------

  private static void putPiece(Position p, int piece, int square) {
    p.pieceBB[piece] |= 1L << square;
  }

  private static void rebuildOccupancy(Position p) {
    long w = 0L;
    long b = 0L;
    for (int i = 0; i < 6; i++) {
      w |= p.pieceBB[i];
      b |= p.pieceBB[6 + i];
    }
    p.occupancyBB[Color.WHITE] = w;
    p.occupancyBB[Color.BLACK] = b;
    p.occupancyBB[2] = w | b;
  }

  private static Position startposManual() {
    Position p = new Position();
    p.pieceBB[Piece.WHITE_PAWN] = 0x000000000000FF00L;
    p.pieceBB[Piece.WHITE_KNIGHT] = (1L << Square.B1) | (1L << Square.G1);
    p.pieceBB[Piece.WHITE_BISHOP] = (1L << Square.C1) | (1L << Square.F1);
    p.pieceBB[Piece.WHITE_ROOK] = (1L << Square.A1) | (1L << Square.H1);
    p.pieceBB[Piece.WHITE_QUEEN] = 1L << Square.D1;
    p.pieceBB[Piece.WHITE_KING] = 1L << Square.E1;
    p.pieceBB[Piece.BLACK_PAWN] = 0x00FF000000000000L;
    p.pieceBB[Piece.BLACK_KNIGHT] = (1L << Square.B8) | (1L << Square.G8);
    p.pieceBB[Piece.BLACK_BISHOP] = (1L << Square.C8) | (1L << Square.F8);
    p.pieceBB[Piece.BLACK_ROOK] = (1L << Square.A8) | (1L << Square.H8);
    p.pieceBB[Piece.BLACK_QUEEN] = 1L << Square.D8;
    p.pieceBB[Piece.BLACK_KING] = 1L << Square.E8;
    rebuildOccupancy(p);
    p.sideToMove = Color.WHITE;
    p.castlingRights = Castling.ALL;
    p.epSquare = Square.NONE;
    p.halfmoveClock = 0;
    p.fullmoveNumber = 1;
    return p;
  }

  private static void fillRandom(Position p, Random rng) {
    for (int i = 0; i < Piece.NB_PIECES; i++) {
      p.pieceBB[i] = rng.nextLong();
    }
    for (int i = 0; i < 3; i++) {
      p.occupancyBB[i] = rng.nextLong();
    }
    p.sideToMove = rng.nextInt(2);
    p.castlingRights = rng.nextInt(16);
    p.epSquare = rng.nextInt(65) - 1; // -1..63
    p.halfmoveClock = rng.nextInt(101);
    p.fullmoveNumber = rng.nextInt(500) + 1;
    p.zobristHash = rng.nextLong();
  }

  private static void assertPositionsEqual(Position a, Position b, String context) {
    for (int i = 0; i < Piece.NB_PIECES; i++) {
      assertThat(a.pieceBB(i)).as("%s : pieceBB[%d]", context, i).isEqualTo(b.pieceBB(i));
    }
    assertThat(a.occupancyBB(Color.WHITE))
        .as("%s : occWhite", context)
        .isEqualTo(b.occupancyBB(Color.WHITE));
    assertThat(a.occupancyBB(Color.BLACK))
        .as("%s : occBlack", context)
        .isEqualTo(b.occupancyBB(Color.BLACK));
    assertThat(a.allOccupancy()).as("%s : occBoth", context).isEqualTo(b.allOccupancy());
    assertThat(a.sideToMove()).as("%s : sideToMove", context).isEqualTo(b.sideToMove());
    assertThat(a.castlingRights()).as("%s : castling", context).isEqualTo(b.castlingRights());
    assertThat(a.epSquare()).as("%s : epSquare", context).isEqualTo(b.epSquare());
    assertThat(a.halfmoveClock()).as("%s : halfmove", context).isEqualTo(b.halfmoveClock());
    assertThat(a.fullmoveNumber()).as("%s : fullmove", context).isEqualTo(b.fullmoveNumber());
    assertThat(a.zobristHash()).as("%s : hash", context).isEqualTo(b.zobristHash());
  }
}
