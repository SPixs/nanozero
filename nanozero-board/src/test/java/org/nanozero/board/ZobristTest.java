package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Tests unitaires de {@link Zobrist} (cf. SPEC §5.6, §14 phase 6).
 *
 * <p>Les critères de complétion phase 6 imposent : reproductibilité de la seed, et invariant {@code
 * I-Pos-5 (zobristHash == computeFull)} sur 1 000 000 de transitions de random play.
 */
class ZobristTest {

  // -----------------------------------------------------------------------------------------
  // Constantes publiques
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Constantes : SEED=0x9E3779B97F4A7C15L, NB_CONSTANTS=781")
  void publicConstants() {
    assertThat(Zobrist.SEED).isEqualTo(0x9E3779B97F4A7C15L);
    assertThat(Zobrist.NB_CONSTANTS).isEqualTo(781);
  }

  // -----------------------------------------------------------------------------------------
  // Reproductibilité : ré-implémentation indépendante du SPEC §5.6.1 / §5.6.2
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Les 781 constantes correspondent au PRNG xorshift64* seedé selon §5.6.2")
  void constantsMatchSpecAlgorithm() {
    long state = Zobrist.SEED;
    long[] expected = new long[Zobrist.NB_CONSTANTS];
    for (int i = 0; i < Zobrist.NB_CONSTANTS; i++) {
      state = xorshift64StarReference(state);
      expected[i] = state;
    }
    int idx = 0;
    // Indices 0..767 : pieceSquare[piece * 64 + square]
    for (int piece = 0; piece < Piece.NB_PIECES; piece++) {
      for (int sq = 0; sq < Square.NB_SQUARES; sq++) {
        assertThat(Zobrist.pieceSquare(piece, sq))
            .as("pieceSquare(%d, %d) at index %d", piece, sq, idx)
            .isEqualTo(expected[idx]);
        idx++;
      }
    }
    // Indices 768..771 : castling [WK, WQ, BK, BQ]
    assertThat(Zobrist.castling(Castling.WHITE_KINGSIDE)).isEqualTo(expected[idx++]);
    assertThat(Zobrist.castling(Castling.WHITE_QUEENSIDE)).isEqualTo(expected[idx++]);
    assertThat(Zobrist.castling(Castling.BLACK_KINGSIDE)).isEqualTo(expected[idx++]);
    assertThat(Zobrist.castling(Castling.BLACK_QUEENSIDE)).isEqualTo(expected[idx++]);
    // Indices 772..779 : enPassantFile[a..h]
    for (int file = 0; file < 8; file++) {
      assertThat(Zobrist.enPassantFile(file))
          .as("enPassantFile(%d) at index %d", file, idx)
          .isEqualTo(expected[idx]);
      idx++;
    }
    // Index 780 : sideBlack
    assertThat(Zobrist.sideBlack()).isEqualTo(expected[idx]);
  }

  /** Réimplémentation de référence (indépendante de Zobrist.java) du PRNG SPEC §5.6.1. */
  private static long xorshift64StarReference(long state) {
    long s = state;
    s ^= s << 13;
    s ^= s >>> 7;
    s ^= s << 17;
    return s * 0x2545F4914F6CDD1DL;
  }

  // -----------------------------------------------------------------------------------------
  // Sanité : distinctes, non-nulles
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Les 781 constantes sont distinctes et non nulles")
  void constantsAreDistinctAndNonZero() {
    Set<Long> seen = new HashSet<>(Zobrist.NB_CONSTANTS * 2);
    for (int piece = 0; piece < Piece.NB_PIECES; piece++) {
      for (int sq = 0; sq < Square.NB_SQUARES; sq++) {
        long c = Zobrist.pieceSquare(piece, sq);
        assertThat(c).isNotZero();
        assertThat(seen.add(c)).as("doublon piece=%d sq=%d", piece, sq).isTrue();
      }
    }
    for (int bit :
        new int[] {
          Castling.WHITE_KINGSIDE,
          Castling.WHITE_QUEENSIDE,
          Castling.BLACK_KINGSIDE,
          Castling.BLACK_QUEENSIDE
        }) {
      long c = Zobrist.castling(bit);
      assertThat(c).isNotZero();
      assertThat(seen.add(c)).as("doublon castling bit=%d", bit).isTrue();
    }
    for (int file = 0; file < 8; file++) {
      long c = Zobrist.enPassantFile(file);
      assertThat(c).isNotZero();
      assertThat(seen.add(c)).as("doublon enPassantFile file=%d", file).isTrue();
    }
    long sb = Zobrist.sideBlack();
    assertThat(sb).isNotZero();
    assertThat(seen.add(sb)).as("doublon sideBlack").isTrue();
    assertThat(seen).hasSize(Zobrist.NB_CONSTANTS);
  }

  // -----------------------------------------------------------------------------------------
  // computeFull : invariants basiques
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "computeFull(empty position) == 0 (aucune pièce, WHITE au trait, pas d'EP, pas de droits)")
  void computeFullEmptyPosition() {
    Position p = new Position();
    assertThat(Zobrist.computeFull(p)).isZero();
  }

  @Test
  @DisplayName("computeFull(STARTPOS) cohérent avec le hash incrémental après Fen.parse")
  void computeFullStartposMatchesParse() {
    Position p = Fen.parse(Fen.STARTPOS);
    assertThat(p.zobristHash()).isEqualTo(Zobrist.computeFull(p));
    assertThat(p.zobristHash()).isNotZero(); // sanity : startpos hash non trivial
  }

  @Test
  @DisplayName("Hash sensible au côté au trait : startpos 'w' ≠ même position 'b'")
  void hashSensitiveToSideToMove() {
    Position pw = Fen.parse(Fen.STARTPOS);
    Position pb = Fen.parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    assertThat(pw.zobristHash()).isNotEqualTo(pb.zobristHash());
    assertThat(pw.zobristHash() ^ pb.zobristHash()).isEqualTo(Zobrist.sideBlack());
  }

  @Test
  @DisplayName("Hash sensible aux droits de roque : suppression d'un droit modifie le hash")
  void hashSensitiveToCastlingRights() {
    Position full = Fen.parse(Fen.STARTPOS);
    Position partial = Fen.parse("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQk - 0 1");
    long diff = full.zobristHash() ^ partial.zobristHash();
    assertThat(diff).isEqualTo(Zobrist.castling(Castling.BLACK_QUEENSIDE));
  }

  // -----------------------------------------------------------------------------------------
  // Critère §14 phase 6 : I-Pos-5 sur 1 000 000 transitions random play
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("I-Pos-5 : zobristHash == computeFull sur 1 000 000 transitions random play")
  void incrementalHashMatchesFullOverOneMillionTransitions() {
    Random rng = new Random(0xBADCAFE12345L);
    int[] buffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int total = 0;
    int gameCount = 0;
    while (total < 1_000_000) {
      Position p = Fen.parse(Fen.STARTPOS);
      // Cap conservateur sur la longueur de partie pour éviter les boucles infinies en jeu
      // aléatoire.
      for (int ply = 0; ply < 250 && total < 1_000_000; ply++) {
        int count = MoveGen.generateMoves(p, buffer, 0);
        if (count == 0) {
          break; // mat ou pat, démarrer une nouvelle partie
        }
        int chosen = buffer[rng.nextInt(count)];
        p.applyMove(chosen);
        total++;
        long expected = Zobrist.computeFull(p);
        if (p.zobristHash() != expected) {
          throw new AssertionError(
              String.format(
                  "I-Pos-5 violé après transition #%d (game %d, ply %d) : incremental=0x%016X vs"
                      + " computeFull=0x%016X%n  FEN: %s",
                  total, gameCount, ply, p.zobristHash(), expected, p.toFen()));
        }
      }
      gameCount++;
    }
    assertThat(total).isEqualTo(1_000_000);
  }

  // -----------------------------------------------------------------------------------------
  // Constructeur privé
  // -----------------------------------------------------------------------------------------

  @Test
  @DisplayName("Constructeur privé non instanciable")
  void constructorIsPrivate() throws Exception {
    var ctor = Zobrist.class.getDeclaredConstructor();
    ctor.setAccessible(true);
    assertThatThrownBy(ctor::newInstance).hasCauseInstanceOf(AssertionError.class);
  }
}
