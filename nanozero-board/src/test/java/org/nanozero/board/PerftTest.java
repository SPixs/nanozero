package org.nanozero.board;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Suite Perft (cf. SPEC §8.2 et §15.1). Compare les node counts produits par {@link
 * MoveGen#perft(Position, int)} aux références issues de chessprogramming.org.
 *
 * <p>Les profondeurs élevées à fort branching factor sont annotées {@code @Tag("slow")} et exclues
 * de la CI rapide ; elles sont ré-incluses via {@code mvn -Pperft-full verify} (cf. SPEC §14 phase
 * 10, profil défini dans {@code nanozero-board/pom.xml}).
 *
 * <p>Cible de durée pour la suite hors {@code @Tag("slow")} : &lt; 60 s en CI (cf. SPEC §8.2).
 */
class PerftTest {

  // -----------------------------------------------------------------------------------------
  // Suite rapide (CI par défaut) — cf. SPEC §15.1
  // -----------------------------------------------------------------------------------------

  @ParameterizedTest(name = "{0} d={1} → {2} nodes")
  @CsvSource({
    // Position 1 — startpos
    "STARTPOS, 1, 20,        rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "STARTPOS, 2, 400,       rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "STARTPOS, 3, 8902,      rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "STARTPOS, 4, 197281,    rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "STARTPOS, 5, 4865609,   rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "STARTPOS, 6, 119060324, rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    // Position 2 — Kiwipete (depth 6 = slow, voir perftSlow)
    "KIWIPETE, 1, 48,         r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "KIWIPETE, 2, 2039,       r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "KIWIPETE, 3, 97862,      r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "KIWIPETE, 4, 4085603,    r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "KIWIPETE, 5, 193690690,  r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    // Position 3 (depth 7 = slow)
    "POS3, 1, 14,        8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "POS3, 2, 191,       8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "POS3, 3, 2812,      8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "POS3, 4, 43238,     8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "POS3, 5, 674624,    8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "POS3, 6, 11030083,  8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    // Position 4 (depth 6 = slow)
    "POS4, 1, 6,        r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "POS4, 2, 264,      r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "POS4, 3, 9467,     r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "POS4, 4, 422333,   r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "POS4, 5, 15833292, r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    // Position 5
    "POS5, 1, 44,         rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "POS5, 2, 1486,       rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "POS5, 3, 62379,      rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "POS5, 4, 2103487,    rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "POS5, 5, 89941194,   rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    // Position 6 (depth 6 = slow)
    "POS6, 1, 46,         r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "POS6, 2, 2079,       r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "POS6, 3, 89890,      r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "POS6, 4, 3894594,    r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "POS6, 5, 164075551,  r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10"
  })
  @DisplayName("Perft suite §15.1 — cas rapides (CI par défaut)")
  void perftFast(String label, int depth, long expectedNodes, String fen) {
    Position position = new Position();
    Fen.parse(fen, position);
    long actual = MoveGen.perft(position, depth);
    assertThat(actual).as("Perft %s depth=%d FEN=%s", label, depth, fen).isEqualTo(expectedNodes);
  }

  // -----------------------------------------------------------------------------------------
  // Suite slow (mvn -Pperft-full verify)
  // -----------------------------------------------------------------------------------------

  @Tag("slow")
  @ParameterizedTest(name = "{0} d={1} → {2} nodes (slow)")
  @CsvSource({
    "KIWIPETE, 6, 8031647685, r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "POS3,     7, 178633661,  8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "POS4,     6, 706045033,  r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "POS6,     6, 6923051137, r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0"
        + " 10"
  })
  @DisplayName("Perft suite §15.1 — cas slow (profil perft-full)")
  void perftSlow(String label, int depth, long expectedNodes, String fen) {
    Position position = new Position();
    Fen.parse(fen, position);
    long actual = MoveGen.perft(position, depth);
    assertThat(actual).as("Perft %s depth=%d FEN=%s", label, depth, fen).isEqualTo(expectedNodes);
  }

  // -----------------------------------------------------------------------------------------
  // Edge cases : profondeur 0 et appel direct sur la position passée
  // -----------------------------------------------------------------------------------------

  @org.junit.jupiter.api.Test
  @DisplayName("perft(p, 0) == 1 (convention SPEC §8.2)")
  void perftDepthZero() {
    Position p = new Position();
    Fen.parse(Fen.STARTPOS, p);
    assertThat(MoveGen.perft(p, 0)).isEqualTo(1L);
  }

  @org.junit.jupiter.api.Test
  @DisplayName("perft NE modifie PAS la position passée")
  void perftDoesNotMutateInput() {
    Position p = new Position();
    Fen.parse(Fen.STARTPOS, p);
    String fenBefore = p.toFen();
    long hashBefore = p.zobristHash();
    MoveGen.perft(p, 4);
    assertThat(p.toFen()).isEqualTo(fenBefore);
    assertThat(p.zobristHash()).isEqualTo(hashBefore);
  }
}
