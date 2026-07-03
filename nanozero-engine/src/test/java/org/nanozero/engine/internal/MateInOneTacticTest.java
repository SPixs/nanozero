package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.nanozero.board.GameState;
import org.nanozero.engine.EngineConfig;
import org.nanozero.engine.SearchBudget;
import org.nanozero.nn.MoveEncoding;
import org.nanozero.nn.NNOutput;

/**
 * Critère §12 phase 8 : sur 20 positions tactiques mate-en-1, l'engine trouve le mat avec {@code
 * SearchBudget.nodes(100)} et {@code result.bestMove} correspond au coup matant.
 *
 * <p>Test paramétré sur 20 positions FEN curées manuellement et vérifiées contre {@code MoveGen +
 * isCheckmate}. Pour chaque position, on lance {@code runSearch(nodes(100))} avec un mock {@link
 * NetworkProvider} retournant des priors uniformes (logits=0) et value=0. On extrait le best move
 * via argmax-visits puis on vérifie qu'après application il y a échec et mat (i.e. l'engine a
 * trouvé un mat ; les positions à mate unique l'identifient sans ambiguïté, celles à plusieurs
 * mates valident n'importe lequel).
 *
 * <p>Mock uniforme retenu (pas le vrai Network) car :
 *
 * <ul>
 *   <li>Détermine totalement par {@code MoveGen} l'ordre d'exploration (PUCT tie-breaks par premier
 *       index, déterministe ADR-010).
 *   <li>Force {@code SearcherCore} à tester chaque coup légal (pas de biais NN).
 *   <li>Permet de vérifier la propagation correcte de {@code terminalValue=-1} via {@code Backup}
 *       jusqu'à la racine (le coup matant doit recevoir {@code Q=+1} après backup).
 * </ul>
 *
 * <p>Pour budget=100 sur ~20 coups légaux par position, chaque coup est visité ≥ 5 fois
 * statistiquement, ce qui permet à l'engine de découvrir le coup matant et de propager sa value=+1
 * (= -terminalValue alterné) à la racine, en faisant l'argmax des visites.
 */
class MateInOneTacticTest {

  // -------------------------------------------------------------------------------------------
  // Mock uniforme : logits=0, value=0
  // -------------------------------------------------------------------------------------------

  private static final class UniformMockProvider implements NetworkProvider {
    @Override
    public void forward(float[] planes, NNOutput output) {
      try {
        java.lang.reflect.Field logitsField = NNOutput.class.getDeclaredField("logits");
        java.lang.reflect.Field valuesField = NNOutput.class.getDeclaredField("values");
        logitsField.setAccessible(true);
        valuesField.setAccessible(true);
        float[] outLogits = (float[]) logitsField.get(output);
        float[] outValues = (float[]) valuesField.get(output);
        java.util.Arrays.fill(outLogits, 0, MoveEncoding.POLICY_INDICES, 0f);
        outValues[0] = 0f;
      } catch (ReflectiveOperationException e) {
        throw new AssertionError(e);
      }
    }
  }

  /** argmax des visites parmi les children instanciés. */
  private static int argmaxVisits(Node root) {
    int best = 0;
    int bestVisits = -1;
    for (int i = 0; i < root.children.length; i++) {
      Node c = root.children[i];
      int v = c == null ? 0 : c.totalVisits.get();
      if (v > bestVisits) {
        bestVisits = v;
        best = i;
      }
    }
    return best;
  }

  // -------------------------------------------------------------------------------------------
  // 20 positions mate-en-1 (curées et vérifiées MoveGen + isCheckmate)
  // -------------------------------------------------------------------------------------------

  /**
   * Liste figée. Chaque entrée : {@code <FEN> ; <description>}. Les positions ont toutes au moins
   * un mate-en-1 disponible pour le côté au trait. Le test ne vérifie pas un coup spécifique mais
   * que <strong>le best move trouvé donne mat</strong> (positions multi-mate acceptées).
   */
  @ParameterizedTest(name = "[{index}] {1}")
  @CsvSource(
      delimiter = ';',
      value = {
        "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1            ; back-rank Ra8#",
        "6k1/5ppp/8/8/8/8/4Q3/7K w - - 0 1           ; back-rank Qe8#",
        "7k/5Kpp/8/8/8/8/4Q3/8 w - - 0 1             ; Qe8# K-supported",
        "6rk/6pp/7N/8/8/8/8/7K w - - 0 1             ; Nf7# smothered-style",
        "7k/8/7K/8/8/8/8/R7 w - - 0 1                ; Ra8# K+R",
        "3r2k1/5ppp/8/8/8/8/8/3R3K w - - 0 1         ; Rd8# back-rank",
        "6kr/5ppp/8/8/8/8/8/3Q3K w - - 0 1           ; Qd8# avec R passive",
        "k7/8/K7/8/8/8/8/Q7 w - - 0 1                ; Qa1-h8# diag",
        "4k3/4P3/4K3/8/R7/8/8/8 w - - 0 1            ; Ra8# avec pion blanc bloquant",
        "6k1/5ppp/4B3/8/8/8/8/3R3K w - - 0 1         ; Rd8# avec B support",
        "6k1/4pppp/8/8/8/8/8/R6K w - - 0 1           ; Ra8# 4 pawns",
        "6k1/3ppppp/8/8/8/8/8/R6K w - - 0 1          ; Ra8# 5 pawns",
        "7k/6pp/8/8/8/8/8/R5RK w - - 0 1             ; Ra8# corner two rooks",
        "k7/p7/K7/8/8/8/8/Q7 w - - 0 1               ; Qa1-h8# avec p noir",
        "6k1/5ppp/8/8/8/8/8/B2Q3K w - - 0 1          ; Qd8# avec B+Q",
        "6rk/6pp/8/4N3/3R4/8/8/7K w - - 0 1          ; Anastasia Nf7#",
        "6k1/5ppp/8/8/8/8/8/4Q2K w - - 0 1           ; Qe8# back-rank simple",
        "4k3/8/4K3/Q7/8/8/8/8 w - - 0 1              ; Qa5-a8# K+Q vs K",
        "7k/5p1p/6pP/8/8/8/8/4R2K w - - 0 1          ; Re8# back-rank avec p blanc",
        "7k/5ppp/8/8/8/8/8/R5RK w - - 0 1            ; Ra8# corner-king",
      })
  @DisplayName("20 mate-en-1 : runSearch(nodes(100)) trouve le mat (best move = mat)")
  void mateInOnePositions(String fen, String description) {
    GameState state = new GameState(fen);
    // Sanity : la position n'est pas déjà terminale, mais elle a au moins un mate-en-1.
    assertThat(state.isTerminal())
        .as("[%s] startState ne doit pas être déjà terminale", description)
        .isFalse();

    SearcherCore core = new SearcherCore(new UniformMockProvider(), EngineConfig.defaults());
    SearchTree tree = new SearchTree(state);
    int sims = core.runSearch(tree, SearchBudget.nodes(100));
    assertThat(sims).isEqualTo(100);

    int bestIdx = argmaxVisits(tree.root());
    int bestMove = tree.root().childMoves[bestIdx];

    // Vérifier qu'après application du best move, la position est mat.
    state.applyMove(bestMove);
    assertThat(state.isCheckmate())
        .as(
            "[%s] best move (idx=%d, move=%s) doit donner mate ; visites=%d",
            description,
            bestIdx,
            org.nanozero.board.Move.toUci(bestMove),
            tree.root().children[bestIdx] == null
                ? 0
                : tree.root().children[bestIdx].totalVisits.get())
        .isTrue();
  }
}
