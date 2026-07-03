package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.engine.EngineConfig;
import org.nanozero.engine.SearchBudget;
import org.nanozero.nn.MoveEncoding;
import org.nanozero.nn.NNOutput;

/**
 * Tests dédiés à la distinction des types de fin de partie dans {@link SearcherCore} (cf. SPEC §5.4
 * conventions terminalValue, §12 phase 8). Vérifie qu'aucun appel NN n'est fait sur un nœud
 * terminal et que la {@code terminalValue} stockée dans {@code Node} est correcte selon le type :
 *
 * <ul>
 *   <li>Mat (joueur au trait est maté) : -1.0f
 *   <li>Pat : 0.0f
 *   <li>Règle des 50 coups : 0.0f
 *   <li>Matériel insuffisant : 0.0f
 *   <li>Répétition triple : 0.0f
 * </ul>
 *
 * <p>La logique de calcul est inline dans {@link SearcherCore} (une ligne : {@code
 * state.isCheckmate() ? -1.0f : 0.0f}), pas extraite dans une classe dédiée. Ces tests exercent la
 * logique via {@code runSearch} avec un mock {@link NetworkProvider} qui compte les appels (un
 * appel NN sur un terminal serait un bug — le critère §12 phase 8 est "skip de l'appel NN").
 */
class TerminalDetectionTest {

  // -------------------------------------------------------------------------------------------
  // Mock NetworkProvider qui compte les appels (tout call sur terminal = bug)
  // -------------------------------------------------------------------------------------------

  private static final class CountingMockProvider implements NetworkProvider {
    int forwardCallCount = 0;

    @Override
    public void forward(float[] planes, NNOutput output) {
      forwardCallCount++;
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

  /** Vérifie sur une position terminale : terminalValue attendue, NN jamais appelé. */
  private static void assertTerminalBehavior(
      GameState terminalState, float expectedTerminalValue, String description) {
    CountingMockProvider mock = new CountingMockProvider();
    SearcherCore core = new SearcherCore(mock, EngineConfig.defaults());
    SearchTree tree = new SearchTree(terminalState);

    int sims = core.runSearch(tree, SearchBudget.nodes(5));

    assertThat(sims).as("[%s] simulations", description).isEqualTo(5);
    assertThat(tree.root().terminal).as("[%s] terminal", description).isTrue();
    assertThat(tree.root().expanded).as("[%s] expanded", description).isTrue();
    assertThat(tree.root().terminalValue)
        .as("[%s] terminalValue", description)
        .isEqualTo(expectedTerminalValue);
    assertThat(mock.forwardCallCount)
        .as("[%s] NN ne doit pas être appelé sur position terminale", description)
        .isZero();
    // Backup propage la même value à chaque sim → totalValueSum = sims × terminalValue.
    assertThat(tree.root().totalValueSum.get())
        .as("[%s] totalValueSum", description)
        .isEqualTo(sims * expectedTerminalValue);
  }

  // -------------------------------------------------------------------------------------------
  // Mat (checkmate) → -1.0f
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Mat (back-rank, white to move maté) → terminalValue = -1.0f")
  void checkmate_minusOne() {
    GameState s = new GameState("4k3/8/8/8/8/8/5PPP/r6K w - - 0 1");
    assertThat(s.isTerminal()).isTrue();
    assertThat(s.isCheckmate()).isTrue();
    assertTerminalBehavior(s, -1.0f, "back-rank checkmate");
  }

  // -------------------------------------------------------------------------------------------
  // Pat (stalemate) → 0.0f
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Pat (Q+K vs K classique, black to move) → terminalValue = 0.0f")
  void stalemate_zero() {
    GameState s = new GameState("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    assertThat(s.isTerminal()).isTrue();
    assertThat(s.isStalemate()).isTrue();
    assertThat(s.isCheckmate()).isFalse();
    assertTerminalBehavior(s, 0.0f, "stalemate");
  }

  // -------------------------------------------------------------------------------------------
  // Règle des 50 coups → 0.0f
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Règle des 50 coups (halfmove clock = 100) → terminalValue = 0.0f")
  void fiftyMoveRule_zero() {
    GameState s = new GameState("8/8/3k4/3p4/3P4/3K4/8/8 w - - 100 1");
    assertThat(s.isFiftyMoveRule()).isTrue();
    assertThat(s.isTerminal()).isTrue();
    assertThat(s.isCheckmate()).isFalse();
    assertTerminalBehavior(s, 0.0f, "fifty-move rule");
  }

  // -------------------------------------------------------------------------------------------
  // Matériel insuffisant → 0.0f
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("K vs K → terminalValue = 0.0f (matériel insuffisant)")
  void insufficientKingVsKing_zero() {
    GameState s = new GameState("8/8/4k3/8/8/4K3/8/8 w - - 0 1");
    assertThat(s.isInsufficientMaterial()).isTrue();
    assertThat(s.isTerminal()).isTrue();
    assertTerminalBehavior(s, 0.0f, "K vs K");
  }

  @Test
  @DisplayName("K+B vs K → terminalValue = 0.0f (matériel insuffisant)")
  void insufficientKingBishopVsKing_zero() {
    GameState s = new GameState("8/8/4k3/8/8/4K3/8/3B4 w - - 0 1");
    assertThat(s.isInsufficientMaterial()).isTrue();
    assertThat(s.isTerminal()).isTrue();
    assertTerminalBehavior(s, 0.0f, "K+B vs K");
  }

  @Test
  @DisplayName("K+N vs K → terminalValue = 0.0f (matériel insuffisant)")
  void insufficientKingKnightVsKing_zero() {
    GameState s = new GameState("8/8/4k3/8/8/4K3/8/3N4 w - - 0 1");
    assertThat(s.isInsufficientMaterial()).isTrue();
    assertThat(s.isTerminal()).isTrue();
    assertTerminalBehavior(s, 0.0f, "K+N vs K");
  }

  // -------------------------------------------------------------------------------------------
  // Répétition triple → 0.0f
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Répétition triple (3-fold) construite par cycle de 4 plies → terminalValue = 0.0f")
  void threefoldRepetition_zero() {
    // Construction : on cycle Nf3-Nb1-Nc3-Nb1 etc. pour répéter la position de départ 3 fois.
    // Plus simple : utiliser Ng1-f3-g1 et Nb8-c6-b8 alternés. Après 4 demi-coups
    // (Nf3, Nc6, Ng1, Nb8) on est revenu à la position de départ : 1ère répétition.
    // Encore 4 demi-coups : 2e répétition. Encore 4 : 3e répétition.
    GameState s = new GameState();
    String[] uciCycle = {"g1f3", "b8c6", "f3g1", "c6b8"};
    int[] buf = new int[org.nanozero.board.MoveGen.RECOMMENDED_BUFFER_SIZE];
    // 2 cycles complets + 1 cycle complet = 3 occurrences de startpos après le 3e cycle.
    // Total : 12 demi-coups après lesquels la position de départ a été visitée 3 fois.
    // Note : isRepetition(3) compte la position courante, donc dès la 3e occurrence on a true.
    for (int cycle = 0; cycle < 3; cycle++) {
      for (String uci : uciCycle) {
        int n = s.generateMoves(buf, 0);
        boolean applied = false;
        for (int i = 0; i < n; i++) {
          if (org.nanozero.board.Move.toUci(buf[i]).equals(uci)) {
            s.applyMove(buf[i]);
            applied = true;
            break;
          }
        }
        if (!applied) {
          throw new AssertionError("Cycle move " + uci + " absent des coups légaux");
        }
      }
    }
    // Après le 3e cycle, la position startpos a été visitée 3 fois (initiale + 2 retours).
    // Wait : 1 cycle = 4 demi-coups, retour à startpos. Après cycle 1 : startpos visitée 2 fois
    // (initiale + retour). Cycle 2 : 3 fois. Cycle 3 : 4 fois.
    // Donc après cycle 2 on a déjà 3-fold. On vérifie après les 3 cycles (4-fold = OK aussi).
    assertThat(s.isRepetition(3)).isTrue();
    assertThat(s.isTerminal()).isTrue();
    assertThat(s.isCheckmate()).isFalse();
    assertTerminalBehavior(s, 0.0f, "threefold repetition");
  }

  // -------------------------------------------------------------------------------------------
  // Sanity : position non-terminale n'est pas traitée comme terminale
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("startpos n'est pas terminale (sanity)")
  void startposIsNotTerminal() {
    GameState s = new GameState();
    assertThat(s.isTerminal()).isFalse();
    assertThat(s.isCheckmate()).isFalse();
    assertThat(s.isStalemate()).isFalse();
  }
}
