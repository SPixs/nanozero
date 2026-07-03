package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.board.Move;
import org.nanozero.board.MoveGen;
import org.nanozero.board.MoveType;
import org.nanozero.board.Square;

/**
 * Tests {@link SearchTree} (cf. SPEC §3.2, §5.5, §12 phase 2).
 *
 * <p>Couvre tous les scénarios de re-rooting :
 *
 * <ul>
 *   <li>single : racine non expandée, terminale, coup absent, child non instancié, succès.
 *   <li>double : firstMove absent, intermédiaire non expandé, intermédiaire terminal, secondMove
 *       absent, succès.
 *   <li>validation : null arguments.
 *   <li>intégration : avec {@code GameState} réel + Move encodé.
 * </ul>
 */
class SearchTreeTest {

  // -------------------------------------------------------------------------------------------
  // Helpers : construction d'arbres dans des configurations contrôlées
  // -------------------------------------------------------------------------------------------

  private static final int MOVE_A = 0x1A;
  private static final int MOVE_B = 0x2B;
  private static final int MOVE_C = 0x3C;
  private static final int MOVE_X = 0x4D;
  private static final int MOVE_Y = 0x5E;

  /**
   * Expand une racine avec les moves donnés. Children[i] reste null pour signifier "non
   * matérialisé" jusqu'à ce que le test l'instancie via {@link #materializeChild}.
   */
  private static void expand(Node node, int... moves) {
    node.expanded = true;
    node.childMoves = moves.clone();
    node.childPriors = new float[moves.length];
    node.children = new Node[moves.length];
  }

  /** Matérialise un enfant à l'index donné et retourne la référence. */
  private static Node materializeChild(Node parent, int idx) {
    Node child = new Node(parent, parent.childMoves[idx]);
    parent.children[idx] = child;
    return child;
  }

  // -------------------------------------------------------------------------------------------
  // Constructor + accesseurs
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Constructor : fresh tree, root non null + non expandée, rootState référencé")
  void constructorFreshTree() {
    GameState state = new GameState();
    SearchTree tree = new SearchTree(state);
    assertThat(tree.root()).isNotNull();
    assertThat(tree.root().expanded).isFalse();
    assertThat(tree.root().parent).isNull();
    assertThat(tree.rootState()).isSameAs(state);
  }

  @Test
  @DisplayName("Constructor null state → NPE")
  void constructorNullStateRejected() {
    assertThatThrownBy(() -> new SearchTree(null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("initialState");
  }

  // -------------------------------------------------------------------------------------------
  // rerootOnMove
  // -------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("rerootOnMove(move, newState)")
  class RerootSingle {

    @Test
    @DisplayName("Racine non expandée → fresh tree")
    void unexpandedRootYieldsFreshTree() {
      GameState s1 = new GameState();
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(s1);
      Node initialRoot = tree.root();

      tree.rerootOnMove(MOVE_A, s2);

      assertThat(tree.root()).isNotSameAs(initialRoot);
      assertThat(tree.root().parent).isNull();
      assertThat(tree.root().expanded).isFalse();
      assertThat(tree.rootState()).isSameAs(s2);
    }

    @Test
    @DisplayName("Racine terminale → fresh tree")
    void terminalRootYieldsFreshTree() {
      GameState s1 = new GameState();
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(s1);
      tree.root().expanded = true;
      tree.root().terminal = true;
      tree.root().terminalValue = -1.0f;
      tree.root().childMoves = new int[0];
      tree.root().childPriors = new float[0];
      tree.root().children = new Node[0];

      Node initialRoot = tree.root();
      tree.rerootOnMove(MOVE_A, s2);

      assertThat(tree.root()).isNotSameAs(initialRoot);
      assertThat(tree.root().expanded).isFalse();
      assertThat(tree.rootState()).isSameAs(s2);
    }

    @Test
    @DisplayName("Coup absent de childMoves → fresh tree")
    void moveAbsentYieldsFreshTree() {
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(new GameState());
      expand(tree.root(), MOVE_A, MOVE_B);
      // Aucun child matérialisé.

      Node initialRoot = tree.root();
      tree.rerootOnMove(MOVE_C, s2);

      assertThat(tree.root()).isNotSameAs(initialRoot);
      assertThat(tree.root().expanded).isFalse();
      assertThat(tree.rootState()).isSameAs(s2);
    }

    @Test
    @DisplayName("Coup présent mais children[i] == null → fresh tree (pas de subtree à réutiliser)")
    void moveInListButChildNullYieldsFreshTree() {
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(new GameState());
      expand(tree.root(), MOVE_A, MOVE_B);
      // children[0] et children[1] restent null (jamais sélectionnés par PUCT).

      Node initialRoot = tree.root();
      tree.rerootOnMove(MOVE_A, s2);

      assertThat(tree.root()).isNotSameAs(initialRoot);
      assertThat(tree.root().expanded).isFalse();
      assertThat(tree.rootState()).isSameAs(s2);
    }

    @Test
    @DisplayName("Succès : child matérialisé promu en racine, parent=null, moveFromParent=0")
    void successPromotesChild() {
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(new GameState());
      expand(tree.root(), MOVE_A, MOVE_B);
      Node child = materializeChild(tree.root(), 0);
      child.totalVisits.set(42);
      child.moveFromParent = MOVE_A;

      tree.rerootOnMove(MOVE_A, s2);

      // Le child devient racine — même objet, pas une nouvelle alloc.
      assertThat(tree.root()).isSameAs(child);
      assertThat(tree.root().parent).isNull();
      assertThat(tree.root().moveFromParent).isZero();
      // Statistiques préservées (essence du re-rooting, économie de travail).
      assertThat(tree.root().totalVisits.get()).isEqualTo(42);
      assertThat(tree.rootState()).isSameAs(s2);
    }

    @Test
    @DisplayName("Succès même quand un autre child est non instancié")
    void successWithOtherChildNull() {
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(new GameState());
      expand(tree.root(), MOVE_A, MOVE_B, MOVE_C);
      // Seul children[1] (MOVE_B) est matérialisé.
      Node childB = materializeChild(tree.root(), 1);

      tree.rerootOnMove(MOVE_B, s2);

      assertThat(tree.root()).isSameAs(childB);
      assertThat(tree.root().parent).isNull();
    }
  }

  // -------------------------------------------------------------------------------------------
  // rerootOnMoves
  // -------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("rerootOnMoves(firstMove, secondMove, newState)")
  class RerootDouble {

    @Test
    @DisplayName("Succès : grandchild matérialisé promu en racine")
    void successPromotesGrandchild() {
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(new GameState());
      expand(tree.root(), MOVE_A, MOVE_B);
      Node childA = materializeChild(tree.root(), 0);
      expand(childA, MOVE_X, MOVE_Y);
      Node grandchild = materializeChild(childA, 0);
      grandchild.totalVisits.set(17);

      tree.rerootOnMoves(MOVE_A, MOVE_X, s2);

      assertThat(tree.root()).isSameAs(grandchild);
      assertThat(tree.root().parent).isNull();
      assertThat(tree.root().moveFromParent).isZero();
      assertThat(tree.root().totalVisits.get()).isEqualTo(17);
      assertThat(tree.rootState()).isSameAs(s2);
    }

    @Test
    @DisplayName("Racine non expandée → fresh tree")
    void unexpandedRootYieldsFreshTree() {
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(new GameState());
      Node initialRoot = tree.root();

      tree.rerootOnMoves(MOVE_A, MOVE_X, s2);

      assertThat(tree.root()).isNotSameAs(initialRoot);
      assertThat(tree.root().expanded).isFalse();
    }

    @Test
    @DisplayName("Racine terminale → fresh tree")
    void terminalRootYieldsFreshTree() {
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(new GameState());
      tree.root().expanded = true;
      tree.root().terminal = true;
      tree.root().childMoves = new int[0];
      tree.root().childPriors = new float[0];
      tree.root().children = new Node[0];
      Node initialRoot = tree.root();

      tree.rerootOnMoves(MOVE_A, MOVE_X, s2);

      assertThat(tree.root()).isNotSameAs(initialRoot);
      assertThat(tree.root().expanded).isFalse();
    }

    @Test
    @DisplayName("firstMove absent → fresh tree")
    void firstMoveMissingYieldsFreshTree() {
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(new GameState());
      expand(tree.root(), MOVE_A);
      materializeChild(tree.root(), 0);

      tree.rerootOnMoves(MOVE_C, MOVE_X, s2);

      assertThat(tree.root().expanded).isFalse();
    }

    @Test
    @DisplayName("firstMove présent mais child non matérialisé → fresh tree")
    void firstMoveChildNullYieldsFreshTree() {
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(new GameState());
      expand(tree.root(), MOVE_A);
      // root.children[0] reste null.

      tree.rerootOnMoves(MOVE_A, MOVE_X, s2);

      assertThat(tree.root().expanded).isFalse();
    }

    @Test
    @DisplayName("Intermédiaire non expandé → fresh tree")
    void intermediateUnexpandedYieldsFreshTree() {
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(new GameState());
      expand(tree.root(), MOVE_A);
      materializeChild(tree.root(), 0);
      // Intermediate n'est PAS expand().

      tree.rerootOnMoves(MOVE_A, MOVE_X, s2);

      assertThat(tree.root().expanded).isFalse();
    }

    @Test
    @DisplayName("Intermédiaire terminal → fresh tree")
    void intermediateTerminalYieldsFreshTree() {
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(new GameState());
      expand(tree.root(), MOVE_A);
      Node childA = materializeChild(tree.root(), 0);
      childA.expanded = true;
      childA.terminal = true;
      childA.terminalValue = -1.0f;
      childA.childMoves = new int[0];
      childA.childPriors = new float[0];
      childA.children = new Node[0];

      tree.rerootOnMoves(MOVE_A, MOVE_X, s2);

      assertThat(tree.root().expanded).isFalse();
    }

    @Test
    @DisplayName("secondMove absent du child intermédiaire → fresh tree")
    void secondMoveMissingYieldsFreshTree() {
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(new GameState());
      expand(tree.root(), MOVE_A);
      Node childA = materializeChild(tree.root(), 0);
      expand(childA, MOVE_Y); // pas de MOVE_X
      materializeChild(childA, 0);

      tree.rerootOnMoves(MOVE_A, MOVE_X, s2);

      assertThat(tree.root().expanded).isFalse();
    }

    @Test
    @DisplayName("secondMove présent mais grandchild non matérialisé → fresh tree")
    void secondMoveGrandchildNullYieldsFreshTree() {
      GameState s2 = new GameState();
      SearchTree tree = new SearchTree(new GameState());
      expand(tree.root(), MOVE_A);
      Node childA = materializeChild(tree.root(), 0);
      expand(childA, MOVE_X);
      // childA.children[0] reste null.

      tree.rerootOnMoves(MOVE_A, MOVE_X, s2);

      assertThat(tree.root().expanded).isFalse();
    }
  }

  // -------------------------------------------------------------------------------------------
  // Validation null
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("rerootOnMove null state → NPE")
  void rerootOnMoveNullState() {
    SearchTree tree = new SearchTree(new GameState());
    assertThatThrownBy(() -> tree.rerootOnMove(MOVE_A, null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("newState");
  }

  @Test
  @DisplayName("rerootOnMoves null state → NPE")
  void rerootOnMovesNullState() {
    SearchTree tree = new SearchTree(new GameState());
    assertThatThrownBy(() -> tree.rerootOnMoves(MOVE_A, MOVE_X, null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("newState");
  }

  // -------------------------------------------------------------------------------------------
  // Intégration GameState réel
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Intégration : GameState startpos + 1.e2-e4 légal, re-root sur ce coup")
  void integrationWithRealGameState() {
    GameState start = new GameState();
    int e2e4 = Move.encode(Square.E2, Square.E4, MoveType.NORMAL, 0);

    // Sanity : e2e4 est bien un des 20 coups légaux de startpos.
    int[] legal = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = start.generateMoves(legal, 0);
    boolean found = false;
    for (int i = 0; i < n; i++) {
      if (legal[i] == e2e4) {
        found = true;
        break;
      }
    }
    assertThat(found).as("e2e4 doit être un coup légal de startpos").isTrue();

    SearchTree tree = new SearchTree(start);
    expand(tree.root(), e2e4);
    Node afterE4Node = materializeChild(tree.root(), 0);
    afterE4Node.totalVisits.set(100);

    // Construire un GameState après e2e4 (mutation in-place ; on récupère un alias après apply).
    GameState afterE4 = new GameState();
    afterE4.applyMove(e2e4);

    tree.rerootOnMove(e2e4, afterE4);

    assertThat(tree.root()).isSameAs(afterE4Node);
    assertThat(tree.root().parent).isNull();
    assertThat(tree.root().totalVisits.get()).isEqualTo(100);
    assertThat(tree.rootState()).isSameAs(afterE4);
  }
}
