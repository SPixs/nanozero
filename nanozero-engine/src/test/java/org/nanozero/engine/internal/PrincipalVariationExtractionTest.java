package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Tests unitaires de {@link ThreadController#extractPrincipalVariation} (cf. SPEC §3.3 I-Result-2,
 * §14 annexe {@code buildSearchResult}, §12 phase 12).
 *
 * <p>Construit des arbres synthétiques minimaux pour valider la sémantique de descente max-visits
 * (tie-break ordre childMoves) et les conditions d'arrêt (terminal, non-expandé, child non
 * instancié, profondeur max).
 */
class PrincipalVariationExtractionTest {

  /**
   * Construit un Node expansé avec {@code numChildren} enfants nommés (childMoves[i] = i+1). Tous
   * les enfants sont {@code null} initialement ; le caller en instancie les nœuds requis via {@link
   * #attachChild}.
   */
  private static Node expandedNode(int numChildren) {
    Node n = new Node();
    n.expanded = true;
    n.childMoves = new int[numChildren];
    n.childPriors = new float[numChildren];
    n.children = new Node[numChildren];
    for (int i = 0; i < numChildren; i++) {
      n.childMoves[i] = i + 1; // 1-based pour éviter 0 = "no move"
      n.childPriors[i] = 1.0f / numChildren;
    }
    return n;
  }

  /** Attache un child synthétique au parent à l'index {@code i}, avec {@code visits} visites. */
  private static Node attachChild(Node parent, int i, int visits) {
    Node child = new Node(parent, parent.childMoves[i]);
    child.totalVisits.set(visits);
    parent.children[i] = child;
    return child;
  }

  @Test
  @DisplayName("PV simple : root → child[0] (50 visits) → grandchild[0] (30 visits)")
  void simpleDescent() {
    Node root = expandedNode(2);
    Node child0 = attachChild(root, 0, 50);
    attachChild(root, 1, 5);

    // child0 expandé avec 2 grandchildren ; on instancie le grandchild[0] et grandchild[1]
    child0.expanded = true;
    child0.childMoves = new int[] {10, 20};
    child0.childPriors = new float[] {0.5f, 0.5f};
    child0.children = new Node[2];
    Node grand0 = new Node(child0, 10);
    grand0.totalVisits.set(30);
    child0.children[0] = grand0;
    Node grand1 = new Node(child0, 20);
    grand1.totalVisits.set(10);
    child0.children[1] = grand1;

    int[] pv = ThreadController.extractPrincipalVariation(root, 0, root.childMoves[0]);

    // Attendu : [moveVersChild0=1, moveVersGrand0=10]
    assertThat(pv).containsExactly(1, 10);
  }

  @Test
  @DisplayName("PV s'arrête sur child terminal (mat trouvé en profondeur 2)")
  void stopsAtTerminal() {
    Node root = expandedNode(1);
    Node child0 = attachChild(root, 0, 100);
    child0.expanded = true;
    child0.terminal = true;
    child0.terminalValue = -1.0f;

    int[] pv = ThreadController.extractPrincipalVariation(root, 0, root.childMoves[0]);

    // PV = [moveVersChild0] : descente s'arrête car child terminal.
    assertThat(pv).containsExactly(1);
  }

  @Test
  @DisplayName("PV s'arrête sur child non-expandé")
  void stopsAtUnexpanded() {
    Node root = expandedNode(1);
    Node child0 = attachChild(root, 0, 100);
    // child0.expanded reste false → fin de descente.
    assertThat(child0.expanded).isFalse();

    int[] pv = ThreadController.extractPrincipalVariation(root, 0, root.childMoves[0]);

    assertThat(pv).containsExactly(1);
  }

  @Test
  @DisplayName("PV s'arrête si bestChild instancié mais aucun de ses enfants ne l'est")
  void stopsWhenChildHasNoInstantiatedChildren() {
    Node root = expandedNode(1);
    Node child0 = attachChild(root, 0, 100);
    child0.expanded = true;
    child0.childMoves = new int[] {10, 20};
    child0.childPriors = new float[] {0.5f, 0.5f};
    child0.children = new Node[2]; // tous null

    int[] pv = ThreadController.extractPrincipalVariation(root, 0, root.childMoves[0]);

    assertThat(pv).containsExactly(1);
  }

  @Test
  @DisplayName("PV bornée à MAX_PV = 64 plies sur arbre profond synthétique")
  void truncatesAtMaxPv() {
    // Construit une chaîne profonde : root → c → cc → ccc → ... avec 1 child à chaque niveau
    // tous instanciés, 80 niveaux au total (= 80 plies au-delà de la racine).
    int totalDepth = 80;
    Node root = expandedNode(1);
    Node prev = attachChild(root, 0, 100);
    for (int d = 1; d < totalDepth; d++) {
      prev.expanded = true;
      prev.childMoves = new int[] {100 + d};
      prev.childPriors = new float[] {1.0f};
      prev.children = new Node[1];
      Node next = new Node(prev, 100 + d);
      next.totalVisits.set(100 - d); // décroît mais reste positif
      prev.children[0] = next;
      prev = next;
    }

    int[] pv = ThreadController.extractPrincipalVariation(root, 0, root.childMoves[0]);

    assertThat(pv).hasSize(ThreadController.MAX_PV);
    assertThat(pv[0]).isEqualTo(1); // bestMove
    assertThat(pv[1]).isEqualTo(101); // first move from child0
  }

  @Test
  @DisplayName("PV max-visits choisit le child le plus visité (pas le premier)")
  void maxVisitsTieBreakingByOrder() {
    Node root = expandedNode(3);
    Node child0 = attachChild(root, 0, 50);
    Node child1 = attachChild(root, 1, 100); // gagnant max-visits au root
    Node child2 = attachChild(root, 2, 30);

    // child1 expandé : son max-visits child est children[2] avec 80 visits.
    child1.expanded = true;
    child1.childMoves = new int[] {7, 8, 9};
    child1.childPriors = new float[] {0.3f, 0.3f, 0.4f};
    child1.children = new Node[3];
    Node grand0 = new Node(child1, 7);
    grand0.totalVisits.set(20);
    child1.children[0] = grand0;
    Node grand1 = new Node(child1, 8);
    grand1.totalVisits.set(50);
    child1.children[1] = grand1;
    Node grand2 = new Node(child1, 9);
    grand2.totalVisits.set(80);
    child1.children[2] = grand2;

    // bestIdx = 1 (root.children[1] a 100 visits, max au root).
    int[] pv = ThreadController.extractPrincipalVariation(root, 1, root.childMoves[1]);

    // Attendu : [moveVersChild1=2, moveVersGrand2=9]
    assertThat(pv).containsExactly(2, 9);
  }

  @Test
  @DisplayName("PV tie-break à visites égales : premier child instancié dans l'ordre childMoves")
  void tieBreakingFirstInOrder() {
    Node root = expandedNode(3);
    attachChild(root, 0, 50);
    Node child1 = attachChild(root, 1, 100);

    // child1.children[0] et children[2] ont les mêmes visites (max).
    child1.expanded = true;
    child1.childMoves = new int[] {7, 8, 9};
    child1.childPriors = new float[] {0.3f, 0.3f, 0.4f};
    child1.children = new Node[3];
    Node grand0 = new Node(child1, 7);
    grand0.totalVisits.set(30);
    child1.children[0] = grand0;
    // children[1] = null
    Node grand2 = new Node(child1, 9);
    grand2.totalVisits.set(30);
    child1.children[2] = grand2;

    int[] pv = ThreadController.extractPrincipalVariation(root, 1, root.childMoves[1]);

    // grand0 (visits=30) sélectionné car premier instancié à visites égales (>-1 strict break).
    assertThat(pv).containsExactly(2, 7);
  }
}
