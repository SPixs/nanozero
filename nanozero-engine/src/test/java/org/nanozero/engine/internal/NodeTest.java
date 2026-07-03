package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Tests {@link Node} (cf. SPEC §3.1, §12 phase 2).
 *
 * <p>{@code Node} est une struct mutable sans validation interne. Ces tests vérifient la
 * <strong>cohérence</strong> des invariants I-Node-1 à I-Node-5 quand le nœud est mis dans un état
 * valide, pas que {@code Node} empêche les états invalides (c'est l'usage MCTS qui maintient les
 * invariants).
 */
class NodeTest {

  @Test
  @DisplayName("Constructeur racine : parent=null, moveFromParent=0, statistiques nulles")
  void constructorRoot() {
    Node root = new Node();
    assertThat(root.parent).isNull();
    assertThat(root.moveFromParent).isZero();
    assertThat(root.totalVisits.get()).isZero();
    assertThat(root.totalValueSum.get()).isZero();
    assertThat(root.expanded).isFalse();
    assertThat(root.terminal).isFalse();
    assertThat(root.terminalValue).isZero();
    assertThat(root.childMoves).isNull();
    assertThat(root.childPriors).isNull();
    assertThat(root.children).isNull();
  }

  @Test
  @DisplayName("Constructeur enfant : parent et moveFromParent affectés, autres champs par défaut")
  void constructorChild() {
    Node parent = new Node();
    Node child = new Node(parent, 0xABCD);
    assertThat(child.parent).isSameAs(parent);
    assertThat(child.moveFromParent).isEqualTo(0xABCD);
    assertThat(child.totalVisits.get()).isZero();
    assertThat(child.totalValueSum.get()).isZero();
    assertThat(child.expanded).isFalse();
    assertThat(child.terminal).isFalse();
    assertThat(child.childMoves).isNull();
    assertThat(child.childPriors).isNull();
    assertThat(child.children).isNull();
  }

  @Test
  @DisplayName("I-Node-2 : nœud non-expandé a tous ses arrays à null")
  void invariantNode2_unexpandedHasNullArrays() {
    Node node = new Node();
    assertThat(node.expanded).isFalse();
    assertThat(node.childMoves).isNull();
    assertThat(node.childPriors).isNull();
    assertThat(node.children).isNull();
  }

  @Test
  @DisplayName("I-Node-1 : nœud expandé a 3 arrays non-null de même longueur")
  void invariantNode1_expandedHasConsistentArrays() {
    Node node = new Node();
    node.expanded = true;
    node.childMoves = new int[] {0x1234, 0x5678, 0x9ABC};
    node.childPriors = new float[] {0.5f, 0.3f, 0.2f};
    node.children = new Node[3];
    assertThat(node.childMoves).isNotNull().hasSize(3);
    assertThat(node.childPriors).isNotNull().hasSize(3);
    assertThat(node.children).isNotNull().hasSize(3);
    assertThat(node.childMoves.length)
        .isEqualTo(node.childPriors.length)
        .isEqualTo(node.children.length);
  }

  @Test
  @DisplayName("I-Node-3 : nœud terminal a expanded=true, arrays vides, terminalValue ∈ {-1, 0}")
  void invariantNode3_terminalIsExpandedWithEmptyChildren() {
    Node node = new Node();
    node.expanded = true;
    node.terminal = true;
    node.terminalValue = -1.0f;
    node.childMoves = new int[0];
    node.childPriors = new float[0];
    node.children = new Node[0];
    assertThat(node.expanded).isTrue();
    assertThat(node.terminal).isTrue();
    assertThat(node.terminalValue).isIn(-1.0f, 0.0f);
    assertThat(node.childMoves).isEmpty();
    assertThat(node.childPriors).isEmpty();
    assertThat(node.children).isEmpty();
  }

  @Test
  @DisplayName("I-Node-3 : variante pat → terminalValue=0.0f")
  void invariantNode3_drawTerminalValueZero() {
    Node node = new Node();
    node.expanded = true;
    node.terminal = true;
    node.terminalValue = 0.0f;
    node.childMoves = new int[0];
    node.childPriors = new float[0];
    node.children = new Node[0];
    assertThat(node.terminalValue).isZero();
  }

  @Test
  @DisplayName("I-Node-4 : enfant visité a totalVisits ≤ parent.totalVisits dans l'usage normal")
  void invariantNode4_visitsConsistencyParentChild() {
    Node parent = new Node();
    parent.expanded = true;
    parent.childMoves = new int[] {0x1234};
    parent.childPriors = new float[] {1.0f};
    parent.children = new Node[1];
    Node child = new Node(parent, 0x1234);
    parent.children[0] = child;

    // Configuration usuelle pendant la recherche : parent.totalVisits.get() >= child.totalVisits
    parent.totalVisits.set(7);
    child.totalVisits.set(5);

    assertThat(child.totalVisits.get()).isLessThanOrEqualTo(parent.totalVisits.get());
    assertThat(parent.children[0]).isSameAs(child);
    assertThat(child.parent).isSameAs(parent);
  }

  @Test
  @DisplayName("I-Node-5 : racine après re-rooting (simulé) a parent=null")
  void invariantNode5_postRerootingHasNullParent() {
    Node oldRoot = new Node();
    oldRoot.expanded = true;
    oldRoot.childMoves = new int[] {0x1234};
    oldRoot.childPriors = new float[] {1.0f};
    oldRoot.children = new Node[] {new Node(oldRoot, 0x1234)};

    Node newRoot = oldRoot.children[0];
    assertThat(newRoot.parent).isSameAs(oldRoot);

    // Simule le re-rooting (mutation directe ; la mécanique est dans SearchTree).
    newRoot.parent = null;
    newRoot.moveFromParent = 0;
    assertThat(newRoot.parent).isNull();
    assertThat(newRoot.moveFromParent).isZero();
  }

  @Test
  @DisplayName("Mutation des statistiques (simulation BACKUP) : totalVisits++, totalValueSum +=")
  void mutateStatsLikeBackup() {
    Node node = new Node();
    assertThat(node.totalVisits.get()).isZero();
    assertThat(node.totalValueSum.get()).isZero();

    // Backup d'une simulation : N(s) += 1, W(s) += value.
    node.totalVisits.incrementAndGet();
    node.totalValueSum.atomicAdd(0.42f);
    assertThat(node.totalVisits.get()).isEqualTo(1);
    assertThat(node.totalValueSum.get()).isEqualTo(0.42f);

    node.totalVisits.incrementAndGet();
    node.totalValueSum.atomicAdd(-0.13f);
    assertThat(node.totalVisits.get()).isEqualTo(2);
    assertThat(node.totalValueSum.get()).isEqualTo(0.42f - 0.13f);
  }
}
