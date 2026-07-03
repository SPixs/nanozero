package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Modifier;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;

/**
 * Tests {@link Backup} (cf. SPEC §5.4, §12 phase 5).
 *
 * <p>Couvre l'alternance du signe sur des chaînes de profondeurs variées (1, 5, 30), l'accumulation
 * de plusieurs backups, l'isolation des sous-arbres non concernés, le comportement post re-rooting,
 * et la propagation des valeurs extrêmes (caller responsibility).
 */
class BackupTest {

  // -------------------------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------------------------

  /**
   * Crée une chaîne linéaire root → child → grandchild → ... de longueur {@code depth}. Retourne la
   * racine ; la feuille est à l'index {@code depth - 1} dans l'ordre top-down. Tous les nœuds sont
   * non expandés (totalVisits=0, totalValueSum=0).
   */
  private static Node[] createChain(int depth) {
    Node[] chain = new Node[depth];
    chain[0] = new Node();
    for (int i = 1; i < depth; i++) {
      chain[i] = new Node(chain[i - 1], 0);
    }
    return chain;
  }

  // -------------------------------------------------------------------------------------------
  // Cas dégénérés et chaînes courtes
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Single node (pas de parent) : visites=1, sum = leafValue")
  void singleNode() {
    Node leaf = new Node();
    Backup.backup(leaf, 0.5f);
    assertThat(leaf.totalVisits.get()).isEqualTo(1);
    assertThat(leaf.totalValueSum.get()).isEqualTo(0.5f);
    assertThat(leaf.parent).isNull();
  }

  @Test
  @DisplayName("Chaîne de 2 (root → leaf) : leaf=+v, root=-v")
  void twoNodesSignFlipped() {
    Node[] chain = createChain(2);
    Node root = chain[0];
    Node leaf = chain[1];
    Backup.backup(leaf, 0.5f);
    assertThat(leaf.totalVisits.get()).isEqualTo(1);
    assertThat(leaf.totalValueSum.get()).isEqualTo(0.5f);
    assertThat(root.totalVisits.get()).isEqualTo(1);
    assertThat(root.totalValueSum.get()).isEqualTo(-0.5f);
  }

  @Test
  @DisplayName("Chaîne de 3 : leaf=+v, child=-v, root=+v (alternance stricte)")
  void threeNodesSignAlternates() {
    Node[] chain = createChain(3);
    Node root = chain[0];
    Node child = chain[1];
    Node grandchild = chain[2];
    Backup.backup(grandchild, 0.8f);
    assertThat(grandchild.totalValueSum.get()).isEqualTo(0.8f);
    assertThat(child.totalValueSum.get()).isEqualTo(-0.8f);
    assertThat(root.totalValueSum.get()).isEqualTo(0.8f);
    assertThat(grandchild.totalVisits.get()).isEqualTo(1);
    assertThat(child.totalVisits.get()).isEqualTo(1);
    assertThat(root.totalVisits.get()).isEqualTo(1);
  }

  @Test
  @DisplayName("Chaîne de 30 plies : alternance correcte ply pair / impair")
  void thirtyNodesSignAlternation() {
    Node[] chain = createChain(30);
    // chain[29] = leaf, chain[0] = root.
    Backup.backup(chain[29], 1.0f);
    for (int i = 0; i < 30; i++) {
      // Distance depuis la feuille = (29 - i). Distance pair → +1.0, impair → -1.0.
      int distFromLeaf = 29 - i;
      float expected = (distFromLeaf % 2 == 0) ? 1.0f : -1.0f;
      assertThat(chain[i].totalValueSum.get())
          .as("Node depth-from-root=%d (depth-from-leaf=%d)", i, distFromLeaf)
          .isEqualTo(expected);
      assertThat(chain[i].totalVisits.get()).isEqualTo(1);
    }
  }

  // -------------------------------------------------------------------------------------------
  // Accumulation et signes
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Multiple backups accumulent visites et sums")
  void multipleBackupsAccumulate() {
    Node[] chain = createChain(2);
    Node root = chain[0];
    Node leaf = chain[1];

    Backup.backup(leaf, 0.3f);
    Backup.backup(leaf, 0.7f);

    assertThat(leaf.totalVisits.get()).isEqualTo(2);
    assertThat(leaf.totalValueSum.get()).isEqualTo(1.0f); // 0.3 + 0.7
    assertThat(root.totalVisits.get()).isEqualTo(2);
    assertThat(root.totalValueSum.get()).isEqualTo(-1.0f); // -0.3 + -0.7
  }

  @Test
  @DisplayName("Value négative : leaf=-v, parent=+v (inversion ramène au positif)")
  void negativeValueInverts() {
    Node[] chain = createChain(2);
    Backup.backup(chain[1], -0.6f);
    assertThat(chain[1].totalValueSum.get()).isEqualTo(-0.6f);
    assertThat(chain[0].totalValueSum.get()).isEqualTo(0.6f);
  }

  @Test
  @DisplayName("Value zéro : sum=0 partout, visites=1 partout")
  void zeroValuePropagation() {
    Node[] chain = createChain(5);
    Backup.backup(chain[4], 0.0f);
    for (Node n : chain) {
      assertThat(n.totalVisits.get()).isEqualTo(1);
      assertThat(n.totalValueSum.get()).isZero();
    }
  }

  // -------------------------------------------------------------------------------------------
  // Isolation des sous-arbres non concernés
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Backup depuis un grandchild : sous-arbre frère intact (visites=0)")
  void siblingSubtreeUntouched() {
    // root
    //   ├─ childA
    //   │    ├─ grandchildAA  (← backup ici)
    //   │    └─ grandchildAB
    //   └─ childB
    //        └─ grandchildBA
    Node root = new Node();
    Node childA = new Node(root, 0);
    Node childB = new Node(root, 0);
    Node grandchildAA = new Node(childA, 0);
    Node grandchildAB = new Node(childA, 0);
    Node grandchildBA = new Node(childB, 0);

    Backup.backup(grandchildAA, 0.5f);

    // Chaîne grandchildAA → childA → root mise à jour.
    assertThat(grandchildAA.totalVisits.get()).isEqualTo(1);
    assertThat(grandchildAA.totalValueSum.get()).isEqualTo(0.5f);
    assertThat(childA.totalVisits.get()).isEqualTo(1);
    assertThat(childA.totalValueSum.get()).isEqualTo(-0.5f);
    assertThat(root.totalVisits.get()).isEqualTo(1);
    assertThat(root.totalValueSum.get()).isEqualTo(0.5f);

    // Sous-arbres non concernés : intacts.
    assertThat(grandchildAB.totalVisits.get()).isZero();
    assertThat(grandchildAB.totalValueSum.get()).isZero();
    assertThat(childB.totalVisits.get()).isZero();
    assertThat(childB.totalValueSum.get()).isZero();
    assertThat(grandchildBA.totalVisits.get()).isZero();
    assertThat(grandchildBA.totalValueSum.get()).isZero();
  }

  @Test
  @DisplayName(
      "Invariant I-Node-4 : pour une chaîne pure, parent.visits == sum(children.visits) après N"
          + " backups")
  void invariantNode4InPureChain() {
    // Construction : root → childA → leaf, et root → childB sans visites.
    Node root = new Node();
    Node childA = new Node(root, 0);
    Node childB = new Node(root, 0);
    Node leaf = new Node(childA, 0);
    // root est un parent avec 2 children (A et B), childA est un parent avec 1 enfant (leaf).

    // 5 backups depuis leaf.
    for (int i = 0; i < 5; i++) {
      Backup.backup(leaf, 0.1f);
    }

    // root.totalVisits.get() == childA.totalVisits + childB.totalVisits (childB=0, donc =5+0=5).
    assertThat(root.totalVisits.get())
        .isEqualTo(childA.totalVisits.get() + childB.totalVisits.get());
    assertThat(childA.totalVisits.get()).isEqualTo(leaf.totalVisits.get());
    assertThat(root.totalVisits.get()).isEqualTo(5);
    assertThat(childA.totalVisits.get()).isEqualTo(5);
    assertThat(leaf.totalVisits.get()).isEqualTo(5);
    assertThat(childB.totalVisits.get()).isZero();
  }

  // -------------------------------------------------------------------------------------------
  // Comportement post re-rooting (chaîne s'arrête au nouveau root)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Post re-rooting : chaîne s'arrête au nouveau root, pas de propagation au-delà")
  void afterRerootingStopsAtNewRoot() {
    // Construit un arbre minimal et utilise SearchTree.rerootOnMove pour réaffecter parent=null.
    GameState state = new GameState();
    SearchTree tree = new SearchTree(state);
    // Expand root manuellement.
    tree.root().expanded = true;
    tree.root().childMoves = new int[] {0xAA};
    tree.root().childPriors = new float[] {1.0f};
    tree.root().children = new Node[1];
    Node child = new Node(tree.root(), 0xAA);
    tree.root().children[0] = child;

    // Expand child + matérialise grandchild.
    child.expanded = true;
    child.childMoves = new int[] {0xBB};
    child.childPriors = new float[] {1.0f};
    child.children = new Node[1];
    Node grandchild = new Node(child, 0xBB);
    child.children[0] = grandchild;

    // 1er backup pré-rerooting depuis grandchild → updates 3 nœuds.
    Backup.backup(grandchild, 0.4f);
    assertThat(grandchild.totalValueSum.get()).isEqualTo(0.4f);
    assertThat(child.totalValueSum.get()).isEqualTo(-0.4f);
    assertThat(tree.root().totalValueSum.get()).isEqualTo(0.4f);

    // Re-rooting sur child : child devient new root, parent=null. Stats conservées.
    Node ancestorRoot = tree.root();
    tree.rerootOnMove(0xAA, new GameState());
    assertThat(tree.root()).isSameAs(child);
    assertThat(child.parent).isNull();
    assertThat(child.totalValueSum.get()).isEqualTo(-0.4f); // inchangé

    // Stats de l'ancêtre figées dans le passé (orphelin GC-collectable).
    assertThat(ancestorRoot.totalValueSum.get()).isEqualTo(0.4f);

    // 2e backup post-rerooting depuis grandchild : la chaîne s'arrête au nouveau root (child).
    // Aucune propagation à ancestorRoot (qui n'est plus dans la chaîne parent du nouveau root).
    float ancestorBefore = ancestorRoot.totalValueSum.get();
    int ancestorVisitsBefore = ancestorRoot.totalVisits.get();
    Backup.backup(grandchild, 0.7f);

    // grandchild incrémenté.
    assertThat(grandchild.totalVisits.get()).isEqualTo(2);
    assertThat(grandchild.totalValueSum.get()).isEqualTo(0.4f + 0.7f); // 1.1
    // child (= newRoot) incrémenté avec inversion.
    assertThat(child.totalVisits.get()).isEqualTo(2);
    assertThat(child.totalValueSum.get()).isEqualTo(-0.4f + -0.7f); // -1.1
    // Ancien root NON modifié (chaîne s'est arrêtée au nouveau root).
    assertThat(ancestorRoot.totalValueSum.get()).isEqualTo(ancestorBefore);
    assertThat(ancestorRoot.totalVisits.get()).isEqualTo(ancestorVisitsBefore);
  }

  // -------------------------------------------------------------------------------------------
  // Validation et valeurs extrêmes
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("backup(null, *) → NPE")
  void nullLeafRejected() {
    assertThatThrownBy(() -> Backup.backup(null, 0.0f))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("leaf");
  }

  @Test
  @DisplayName("Float.MAX_VALUE propagé sans crash (caller responsibility)")
  void extremeFiniteValuePropagates() {
    Node[] chain = createChain(3);
    Backup.backup(chain[2], Float.MAX_VALUE);
    assertThat(chain[2].totalValueSum.get()).isEqualTo(Float.MAX_VALUE);
    assertThat(chain[1].totalValueSum.get()).isEqualTo(-Float.MAX_VALUE);
    assertThat(chain[0].totalValueSum.get()).isEqualTo(Float.MAX_VALUE);
  }

  @Test
  @DisplayName("Float.NaN propagé sans crash (NaN reste NaN, -NaN reste NaN)")
  void nanValuePropagates() {
    Node[] chain = createChain(3);
    Backup.backup(chain[2], Float.NaN);
    assertThat(Float.isNaN(chain[2].totalValueSum.get())).isTrue();
    assertThat(Float.isNaN(chain[1].totalValueSum.get())).isTrue();
    assertThat(Float.isNaN(chain[0].totalValueSum.get())).isTrue();
    // Les visites incrémentent normalement quand même.
    for (Node n : chain) {
      assertThat(n.totalVisits.get()).isEqualTo(1);
    }
  }

  @Test
  @DisplayName("Float.POSITIVE_INFINITY propagé : alternance +Inf / -Inf")
  void infinityValuePropagates() {
    Node[] chain = createChain(3);
    Backup.backup(chain[2], Float.POSITIVE_INFINITY);
    assertThat(chain[2].totalValueSum.get()).isEqualTo(Float.POSITIVE_INFINITY);
    assertThat(chain[1].totalValueSum.get()).isEqualTo(Float.NEGATIVE_INFINITY);
    assertThat(chain[0].totalValueSum.get()).isEqualTo(Float.POSITIVE_INFINITY);
  }

  // -------------------------------------------------------------------------------------------
  // Utility class non instanciable
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "Constructeur privé (utility class), instanciation par réflexion lève AssertionError")
  void utilityClassNotInstantiable() throws NoSuchMethodException {
    Constructor<Backup> ctor = Backup.class.getDeclaredConstructor();
    assertThat(Modifier.isPrivate(ctor.getModifiers())).isTrue();
    ctor.setAccessible(true);
    assertThatThrownBy(ctor::newInstance)
        .isInstanceOf(InvocationTargetException.class)
        .cause()
        .isInstanceOf(AssertionError.class);
  }
}
