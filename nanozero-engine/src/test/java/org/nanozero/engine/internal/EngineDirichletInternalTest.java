package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayDeque;
import java.util.Deque;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.board.MoveGen;
import org.nanozero.engine.EngineConfig;
import org.nanozero.engine.SearchBudget;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/**
 * Tests d'intégration phase 1.1.0-3 axes B (ponder + Dirichlet) et C (tree reuse + Dirichlet +
 * invariant root-only). Accès package-private au {@link SearchTree} et {@link Node} via {@link
 * ThreadController}. Pattern blanc-box cohérent avec les autres tests {@code engine.internal}.
 *
 * <p>Tag {@code slow} : classe d'intégration NN end-to-end (~6 s, 4 tests). Skipped par défaut en
 * CI rapide ; activable via {@code mvn verify -DexcludedGroups=}.
 */
@Tag("slow")
class EngineDirichletInternalTest {

  private static Network sharedNetwork;

  @BeforeAll
  static void loadNetwork() throws IOException {
    var url = EngineDirichletInternalTest.class.getResource("/npz/parity-model.npz");
    if (url == null) {
      throw new AssertionError("Fixture introuvable : /npz/parity-model.npz");
    }
    try {
      Path path = Paths.get(url.toURI());
      sharedNetwork = NetworkLoader.load(path, LoadOptions.defaults());
    } catch (Exception e) {
      throw new AssertionError("Impossible de charger parity-model.npz", e);
    }
  }

  @AfterAll
  static void clearNetwork() {
    sharedNetwork = null;
  }

  private static ThreadController newController(EngineConfig config) {
    SearcherCore searcher = new SearcherCore(new NetworkProviderImpl(sharedNetwork), config);
    return new ThreadController(searcher);
  }

  private static int firstLegalMove(GameState state) {
    int[] buffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = state.generateMoves(buffer, 0);
    assertThat(n).isGreaterThan(0);
    return buffer[0];
  }

  // -------------------------------------------------------------------------------------------
  // Axe B — Ponder + Dirichlet
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Axe B : startPonder + Dirichlet → root.dirichletNoise non-null après quelques sims")
  void dirichletDuringPonder_sampleNormallyOnRootExpansion() throws Exception {
    EngineConfig config = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);
    GameState startpos = new GameState();
    int predictedMove = firstLegalMove(startpos);

    try (ThreadController controller = newController(config)) {
      controller.submitPonder(startpos, predictedMove);
      // Attendre au moins quelques sims pour que le sampling se déclenche (après expand du root).
      awaitAtLeastSims(controller, 5);

      SearchTree tree = controller.tree;
      assertThat(tree).as("tree créé par submitPonder").isNotNull();
      assertThat(tree.root().dirichletNoise)
          .as("Pendant ponder, le root a un noise sample après la 1ère expansion")
          .isNotNull();
      assertThat(tree.root().dirichletNoise).hasSize(tree.root().childMoves.length);

      controller.stop();
    }
  }

  @Test
  @DisplayName(
      "Axe B : ponderhit conserve le noise (documented v1.1.0 behavior, divergence assumée)")
  void dirichletPreservedOnPonderhit_documentedBehavior() throws Exception {
    EngineConfig config = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);
    GameState startpos = new GameState();
    int predictedMove = firstLegalMove(startpos);

    try (ThreadController controller = newController(config)) {
      controller.submitPonder(startpos, predictedMove);
      awaitAtLeastSims(controller, 10);

      // Capture le noise sampled pendant ponder.
      float[] noiseSampledDuringPonder = controller.tree.root().dirichletNoise;
      assertThat(noiseSampledDuringPonder).isNotNull();

      // Transition ponderhit avec budget limité pour terminer la search proprement.
      controller.ponderhit(SearchBudget.nodes(30));
      controller.awaitDone();

      // Le tree est conservé sur ponderhit (cf. §5.5 amendement v1.1.0) :
      // le root reste le MÊME node, donc dirichletNoise est préservé (même référence).
      assertThat(controller.tree.root().dirichletNoise)
          .as("Documented v1.1.0 behavior: noise preserved on ponderhit (tree conservé)")
          .isSameAs(noiseSampledDuringPonder);

      controller.stop();
    }
  }

  // -------------------------------------------------------------------------------------------
  // Axe C — Tree reuse + Dirichlet + invariant root-only
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "Axe C : tree reuse → nouveau root a noise re-sampled (différent du noise précédent)")
  void dirichletTreeReuse_newRootReSamplesNoise() throws Exception {
    EngineConfig config = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);
    GameState startpos = new GameState();

    try (ThreadController controller = newController(config)) {
      // 1ère search sur startpos.
      controller.submitSearch(startpos, SearchBudget.nodes(64));
      controller.awaitDone();
      float[] noiseFirstRoot = controller.tree.root().dirichletNoise;
      assertThat(noiseFirstRoot).as("1ère search : root1 a noise sample").isNotNull();
      controller.stop();

      // 2e search sur startpos + 1er coup légal.
      // Doit déclencher tree reuse 1-ply via Zobrist (le sous-arbre du 1er coup
      // devient le nouveau root) OU fresh tree si tree reuse échoue.
      int firstMove = firstLegalMove(startpos);
      GameState afterFirstMove = new GameState();
      afterFirstMove.applyMove(firstMove);

      controller.submitSearch(afterFirstMove, SearchBudget.nodes(64));
      controller.awaitDone();
      float[] noiseSecondRoot = controller.tree.root().dirichletNoise;

      assertThat(noiseSecondRoot).as("2e search : root2 a noise re-sampled").isNotNull();
      assertThat(noiseSecondRoot)
          .as("noise re-sampled sur le nouveau root, référence différente")
          .isNotSameAs(noiseFirstRoot);

      controller.stop();
    }
  }

  @Test
  @DisplayName("Axe C : invariant root-only : seul le root peut avoir dirichletNoise != null")
  void invariantRootOnlyHasDirichletNoise() throws Exception {
    EngineConfig config = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);

    try (ThreadController controller = newController(config)) {
      controller.submitSearch(new GameState(), SearchBudget.nodes(128));
      controller.awaitDone();

      Node root = controller.tree.root();
      assertThat(root.dirichletNoise)
          .as("Le root doit avoir dirichletNoise non-null (invariant ok)")
          .isNotNull();

      int nonRootNodesWithNoise = countNonRootNodesWithNoise(root);
      assertThat(nonRootNodesWithNoise)
          .as("INVARIANT v1.1.0 : seul le root peut avoir dirichletNoise != null")
          .isZero();

      controller.stop();
    }
  }

  // -------------------------------------------------------------------------------------------
  // v1.1.2 — Bug 3 isolation cross-search : 0-ply tree reuse re-samples noise
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "v1.1.2 : 0-ply tree reuse (même position) → noise re-sampled, pas réutilisé entre searches")
  void dirichletZeroPlyReuse_reSamplesNoise() throws Exception {
    EngineConfig config = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);
    GameState startpos = new GameState();

    try (ThreadController controller = newController(config)) {
      // 1ère search sur startpos.
      controller.submitSearch(startpos, SearchBudget.nodes(64));
      controller.awaitDone();
      float[] noiseFirstSearch = controller.tree.root().dirichletNoise;
      assertThat(noiseFirstSearch).as("1ère search : root a noise sample").isNotNull();
      controller.stop();

      // 2e search sur EXACTEMENT la même position startpos (0-ply tree reuse cas).
      // Sans le fix v1.1.2 : noise réutilisé entre les 2 searches (BUG isolation cross-search).
      // Avec le fix v1.1.2 : IdleState.onSubmit clear -> re-sample à la 1ère sim.
      controller.submitSearch(new GameState(), SearchBudget.nodes(64));
      controller.awaitDone();
      float[] noiseSecondSearch = controller.tree.root().dirichletNoise;

      assertThat(noiseSecondSearch).as("2e search : root a noise re-sampled").isNotNull();
      assertThat(noiseSecondSearch)
          .as("v1.1.2 fix : noise re-sampled sur 0-ply reuse (référence différente)")
          .isNotSameAs(noiseFirstSearch);
      // Vérification contenu : avec déterminisme (seed=42), si le re-sample s'opère vraiment,
      // la séquence Random a avancé donc les valeurs diffèrent du premier sample.
      assertThat(noiseSecondSearch)
          .as("Contenu noise différent (Random séquence a avancé entre les 2 samples)")
          .isNotEqualTo(noiseFirstSearch);

      controller.stop();
    }
  }

  /**
   * Parcours itératif de l'arbre depuis les enfants de {@code root}. Compte les nodes non-root où
   * {@code dirichletNoise != null}. Retourne 0 si l'invariant tient.
   */
  private static int countNonRootNodesWithNoise(Node root) {
    int count = 0;
    Deque<Node> stack = new ArrayDeque<>();
    if (root.children != null) {
      for (Node child : root.children) {
        if (child != null) {
          stack.push(child);
        }
      }
    }
    while (!stack.isEmpty()) {
      Node n = stack.pop();
      if (n.dirichletNoise != null) {
        count++;
      }
      if (n.children != null) {
        for (Node child : n.children) {
          if (child != null) {
            stack.push(child);
          }
        }
      }
    }
    return count;
  }

  /**
   * Bloque (avec timeout) jusqu'à ce que {@code currentBest().simulationsCount() >= minSims}. Utile
   * pour observer un état stable pendant ponder (worker thread asynchrone).
   */
  private static void awaitAtLeastSims(ThreadController controller, int minSims) throws Exception {
    long deadline = System.nanoTime() + 5_000_000_000L; // 5s timeout
    while (controller.currentBest().simulationsCount() < minSims) {
      if (System.nanoTime() > deadline) {
        throw new AssertionError(
            "Timeout : currentBest().simulationsCount() < "
                + minSims
                + " après 5s (vu : "
                + controller.currentBest().simulationsCount()
                + ")");
      }
      Thread.sleep(20);
    }
  }
}
