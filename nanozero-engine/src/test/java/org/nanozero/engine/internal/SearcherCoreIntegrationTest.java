package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.board.Move;
import org.nanozero.board.MoveGen;
import org.nanozero.board.MoveType;
import org.nanozero.board.Square;
import org.nanozero.engine.EngineConfig;
import org.nanozero.engine.SearchBudget;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/**
 * Tests d'intégration {@link SearcherCore} avec un {@link Network} réel chargé depuis le fixture
 * {@code parity-model.npz} (cf. SPEC §12 phase 7).
 *
 * <p>Substitue le mock de phase 6 par un vrai {@link NetworkProviderImpl} encapsulant le réseau
 * Fixup-init de la phase 9a nn (mean value=-0.68 sur 100 random positions, std=0.43, logits
 * std=1.241). Critère §12 phase 7 permissif : 1000 simulations sur startpos produisent un arbre
 * cohérent (best move légal, value bornée, pas de NaN, déterminisme).
 *
 * <p>La construction de {@code SearchResult} (avec PV via descente max-visits) est différée à la
 * phase 12 ; les tests ici lisent directement l'arbre via {@code tree.root().children/childMoves}.
 */
class SearcherCoreIntegrationTest {

  // -------------------------------------------------------------------------------------------
  // Setup helpers
  // -------------------------------------------------------------------------------------------

  private static Path fixturePath() {
    var url = SearcherCoreIntegrationTest.class.getResource("/npz/parity-model.npz");
    if (url == null) {
      throw new AssertionError("Fixture introuvable : /npz/parity-model.npz");
    }
    try {
      return Paths.get(url.toURI());
    } catch (Exception e) {
      throw new AssertionError("Impossible de résoudre /npz/parity-model.npz", e);
    }
  }

  private static Network loadParityNetwork() throws IOException {
    return NetworkLoader.load(fixturePath(), LoadOptions.defaults());
  }

  private static SearcherCore newCore(Network net, EngineConfig cfg) {
    return new SearcherCore(new NetworkProviderImpl(net), cfg);
  }

  /** Calcule l'index du child avec le plus de visites (best move standard, ADR-007). */
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
  // Tests d'intégration
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Intégration : runSearch(100) sur startpos avec NN réel ne crash pas")
  void runSearchNoCrash() throws IOException {
    Network net = loadParityNetwork();
    SearchTree tree = new SearchTree(new GameState());
    SearcherCore core = newCore(net, EngineConfig.defaults());

    int sims = core.runSearch(tree, SearchBudget.nodes(100));

    assertThat(sims).isEqualTo(100);
    assertThat(tree.root().expanded).isTrue();
    assertThat(tree.root().totalVisits.get()).isEqualTo(100);
  }

  @Test
  @DisplayName(
      "Intégration : 1000 sims, arbre cohérent (priors normalisés, exploration + exploitation)")
  void runSearchTreeCoherence() throws IOException {
    Network net = loadParityNetwork();
    SearchTree tree = new SearchTree(new GameState());
    SearcherCore core = newCore(net, EngineConfig.defaults());

    core.runSearch(tree, SearchBudget.nodes(1000));

    // 20 coups légaux à startpos.
    assertThat(tree.root().childMoves).hasSize(20);
    assertThat(tree.root().childPriors).hasSize(20);

    // Priors sum ≈ 1 (post-softmax masqué via decodePolicy). Tolérance 1e-4 cumulative.
    double sumPriors = 0;
    for (float p : tree.root().childPriors) {
      assertThat(p).isGreaterThanOrEqualTo(0f);
      sumPriors += p;
    }
    assertThat((float) sumPriors).isCloseTo(1.0f, within(1e-4f));

    // PUCT explore : ≥ 2 children visités. Avec un réseau Fixup-init (non entraîné), les priors
    // sont quasi-uniformes (≈ 1/20 chacun) — PUCT explore largement, distribution relativement
    // plate. On vérifie juste que la distribution n'est pas uniforme stricte (max > min).
    int instantiated = 0;
    int maxVisits = 0;
    int minVisits = Integer.MAX_VALUE;
    for (Node c : tree.root().children) {
      if (c != null) {
        instantiated++;
        maxVisits = Math.max(maxVisits, c.totalVisits.get());
        minVisits = Math.min(minVisits, c.totalVisits.get());
      }
    }
    assertThat(instantiated).isGreaterThanOrEqualTo(2);
    assertThat(maxVisits).isGreaterThan(minVisits);
  }

  @Test
  @DisplayName("Intégration : best move (argmax visits) est un coup légal de startpos")
  void bestMoveLegal() throws IOException {
    Network net = loadParityNetwork();
    SearchTree tree = new SearchTree(new GameState());
    SearcherCore core = newCore(net, EngineConfig.defaults());

    core.runSearch(tree, SearchBudget.nodes(500));

    int bestIdx = argmaxVisits(tree.root());
    int bestMove = tree.root().childMoves[bestIdx];

    // Confronter à la liste légale fraîche depuis MoveGen.
    int[] legal = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = new GameState().generateMoves(legal, 0);
    Set<Integer> legalSet = new HashSet<>();
    for (int i = 0; i < n; i++) {
      legalSet.add(legal[i]);
    }
    assertThat(legalSet).contains(bestMove);
  }

  @Test
  @DisplayName("Intégration : value du best child bornée dans [-1, +1] et finie")
  void bestChildValueBounded() throws IOException {
    Network net = loadParityNetwork();
    SearchTree tree = new SearchTree(new GameState());
    SearcherCore core = newCore(net, EngineConfig.defaults());

    core.runSearch(tree, SearchBudget.nodes(500));

    int bestIdx = argmaxVisits(tree.root());
    Node best = tree.root().children[bestIdx];
    assertThat(best).isNotNull();
    float q = best.totalValueSum.get() / best.totalVisits.get();
    assertThat(Float.isFinite(q)).isTrue();
    assertThat(q).isBetween(-1.0f, 1.0f);
  }

  @Test
  @DisplayName("Intégration : aucun NaN/Inf dans l'arbre après 1000 sims (BFS sur sous-arbre)")
  void noNaNAnywhere() throws IOException {
    Network net = loadParityNetwork();
    SearchTree tree = new SearchTree(new GameState());
    SearcherCore core = newCore(net, EngineConfig.defaults());

    core.runSearch(tree, SearchBudget.nodes(1000));

    // BFS depuis root, vérifier sums et priors.
    java.util.ArrayDeque<Node> queue = new java.util.ArrayDeque<>();
    queue.add(tree.root());
    while (!queue.isEmpty()) {
      Node n = queue.poll();
      assertThat(Float.isFinite(n.totalValueSum.get())).as("totalValueSum finite").isTrue();
      if (n.childPriors != null) {
        for (float p : n.childPriors) {
          assertThat(Float.isFinite(p)).isTrue();
          assertThat(p).isGreaterThanOrEqualTo(0f);
        }
      }
      if (n.children != null) {
        for (Node c : n.children) {
          if (c != null) {
            queue.add(c);
          }
        }
      }
    }
  }

  @Test
  @DisplayName("Intégration : déterminisme — 2 runs identiques produisent les mêmes visites")
  void determinismOnSamePosition() throws IOException {
    Network net1 = loadParityNetwork();
    Network net2 = loadParityNetwork();

    SearchTree tree1 = new SearchTree(new GameState());
    SearchTree tree2 = new SearchTree(new GameState());

    SearcherCore core1 = newCore(net1, EngineConfig.defaults());
    SearcherCore core2 = newCore(net2, EngineConfig.defaults());

    int budget = 200;
    core1.runSearch(tree1, SearchBudget.nodes(budget));
    core2.runSearch(tree2, SearchBudget.nodes(budget));

    int n = tree1.root().childMoves.length;
    assertThat(tree2.root().childMoves.length).isEqualTo(n);
    for (int i = 0; i < n; i++) {
      int v1 = tree1.root().children[i] == null ? 0 : tree1.root().children[i].totalVisits.get();
      int v2 = tree2.root().children[i] == null ? 0 : tree2.root().children[i].totalVisits.get();
      assertThat(v2).as("child[%d] visits", i).isEqualTo(v1);
    }
  }

  @Test
  @DisplayName("Intégration : position après 1.e4 (BLACK to move), best move = coup légal noir")
  void blackToMoveBestLegal() throws IOException {
    Network net = loadParityNetwork();
    GameState afterE4 = new GameState();
    afterE4.applyMove(Move.encode(Square.E2, Square.E4, MoveType.NORMAL, 0));

    SearchTree tree = new SearchTree(afterE4);
    SearcherCore core = newCore(net, EngineConfig.defaults());
    core.runSearch(tree, SearchBudget.nodes(200));

    // Priors sum ≈ 1 sur les 20 coups noirs.
    assertThat(tree.root().childMoves).hasSize(20);
    double sumPriors = 0;
    for (float p : tree.root().childPriors) {
      sumPriors += p;
    }
    assertThat((float) sumPriors).isCloseTo(1.0f, within(1e-4f));

    // Best move dans la liste légale noire.
    int[] legal = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    GameState afterE4Probe = new GameState();
    afterE4Probe.applyMove(Move.encode(Square.E2, Square.E4, MoveType.NORMAL, 0));
    int n = afterE4Probe.generateMoves(legal, 0);
    Set<Integer> legalSet = new HashSet<>();
    for (int i = 0; i < n; i++) {
      legalSet.add(legal[i]);
    }
    int bestMove = tree.root().childMoves[argmaxVisits(tree.root())];
    assertThat(legalSet).contains(bestMove);
  }

  @Test
  @DisplayName("Intégration : rootState terminale (mat) → terminal détecté, value=-1, NN robuste")
  void terminalDetectionFromRoot() throws IOException {
    Network net = loadParityNetwork();
    GameState terminal = new GameState("4k3/8/8/8/8/8/5PPP/r6K w - - 0 1");
    assertThat(terminal.isTerminal()).isTrue();
    assertThat(terminal.isCheckmate()).isTrue();

    SearchTree tree = new SearchTree(terminal);
    SearcherCore core = newCore(net, EngineConfig.defaults());
    core.runSearch(tree, SearchBudget.nodes(10));

    assertThat(tree.root().terminal).isTrue();
    assertThat(tree.root().terminalValue).isEqualTo(-1.0f);
    assertThat(tree.root().totalVisits.get()).isEqualTo(10);
  }

  @Test
  @DisplayName("Intégration : rootState restauré (make-undo strict) après 200 sims NN réel")
  void rootStateRestoredWithRealNN() throws IOException {
    Network net = loadParityNetwork();
    GameState start = new GameState();
    String fenBefore = start.toFen();

    SearchTree tree = new SearchTree(start);
    SearcherCore core = newCore(net, EngineConfig.defaults());
    core.runSearch(tree, SearchBudget.nodes(200));

    assertThat(start.toFen()).isEqualTo(fenBefore);
  }

  // -------------------------------------------------------------------------------------------
  // Bench informatif (non bloquant)
  // -------------------------------------------------------------------------------------------

  @Test
  @Tag("perf")
  @DisplayName("Bench : throughput SearcherCore avec NN réel sur startpos (sanity > 5 sims/s)")
  void throughputWithRealNN() throws IOException {
    Network net = loadParityNetwork();
    SearchTree warmup = new SearchTree(new GameState());
    SearcherCore core = newCore(net, EngineConfig.defaults());

    // Warmup.
    core.runSearch(warmup, SearchBudget.nodes(20));

    SearchTree tree = new SearchTree(new GameState());
    long t0 = System.nanoTime();
    int sims = core.runSearch(tree, SearchBudget.nodes(100));
    long elapsedNs = System.nanoTime() - t0;
    double simsPerSec = sims * 1e9 / elapsedNs;
    System.out.printf(
        "SearcherCore + Network réel (parity-model) : %.1f sims/s (%d sims en %.2f ms)%n",
        simsPerSec, sims, elapsedNs / 1e6);
    // Sanity floor très permissif. Bench normatif : phase 11 engine.
    assertThat(simsPerSec).isGreaterThan(5.0);
  }
}
