package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.board.MoveGen;
import org.nanozero.engine.EngineConfig;
import org.nanozero.engine.SearchBudget;
import org.nanozero.nn.MoveEncoding;
import org.nanozero.nn.NNOutput;

/**
 * Tests {@link SearcherCore} (cf. SPEC §5.1, §12 phase 6).
 *
 * <p>Critère §12 phase 6 : avec NN mocké constant (value=0, priors uniformes), 1000 simulations sur
 * position de départ produisent un arbre cohérent (visites concentrées sur quelques branches via
 * PUCT, pas réparties uniformément si {@code c_puct < ∞}).
 *
 * <p>Le mock {@link MockNetworkProvider} est self-contained (pas de Mockito) et configurable :
 * uniforme, focus sur un index, value variable, compteur d'appels.
 */
class SearcherCoreTest {

  // -------------------------------------------------------------------------------------------
  // Mock NetworkProvider self-contained
  // -------------------------------------------------------------------------------------------

  private static final class MockNetworkProvider implements NetworkProvider {

    final float[] logitsForSample0;
    float value;
    int forwardCallCount = 0;

    MockNetworkProvider() {
      this.logitsForSample0 = new float[MoveEncoding.POLICY_INDICES];
      this.value = 0.0f;
    }

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
        System.arraycopy(logitsForSample0, 0, outLogits, 0, MoveEncoding.POLICY_INDICES);
        outValues[0] = value;
      } catch (ReflectiveOperationException e) {
        throw new AssertionError("MockNetworkProvider reflection setup failed", e);
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------------------------

  private static MockNetworkProvider mockUniform(float value) {
    MockNetworkProvider mock = new MockNetworkProvider();
    java.util.Arrays.fill(mock.logitsForSample0, 0f);
    mock.value = value;
    return mock;
  }

  // -------------------------------------------------------------------------------------------
  // Budget zéro / un / N
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Budget zéro : 0 simulations, root inchangée")
  void zeroBudgetZeroSimulations() {
    SearchTree tree = new SearchTree(new GameState());
    MockNetworkProvider mock = mockUniform(0.0f);
    SearcherCore core = new SearcherCore(mock, EngineConfig.defaults());

    int sims = core.runSearch(tree, SearchBudget.nodes(0));

    assertThat(sims).isZero();
    assertThat(mock.forwardCallCount).isZero();
    assertThat(tree.root().expanded).isFalse();
    assertThat(tree.root().totalVisits.get()).isZero();
  }

  @Test
  @DisplayName("Budget 1 : 1 simulation, root expandée + visitée, value=0 propagée")
  void oneBudgetOneSimulation() {
    SearchTree tree = new SearchTree(new GameState());
    MockNetworkProvider mock = mockUniform(0.0f);
    SearcherCore core = new SearcherCore(mock, EngineConfig.defaults());

    int sims = core.runSearch(tree, SearchBudget.nodes(1));

    assertThat(sims).isEqualTo(1);
    assertThat(mock.forwardCallCount).isEqualTo(1);
    assertThat(tree.root().expanded).isTrue();
    assertThat(tree.root().terminal).isFalse();
    assertThat(tree.root().totalVisits.get()).isEqualTo(1);
    assertThat(tree.root().totalValueSum.get()).isEqualTo(0.0f);
    assertThat(tree.root().childMoves).hasSize(20); // startpos
    assertThat(tree.root().childPriors).hasSize(20);
    assertThat(tree.root().children).hasSize(20);

    // Priors uniformes ≈ 1/20 = 0.05.
    double sumPriors = 0;
    for (float p : tree.root().childPriors) {
      sumPriors += p;
    }
    assertThat((float) sumPriors).isCloseTo(1.0f, within(1e-6f));
  }

  @Test
  @DisplayName("Budget 10 : tree grows, sum visits children = 9 (root visite première sim)")
  void tenBudgetTreeGrows() {
    SearchTree tree = new SearchTree(new GameState());
    MockNetworkProvider mock = mockUniform(0.0f);
    SearcherCore core = new SearcherCore(mock, EngineConfig.defaults());

    int sims = core.runSearch(tree, SearchBudget.nodes(10));

    assertThat(sims).isEqualTo(10);
    assertThat(tree.root().totalVisits.get()).isEqualTo(10);

    // Sim 1 visite seulement root (pas de child instancié).
    // Sims 2-10 (9 sims) descendent vers un child et le visitent.
    int sumChildVisits = 0;
    int childrenInstantiated = 0;
    for (Node child : tree.root().children) {
      if (child != null) {
        sumChildVisits += child.totalVisits.get();
        childrenInstantiated++;
      }
    }
    assertThat(sumChildVisits).isEqualTo(9);
    assertThat(childrenInstantiated).isGreaterThanOrEqualTo(1);
  }

  // -------------------------------------------------------------------------------------------
  // Critère §12 phase 6 : 1000 sims, arbre cohérent (concentration via PUCT)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "Critère §12 phase 6 : 1000 simulations uniform mock → arbre cohérent (concentration PUCT)")
  void thousandSimulationsCoherentTree() {
    SearchTree tree = new SearchTree(new GameState());
    MockNetworkProvider mock = mockUniform(0.0f);
    SearcherCore core = new SearcherCore(mock, EngineConfig.defaults());

    int sims = core.runSearch(tree, SearchBudget.nodes(1000));

    assertThat(sims).isEqualTo(1000);
    assertThat(tree.root().totalVisits.get()).isEqualTo(1000);

    // Tous les 20 children doivent avoir été matérialisés à 1000 sims.
    int instantiated = 0;
    int totalChildVisits = 0;
    for (Node child : tree.root().children) {
      if (child != null) {
        instantiated++;
        totalChildVisits += child.totalVisits.get();
      }
    }
    assertThat(instantiated)
        .isGreaterThanOrEqualTo(2); // PUCT distribue, plusieurs branches visitées
    // Sum des visites enfants = 999 (root visité au sim 1 sans descendre).
    assertThat(totalChildVisits).isEqualTo(999);

    // Concentration via PUCT : avec uniform mock, la première branche
    // (tie-breaking premier index) attire plus de visites que les autres.
    // Les visites NE sont PAS uniformément réparties (sinon PUCT serait inutile).
    int max = 0;
    int min = Integer.MAX_VALUE;
    for (Node child : tree.root().children) {
      if (child != null) {
        max = Math.max(max, child.totalVisits.get());
        min = Math.min(min, child.totalVisits.get());
      }
    }
    // Au moins un écart visible (max strictement > min) ; PUCT ne distribue pas uniformément.
    assertThat(max).isGreaterThan(min);
  }

  // -------------------------------------------------------------------------------------------
  // Mock favorisant un coup spécifique
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Mock favorisant l'index 5 : ce child reçoit majorité de visites")
  void favorIndex5GetsMostVisits() {
    SearchTree tree = new SearchTree(new GameState());
    MockNetworkProvider mock = new MockNetworkProvider();
    java.util.Arrays.fill(mock.logitsForSample0, 0f);
    // On expand fictivement root pour connaître l'ordre des moves, puis on remet à false.
    int[] legals = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = new GameState().generateMoves(legals, 0);
    int favoriteMove = legals[5];
    int favoriteIdx = MoveEncoding.encode(favoriteMove, org.nanozero.board.Color.WHITE);
    mock.logitsForSample0[favoriteIdx] = 100f;
    mock.value = 0.0f;

    SearcherCore core = new SearcherCore(mock, EngineConfig.defaults());
    core.runSearch(tree, SearchBudget.nodes(200));

    assertThat(tree.root().children[5]).isNotNull();
    int favoriteVisits = tree.root().children[5].totalVisits.get();
    int otherMaxVisits = 0;
    for (int i = 0; i < n; i++) {
      if (i != 5 && tree.root().children[i] != null) {
        otherMaxVisits = Math.max(otherMaxVisits, tree.root().children[i].totalVisits.get());
      }
    }
    assertThat(favoriteVisits)
        .as("favorite (5) %d > all others max %d", favoriteVisits, otherMaxVisits)
        .isGreaterThan(otherMaxVisits);
  }

  // -------------------------------------------------------------------------------------------
  // Backup avec value non-zero : alternation correcte
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Mock value=0.5, 2 sims : root sum=0 (alternation), child sum=0.5")
  void valueBackupAlternatesAcrossSimulations() {
    SearchTree tree = new SearchTree(new GameState());
    MockNetworkProvider mock = mockUniform(0.5f);
    SearcherCore core = new SearcherCore(mock, EngineConfig.defaults());

    // Sim 1 : root non expandé, l'expand donne value=0.5 → backup(root, 0.5)
    // → root.totalValueSum = 0.5, root.totalVisits = 1.
    core.runSearch(tree, SearchBudget.nodes(1));
    assertThat(tree.root().totalValueSum.get()).isEqualTo(0.5f);
    assertThat(tree.root().totalVisits.get()).isEqualTo(1);

    // Sim 2 : root expandé, descend vers child[0] (tie-breaking), child non-expandé,
    // expand → value=0.5 → backup(child, 0.5) propage : child=+0.5, root=+0.5+(-0.5)=0.
    core.runSearch(tree, SearchBudget.nodes(1)); // budget RELATIF, mais on a passé nodes(1)
    // budget recalculé chaque appel → 1 sim de plus → total 2 sims
    assertThat(tree.root().totalVisits.get()).isEqualTo(2);
    assertThat(tree.root().totalValueSum.get()).isEqualTo(0.0f); // 0.5 + (-0.5)
    Node firstChild = tree.root().children[0];
    assertThat(firstChild).isNotNull();
    assertThat(firstChild.totalVisits.get()).isEqualTo(1);
    assertThat(firstChild.totalValueSum.get()).isEqualTo(0.5f);
  }

  // -------------------------------------------------------------------------------------------
  // Terminal handling
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Position de mat (rootState terminale) : aucun appel NN, terminalValue=-1 propagée")
  void terminalRootStateNoNNCallTerminalValueMinusOne() {
    // Mat du couloir : noir avec roi e8, blanc avec pions f2 g2 h2 et roi h1, tour noire a1.
    // Black to move? non — FEN " w " : white to move, white est maté (back-rank mate).
    // Vérifier sideToMove : currentPosition().sideToMove() == WHITE et isCheckmate() == true.
    GameState terminal = new GameState("4k3/8/8/8/8/8/5PPP/r6K w - - 0 1");
    assertThat(terminal.isTerminal()).isTrue();
    assertThat(terminal.isCheckmate()).isTrue();
    assertThat(terminal.currentPosition().sideToMove()).isEqualTo(org.nanozero.board.Color.WHITE);

    SearchTree tree = new SearchTree(terminal);
    MockNetworkProvider mock = mockUniform(0.7f); // value que l'on ne devrait JAMAIS voir
    SearcherCore core = new SearcherCore(mock, EngineConfig.defaults());

    int sims = core.runSearch(tree, SearchBudget.nodes(10));

    assertThat(sims).isEqualTo(10);
    assertThat(mock.forwardCallCount).as("NN ne doit pas être appelé sur terminal").isZero();
    assertThat(tree.root().expanded).isTrue();
    assertThat(tree.root().terminal).isTrue();
    assertThat(tree.root().terminalValue).isEqualTo(-1.0f); // côté au trait maté
    assertThat(tree.root().totalVisits.get()).isEqualTo(10);
    assertThat(tree.root().totalValueSum.get()).isEqualTo(-10.0f); // 10 × (-1)
  }

  @Test
  @DisplayName("Position de pat (stalemate) : terminalValue=0, NN pas appelé")
  void stalemateRootStateTerminalValueZero() {
    // Pat construit (Black king h8, white queen f7, white king g6 : black to move, pas en échec).
    GameState terminal = new GameState("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1");
    assertThat(terminal.isTerminal()).isTrue();
    assertThat(terminal.isStalemate()).isTrue();
    assertThat(terminal.isCheckmate()).isFalse();

    SearchTree tree = new SearchTree(terminal);
    MockNetworkProvider mock = mockUniform(0.42f);
    SearcherCore core = new SearcherCore(mock, EngineConfig.defaults());

    int sims = core.runSearch(tree, SearchBudget.nodes(5));

    assertThat(sims).isEqualTo(5);
    assertThat(mock.forwardCallCount).isZero();
    assertThat(tree.root().terminal).isTrue();
    assertThat(tree.root().terminalValue).isEqualTo(0.0f);
    assertThat(tree.root().totalValueSum.get()).isEqualTo(0.0f);
    assertThat(tree.root().totalVisits.get()).isEqualTo(5);
  }

  // -------------------------------------------------------------------------------------------
  // make-undo : rootState restauré après runSearch
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("rootState restauré après runSearch (make-undo strict)")
  void rootStateRestoredAfterRunSearch() {
    GameState start = new GameState();
    String fenBefore = start.toFen();

    SearchTree tree = new SearchTree(start);
    SearcherCore core = new SearcherCore(mockUniform(0.0f), EngineConfig.defaults());
    core.runSearch(tree, SearchBudget.nodes(100));

    String fenAfter = start.toFen();
    assertThat(fenAfter).as("rootState doit être restauré post-search").isEqualTo(fenBefore);
  }

  // -------------------------------------------------------------------------------------------
  // Validation arguments + budget composé
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Constructor null provider → NPE")
  void constructorNullProviderRejected() {
    assertThatThrownBy(() -> new SearcherCore(null, EngineConfig.defaults()))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("networkProvider");
  }

  @Test
  @DisplayName("Constructor null config → NPE")
  void constructorNullConfigRejected() {
    assertThatThrownBy(() -> new SearcherCore(mockUniform(0.0f), null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("config");
  }

  @Test
  @DisplayName("runSearch null tree → NPE")
  void runSearchNullTreeRejected() {
    SearcherCore core = new SearcherCore(mockUniform(0.0f), EngineConfig.defaults());
    assertThatThrownBy(() -> core.runSearch(null, SearchBudget.nodes(1)))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("tree");
  }

  @Test
  @DisplayName("runSearch null budget → NPE")
  void runSearchNullBudgetRejected() {
    SearcherCore core = new SearcherCore(mockUniform(0.0f), EngineConfig.defaults());
    assertThatThrownBy(() -> core.runSearch(new SearchTree(new GameState()), null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("budget");
  }

  @Test
  @DisplayName("Budget untilStopped(true) initial → 0 simulations, tree intact")
  void budgetEarlyStopReturnsZero() {
    AtomicBoolean stop = new AtomicBoolean(true);
    SearchTree tree = new SearchTree(new GameState());
    MockNetworkProvider mock = mockUniform(0.0f);
    SearcherCore core = new SearcherCore(mock, EngineConfig.defaults());

    int sims = core.runSearch(tree, SearchBudget.untilStopped(stop));

    assertThat(sims).isZero();
    assertThat(tree.root().expanded).isFalse();
    assertThat(mock.forwardCallCount).isZero();
  }

  @Test
  @DisplayName("Budget composite firstOf(nodes(50), untilStopped) : nodes triggered first")
  void compositeBudgetNodesTriggers() {
    AtomicBoolean stop = new AtomicBoolean(false);
    SearchTree tree = new SearchTree(new GameState());
    SearcherCore core = new SearcherCore(mockUniform(0.0f), EngineConfig.defaults());

    SearchBudget composite =
        SearchBudget.firstOf(SearchBudget.nodes(50), SearchBudget.untilStopped(stop));
    int sims = core.runSearch(tree, composite);

    assertThat(sims).isEqualTo(50);
    assertThat(tree.root().totalVisits.get()).isEqualTo(50);
  }

  // -------------------------------------------------------------------------------------------
  // Déterminisme (ADR-010)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Déterminisme : 2 runs identiques produisent la même distribution de visites")
  void deterministicAcrossRuns() {
    int budgetN = 500;

    SearchTree tree1 = new SearchTree(new GameState());
    SearcherCore core1 = new SearcherCore(mockUniform(0.0f), EngineConfig.defaults());
    core1.runSearch(tree1, SearchBudget.nodes(budgetN));

    SearchTree tree2 = new SearchTree(new GameState());
    SearcherCore core2 = new SearcherCore(mockUniform(0.0f), EngineConfig.defaults());
    core2.runSearch(tree2, SearchBudget.nodes(budgetN));

    int n = tree1.root().childMoves.length;
    assertThat(tree2.root().childMoves.length).isEqualTo(n);
    for (int i = 0; i < n; i++) {
      int v1 = tree1.root().children[i] == null ? 0 : tree1.root().children[i].totalVisits.get();
      int v2 = tree2.root().children[i] == null ? 0 : tree2.root().children[i].totalVisits.get();
      assertThat(v2).as("child[%d] visits", i).isEqualTo(v1);
    }
  }

  // -------------------------------------------------------------------------------------------
  // v1.1.0 — Dirichlet sampling au root (cf. SPEC §5.3 amendement, ADR-012)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Dirichlet désactivé (defaults epsilon=0) : root.dirichletNoise reste null")
  void dirichletEpsilonZero_noSampling() {
    SearchTree tree = new SearchTree(new GameState());
    SearcherCore core = new SearcherCore(mockUniform(0.0f), EngineConfig.defaults());
    core.runSearch(tree, SearchBudget.nodes(100));
    assertThat(tree.root().dirichletNoise).isNull();
  }

  @Test
  @DisplayName(
      "Dirichlet activé : root.dirichletNoise non-null après expansion, somme=1, valeurs>=0")
  void dirichletEpsilonPositive_samplesOnce() {
    SearchTree tree = new SearchTree(new GameState());
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);
    SearcherCore core = new SearcherCore(mockUniform(0.0f), cfg);

    // 2 sims : sim 1 expand le root, sim 2 sample le noise au début (avant SELECT).
    core.runSearch(tree, SearchBudget.nodes(2));

    assertThat(tree.root().dirichletNoise).isNotNull();
    assertThat(tree.root().dirichletNoise).hasSize(tree.root().childMoves.length);
    float sum = 0f;
    for (float v : tree.root().dirichletNoise) {
      assertThat(v).isGreaterThanOrEqualTo(0f);
      sum += v;
    }
    assertThat(sum).isCloseTo(1.0f, within(1e-5f));
  }

  @Test
  @DisplayName("Dirichlet idempotent : pas de re-sampling entre sims (référence du tableau stable)")
  void dirichletIdempotent_noResamplingBetweenSims() {
    SearchTree tree = new SearchTree(new GameState());
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);
    SearcherCore core = new SearcherCore(mockUniform(0.0f), cfg);

    // 2 sims pour expand + sample, capture du tableau.
    core.runSearch(tree, SearchBudget.nodes(2));
    float[] noiseAfterSim2 = tree.root().dirichletNoise;
    assertThat(noiseAfterSim2).isNotNull();

    // 50 sims supplémentaires : le sample doit rester identique (référence et contenu).
    core.runSearch(tree, SearchBudget.nodes(50));
    assertThat(tree.root().dirichletNoise).isSameAs(noiseAfterSim2);
  }

  @Test
  @DisplayName("Dirichlet déterministe : seed fixée → noise identique entre 2 SearcherCore")
  void dirichletDeterministicWithFixedSeed() {
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);

    SearchTree tree1 = new SearchTree(new GameState());
    SearcherCore core1 = new SearcherCore(mockUniform(0.0f), cfg);
    core1.runSearch(tree1, SearchBudget.nodes(2));

    SearchTree tree2 = new SearchTree(new GameState());
    SearcherCore core2 = new SearcherCore(mockUniform(0.0f), cfg);
    core2.runSearch(tree2, SearchBudget.nodes(2));

    assertThat(tree1.root().dirichletNoise).isEqualTo(tree2.root().dirichletNoise);
  }

  @Test
  @DisplayName("Dirichlet non-déterministe : seeds différentes → noise différent")
  void dirichletNonDeterministicWithDifferentSeeds() {
    EngineConfig cfg1 = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 1L);
    EngineConfig cfg2 = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 2L);

    SearchTree tree1 = new SearchTree(new GameState());
    new SearcherCore(mockUniform(0.0f), cfg1).runSearch(tree1, SearchBudget.nodes(2));

    SearchTree tree2 = new SearchTree(new GameState());
    new SearcherCore(mockUniform(0.0f), cfg2).runSearch(tree2, SearchBudget.nodes(2));

    assertThat(tree1.root().dirichletNoise).isNotEqualTo(tree2.root().dirichletNoise);
  }

  // -------------------------------------------------------------------------------------------
  // v1.1.2 — Bug 1 : skip sampling on terminal root (cf. SPEC §5.3 mise à jour v1.1.2, ADR-012)
  // -------------------------------------------------------------------------------------------

  /**
   * Construit un SearchTree dont le root est pré-marqué comme expanded + terminal (childMoves
   * vide). Simule l'état après détection d'une position terminale par {@code SearcherCore} lors
   * d'une sim précédente.
   */
  private static SearchTree terminalRootTree() {
    SearchTree tree = new SearchTree(new GameState());
    Node root = tree.root();
    root.expanded = true;
    root.terminal = true;
    root.terminalValue = 0.0f; // pat / nul
    root.childMoves = new int[0];
    root.childPriors = new float[0];
    root.children = new Node[0];
    return tree;
  }

  @Test
  @DisplayName(
      "v1.1.2 : Dirichlet activé + root terminal → pas de crash, dirichletNoise reste null")
  void dirichletEnabled_terminalRoot_skipsSamplingSilently() {
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);
    SearchTree tree = terminalRootTree();
    SearcherCore core = new SearcherCore(mockUniform(0.0f), cfg);

    // Sans le fix v1.1.2 : DirichletSampler.sample(0.3, 0, rng) lèverait IllegalArgumentException.
    // Avec le fix : sampling skipé via !terminal && childMoves.length > 0, runSearch passe.
    core.runSearch(tree, SearchBudget.nodes(5));

    assertThat(tree.root().dirichletNoise)
        .as("v1.1.2 : root terminal → sampling skip, dirichletNoise reste null")
        .isNull();
  }

  @Test
  @DisplayName("v1.1.2 : Dirichlet activé + root terminal + plusieurs sims → toujours pas de crash")
  void dirichletEnabled_terminalRoot_multipleSimsStable() {
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);
    SearchTree tree = terminalRootTree();
    SearcherCore core = new SearcherCore(mockUniform(0.0f), cfg);

    // 20 sims successives ; la condition v1.1.2 doit rester stable, pas de crash sporadique.
    int sims = core.runSearch(tree, SearchBudget.nodes(20));
    assertThat(sims).isEqualTo(20);
    assertThat(tree.root().dirichletNoise).isNull();
    // Backup s'est exécuté normalement sur le node terminal : totalVisits incrémenté.
    assertThat(tree.root().totalVisits.get()).isEqualTo(20);
  }

  @Test
  @DisplayName(
      "v1.1.2 : Dirichlet désactivé + root terminal → comportement v1.0.0 préservé (pas de crash)")
  void dirichletDisabled_terminalRoot_v100BehaviorPreserved() {
    // Sanity : sans Dirichlet, la condition isn't reached, terminal root reste no-op (v1.0.0 OK).
    EngineConfig cfg = EngineConfig.defaults();
    SearchTree tree = terminalRootTree();
    SearcherCore core = new SearcherCore(mockUniform(0.0f), cfg);
    core.runSearch(tree, SearchBudget.nodes(10));
    assertThat(tree.root().dirichletNoise).isNull();
    assertThat(tree.root().totalVisits.get()).isEqualTo(10);
  }

  // -------------------------------------------------------------------------------------------
  // (v1.3.0 fix M4) Garde-fou de profondeur du pathIndicesBuffer
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("M4 : capacité buffer réduite → garde-fou profondeur, pas d'AIOOBE ni leak vloss")
  void m4PathBufferDepthGuard() {
    // vloss=1.0 → vlossActive ; on réduit la capacité du pathIndicesBuffer à 1 (seam de test) pour
    // exercer la branche `depth >= length` du garde-fou M4 sans construire un arbre profond de 256.
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, 1, 1.0f);
    SearchTree tree = new SearchTree(new GameState());
    SearcherCore core = new SearcherCore(mockUniform(0.0f), cfg);
    core.pathIndicesBuffer = new int[1]; // capacité 1 : tout depth >= 1 déclenche le garde-fou

    // ~100 sims : après l'expansion en largeur des ~20 coups racine, la descente atteint depth >= 1
    // → la branche `depth < pathIndicesBuffer.length == false` est exercée. Aucun AIOOBE ne doit
    // remonter ; le release vloss (childInFlight) + le make-undo doivent rester symétriques (sinon
    // simulations échouerait ou totalVisits dériverait).
    int sims = core.runSearch(tree, SearchBudget.nodes(100));

    assertThat(sims).isEqualTo(100);
    assertThat(tree.root().expanded).isTrue();
    assertThat(tree.root().totalVisits.get()).isEqualTo(100);
    // L'arbre a bien dépassé depth 0 (au moins un child racine expandé) → garde-fou réellement
    // exercé.
    boolean someChildExpanded = false;
    for (Node c : tree.root().children) {
      if (c != null && c.expanded) {
        someChildExpanded = true;
        break;
      }
    }
    assertThat(someChildExpanded).as("la descente devait dépasser depth 0").isTrue();
  }
}
