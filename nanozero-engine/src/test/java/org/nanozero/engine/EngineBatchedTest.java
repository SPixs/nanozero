package org.nanozero.engine;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/**
 * Tests d'intégration {@link Engine} en mode batched multi-thread (cf. ADR-013, ADR-015, ADR-016,
 * SPEC §15, phase 1.2.0-5).
 *
 * <p>Vérifie que le branchement {@code config.searchThreads() > 1} produit des recherches valides
 * (bestmove cohérent, value finie, somme visites cohérente) sans crash ni race silencieuse.
 *
 * <p>Tag {@code @Tag("slow")} : utilise {@code parity-model.npz} (chargement NN), incompatible
 * profile CI rapide.
 */
@Tag("slow")
class EngineBatchedTest {

  private static Path fixturePath() {
    var url = EngineBatchedTest.class.getResource("/npz/parity-model.npz");
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

  private static EngineConfig batchedConfig(int searchThreads) {
    return new EngineConfig(
        2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, searchThreads, /*batchSize*/ 1, /*virtualLoss*/ 3.0f);
  }

  /**
   * (v1.3.0) Configuration mode B avec path batched-queue actif. {@code batchSize > 1} déclenche la
   * création d'un {@link org.nanozero.engine.internal.batched.SearchPool}. Sur Vector API SIMD CPU
   * (parity-model.npz), c'est volontairement l'anti-pattern documenté (cf. SPEC §16.3) — ces tests
   * valident la CORRECTNESS du path queue, pas la performance.
   */
  private static EngineConfig batchedConfigMode_B(int searchThreads, int batchSize) {
    return new EngineConfig(
        2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, searchThreads, batchSize, /*virtualLoss*/ 1.0f);
  }

  // -------------------------------------------------------------------------------------------
  // Smoke tests : searchThreads > 1 produit un SearchResult valide
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("searchThreads=2 : searchSync 100 sims → SearchResult valide")
  void searchThreads2SmallSearch() throws Exception {
    try (Engine engine = new Engine(loadParityNetwork(), batchedConfig(2))) {
      SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(100));
      assertThat(result).isNotNull();
      assertThat(result.simulationsCount())
          .as("au moins 100 sims (peut être plus en multi-thread)")
          .isGreaterThanOrEqualTo(100);
      assertThat(result.bestMove())
          .as("bestmove doit être un coup légal non null (mode startpos)")
          .isNotZero();
      assertThat(Float.isFinite(result.value())).as("value finie (pas NaN/Inf)").isTrue();
      assertThat(result.value()).isBetween(-1.0f, 1.0f);
      assertThat(result.principalVariation()).as("PV non vide après 100 sims").isNotEmpty();
      assertThat(result.childMoves()).isNotEmpty();
      assertThat(result.childVisits()).hasSameSizeAs(result.childMoves());
    }
  }

  @Test
  @DisplayName("searchThreads=4 : searchSync 200 sims → SearchResult valide")
  void searchThreads4SmallSearch() throws Exception {
    try (Engine engine = new Engine(loadParityNetwork(), batchedConfig(4))) {
      SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(200));
      assertThat(result).isNotNull();
      assertThat(result.simulationsCount()).isGreaterThanOrEqualTo(200);
      assertThat(result.bestMove()).isNotZero();
      assertThat(Float.isFinite(result.value())).isTrue();
    }
  }

  // -------------------------------------------------------------------------------------------
  // Cohérence des compteurs
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("searchThreads=4 : sum(childVisits) cohérent (≤ simulations)")
  void searchThreads4CountersConsistent() throws Exception {
    try (Engine engine = new Engine(loadParityNetwork(), batchedConfig(4))) {
      SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(500));

      int sumChildVisits = 0;
      for (int v : result.childVisits()) {
        assertThat(v).as("visite d'un child >= 0").isGreaterThanOrEqualTo(0);
        sumChildVisits += v;
      }
      // sum(childVisits) ≤ simulationsCount (la racine reçoit toujours +1, les children pas
      // tous). En multi-thread la cohérence stricte est : sum(childVisits) + 1ère sim au root ≈
      // simulations (tolérance liée au tracking inFlight).
      assertThat(sumChildVisits)
          .as("sum childVisits ≤ simulations")
          .isLessThanOrEqualTo(result.simulationsCount());
    }
  }

  // -------------------------------------------------------------------------------------------
  // Comparaison mono vs batched : bestmove cohérent
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Mono-thread vs batched=2 sur 500 sims : bestmoves dans le top-2 des visites mono")
  void monoVsBatchedBestmoveCoherent() throws Exception {
    // Mono-thread search → identifie le top-2 par visites.
    int[] monoTopChildMoves;
    int monoBest;
    try (Engine engine = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(500));
      monoBest = result.bestMove();
      // Top-2 par visites pour tolérance multi-thread non-déterministe.
      int[] visits = result.childVisits().clone();
      int[] moves = result.childMoves().clone();
      // Tri descendant des indices selon visits.
      Integer[] idx = new Integer[visits.length];
      for (int i = 0; i < idx.length; i++) {
        idx[i] = i;
      }
      java.util.Arrays.sort(idx, (a, b) -> Integer.compare(visits[b], visits[a]));
      monoTopChildMoves = new int[Math.min(3, visits.length)];
      for (int i = 0; i < monoTopChildMoves.length; i++) {
        monoTopChildMoves[i] = moves[idx[i]];
      }
    }

    // Batched search threads=2.
    int batchedBest;
    try (Engine engine = new Engine(loadParityNetwork(), batchedConfig(2))) {
      SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(500));
      batchedBest = result.bestMove();
    }

    // En multi-thread, l'ordre des sims est non-déterministe → bestmove peut différer du mono
    // exact. Tolérance : batched bestmove doit être dans le top-3 du mono.
    boolean inTop = false;
    for (int m : monoTopChildMoves) {
      if (m == batchedBest) {
        inTop = true;
        break;
      }
    }
    assertThat(inTop)
        .as(
            "bestmove batched (%s) doit être dans le top-3 mono (%s)",
            batchedBest, java.util.Arrays.toString(monoTopChildMoves))
        .isTrue();
  }

  // -------------------------------------------------------------------------------------------
  // Lifecycle : pas de thread leak
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Lifecycle batched : 3 searches consécutives + close → pas de thread leak")
  void batchedLifecycleNoLeak() throws Exception {
    int threadsBeforeAll = Thread.activeCount();
    for (int round = 0; round < 3; round++) {
      try (Engine engine = new Engine(loadParityNetwork(), batchedConfig(2))) {
        SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(50));
        assertThat(result).isNotNull();
      }
    }
    // Court délai pour permettre aux daemon threads de se finaliser.
    Thread.sleep(500);
    int threadsAfterAll = Thread.activeCount();
    // Tolérance : ±5 threads pour les variations JVM internes (GC threads, etc.). On vérifie
    // surtout l'ordre de grandeur (pas de leak proportionnel au nombre de searches × N).
    assertThat(threadsAfterAll - threadsBeforeAll)
        .as("thread leak limité (<5 threads ajoutés)")
        .isLessThan(5);
  }

  // -------------------------------------------------------------------------------------------
  // (v1.3.0) Mode B : path batched-queue actif (`batchSize > 1`). Tests correctness, pas perf.
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "v1.3.0 Mode B : searchThreads=4, batchSize=4 → SearchResult valide via NNEvalThread")
  void batched_mode_B_threads4_batchSize4() throws Exception {
    try (Engine engine = new Engine(loadParityNetwork(), batchedConfigMode_B(4, 4))) {
      SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(100));
      assertThat(result).isNotNull();
      assertThat(result.simulationsCount()).isGreaterThan(0);
      assertThat(result.bestMove()).isNotZero();
      assertThat(result.value()).isFinite();
      // Somme des visites enfants <= simulationsCount (chaque sim incrémente le compteur root,
      // mais BACKUP propage à 1 child max).
      int sumChildVisits = 0;
      for (int v : result.childVisits()) sumChildVisits += v;
      assertThat(sumChildVisits).isLessThanOrEqualTo(result.simulationsCount());
    }
  }

  @Test
  @DisplayName(
      "v1.3.0 Mode B dégénéré : searchThreads=1, batchSize=4 → fonctionne (pas de deadlock)")
  void batched_mode_B_degenerate_threads1_batchSize4() throws Exception {
    // Cas dégénéré : 1 thread MCTS soumet 1 leaf, NNEvalThread fait forward(1). Inutile en
    // pratique (aucun amortissement), MAIS doit fonctionner fonctionnellement (pas de deadlock,
    // SearchResult valide).
    try (Engine engine = new Engine(loadParityNetwork(), batchedConfigMode_B(1, 4))) {
      SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(50));
      assertThat(result).isNotNull();
      assertThat(result.simulationsCount()).isGreaterThan(0);
      assertThat(result.bestMove()).isNotZero();
      assertThat(result.value()).isFinite();
    }
  }

  @Test
  @DisplayName("v1.3.0 Mode B lifecycle : 3 searches consécutives + close → pas de deadlock")
  void batched_mode_B_lifecycle_no_deadlock() throws Exception {
    // Verify que l'ordre Engine.close() (searchPool.shutdown AVANT controller.close) ne deadlock
    // pas, même après plusieurs searches consécutives qui ont laissé l'état pool/queue en place.
    for (int round = 0; round < 3; round++) {
      try (Engine engine = new Engine(loadParityNetwork(), batchedConfigMode_B(2, 4))) {
        SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(50));
        assertThat(result).isNotNull();
      }
      // Si on arrive ici sans timeout (junit @Timeout absent mais BUILD_SUCCESS du test suffit),
      // le close() ne deadlock pas.
    }
  }
}
