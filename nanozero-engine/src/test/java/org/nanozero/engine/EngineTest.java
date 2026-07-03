package org.nanozero.engine;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.board.Move;
import org.nanozero.board.MoveGen;
import org.nanozero.board.MoveType;
import org.nanozero.board.Square;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/**
 * Tests {@link Engine} (cf. SPEC §4.2, §4.3, §12 phase 9).
 *
 * <p>Couvre l'API publique synchrone phase 9 : constructor, searchSync, state, close,
 * try-with-resources, transitions IDLE → SEARCHING → IDLE et IllegalStateException sur transitions
 * invalides (CLOSED, SEARCHING concurrent), stop via budget.
 */
class EngineTest {

  // -------------------------------------------------------------------------------------------
  // Setup helpers
  // -------------------------------------------------------------------------------------------

  private static Path fixturePath() {
    var url = EngineTest.class.getResource("/npz/parity-model.npz");
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

  // -------------------------------------------------------------------------------------------
  // Constructor
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Constructor : valid args → IDLE")
  void constructorValidArgs() throws IOException {
    Network net = loadParityNetwork();
    Engine e = new Engine(net, EngineConfig.defaults());
    assertThat(e.state()).isEqualTo(EngineState.IDLE);
    e.close();
  }

  @Test
  @DisplayName("Constructor null network → NPE")
  void constructorNullNetwork() {
    assertThatThrownBy(() -> new Engine(null, EngineConfig.defaults()))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("network");
  }

  @Test
  @DisplayName("Constructor null config → NPE")
  void constructorNullConfig() throws IOException {
    Network net = loadParityNetwork();
    assertThatThrownBy(() -> new Engine(net, null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("config");
  }

  // -------------------------------------------------------------------------------------------
  // searchSync : transitions et résultat
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("searchSync IDLE → SEARCHING → IDLE, result.simulationsCount == budget")
  void searchSyncTransitions() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
      SearchResult r = e.searchSync(new GameState(), SearchBudget.nodes(50));
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
      assertThat(r.simulationsCount()).isEqualTo(50);
    }
  }

  @Test
  @DisplayName("ADR-018 : nnCacheSize>0 (mode A) → recherche complète, bestMove légal")
  void searchSyncWithNnCacheEnabled() throws IOException {
    // Config par défaut MAIS cache d'évaluation NN activé (nnCacheSize=4096) — exerce la branche
    // Engine `config.nnCacheSize() > 0`, le câblage SearcherCore→LeafEvaluator avec cache non-null,
    // et les lookup/store en cours de recherche réelle (transpositions de l'arbre MCTS).
    EngineConfig cached = new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, 1, 0.0f, 4096);
    try (Engine e = new Engine(loadParityNetwork(), cached)) {
      SearchResult r = e.searchSync(new GameState(), SearchBudget.nodes(200));
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
      assertThat(r.simulationsCount()).isEqualTo(200);

      int[] legal = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
      int n = new GameState().generateMoves(legal, 0);
      Set<Integer> legalSet = new HashSet<>();
      for (int i = 0; i < n; i++) {
        legalSet.add(legal[i]);
      }
      assertThat(legalSet).as("bestMove avec cache reste légal").contains(r.bestMove());
    }
  }

  @Test
  @DisplayName("ADR-018 : mode A déterministe avec cache activé (même clé → même éval)")
  void searchSyncWithNnCacheIsDeterministic() throws IOException {
    EngineConfig cached = new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, 1, 0.0f, 8192);
    int best1;
    int best2;
    try (Engine e = new Engine(loadParityNetwork(), cached)) {
      best1 = e.searchSync(new GameState(), SearchBudget.nodes(150)).bestMove();
    }
    try (Engine e = new Engine(loadParityNetwork(), cached)) {
      best2 = e.searchSync(new GameState(), SearchBudget.nodes(150)).bestMove();
    }
    assertThat(best1).as("cache activé : déterminisme par clé préservé en mode A").isEqualTo(best2);
  }

  @Test
  @DisplayName("searchSync : SearchResult cohérent (champs valides, value bornée)")
  void searchSyncResultIsValid() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      SearchResult r = e.searchSync(new GameState(), SearchBudget.nodes(100));

      // Best move ∈ legalMoves(start).
      int[] legal = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
      int n = new GameState().generateMoves(legal, 0);
      Set<Integer> legalSet = new HashSet<>();
      for (int i = 0; i < n; i++) {
        legalSet.add(legal[i]);
      }
      assertThat(legalSet).contains(r.bestMove());

      // Champs structurels.
      assertThat(r.simulationsCount()).isEqualTo(100);
      assertThat(r.elapsedNanos()).isPositive();
      assertThat(r.childMoves()).hasSize(20);
      assertThat(r.childVisits()).hasSize(20);

      // Sum childVisits ≤ simulations (chaque sim passe par au plus 1 child à la 1ère descente,
      // ou aucun si elle s'arrête à root, donc 99 ou 100).
      int sum = 0;
      for (int v : r.childVisits()) {
        sum += v;
      }
      assertThat(sum).isLessThanOrEqualTo(r.simulationsCount());

      // Value ∈ [-1, 1] et finie.
      assertThat(Float.isFinite(r.value())).isTrue();
      assertThat(r.value()).isBetween(-1.0f, 1.0f);

      // PV par descente max-visits (phase 12, cf. SPEC §3.3 I-Result-2). Au moins
      // [bestMove] ; longueur effective dépend de la profondeur explorée.
      assertThat(r.principalVariation()).isNotEmpty();
      assertThat(r.principalVariation().length).isLessThanOrEqualTo(64);
      assertThat(r.principalVariation()[0]).isEqualTo(r.bestMove());

      // Phase 9 simplifiée : terminated = true dès simulations > 0.
      assertThat(r.terminated()).isTrue();
    }
  }

  @Test
  @DisplayName("searchSync : 2 appels successifs sur positions liées (tree reuse phase 11)")
  void searchSyncMultipleCalls() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      SearchResult r1 = e.searchSync(new GameState(), SearchBudget.nodes(30));
      assertThat(e.state()).isEqualTo(EngineState.IDLE);

      GameState afterE4 = new GameState();
      afterE4.applyMove(Move.encode(Square.E2, Square.E4, MoveType.NORMAL, 0));
      SearchResult r2 = e.searchSync(afterE4, SearchBudget.nodes(30));
      assertThat(e.state()).isEqualTo(EngineState.IDLE);

      assertThat(r1.simulationsCount()).isEqualTo(30);
      assertThat(r2.simulationsCount()).isEqualTo(30);
      // Best move différent attendu (positions différentes, sides différents).
      // Pas d'assertion stricte ici ; le sanity du fresh tree est juste l'absence de crash.
    }
  }

  @Test
  @DisplayName("searchSync zero budget : sims=0, bestMove=0, value=NaN, arrays vides")
  void searchSyncZeroBudget() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      SearchResult r = e.searchSync(new GameState(), SearchBudget.nodes(0));
      assertThat(r.simulationsCount()).isZero();
      assertThat(r.bestMove()).isZero();
      assertThat(Float.isNaN(r.value())).isTrue();
      assertThat(r.childMoves()).isEmpty();
      assertThat(r.childVisits()).isEmpty();
      assertThat(r.principalVariation()).isEmpty();
      assertThat(r.terminated()).isFalse();
    }
  }

  @Test
  @DisplayName("searchSync : rootState restauré (make-undo strict via SearcherCore)")
  void searchSyncRootStateRestored() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      GameState start = new GameState();
      String fenBefore = start.toFen();
      e.searchSync(start, SearchBudget.nodes(100));
      assertThat(start.toFen()).isEqualTo(fenBefore);
    }
  }

  @Test
  @DisplayName("searchSync null position → NPE")
  void searchSyncNullPosition() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      assertThatThrownBy(() -> e.searchSync(null, SearchBudget.nodes(1)))
          .isInstanceOf(NullPointerException.class)
          .hasMessageContaining("position");
    }
  }

  @Test
  @DisplayName("searchSync null budget → NPE")
  void searchSyncNullBudget() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      assertThatThrownBy(() -> e.searchSync(new GameState(), null))
          .isInstanceOf(NullPointerException.class)
          .hasMessageContaining("budget");
    }
  }

  // -------------------------------------------------------------------------------------------
  // Concurrence : 2 threads tentent searchSync simultanément
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "searchSync concurrent : thread B reçoit IllegalStateException pendant que A est SEARCHING")
  void searchSyncConcurrentRejected() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      AtomicBoolean stopFlag = new AtomicBoolean(false);
      CountDownLatch threadAStarted = new CountDownLatch(1);
      AtomicReference<Throwable> threadBError = new AtomicReference<>();

      // Thread A : recherche bloquée sur untilStopped jusqu'à flag.set(true).
      Thread threadA =
          new Thread(
              () -> {
                threadAStarted.countDown();
                e.searchSync(new GameState(), SearchBudget.untilStopped(stopFlag));
              });
      threadA.start();

      threadAStarted.await();
      // Petite attente pour laisser thread A entrer dans synchronized(this) et flipper l'état.
      Thread.sleep(50);
      assertThat(e.state()).isEqualTo(EngineState.SEARCHING);

      // Thread B : essaie searchSync, doit recevoir ISE.
      Thread threadB =
          new Thread(
              () -> {
                try {
                  e.searchSync(new GameState(), SearchBudget.nodes(1));
                } catch (Throwable t) {
                  threadBError.set(t);
                }
              });
      threadB.start();
      threadB.join(2000);
      assertThat(threadBError.get())
          .isNotNull()
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("SEARCHING");

      // Stop thread A.
      stopFlag.set(true);
      threadA.join(2000);
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
    }
  }

  // -------------------------------------------------------------------------------------------
  // close
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("close : idempotent (2 appels successifs OK)")
  void closeIdempotent() throws IOException {
    Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults());
    e.close();
    e.close();
    assertThat(e.state()).isEqualTo(EngineState.CLOSED);
  }

  @Test
  @DisplayName("close → state CLOSED")
  void closeTransitionsToClosed() throws IOException {
    Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults());
    assertThat(e.state()).isEqualTo(EngineState.IDLE);
    e.close();
    assertThat(e.state()).isEqualTo(EngineState.CLOSED);
  }

  @Test
  @DisplayName("searchSync sur engine CLOSED → IllegalStateException")
  void searchSyncOnClosedRejected() throws IOException {
    Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults());
    e.close();
    assertThatThrownBy(() -> e.searchSync(new GameState(), SearchBudget.nodes(1)))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("CLOSED");
  }

  // -------------------------------------------------------------------------------------------
  // try-with-resources
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("try-with-resources : close auto-appelé, state CLOSED après le bloc")
  void tryWithResourcesAutoClose() throws IOException {
    Engine handle;
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      handle = e;
      e.searchSync(new GameState(), SearchBudget.nodes(20));
    }
    assertThat(handle.state()).isEqualTo(EngineState.CLOSED);
  }

  // -------------------------------------------------------------------------------------------
  // Stop via budget (untilStopped flag)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Stop via untilStopped flag : recherche s'arrête en ~100ms après le flip")
  void stopViaUntilStoppedFlag() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      AtomicBoolean flag = new AtomicBoolean(false);
      AtomicReference<SearchResult> result = new AtomicReference<>();
      CountDownLatch done = new CountDownLatch(1);

      long t0 = System.nanoTime();
      Thread t =
          new Thread(
              () -> {
                result.set(e.searchSync(new GameState(), SearchBudget.untilStopped(flag)));
                done.countDown();
              });
      t.start();

      Thread.sleep(100);
      flag.set(true);
      assertThat(done.await(5, TimeUnit.SECONDS)).isTrue();
      long elapsedMs = (System.nanoTime() - t0) / 1_000_000;

      // L'arrêt doit arriver dans ~100-500ms (sleep 100ms + temps de fin de simulation +
      // négociation thread). Permissif pour CI variée.
      assertThat(elapsedMs).isBetween(100L, 5000L);
      assertThat(result.get()).isNotNull();
      assertThat(result.get().simulationsCount()).isPositive();
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
    }
  }

  // -------------------------------------------------------------------------------------------
  // PV extraction (cf. SPEC §3.3 I-Result-2, §14 annexe, §12 phase 12)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("PV extraction : longueur > 1 sur recherche réelle de profondeur 200 sims")
  void pvExtractionFromRealSearchHasDepth() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      SearchResult r = e.searchSync(new GameState(), SearchBudget.nodes(200));
      assertThat(r.simulationsCount()).isEqualTo(200);
      // Avec 200 sims sur la position de départ, l'arbre se développe en profondeur.
      // Borne inférieure permissive : la PV doit avoir au moins 2 plies (le bestMove + 1
      // descente). Si seulement 1, c'est que l'arbre n'a pas développé de profondeur — anormal.
      assertThat(r.principalVariation()).hasSizeGreaterThanOrEqualTo(2);
      // Cohérence I-Result-2 : pv[0] == bestMove.
      assertThat(r.principalVariation()[0]).isEqualTo(r.bestMove());
    }
  }

  @Test
  @DisplayName("PV extraction : tous les coups de la PV sont légaux dans la séquence de positions")
  void pvExtractionLegalChain() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      GameState start = new GameState();
      SearchResult r = e.searchSync(start, SearchBudget.nodes(200));
      int[] pv = r.principalVariation();
      assertThat(pv).isNotEmpty();

      // Rejouer la PV pas à pas et vérifier la légalité à chaque étape.
      GameState walker = new GameState(start.toFen());
      int[] legalBuf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
      for (int i = 0; i < pv.length; i++) {
        int n = walker.generateMoves(legalBuf, 0);
        boolean found = false;
        for (int j = 0; j < n; j++) {
          if (legalBuf[j] == pv[i]) {
            found = true;
            break;
          }
        }
        assertThat(found)
            .as("pv[%d] = 0x%04x doit être un coup légal dans la position courante", i, pv[i])
            .isTrue();
        walker.applyMove(pv[i]);
      }
    }
  }

  @Test
  @DisplayName("PV extraction : longueur ≤ MAX_PV (64 plies)")
  void pvExtractionBoundedByMaxPv() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      SearchResult r = e.searchSync(new GameState(), SearchBudget.nodes(500));
      // Borne stricte SPEC : MAX_PV = 64. PV ne peut jamais dépasser cette limite.
      assertThat(r.principalVariation().length).isLessThanOrEqualTo(64);
    }
  }
}
