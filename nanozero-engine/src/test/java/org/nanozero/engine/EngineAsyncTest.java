package org.nanozero.engine;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/**
 * Tests de l'API asynchrone d'{@link Engine} (cf. SPEC §4.2, §4.4, §12 phase 10).
 *
 * <p>Couvre : {@code startSearch} non bloquant, {@code stop} bloquant jusqu'à arrêt propre, {@code
 * currentBest} lock-free, transitions d'état observables, worker daemon, {@code close} interrompt
 * en &lt; 1 s, refus des opérations sur engine CLOSED.
 */
class EngineAsyncTest {

  private static Path fixturePath() {
    var url = EngineAsyncTest.class.getResource("/npz/parity-model.npz");
    try {
      return Paths.get(url.toURI());
    } catch (Exception e) {
      throw new AssertionError("Impossible de résoudre /npz/parity-model.npz", e);
    }
  }

  private static Network loadParityNetwork() throws IOException {
    return NetworkLoader.load(fixturePath(), LoadOptions.defaults());
  }

  /**
   * Attend, sans dépasser timeoutMs, que {@code state()} atteigne {@code expected}. Polling 10 ms.
   *
   * <p><strong>Choix du timeout</strong> : pour les transitions vers SEARCHING (réveil du worker +
   * lecture de la requête), 1 s suffit. Pour les transitions vers DONE qui dépendent du nombre de
   * simulations à terminer (~55 sims/s nominal mono-thread + warmup JIT au premier forward d'un
   * engine fresh, accentué sous charge JVM cumulée par la suite Surefire), prévoir au moins 15 s
   * pour absorber la variabilité CI. Les timeouts trop serrés (~5 s) ont produit des échecs
   * sporadiques en phase 10-11 sur le test {@link #stopFromDoneReturnsResult} et {@link
   * #startSearchThenStopReturnsResult}.
   */
  private static void awaitState(Engine engine, EngineState expected, long timeoutMs)
      throws InterruptedException {
    long deadline = System.nanoTime() + timeoutMs * 1_000_000L;
    while (engine.state() != expected) {
      if (System.nanoTime() > deadline) {
        throw new AssertionError(
            "Timeout waiting for state " + expected + ", got " + engine.state());
      }
      Thread.sleep(10);
    }
  }

  // -------------------------------------------------------------------------------------------
  // startSearch async : retour immédiat, état SEARCHING
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("startSearch retourne en < 50 ms (non bloquant), état SEARCHING juste après")
  void startSearchReturnsImmediately() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      AtomicBoolean stop = new AtomicBoolean(false);

      long t0 = System.nanoTime();
      e.startSearch(new GameState(), SearchBudget.untilStopped(stop));
      long elapsedMs = (System.nanoTime() - t0) / 1_000_000;

      assertThat(elapsedMs).as("startSearch latency").isLessThan(50L);
      // L'état devient SEARCHING (race possible si worker n'a pas démarré, attente brève).
      awaitState(e, EngineState.SEARCHING, 1000);

      stop.set(true);
      e.stop();
    }
  }

  @Test
  @DisplayName("startSearch + stop : recherche complète, simulationsCount > 0")
  void startSearchThenStopReturnsResult() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      e.startSearch(new GameState(), SearchBudget.nodes(50));
      // Attente que le budget soit naturellement épuisé. Timeout généreux 15 s : 50 sims @
      // ~55 sims/s = ~900 ms steady-state, plus warmup JIT (~1-2 s) sur engine fresh, plus
      // variabilité CI sous charge Surefire — voir Javadoc awaitState pour le rationale.
      awaitState(e, EngineState.DONE, 15_000);
      SearchResult r = e.stop();
      assertThat(r.simulationsCount()).isEqualTo(50);
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
    }
  }

  // -------------------------------------------------------------------------------------------
  // stop bloque jusqu'à arrêt propre
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("stop : interrompt activement (signal STOPPING) + bloque jusqu'à arrêt propre")
  void stopInterruptsActivelyAndBlocksUntilDone() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      AtomicBoolean stopFlag = new AtomicBoolean(false);
      // Budget untilStopped : sans stop() actif, la recherche tournerait indéfiniment puisque
      // stopFlag reste à false. C'est exactement le scénario UCI "stop now" : le caller doit
      // pouvoir interrompre la recherche sans dépendre du budget.
      e.startSearch(new GameState(), SearchBudget.untilStopped(stopFlag));
      awaitState(e, EngineState.SEARCHING, 1000);

      AtomicReference<SearchResult> result = new AtomicReference<>();
      CountDownLatch stopReturned = new CountDownLatch(1);
      Thread stopper =
          new Thread(
              () -> {
                result.set(e.stop());
                stopReturned.countDown();
              });
      stopper.start();

      // Avec stop() actif (pattern UCI), le worker check state=STOPPING entre simulations et
      // sort en quelques ms (durée d'une sim ~10-20 ms post-warmup, plus 1er forward NN warmup
      // peut être ~1-2 s). Tolérance permissive : 5 s.
      assertThat(stopReturned.await(5, TimeUnit.SECONDS))
          .as("stop doit débloquer en < 5 s, même si budget reste actif")
          .isTrue();

      assertThat(result.get()).isNotNull();
      assertThat(result.get().simulationsCount()).isPositive();
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
      // stopFlag n'a jamais été flippé : prouve que stop() interrompt par état, pas via le budget.
      assertThat(stopFlag.get()).isFalse();
    }
  }

  @Test
  @DisplayName("stop depuis IDLE : SearchResult vide, pas d'exception")
  void stopFromIdleReturnsEmptyResult() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
      SearchResult r = e.stop();
      assertThat(r.simulationsCount()).isZero();
      assertThat(r.bestMove()).isZero();
      assertThat(Float.isNaN(r.value())).isTrue();
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
    }
  }

  @Test
  @DisplayName("stop depuis DONE (budget naturellement épuisé) : retourne result + transition IDLE")
  void stopFromDoneReturnsResult() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      e.startSearch(new GameState(), SearchBudget.nodes(30));
      // Timeout généreux 15 s — voir Javadoc awaitState pour rationale (warmup + charge CI).
      awaitState(e, EngineState.DONE, 15_000);
      SearchResult r = e.stop();
      assertThat(r.simulationsCount()).isEqualTo(30);
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
    }
  }

  // -------------------------------------------------------------------------------------------
  // currentBest lock-free
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("currentBest pendant une recherche : snapshot lock-free, pas de crash")
  void currentBestLockFree() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      AtomicBoolean stop = new AtomicBoolean(false);
      e.startSearch(new GameState(), SearchBudget.untilStopped(stop));

      // Sleep court pour laisser au moins un snapshot interval (16 sims) s'accumuler.
      Thread.sleep(500);
      SearchResult mid = e.currentBest();
      // Snapshot mid à au moins 1 sim (peut être 0 si la recherche est très lente sur CI faible ;
      // borne inférieure permissive : sims >= 0, pas de crash).
      assertThat(mid).isNotNull();

      Thread.sleep(200);
      SearchResult later = e.currentBest();
      assertThat(later).isNotNull();
      // Les sims du later devraient être >= mid (monotone) si snapshot mis à jour entre les deux.
      assertThat(later.simulationsCount()).isGreaterThanOrEqualTo(mid.simulationsCount());

      stop.set(true);
      e.stop();
    }
  }

  @Test
  @DisplayName("currentBest avant tout startSearch : snapshot vide non null")
  void currentBestBeforeStartReturnsEmpty() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      SearchResult r = e.currentBest();
      assertThat(r).isNotNull();
      assertThat(r.simulationsCount()).isZero();
    }
  }

  // -------------------------------------------------------------------------------------------
  // Concurrence : startSearch concurrent rejected
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("startSearch concurrent : 2e appel pendant SEARCHING → IllegalStateException")
  void startSearchConcurrentRejected() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      AtomicBoolean stop = new AtomicBoolean(false);
      e.startSearch(new GameState(), SearchBudget.untilStopped(stop));
      awaitState(e, EngineState.SEARCHING, 1000);

      assertThatThrownBy(() -> e.startSearch(new GameState(), SearchBudget.nodes(1)))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("IDLE");

      stop.set(true);
      e.stop();
    }
  }

  // -------------------------------------------------------------------------------------------
  // close : interrompt + idempotent + rejette opérations subséquentes
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("close interrompt une recherche en cours en < 1 s")
  void closeInterruptsRunningSearch() throws Exception {
    Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults());
    AtomicBoolean neverFlipped = new AtomicBoolean(false);
    e.startSearch(new GameState(), SearchBudget.untilStopped(neverFlipped));
    awaitState(e, EngineState.SEARCHING, 1000);

    long t0 = System.nanoTime();
    e.close();
    long elapsedMs = (System.nanoTime() - t0) / 1_000_000;

    assertThat(elapsedMs).as("close latency").isLessThan(1000L);
    assertThat(e.state()).isEqualTo(EngineState.CLOSED);
  }

  @Test
  @DisplayName("close idempotent : 2 appels successifs OK")
  void closeIdempotent() throws IOException {
    Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults());
    e.close();
    e.close();
    assertThat(e.state()).isEqualTo(EngineState.CLOSED);
  }

  @Test
  @DisplayName("startSearch sur engine CLOSED → IllegalStateException")
  void startSearchOnClosedRejected() throws IOException {
    Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults());
    e.close();
    assertThatThrownBy(() -> e.startSearch(new GameState(), SearchBudget.nodes(1)))
        .isInstanceOf(IllegalStateException.class);
  }

  @Test
  @DisplayName("stop sur engine CLOSED → IllegalStateException")
  void stopOnClosedRejected() throws IOException {
    Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults());
    e.close();
    assertThatThrownBy(e::stop).isInstanceOf(IllegalStateException.class);
  }

  // -------------------------------------------------------------------------------------------
  // Worker daemon
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Worker thread est marqué daemon (permet JVM exit propre)")
  void workerThreadIsDaemon() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      // Inspect via réflexion sur le ThreadController (champ workerThread accessible via
      // workerThreadForTest ; on accède au controller via reflection).
      java.lang.reflect.Field controllerField = Engine.class.getDeclaredField("controller");
      controllerField.setAccessible(true);
      Object controller = controllerField.get(e);
      java.lang.reflect.Method m = controller.getClass().getDeclaredMethod("workerThreadForTest");
      m.setAccessible(true);
      Thread worker = (Thread) m.invoke(controller);
      assertThat(worker.isDaemon()).isTrue();
      assertThat(worker.getName()).startsWith("nanozero-engine-worker-");
    }
  }

  // -------------------------------------------------------------------------------------------
  // Non-régression searchSync (phase 9)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("searchSync (convenience) reste fonctionnel post-phase-10 (non-régression phase 9)")
  void searchSyncStillWorks() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      SearchResult r = e.searchSync(new GameState(), SearchBudget.nodes(50));
      assertThat(r.simulationsCount()).isEqualTo(50);
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
    }
  }
}
