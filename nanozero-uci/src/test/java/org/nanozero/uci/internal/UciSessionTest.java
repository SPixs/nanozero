package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.engine.Engine;
import org.nanozero.engine.EngineConfig;
import org.nanozero.engine.EngineState;
import org.nanozero.engine.SearchBudget;
import org.nanozero.engine.SearchResult;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/**
 * Tests unitaires non-race de {@link UciSession} (cf. SPEC §3.4, §5.6, §5.7, §12 phase 6).
 *
 * <p>Tests fondamentaux : construction de {@link InfoFields} depuis {@link SearchResult},
 * comportement de {@link UciSession#emitBestmove} (null no-op, valid emit, idempotence CAS),
 * démarrage et arrêt du thread {@code InfoReporter} (daemon, idempotent).
 *
 * <p>Les tests race conditions exhaustifs sont dans {@code UciSessionRaceConditionsTest}.
 */
class UciSessionTest {

  private static Network sharedNetwork;

  @BeforeAll
  static void loadNetwork() throws IOException {
    var url = UciSessionTest.class.getResource("/npz/parity-model.npz");
    Path path;
    try {
      path = Paths.get(url.toURI());
    } catch (Exception e) {
      throw new AssertionError(e);
    }
    sharedNetwork = NetworkLoader.load(path, LoadOptions.defaults());
  }

  @AfterAll
  static void clearNetwork() {
    sharedNetwork = null;
  }

  // -------------------------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------------------------

  /** Crée un writer sur ByteArrayOutputStream + PrintStream auto-flush UTF_8. */
  private static UciResponseWriter writerOver(ByteArrayOutputStream baos) {
    PrintStream ps = new PrintStream(baos, /* autoFlush */ true, StandardCharsets.UTF_8);
    return new UciResponseWriter(ps);
  }

  private static Engine newEngine() {
    return new Engine(sharedNetwork, EngineConfig.defaults());
  }

  /** Attend, sans dépasser timeoutMs, que {@code engine.state()} atteigne {@code expected}. */
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
  // buildInfoFields
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("buildInfoFields : depth, nodes, nps, timeMs, scoreCp ou scoreMate présents")
  void buildInfoFieldsFromRunningSearch() throws Exception {
    try (Engine e = newEngine()) {
      AtomicBoolean stop = new AtomicBoolean(false);
      e.startSearch(new GameState(), SearchBudget.untilStopped(stop));
      awaitState(e, EngineState.SEARCHING, 1000);
      // Attendre que des sims s'accumulent (au moins 1 snapshot).
      Thread.sleep(800);

      var session = new UciSession(e, writerOver(new ByteArrayOutputStream()), false);
      SearchResult snap = e.currentBest();
      var fields = session.buildInfoFields(snap);

      // Champs minimaux attendus quand sims > 0
      if (snap.simulationsCount() > 0) {
        assertThat(fields.nodes()).isNotEmpty();
        assertThat(fields.depth()).isNotEmpty();
        assertThat(fields.timeMs()).isNotEmpty();
        // Score cp ou mate : au moins l'un des deux selon value
        boolean hasScore = fields.scoreCp().isPresent() || fields.scoreMate().isPresent();
        if (!Float.isNaN(snap.value())) {
          assertThat(hasScore).as("score cp ou mate doit être présent si value non-NaN").isTrue();
        }
        assertThat(fields.multipv()).hasValue(1);
      }

      stop.set(true);
      e.stop();
    }
  }

  @Test
  @DisplayName("buildInfoFields : emptyResult → champs vides cohérents")
  void buildInfoFieldsFromEmpty() throws Exception {
    try (Engine e = newEngine()) {
      var session = new UciSession(e, writerOver(new ByteArrayOutputStream()), false);
      // SearchResult vide via stop() depuis IDLE → null. On utilise un SearchResult zéro fabriqué
      // via e.currentBest() avant tout startSearch (qui retourne le snapshot vide).
      SearchResult empty = e.currentBest();
      var fields = session.buildInfoFields(empty);

      // Attendu : nodes/depth empty (sims=0, pv vide), nps/timeMs empty (elapsed=0).
      assertThat(fields.nodes()).isEmpty();
      assertThat(fields.depth()).isEmpty();
      assertThat(fields.nps()).isEmpty();
      assertThat(fields.timeMs()).isEmpty();
      // value est NaN → pas de score
      assertThat(fields.scoreCp()).isEmpty();
      assertThat(fields.scoreMate()).isEmpty();
      // multipv toujours présent à 1
      assertThat(fields.multipv()).hasValue(1);
    }
  }

  // -------------------------------------------------------------------------------------------
  // emitBestmove : null no-op / emit / idempotence
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("emitBestmove(null) : no-op silencieux, aucune ligne, bestmoveEmitted=false")
  void emitBestmoveNullIsNoop() throws IOException {
    try (Engine e = newEngine()) {
      var baos = new ByteArrayOutputStream();
      var session = new UciSession(e, writerOver(baos), false);
      session.emitBestmove(null);
      assertThat(baos.toString(StandardCharsets.UTF_8)).isEmpty();
      assertThat(session.isBestmoveEmitted()).isFalse();
    }
  }

  @Test
  @DisplayName(
      "emitBestmove(valid) : info string visits + bestmove (2 lignes v1.2.0+),"
          + " bestmoveEmitted=true")
  void emitBestmoveValid() throws Exception {
    try (Engine e = newEngine()) {
      var baos = new ByteArrayOutputStream();
      var session = new UciSession(e, writerOver(baos), false);

      e.startSearch(new GameState(), SearchBudget.nodes(20));
      awaitState(e, EngineState.DONE, 15_000);
      SearchResult result = e.stop();

      session.emitBestmove(result);

      String captured = baos.toString(StandardCharsets.UTF_8);
      // v1.2.0 : "info string visits ..." émis AVANT "bestmove" (cf. SPEC §6.5).
      String[] lines = captured.split("\n");
      assertThat(lines).as("2 lignes : info string visits + bestmove (v1.2.0+)").hasSize(2);
      assertThat(lines[0]).startsWith("info string visits");
      assertThat(lines[1]).startsWith("bestmove ");
      assertThat(session.isBestmoveEmitted()).isTrue();
    }
  }

  @Test
  @DisplayName(
      "emitBestmove idempotent : 2 appels produisent 2 lignes (info string visits + bestmove,"
          + " CAS protect)")
  void emitBestmoveIdempotent() throws Exception {
    try (Engine e = newEngine()) {
      var baos = new ByteArrayOutputStream();
      var session = new UciSession(e, writerOver(baos), false);

      e.startSearch(new GameState(), SearchBudget.nodes(20));
      awaitState(e, EngineState.DONE, 15_000);
      SearchResult result = e.stop();

      session.emitBestmove(result);
      session.emitBestmove(result); // 2e appel : CAS échoue → no-op (ni info string ni bestmove)

      String captured = baos.toString(StandardCharsets.UTF_8);
      // v1.2.0 : 2 lignes (info string visits + bestmove), pas 4 — la 2e émission est skip via CAS.
      assertThat(captured.split("\n"))
          .as("2 lignes (info string visits + bestmove) malgré 2 appels")
          .hasSize(2);
    }
  }

  // -------------------------------------------------------------------------------------------
  // startInfoReporter : émission info + daemon
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("startInfoReporter : émet au moins 1 ligne info pendant SEARCHING")
  void startInfoReporterEmitsInfo() throws Exception {
    try (Engine e = newEngine()) {
      var baos = new ByteArrayOutputStream();
      var session = new UciSession(e, writerOver(baos), false);

      AtomicBoolean stop = new AtomicBoolean(false);
      e.startSearch(new GameState(), SearchBudget.untilStopped(stop));
      session.startInfoReporter();
      // Attendre 3 cycles de poll INFO_POLL_MS (1.5s) + warmup JIT du premier forward.
      Thread.sleep(3_000);

      stop.set(true);
      session.onUciStop();

      String captured = baos.toString(StandardCharsets.UTF_8);
      String[] lines = captured.split("\n");
      long infoCount = java.util.Arrays.stream(lines).filter(l -> l.startsWith("info ")).count();
      long bestmoveCount =
          java.util.Arrays.stream(lines).filter(l -> l.startsWith("bestmove ")).count();
      assertThat(infoCount).as("au moins 1 info émise").isGreaterThanOrEqualTo(1);
      assertThat(bestmoveCount).as("exactement 1 bestmove final").isEqualTo(1);
    }
  }

  @Test
  @DisplayName("Phase 12 hotfix-005: bestmove émis dans <2× le budget (no 500ms poll latency)")
  void infoReporterEmitsBestmoveQuicklyAfterDone() throws Exception {
    // Régression du bug observé en Phase 12 prod : avec Thread.sleep(INFO_POLL_MS=500ms) avant
    // de check EngineState.DONE, le bestmove était systématiquement émis 400-500ms APRÈS la fin
    // réelle du search. Conséquence : time forfeits massifs à TC=3+0.03 (budget calculé 100ms,
    // wall-clock observé 500ms).
    // Fix : engine.awaitDone(timeout) qui se réveille immédiatement via lock.notifyAll() du worker
    // quand transition DONE. Latence émission bestmove ≤ ~quelques ms post-DONE.
    final long budgetMs = 100;
    final long maxWallMs = 400; // budget * 4 (marge pour JIT warmup + CI lent)

    try (Engine e = newEngine()) {
      // Warmup : un search préliminaire pour amortir JIT compile + premier forward NN.
      e.searchSync(new GameState(), SearchBudget.nodes(10));

      var baos = new ByteArrayOutputStream();
      var session = new UciSession(e, writerOver(baos), false);

      long t0 = System.nanoTime();
      e.startSearch(new GameState(), SearchBudget.duration(java.time.Duration.ofMillis(budgetMs)));
      session.startInfoReporter();

      // Attendre que bestmove apparaisse dans le buffer (polling court côté test).
      long deadline = t0 + maxWallMs * 1_000_000L;
      while (!baos.toString(StandardCharsets.UTF_8).contains("bestmove ")) {
        if (System.nanoTime() > deadline) {
          throw new AssertionError(
              "bestmove not emitted within "
                  + maxWallMs
                  + "ms (regression: poll latency back?). Captured: "
                  + baos.toString(StandardCharsets.UTF_8));
        }
        Thread.sleep(5);
      }
      long wallMs = (System.nanoTime() - t0) / 1_000_000L;

      assertThat(wallMs)
          .as(
              "wall-clock bestmove (%d ms) doit rester proche du budget (%d ms) — pas de"
                  + " latence de poll 500ms",
              wallMs, budgetMs)
          .isLessThan(maxWallMs);
    }
  }

  @Test
  @DisplayName("startInfoReporter : thread est daemon (oublier close ne bloque pas JVM)")
  void infoReporterThreadIsDaemon() throws Exception {
    try (Engine e = newEngine()) {
      var session = new UciSession(e, writerOver(new ByteArrayOutputStream()), false);
      AtomicBoolean stop = new AtomicBoolean(false);
      e.startSearch(new GameState(), SearchBudget.untilStopped(stop));
      session.startInfoReporter();

      Thread t = session.infoReporterThreadForTest();
      assertThat(t).isNotNull();
      assertThat(t.isDaemon()).isTrue();
      assertThat(t.getName()).isEqualTo(UciSession.INFO_REPORTER_THREAD_NAME);

      stop.set(true);
      session.onUciStop();
    }
  }

  @Test
  @DisplayName("startInfoReporter : 2e appel sur même session → IllegalStateException")
  void startInfoReporterRejectsSecondCall() throws Exception {
    try (Engine e = newEngine()) {
      var session = new UciSession(e, writerOver(new ByteArrayOutputStream()), false);
      AtomicBoolean stop = new AtomicBoolean(false);
      e.startSearch(new GameState(), SearchBudget.untilStopped(stop));
      session.startInfoReporter();
      try {
        org.assertj.core.api.Assertions.assertThatThrownBy(session::startInfoReporter)
            .isInstanceOf(IllegalStateException.class)
            .hasMessageContaining("already started");
      } finally {
        stop.set(true);
        session.onUciStop();
      }
    }
  }

  // -------------------------------------------------------------------------------------------
  // stopInfoReporter idempotent
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("stopInfoReporter idempotent : 2 appels successifs sans exception")
  void stopInfoReporterIdempotent() throws Exception {
    try (Engine e = newEngine()) {
      var session = new UciSession(e, writerOver(new ByteArrayOutputStream()), false);
      AtomicBoolean stop = new AtomicBoolean(false);
      e.startSearch(new GameState(), SearchBudget.untilStopped(stop));
      session.startInfoReporter();
      Thread.sleep(100);

      session.stopInfoReporter();
      session.stopInfoReporter(); // 2e appel : no-op

      stop.set(true);
      e.stop();
    }
  }

  @Test
  @DisplayName("stopInfoReporter avant startInfoReporter : no-op silencieux")
  void stopInfoReporterBeforeStartIsNoop() throws IOException {
    try (Engine e = newEngine()) {
      var session = new UciSession(e, writerOver(new ByteArrayOutputStream()), false);
      session.stopInfoReporter(); // pas d'exception
    }
  }

  // -------------------------------------------------------------------------------------------
  // buildInfoFields heuristique mate vs cp (couverture des 3 branches)
  // -------------------------------------------------------------------------------------------

  /**
   * Construit un {@link SearchResult} fabriqué manuellement pour les tests de buildInfoFields.
   * Permet d'exercer les branches mate-positive / mate-negative / cp normal sans dépendre du
   * comportement runtime de l'engine.
   */
  private static SearchResult fabricateResult(float value, int[] pv) {
    int bestMove = pv.length > 0 ? pv[0] : 0;
    int sims = pv.length > 0 ? 100 : 0;
    int[] childMoves = pv.length > 0 ? new int[] {pv[0]} : new int[0];
    int[] childVisits = pv.length > 0 ? new int[] {sims} : new int[0];
    long elapsedNanos = pv.length > 0 ? 1_000_000_000L : 0L;
    return new SearchResult(
        bestMove, pv, value, sims, elapsedNanos, childVisits, childMoves, /* terminated */ true);
  }

  @Test
  @DisplayName("buildInfoFields : value > 0.95 + pv courte → score mate positive")
  void buildInfoFieldsMatePositive() throws IOException {
    try (Engine e = newEngine()) {
      var session = new UciSession(e, writerOver(new ByteArrayOutputStream()), false);
      // value=0.99, pv = 5 plies → score mate (5+1)/2 = 3
      var result = fabricateResult(0.99f, new int[] {1, 2, 3, 4, 5});
      var fields = session.buildInfoFields(result);
      assertThat(fields.scoreMate()).hasValue(3);
      assertThat(fields.scoreCp()).isEmpty();
    }
  }

  @Test
  @DisplayName("buildInfoFields : value < -0.95 + pv courte → score mate negative")
  void buildInfoFieldsMateNegative() throws IOException {
    try (Engine e = newEngine()) {
      var session = new UciSession(e, writerOver(new ByteArrayOutputStream()), false);
      // value=-0.99, pv = 4 plies → score mate -(4+1)/2 = -2
      var result = fabricateResult(-0.99f, new int[] {1, 2, 3, 4});
      var fields = session.buildInfoFields(result);
      assertThat(fields.scoreMate()).hasValue(-2);
      assertThat(fields.scoreCp()).isEmpty();
    }
  }

  @Test
  @DisplayName("buildInfoFields : value modérée (cp normal Lc0 mapping), pas de scoreMate")
  void buildInfoFieldsCpNormalRange() throws IOException {
    try (Engine e = newEngine()) {
      var session = new UciSession(e, writerOver(new ByteArrayOutputStream()), false);
      var result = fabricateResult(0.42f, new int[] {1, 2, 3});
      var fields = session.buildInfoFields(result);
      // Mapping Lc0 : v=0.42 → cp = round(111.714640912 * tan(1.5620688421 * 0.42)) = 86
      assertThat(fields.scoreCp()).hasValue(86);
      assertThat(fields.scoreMate()).isEmpty();
    }
  }

  @Test
  @DisplayName("buildInfoFields : value extrême mais pv longue (≥ 20) → cp pas mate")
  void buildInfoFieldsExtremeValueButLongPvFallsThroughToCp() throws IOException {
    try (Engine e = newEngine()) {
      var session = new UciSession(e, writerOver(new ByteArrayOutputStream()), false);
      // pv = 20 plies, hors borne mate (anti-faux-positif).
      int[] longPv = new int[20];
      for (int i = 0; i < longPv.length; i++) {
        longPv[i] = i + 1;
      }
      var result = fabricateResult(0.99f, longPv);
      var fields = session.buildInfoFields(result);
      // Mapping Lc0 : v=0.99 → cp = round(111.714640912 * tan(1.5620688421 * 0.99)) = 4587
      assertThat(fields.scoreCp()).hasValue(4587);
      assertThat(fields.scoreMate()).isEmpty();
    }
  }

  @Test
  @DisplayName("isPonder() getter retourne la valeur du constructor")
  void isPonderGetter() throws IOException {
    try (Engine e = newEngine()) {
      var s1 = new UciSession(e, writerOver(new ByteArrayOutputStream()), true);
      assertThat(s1.isPonder()).isTrue();
    }
    try (Engine e = newEngine()) {
      var s2 = new UciSession(e, writerOver(new ByteArrayOutputStream()), false);
      assertThat(s2.isPonder()).isFalse();
    }
  }
}
