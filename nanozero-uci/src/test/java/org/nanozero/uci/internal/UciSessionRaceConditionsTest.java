package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.Arrays;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.board.MoveGen;
import org.nanozero.engine.Engine;
import org.nanozero.engine.EngineConfig;
import org.nanozero.engine.EngineState;
import org.nanozero.engine.SearchBudget;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/**
 * Tests de race conditions de {@link UciSession} (cf. SPEC §8.5, §12 phase 6).
 *
 * <p>8 des 14 tests obligatoires §8.5 sont implémentés ici. Les 6 autres dépendent de l'intégration
 * UCI complète (boucle main + handler) et seront ajoutés en phase 7 : {@code
 * testRapidGoStopCycles}, {@code testQuitDuringActiveSearch}, {@code testEofDuringActiveSearch},
 * {@code testConcurrentSetoptionDuringSearch}, {@code testMultipleGoBeforeFirstBestmove}, {@code
 * testRandomCommandSequenceFuzz}.
 *
 * <p><strong>Critère exigeant</strong> : aucune flakiness tolérée. Le test {@link
 * #testBestmoveEmittedExactlyOnce_raceStopVsCompletion} doit passer 100× consécutifs sans échec
 * (validation post-livraison).
 */
class UciSessionRaceConditionsTest {

  private static final long DEFAULT_TIMEOUT_MS = 15_000;

  private static Network sharedNetwork;

  @BeforeAll
  static void loadNetwork() throws IOException {
    var url = UciSessionRaceConditionsTest.class.getResource("/npz/parity-model.npz");
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

  private static UciResponseWriter writerOver(ByteArrayOutputStream baos) {
    PrintStream ps = new PrintStream(baos, /* autoFlush */ true, StandardCharsets.UTF_8);
    return new UciResponseWriter(ps);
  }

  private static Engine newEngine() {
    return new Engine(sharedNetwork, EngineConfig.defaults());
  }

  /** Attend, sans dépasser timeoutMs, que {@code condition} devienne vraie. */
  private static void awaitCondition(java.util.function.BooleanSupplier condition, long timeoutMs)
      throws InterruptedException {
    long deadline = System.nanoTime() + timeoutMs * 1_000_000L;
    while (!condition.getAsBoolean()) {
      if (System.nanoTime() > deadline) {
        throw new AssertionError("Timeout after " + timeoutMs + "ms");
      }
      Thread.sleep(10);
    }
  }

  /** Attend la transition d'état avec timeout. */
  private static void awaitState(Engine e, EngineState s, long timeoutMs)
      throws InterruptedException {
    awaitCondition(() -> e.state() == s, timeoutMs);
  }

  /** Compte les threads daemon vivants nommés exactement {@code name}. */
  private static long countLiveThreadsNamed(String name) {
    return Thread.getAllStackTraces().keySet().stream()
        .filter(t -> t.isAlive() && name.equals(t.getName()))
        .count();
  }

  /** Asserte exactement 1 ligne {@code bestmove ...} dans la sortie capturée. */
  private static void assertExactlyOneBestmove(String captured) {
    long count = Arrays.stream(captured.split("\n")).filter(l -> l.startsWith("bestmove ")).count();
    assertThat(count).as("exactement 1 ligne bestmove (captured: <<<%s>>>)", captured).isEqualTo(1);
  }

  /** Mots-clés UCI valides en début de ligne. */
  private static final Set<String> UCI_LINE_KEYWORDS =
      Set.of("info", "bestmove", "id", "option", "uciok", "readyok");

  /** Asserte que chaque ligne capturée commence par un keyword UCI valide (pas de torn lines). */
  private static void assertAllLinesAreUciValid(String captured) {
    for (String line : captured.split("\n")) {
      if (line.isEmpty()) continue;
      String firstToken = line.split("\\s+")[0];
      assertThat(UCI_LINE_KEYWORDS)
          .as("ligne UCI valide attendue, vu: <<<%s>>>", line)
          .contains(firstToken);
      // Pas de \n interne (single line par println)
      assertThat(line).doesNotContain("\n");
    }
  }

  /** Premier coup légal de {@code state} (utile pour tests ponder). */
  private static int firstLegalMove(GameState state) {
    int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = state.generateMoves(buf, 0);
    if (n == 0) {
      throw new AssertionError("position has no legal moves");
    }
    return buf[0];
  }

  // -------------------------------------------------------------------------------------------
  // Test 1 — bestmove unique sur fin naturelle (budget épuisé)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Test 1 : bestmove émis exactement 1 fois sur fin naturelle (nodes budget)")
  void testBestmoveEmittedExactlyOnce_naturalCompletion() throws Exception {
    try (Engine e = newEngine()) {
      var baos = new ByteArrayOutputStream();
      var session = new UciSession(e, writerOver(baos), false);

      e.startSearch(new GameState(), SearchBudget.nodes(50));
      session.startInfoReporter();

      // Attendre que InfoReporter détecte DONE et émette bestmove.
      awaitCondition(session::isBestmoveEmitted, DEFAULT_TIMEOUT_MS);

      // Petit délai pour permettre à un éventuel double-emit de se manifester.
      Thread.sleep(200);

      String captured = baos.toString(StandardCharsets.UTF_8);
      assertExactlyOneBestmove(captured);
      assertAllLinesAreUciValid(captured);
    }
  }

  // -------------------------------------------------------------------------------------------
  // Test 2 — bestmove unique sur stop UCI utilisateur
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Test 2 : bestmove émis exactement 1 fois sur stop UCI utilisateur")
  void testBestmoveEmittedExactlyOnce_userStop() throws Exception {
    try (Engine e = newEngine()) {
      var baos = new ByteArrayOutputStream();
      var session = new UciSession(e, writerOver(baos), false);
      AtomicBoolean stopFlag = new AtomicBoolean(false);

      e.startSearch(new GameState(), SearchBudget.untilStopped(stopFlag));
      session.startInfoReporter();
      awaitState(e, EngineState.SEARCHING, 1_000);
      Thread.sleep(200); // sims accumulées

      session.onUciStop(); // simule réception "stop" UCI côté main thread

      assertThat(session.isBestmoveEmitted()).isTrue();
      Thread.sleep(200); // permet à un double-emit de se manifester

      String captured = baos.toString(StandardCharsets.UTF_8);
      assertExactlyOneBestmove(captured);
      assertAllLinesAreUciValid(captured);
      // stopFlag JAMAIS flippé : le stop est piloté par état, pas par budget
      assertThat(stopFlag.get()).isFalse();
    }
  }

  // -------------------------------------------------------------------------------------------
  // Test 3 — race stop vs natural completion (200 répétitions, 0 échec)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "Test 3 : race stop vs natural completion — 200 répétitions sur 4 timings, 0 échec toléré")
  void testBestmoveEmittedExactlyOnce_raceStopVsCompletion() throws Exception {
    int[] timings = {50, 100, 150, 200};
    int reps = 50;
    int totalRuns = timings.length * reps;
    long t0 = System.nanoTime();
    for (int timingMs : timings) {
      for (int rep = 0; rep < reps; rep++) {
        try (Engine e = newEngine()) {
          var baos = new ByteArrayOutputStream();
          var session = new UciSession(e, writerOver(baos), false);

          e.startSearch(new GameState(), SearchBudget.duration(Duration.ofMillis(100)));
          session.startInfoReporter();
          Thread.sleep(timingMs);
          session.onUciStop(); // race avec fin naturelle si timingMs > 100

          // Attendre que la session ait émis (l'un des deux threads)
          awaitCondition(session::isBestmoveEmitted, DEFAULT_TIMEOUT_MS);
          Thread.sleep(50); // permet double-emit éventuel de se manifester

          String captured = baos.toString(StandardCharsets.UTF_8);
          long bmCount =
              Arrays.stream(captured.split("\n")).filter(l -> l.startsWith("bestmove ")).count();
          assertThat(bmCount)
              .as("iter %d/%d, timing=%dms, captured: <<<%s>>>", rep + 1, reps, timingMs, captured)
              .isEqualTo(1);
        }
      }
    }
    long elapsedSec = (System.nanoTime() - t0) / 1_000_000_000L;
    System.err.println(
        "Test 3 race: "
            + totalRuns
            + " runs OK in "
            + elapsedSec
            + "s (avg "
            + (elapsedSec * 1000 / totalRuns)
            + "ms/run)");
  }

  // -------------------------------------------------------------------------------------------
  // Test 4 — pas d'interleaving stdout (toutes lignes UCI valides single-line)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Test 4 : pas d'interleaving stdout (lignes UCI valides single-line)")
  void testNoStdoutInterleaving() throws Exception {
    try (Engine e = newEngine()) {
      var baos = new ByteArrayOutputStream();
      var session = new UciSession(e, writerOver(baos), false);
      AtomicBoolean stopFlag = new AtomicBoolean(false);

      e.startSearch(new GameState(), SearchBudget.untilStopped(stopFlag));
      session.startInfoReporter();
      Thread.sleep(3_000); // accumulations infos

      session.onUciStop();
      Thread.sleep(200);

      String captured = baos.toString(StandardCharsets.UTF_8);
      assertAllLinesAreUciValid(captured);
      assertExactlyOneBestmove(captured);
    }
  }

  // -------------------------------------------------------------------------------------------
  // Test 5 — InfoReporter terminé après bestmove
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Test 5 : thread InfoReporter mort après émission bestmove")
  void testInfoReporterStopsAfterBestmove() throws Exception {
    try (Engine e = newEngine()) {
      var session = new UciSession(e, writerOver(new ByteArrayOutputStream()), false);

      e.startSearch(new GameState(), SearchBudget.nodes(30));
      session.startInfoReporter();

      awaitCondition(session::isBestmoveEmitted, DEFAULT_TIMEOUT_MS);
      // Attendre la terminaison effective du thread InfoReporter
      Thread t = session.infoReporterThreadForTest();
      assertThat(t).isNotNull();
      t.join(2_000);
      assertThat(t.isAlive()).as("InfoReporter terminé après bestmove").isFalse();
    }
  }

  // -------------------------------------------------------------------------------------------
  // Test 6 — pas d'émission posthume après bestmove
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Test 6 : pas d'émission après bestmove (info posthume bloqué)")
  void testInfoReporterNoEmissionAfterStop() throws Exception {
    try (Engine e = newEngine()) {
      var baos = new ByteArrayOutputStream();
      var session = new UciSession(e, writerOver(baos), false);
      AtomicBoolean stopFlag = new AtomicBoolean(false);

      e.startSearch(new GameState(), SearchBudget.untilStopped(stopFlag));
      session.startInfoReporter();
      Thread.sleep(800); // au moins 1 cycle info

      session.onUciStop();
      // Snapshot N1 immédiatement après le stop
      Thread.sleep(100);
      int n1 = baos.toString(StandardCharsets.UTF_8).length();

      // Attendre 1.5s sans nouvelle activité
      Thread.sleep(1_500);
      int n2 = baos.toString(StandardCharsets.UTF_8).length();

      assertThat(n2).as("aucune ligne supplémentaire après bestmove").isEqualTo(n1);
    }
  }

  // -------------------------------------------------------------------------------------------
  // Test 7 — stop sans recherche active (engine IDLE)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Test 7 : onUciStop sur engine IDLE → no-op silencieux, aucune ligne")
  void testStopWithoutActiveSearch() throws IOException {
    try (Engine e = newEngine()) {
      var baos = new ByteArrayOutputStream();
      var session = new UciSession(e, writerOver(baos), false);
      // Pas de startSearch ; engine.state() == IDLE
      assertThat(e.state()).isEqualTo(EngineState.IDLE);

      session.onUciStop(); // engine.stop() retourne null → emitBestmove no-op

      assertThat(baos.toString(StandardCharsets.UTF_8)).isEmpty();
      assertThat(session.isBestmoveEmitted()).isFalse();
    }
  }

  // -------------------------------------------------------------------------------------------
  // Test 8 — ponderhit immédiat après startPonder (transition rapide PONDERING → SEARCHING)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Test 8 : ponderhit immédiat après startPonder, bestmove unique")
  void testPonderhitImmediatelyAfterStartPonder() throws Exception {
    try (Engine e = newEngine()) {
      var baos = new ByteArrayOutputStream();
      var session = new UciSession(e, writerOver(baos), true /* isPonder */);

      GameState start = new GameState();
      int predicted = firstLegalMove(start);
      e.startPonder(start, predicted);
      session.startInfoReporter();
      // Pas de sleep : transition immédiate PONDERING → SEARCHING via ponderhit
      e.ponderhit(SearchBudget.nodes(30));

      // Attendre fin naturelle sur le nouveau budget
      awaitCondition(session::isBestmoveEmitted, DEFAULT_TIMEOUT_MS);
      Thread.sleep(200);

      String captured = baos.toString(StandardCharsets.UTF_8);
      assertExactlyOneBestmove(captured);
      assertAllLinesAreUciValid(captured);
    }
  }
}
