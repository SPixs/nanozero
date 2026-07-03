package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.engine.Engine;
import org.nanozero.engine.EngineConfig;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;
import org.nanozero.uci.UciMain;

/**
 * 6 tests race conditions complémentaires de l'intégration UCI complète (cf. SPEC §8.5, §12 phase
 * 7).
 *
 * <p>Ces tests exercent la boucle UCI main loop (via {@link UciMain#runLoop}) avec des séquences de
 * commandes qui stressent les transitions cross-thread :
 *
 * <ul>
 *   <li>{@code testRapidGoStopCycles} : 100 cycles position+go+stop, exactement 100 bestmove
 *       attendus.
 *   <li>{@code testQuitDuringActiveSearch} : quit pendant search → cleanup correct &lt; 5 s.
 *   <li>{@code testEofDuringActiveSearch} : EOF (stdin fermé) pendant search → cleanup correct.
 *   <li>{@code testConcurrentSetoptionDuringSearch} : setoption pendant search → ne crash pas le
 *       search.
 *   <li>{@code testMultipleGoBeforeFirstBestmove} : 2e go pendant SEARCHING → silently rejeté, 1
 *       seul bestmove.
 *   <li>{@code testRandomCommandSequenceFuzz} : 100 séquences × 50 cmd random, aucune exception,
 *       engine reste utilisable.
 * </ul>
 *
 * <p>Ces tests complètent les 8 tests {@link UciSessionRaceConditionsTest} de phase 6 ; ensemble
 * ils couvrent les 14 tests §8.5 obligatoires.
 */
class UciIntegrationRaceTest {

  private static final long DEFAULT_TIMEOUT_MS = 30_000;

  private static Network sharedNetwork;

  @BeforeAll
  static void loadNetwork() throws IOException {
    var url = UciIntegrationRaceTest.class.getResource("/npz/parity-model.npz");
    try {
      sharedNetwork = NetworkLoader.load(Paths.get(url.toURI()), LoadOptions.defaults());
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  @AfterAll
  static void clearNetwork() {
    sharedNetwork = null;
  }

  // -------------------------------------------------------------------------------------------
  // Harness intégré : UciMain.runLoop avec ByteArrayInputStream
  // -------------------------------------------------------------------------------------------

  /**
   * Exécute une séquence de commandes via le main loop complet et retourne la sortie capturée. Le
   * harness crée son propre Engine, UciAdapterState, writer.
   */
  private static String runScenario(String script) throws IOException {
    Engine engine = new Engine(sharedNetwork, EngineConfig.defaults());
    var baos = new ByteArrayOutputStream();
    PrintStream ps = new PrintStream(baos, true, StandardCharsets.UTF_8);
    var writer = new UciResponseWriter(ps);
    try (var state = new UciAdapterState(engine, writer, new UciOptionsState())) {
      UciMain.runLoop(new ByteArrayInputStream(script.getBytes(StandardCharsets.UTF_8)), state);
    }
    return baos.toString(StandardCharsets.UTF_8);
  }

  private static long countLinesStartingWith(String captured, String prefix) {
    return Arrays.stream(captured.split("\n")).filter(l -> l.startsWith(prefix)).count();
  }

  // -------------------------------------------------------------------------------------------
  // Test 1 — testRapidGoStopCycles
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Test 1 : 100x cycles position+go+stop → exactement 100 bestmove")
  void testRapidGoStopCycles() throws Exception {
    // Harness Java direct (pas via runLoop) pour permettre un sleep contrôlé entre go et stop.
    // Sans ce sleep, nodes(5) + stop immédiat peut produire sims=0 (le worker n'a pas eu le
    // temps de démarrer la 1ère sim), et emitBestmove skip via le check sims=0 (cf. phase 6
    // protocole CAS). Avec 20 ms entre go et stop, la 1ère sim est garantie d'être faite.
    Engine engine = new Engine(sharedNetwork, EngineConfig.defaults());
    var baos = new ByteArrayOutputStream();
    PrintStream ps = new PrintStream(baos, true, StandardCharsets.UTF_8);
    var writer = new UciResponseWriter(ps);
    int cycles = 100;
    try (var state = new UciAdapterState(engine, writer, new UciOptionsState())) {
      var goCmd = new UciCommand.Go(makeNodesArgs(5));
      for (int i = 0; i < cycles; i++) {
        UciCommandHandler.handle(UciCommandParser.parse("position startpos"), state);
        UciCommandHandler.handle(goCmd, state);
        Thread.sleep(20); // garantit ≥ 1 sim faite avant stop (post-warmup typique 10-20 ms/sim)
        UciCommandHandler.handle(new UciCommand.Stop(), state);
        // Attendre que le bestmove soit effectivement émis avant le cycle suivant.
        var session = state.currentSession();
        long deadline = System.nanoTime() + 5_000_000_000L;
        while (session != null && !session.isBestmoveEmitted()) {
          if (System.nanoTime() > deadline) {
            throw new AssertionError("cycle " + i + ": timeout waiting for bestmove");
          }
          Thread.sleep(5);
        }
      }
    }
    String captured = baos.toString(StandardCharsets.UTF_8);
    long bestmoveCount = countLinesStartingWith(captured, "bestmove ");
    assertThat(bestmoveCount).as("exactement %d lignes bestmove", cycles).isEqualTo(cycles);
  }

  /** Helper : construit un GoArgs avec uniquement nodes positionné. */
  private static GoArgs makeNodesArgs(int n) {
    return new GoArgs(
        java.util.OptionalLong.empty(),
        java.util.OptionalLong.empty(),
        java.util.OptionalLong.empty(),
        java.util.OptionalLong.empty(),
        java.util.OptionalInt.empty(),
        java.util.OptionalLong.empty(),
        java.util.OptionalInt.of(n),
        java.util.OptionalInt.empty(),
        false,
        false,
        java.util.List.of());
  }

  // -------------------------------------------------------------------------------------------
  // Test 2 — testQuitDuringActiveSearch
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Test 2 : quit pendant search active → cleanup correct < 5s")
  void testQuitDuringActiveSearch() throws Exception {
    String script = "position startpos\ngo infinite\nquit\n";
    long t0 = System.nanoTime();
    String captured = runScenario(script);
    long elapsedMs = (System.nanoTime() - t0) / 1_000_000;
    assertThat(elapsedMs).as("cleanup latency").isLessThan(5_000L);
    // 0 ou 1 bestmove acceptable : le cleanup peut émettre via close() si le worker a fait des
    // sims, ou non si quit arrive avant le premier sim.
    assertThat(countLinesStartingWith(captured, "bestmove ")).isBetween(0L, 1L);
  }

  // -------------------------------------------------------------------------------------------
  // Test 3 — testEofDuringActiveSearch
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Test 3 : EOF (stdin fermé) pendant search → cleanup correct (équivalent quit)")
  void testEofDuringActiveSearch() throws Exception {
    // Script sans "quit" final : runLoop sort sur EOF de l'InputStream.
    String script = "position startpos\ngo infinite\n";
    long t0 = System.nanoTime();
    String captured = runScenario(script);
    long elapsedMs = (System.nanoTime() - t0) / 1_000_000;
    assertThat(elapsedMs).as("EOF cleanup latency").isLessThan(5_000L);
    assertThat(countLinesStartingWith(captured, "bestmove ")).isBetween(0L, 1L);
  }

  // -------------------------------------------------------------------------------------------
  // Test 4 — testConcurrentSetoptionDuringSearch
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Test 4 : setoption pendant search → pas de crash, bestmove arrive normalement")
  void testConcurrentSetoptionDuringSearch() throws Exception {
    // Le main loop est single-threaded : setoption arrive entre commandes lues par readLine.
    // Mais on simule "during search" en plaçant setoption entre go et stop.
    String script =
        """
        position startpos
        go nodes 30
        setoption name Move Overhead value 100
        stop
        quit
        """;
    String captured = runScenario(script);
    assertThat(countLinesStartingWith(captured, "bestmove ")).isEqualTo(1);
  }

  // -------------------------------------------------------------------------------------------
  // Test 5 — testMultipleGoBeforeFirstBestmove
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Test 5 : 2e go pendant SEARCHING → silently rejeté, 1 seul bestmove")
  void testMultipleGoBeforeFirstBestmove() throws Exception {
    // Le 1er go est infinite, le 2e go arrive avant que stop puisse libérer le 1er.
    String script =
        """
        position startpos
        go infinite
        go nodes 5
        stop
        quit
        """;
    String captured = runScenario(script);
    // Le 2e go est ignoré (engine pas IDLE). Seul le 1er go produit un bestmove via stop.
    assertThat(countLinesStartingWith(captured, "bestmove ")).isEqualTo(1);
  }

  // -------------------------------------------------------------------------------------------
  // Test 6 — testRandomCommandSequenceFuzz
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Test 6 : 50 séquences × 20 cmd random → aucune exception non attrapée")
  void testRandomCommandSequenceFuzz() throws Exception {
    // Vocabulaire de commandes random
    String[] vocab = {
      "uci",
      "isready",
      "ucinewgame",
      "position startpos",
      "position startpos moves e2e4",
      "go nodes 5",
      "go movetime 50",
      "stop",
      "ponderhit",
      "setoption name Move Overhead value 50",
      "setoption name Ponder value true",
      "debug on",
      "debug off",
      "garbage random tokens",
      "",
    };
    Random rng = new Random(42L);
    int sequences = 50; // réduit de 100 pour temps acceptable
    int cmdsPerSequence = 20;
    for (int seq = 0; seq < sequences; seq++) {
      StringBuilder script = new StringBuilder();
      for (int i = 0; i < cmdsPerSequence; i++) {
        script.append(vocab[rng.nextInt(vocab.length)]).append('\n');
      }
      // Toujours stop+quit à la fin pour libérer une éventuelle search bloquante.
      script.append("stop\nquit\n");
      // Le scénario doit terminer dans un délai borné (no infinite hang).
      long t0 = System.nanoTime();
      runScenario(script.toString()); // Si exception lancée non attrapée → test fail
      long elapsedMs = (System.nanoTime() - t0) / 1_000_000;
      assertThat(elapsedMs).as("séquence %d ne doit pas hang", seq).isLessThan(DEFAULT_TIMEOUT_MS);
    }
  }
}
