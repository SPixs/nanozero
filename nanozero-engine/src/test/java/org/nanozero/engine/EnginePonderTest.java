package org.nanozero.engine;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.board.MoveGen;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/**
 * Tests des fonctionnalités ponder + tree reuse de l'{@link Engine} (cf. SPEC §4.2, §4.3, §5.5, §12
 * phase 11).
 *
 * <p>Couvre :
 *
 * <ul>
 *   <li>Transitions IDLE → PONDERING via {@code startPonder} + validation entrée.
 *   <li>Transitions PONDERING → SEARCHING via {@code ponderhit} (avec rétention de l'arbre).
 *   <li>Workflow UCI complet (search + ponder + ponderhit + stop).
 *   <li>Ponder miss (stop + startSearch sur position différente).
 *   <li>Tree reuse sur startSearch successifs.
 *   <li>{@code close} pendant ponder.
 *   <li>Rejets {@code IllegalStateException} sur transitions invalides.
 * </ul>
 */
class EnginePonderTest {

  private static Path fixturePath() {
    var url = EnginePonderTest.class.getResource("/npz/parity-model.npz");
    try {
      return Paths.get(url.toURI());
    } catch (Exception e) {
      throw new AssertionError("Impossible de résoudre /npz/parity-model.npz", e);
    }
  }

  private static Network loadParityNetwork() throws IOException {
    return NetworkLoader.load(fixturePath(), LoadOptions.defaults());
  }

  /** Attend, sans dépasser timeoutMs, que {@code state()} atteigne {@code expected}. */
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

  /** Attend, sans dépasser timeoutMs, que {@code currentBest().simulationsCount} dépasse min. */
  private static void awaitSimulations(Engine engine, int min, long timeoutMs)
      throws InterruptedException {
    long deadline = System.nanoTime() + timeoutMs * 1_000_000L;
    while (engine.currentBest().simulationsCount() < min) {
      if (System.nanoTime() > deadline) {
        throw new AssertionError(
            "Timeout waiting for "
                + min
                + " simulations, got "
                + engine.currentBest().simulationsCount());
      }
      Thread.sleep(20);
    }
  }

  /** Premier coup légal depuis {@code position} (pour utilisation comme predictedOpponentMove). */
  private static int firstLegalMove(GameState position) {
    int[] buf = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = position.generateMoves(buf, 0);
    if (n == 0) {
      throw new AssertionError("Position has no legal moves");
    }
    return buf[0];
  }

  // -------------------------------------------------------------------------------------------
  // startPonder : transitions et validation
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("startPonder transitionne IDLE → PONDERING")
  void startPonderTransitionsToPondering() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
      GameState start = new GameState();
      int predicted = firstLegalMove(start);

      e.startPonder(start, predicted);
      awaitState(e, EngineState.PONDERING, 1000);

      // Cleanup : stop pour quitter PONDERING (UNLIMITED budget ne s'arrête pas seul).
      e.stop();
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
    }
  }

  @Test
  @DisplayName("startPonder avec coup illégal → IllegalArgumentException, état reste IDLE")
  void startPonderInvalidMoveRejectsImmediately() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      GameState start = new GameState();
      int illegalMove = 0xFFFF; // encodage non valide

      assertThatThrownBy(() -> e.startPonder(start, illegalMove))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("not a legal move");

      assertThat(e.state()).isEqualTo(EngineState.IDLE);
    }
  }

  @Test
  @DisplayName("startPonder avec position null → NPE")
  void startPonderRejectsNullPosition() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      assertThatThrownBy(() -> e.startPonder(null, 0))
          .isInstanceOf(NullPointerException.class)
          .hasMessageContaining("position");
    }
  }

  @Test
  @DisplayName("startPonder pendant SEARCHING → IllegalStateException")
  void startPonderFromSearchingThrows() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      AtomicBoolean stop = new AtomicBoolean(false);
      e.startSearch(new GameState(), SearchBudget.untilStopped(stop));
      awaitState(e, EngineState.SEARCHING, 1000);

      GameState pos = new GameState();
      int predicted = firstLegalMove(pos);
      assertThatThrownBy(() -> e.startPonder(pos, predicted))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("IDLE");

      stop.set(true);
      e.stop();
    }
  }

  @Test
  @DisplayName("startPonder pendant PONDERING → IllegalStateException")
  void startPonderFromPonderingThrows() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      GameState start = new GameState();
      int predicted = firstLegalMove(start);
      e.startPonder(start, predicted);
      awaitState(e, EngineState.PONDERING, 1000);

      assertThatThrownBy(() -> e.startPonder(start, predicted))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("PONDERING");

      e.stop();
    }
  }

  // -------------------------------------------------------------------------------------------
  // ponderhit : transitions, rétention de l'arbre, tree reuse
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("ponderhit conserve les sims du ponder + ajoute le budget réel")
  void ponderhitContinuesSearch() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      GameState start = new GameState();
      int predicted = firstLegalMove(start);

      e.startPonder(start, predicted);
      awaitState(e, EngineState.PONDERING, 1000);

      // Attente que le ponder ait fait au moins quelques sims (tolérance large pour CI lent).
      awaitSimulations(e, 1, 10_000);
      SearchResult ponderSnap = e.currentBest();
      int simsPondered = ponderSnap.simulationsCount();
      assertThat(simsPondered).isPositive();

      // ponderhit avec budget cumulatif : nodes(simsPondered + 50) → 50 sims supplémentaires.
      e.ponderhit(SearchBudget.nodes(simsPondered + 50));
      awaitState(e, EngineState.DONE, 30_000);

      SearchResult finalResult = e.stop();
      // Les sims du ponder sont conservées + 50 sims supplémentaires.
      assertThat(finalResult.simulationsCount()).isGreaterThan(simsPondered);
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
    }
  }

  @Test
  @DisplayName("ponderhit depuis IDLE → IllegalStateException")
  void ponderhitFromIdleThrows() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
      assertThatThrownBy(() -> e.ponderhit(SearchBudget.nodes(10)))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("PONDERING");
    }
  }

  @Test
  @DisplayName("ponderhit depuis SEARCHING → IllegalStateException")
  void ponderhitFromSearchingThrows() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      AtomicBoolean stop = new AtomicBoolean(false);
      e.startSearch(new GameState(), SearchBudget.untilStopped(stop));
      awaitState(e, EngineState.SEARCHING, 1000);

      assertThatThrownBy(() -> e.ponderhit(SearchBudget.nodes(10)))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("PONDERING");

      stop.set(true);
      e.stop();
    }
  }

  @Test
  @DisplayName("ponderhit avec realBudget null → NPE")
  void ponderhitRejectsNullBudget() throws IOException {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      assertThatThrownBy(() -> e.ponderhit(null))
          .isInstanceOf(NullPointerException.class)
          .hasMessageContaining("realBudget");
    }
  }

  // -------------------------------------------------------------------------------------------
  // Ponder miss : stop + startSearch sur position différente
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Ponder miss : stop puis startSearch sur position différente fonctionne")
  void ponderMissThenFreshSearch() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      GameState start = new GameState();
      int predicted = firstLegalMove(start);

      e.startPonder(start, predicted);
      awaitState(e, EngineState.PONDERING, 1000);
      awaitSimulations(e, 1, 10_000);

      // Adversaire a joué autre chose : on jette le ponder.
      e.stop();
      assertThat(e.state()).isEqualTo(EngineState.IDLE);

      // Position complètement différente (FEN : KP vs K endgame).
      GameState unrelated = new GameState("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
      SearchResult result = e.searchSync(unrelated, SearchBudget.nodes(20));
      assertThat(result.simulationsCount()).isEqualTo(20);
      assertThat(result.bestMove()).isNotZero();
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
    }
  }

  // -------------------------------------------------------------------------------------------
  // Tree reuse sur startSearch successifs
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("startSearch successifs sur positions liées : pas de crash, résultats valides")
  void startSearchTreeReuseAfterPlayedMoves() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      GameState start = new GameState();
      SearchResult r1 = e.searchSync(start, SearchBudget.nodes(20));
      assertThat(r1.simulationsCount()).isEqualTo(20);

      // Jouer le coup choisi par l'engine côté caller, puis chercher la position résultante.
      // Le tryReroot interne devrait promouvoir le sous-arbre correspondant.
      GameState afterEngineMove = new GameState(start.toFen());
      afterEngineMove.applyMove(r1.bestMove());

      SearchResult r2 = e.searchSync(afterEngineMove, SearchBudget.nodes(20));
      assertThat(r2.simulationsCount()).isEqualTo(20);
      assertThat(r2.bestMove()).isNotZero();
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
    }
  }

  @Test
  @DisplayName("startSearch sur position non-liée : fresh tree, pas de crash")
  void startSearchFreshTreeOnUnreachablePosition() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      GameState start = new GameState();
      e.searchSync(start, SearchBudget.nodes(20));

      // Position complètement différente, non atteignable depuis start en 0/1/2 plies.
      GameState unrelated = new GameState("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1");
      SearchResult r = e.searchSync(unrelated, SearchBudget.nodes(20));
      assertThat(r.simulationsCount()).isEqualTo(20);
      assertThat(r.bestMove()).isNotZero();
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
    }
  }

  // -------------------------------------------------------------------------------------------
  // Workflow UCI complet
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Workflow UCI complet : search → startPonder → ponderhit → stop")
  void fullPonderCycleUciLikeWorkflow() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      GameState start = new GameState();

      // 1. Engine cherche son coup.
      SearchResult r1 = e.searchSync(start, SearchBudget.nodes(20));
      int engineMove = r1.bestMove();
      assertThat(engineMove).isNotZero();

      // 2. Le caller applique le coup engine et prédit la réponse adverse.
      GameState afterEngineMove = new GameState(start.toFen());
      afterEngineMove.applyMove(engineMove);
      int predictedOpp = firstLegalMove(afterEngineMove);

      // 3. Engine ponder en supposant que l'adversaire jouera predictedOpp.
      e.startPonder(afterEngineMove, predictedOpp);
      awaitState(e, EngineState.PONDERING, 1000);
      awaitSimulations(e, 1, 10_000);

      // 4. L'adversaire joue effectivement predictedOpp → ponderhit.
      e.ponderhit(SearchBudget.nodes(50));
      awaitState(e, EngineState.DONE, 30_000);

      // 5. Récupère le résultat — la racine du tree pondéré + ponderhit est exactement la
      //    position à chercher.
      SearchResult r2 = e.stop();
      assertThat(r2.simulationsCount()).isPositive();
      assertThat(r2.bestMove()).isNotZero();
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
    }
  }

  // -------------------------------------------------------------------------------------------
  // State transitions : 2 cycles complets PONDERING
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Transitions complètes : IDLE → PONDERING → SEARCHING → DONE → IDLE → PONDERING")
  void stateTransitionsAcrossPonderCycles() throws Exception {
    try (Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults())) {
      GameState start = new GameState();
      int predicted = firstLegalMove(start);

      // Cycle 1 : ponder + ponderhit
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
      e.startPonder(start, predicted);
      awaitState(e, EngineState.PONDERING, 1000);
      awaitSimulations(e, 1, 10_000);

      e.ponderhit(SearchBudget.nodes(50));
      awaitState(e, EngineState.DONE, 30_000);
      e.stop();
      assertThat(e.state()).isEqualTo(EngineState.IDLE);

      // Cycle 2 : ponder + stop direct (sans ponderhit)
      e.startPonder(start, predicted);
      awaitState(e, EngineState.PONDERING, 1000);
      e.stop();
      assertThat(e.state()).isEqualTo(EngineState.IDLE);
    }
  }

  // -------------------------------------------------------------------------------------------
  // close pendant ponder
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("close pendant PONDERING transitionne CLOSED en < 1 s")
  void closeDuringPondering() throws Exception {
    Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults());
    GameState start = new GameState();
    int predicted = firstLegalMove(start);

    e.startPonder(start, predicted);
    awaitState(e, EngineState.PONDERING, 1000);

    long t0 = System.nanoTime();
    e.close();
    long elapsedMs = TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - t0);

    assertThat(elapsedMs).as("close latency").isLessThan(1000L);
    assertThat(e.state()).isEqualTo(EngineState.CLOSED);
  }

  @Test
  @DisplayName("startPonder sur engine CLOSED → IllegalStateException")
  void startPonderOnClosedRejected() throws IOException {
    Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults());
    e.close();
    GameState start = new GameState();
    int predicted = firstLegalMove(start);
    assertThatThrownBy(() -> e.startPonder(start, predicted))
        .isInstanceOf(IllegalStateException.class);
  }

  @Test
  @DisplayName("ponderhit sur engine CLOSED → IllegalStateException")
  void ponderhitOnClosedRejected() throws IOException {
    Engine e = new Engine(loadParityNetwork(), EngineConfig.defaults());
    e.close();
    assertThatThrownBy(() -> e.ponderhit(SearchBudget.nodes(10)))
        .isInstanceOf(IllegalStateException.class);
  }
}
