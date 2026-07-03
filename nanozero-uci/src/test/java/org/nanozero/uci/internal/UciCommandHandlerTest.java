package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.Arrays;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.engine.Engine;
import org.nanozero.engine.EngineConfig;
import org.nanozero.engine.EngineState;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/** Tests unitaires de {@link UciCommandHandler} (cf. SPEC §5.1, §5.7, §12 phase 7). */
class UciCommandHandlerTest {

  private static final long DEFAULT_TIMEOUT_MS = 15_000;
  private static Network sharedNetwork;

  @BeforeAll
  static void loadNetwork() throws IOException {
    var url = UciCommandHandlerTest.class.getResource("/npz/parity-model.npz");
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
  // Test harness
  // -------------------------------------------------------------------------------------------

  /** Bundle ByteArrayOutputStream + UciAdapterState pour les tests. */
  private static final class Harness implements AutoCloseable {
    final ByteArrayOutputStream baos;
    final UciAdapterState state;
    final Engine engine;

    Harness() {
      this.baos = new ByteArrayOutputStream();
      PrintStream ps = new PrintStream(baos, true, StandardCharsets.UTF_8);
      this.engine = new Engine(sharedNetwork, EngineConfig.defaults());
      this.state = new UciAdapterState(engine, new UciResponseWriter(ps), new UciOptionsState());
    }

    String captured() {
      return baos.toString(StandardCharsets.UTF_8);
    }

    @Override
    public void close() {
      state.close();
    }
  }

  private static void awaitCondition(java.util.function.BooleanSupplier cond, long timeoutMs)
      throws InterruptedException {
    long deadline = System.nanoTime() + timeoutMs * 1_000_000L;
    while (!cond.getAsBoolean()) {
      if (System.nanoTime() > deadline) {
        throw new AssertionError("Timeout after " + timeoutMs + "ms");
      }
      Thread.sleep(10);
    }
  }

  // -------------------------------------------------------------------------------------------
  // uci handshake
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("handle Uci : émet id name + id author + options + uciok")
  void handleUciEmitsHandshake() {
    try (var h = new Harness()) {
      UciCommandHandler.HandleAction action =
          UciCommandHandler.handle(new UciCommand.Uci(), h.state);
      assertThat(action).isEqualTo(UciCommandHandler.HandleAction.CONTINUE);
      String[] lines = h.captured().split("\n");
      assertThat(lines[0]).startsWith("id name NanoZero");
      assertThat(lines[1]).startsWith("id author ");
      assertThat(Arrays.asList(lines)).anyMatch(l -> l.startsWith("option name Ponder"));
      assertThat(Arrays.asList(lines)).anyMatch(l -> l.startsWith("option name Move Overhead"));
      assertThat(lines[lines.length - 1]).isEqualTo("uciok");
    }
  }

  @Test
  @DisplayName("handle IsReady : émet readyok")
  void handleIsReadyEmitsReadyOk() {
    try (var h = new Harness()) {
      var action = UciCommandHandler.handle(new UciCommand.IsReady(), h.state);
      assertThat(action).isEqualTo(UciCommandHandler.HandleAction.CONTINUE);
      assertThat(h.captured()).isEqualTo("readyok\n");
    }
  }

  @Test
  @DisplayName("handle UciNewGame : no-op, aucune ligne émise")
  void handleUciNewGameIsNoop() {
    try (var h = new Harness()) {
      var action = UciCommandHandler.handle(new UciCommand.UciNewGame(), h.state);
      assertThat(action).isEqualTo(UciCommandHandler.HandleAction.CONTINUE);
      assertThat(h.captured()).isEmpty();
    }
  }

  // -------------------------------------------------------------------------------------------
  // position : stocke pour go suivant
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("handle Position : stocke lastPosition + lastPlayedMoves")
  void handlePositionStores() {
    try (var h = new Harness()) {
      GameState pos = new GameState();
      int[] moves = {1, 2, 3};
      var cmd = new UciCommand.Position(pos, moves);
      UciCommandHandler.handle(cmd, h.state);
      assertThat(h.state.lastPosition()).isSameAs(pos);
      assertThat(h.state.lastPlayedMoves()).containsExactly(1, 2, 3);
      assertThat(h.captured()).isEmpty();
    }
  }

  // -------------------------------------------------------------------------------------------
  // go : démarre une recherche
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("handle Go avec position préalable : démarre search et émet bestmove final")
  void handleGoStartsSearch() throws Exception {
    try (var h = new Harness()) {
      h.state.setLastPosition(new GameState(), new int[0]);
      var goArgs =
          new GoArgs(
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.of(20),
              java.util.OptionalInt.empty(),
              false,
              false,
              java.util.List.of());
      UciCommandHandler.handle(new UciCommand.Go(goArgs), h.state);
      assertThat(h.state.currentSession()).isNotNull();
      // Attendre fin naturelle (nodes=20, ~400ms) + emit bestmove
      awaitCondition(() -> h.state.currentSession().isBestmoveEmitted(), DEFAULT_TIMEOUT_MS);
      Thread.sleep(100);
      assertThat(h.captured()).contains("bestmove ");
    }
  }

  @Test
  @DisplayName("handle Go sans position préalable : fallback startpos, debug si activé")
  void handleGoFallbackStartpos() throws Exception {
    try (var h = new Harness()) {
      h.state.setDebugMode(true);
      var goArgs =
          new GoArgs(
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.of(20),
              java.util.OptionalInt.empty(),
              false,
              false,
              java.util.List.of());
      UciCommandHandler.handle(new UciCommand.Go(goArgs), h.state);
      assertThat(h.state.currentSession()).isNotNull();
      awaitCondition(() -> h.state.currentSession().isBestmoveEmitted(), DEFAULT_TIMEOUT_MS);
    }
  }

  @Test
  @DisplayName("handle Go pendant SEARCHING : silently ignored (tolérance)")
  void handleGoWhileSearchingIsIgnored() throws Exception {
    try (var h = new Harness()) {
      h.state.setLastPosition(new GameState(), new int[0]);
      var goInfinite =
          new GoArgs(
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalInt.empty(),
              true,
              false,
              java.util.List.of());
      UciCommandHandler.handle(new UciCommand.Go(goInfinite), h.state);
      var firstSession = h.state.currentSession();
      assertThat(firstSession).isNotNull();
      // 2e go pendant SEARCHING
      var goNodes =
          new GoArgs(
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.of(1),
              java.util.OptionalInt.empty(),
              false,
              false,
              java.util.List.of());
      UciCommandHandler.handle(new UciCommand.Go(goNodes), h.state);
      // Session inchangée : le 2e go a été ignoré
      assertThat(h.state.currentSession()).isSameAs(firstSession);
      // Cleanup
      UciCommandHandler.handle(new UciCommand.Stop(), h.state);
      awaitCondition(firstSession::isBestmoveEmitted, DEFAULT_TIMEOUT_MS);
    }
  }

  // -------------------------------------------------------------------------------------------
  // stop
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("handle Stop avec session active : émet bestmove")
  void handleStopEmitsBestmove() throws Exception {
    try (var h = new Harness()) {
      h.state.setLastPosition(new GameState(), new int[0]);
      var goInfinite =
          new GoArgs(
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalInt.empty(),
              true,
              false,
              java.util.List.of());
      UciCommandHandler.handle(new UciCommand.Go(goInfinite), h.state);
      Thread.sleep(200);
      UciCommandHandler.handle(new UciCommand.Stop(), h.state);
      // Émission bestmove asynchrone (control thread) — attendre comme les autres
      // tests stop/emit (cf. lignes 165, 191, 235...) au lieu d'asserter immédiatement.
      // Sinon flaky sous charge (build reactor complet) : le control thread n'a pas
      // encore émis quand on vérifie le flag.
      awaitCondition(() -> h.state.currentSession().isBestmoveEmitted(), DEFAULT_TIMEOUT_MS);
      assertThat(h.state.currentSession().isBestmoveEmitted()).isTrue();
      assertThat(h.captured()).contains("bestmove ");
    }
  }

  @Test
  @DisplayName("handle Stop sans session active : no-op, aucune ligne, aucune exception")
  void handleStopWithoutSessionIsNoop() {
    try (var h = new Harness()) {
      var action = UciCommandHandler.handle(new UciCommand.Stop(), h.state);
      assertThat(action).isEqualTo(UciCommandHandler.HandleAction.CONTINUE);
      assertThat(h.captured()).isEmpty();
    }
  }

  // -------------------------------------------------------------------------------------------
  // setoption
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("handle SetOption Move Overhead : met à jour state.options()")
  void handleSetOptionUpdatesState() {
    try (var h = new Harness()) {
      UciCommandHandler.handle(new UciCommand.SetOption("Move Overhead", "100"), h.state);
      assertThat(h.state.options().moveOverheadMs()).isEqualTo(100);
    }
  }

  @Test
  @DisplayName("handle SetOption Ponder : met à jour state.options()")
  void handleSetOptionPonder() {
    try (var h = new Harness()) {
      UciCommandHandler.handle(new UciCommand.SetOption("Ponder", "false"), h.state);
      assertThat(h.state.options().ponder()).isFalse();
    }
  }

  // -------------------------------------------------------------------------------------------
  // debug
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("handle Debug on : flip debugMode")
  void handleDebugOnFlipsState() {
    try (var h = new Harness()) {
      UciCommandHandler.handle(new UciCommand.Debug(true), h.state);
      assertThat(h.state.debugMode()).isTrue();
      UciCommandHandler.handle(new UciCommand.Debug(false), h.state);
      assertThat(h.state.debugMode()).isFalse();
    }
  }

  // -------------------------------------------------------------------------------------------
  // unknown
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("handle Unknown : no-op, aucune émission stdout, aucune exception")
  void handleUnknownIsNoop() {
    try (var h = new Harness()) {
      var action = UciCommandHandler.handle(new UciCommand.Unknown("garbage"), h.state);
      assertThat(action).isEqualTo(UciCommandHandler.HandleAction.CONTINUE);
      assertThat(h.captured()).isEmpty();
    }
  }

  // -------------------------------------------------------------------------------------------
  // quit
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("handle Quit : retourne HandleAction.QUIT")
  void handleQuitReturnsQuit() {
    try (var h = new Harness()) {
      var action = UciCommandHandler.handle(new UciCommand.Quit(), h.state);
      assertThat(action).isEqualTo(UciCommandHandler.HandleAction.QUIT);
      assertThat(h.captured()).isEmpty();
    }
  }

  // -------------------------------------------------------------------------------------------
  // Engine state observable après handle Go
  // -------------------------------------------------------------------------------------------

  // -------------------------------------------------------------------------------------------
  // ponderhit + go ponder (couvre branches handlePonderHit + startPonderSearch fallback)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("handle PonderHit sans session active : no-op silencieux")
  void handlePonderHitWithoutSessionIsNoop() {
    try (var h = new Harness()) {
      var action = UciCommandHandler.handle(new UciCommand.PonderHit(), h.state);
      assertThat(action).isEqualTo(UciCommandHandler.HandleAction.CONTINUE);
      assertThat(h.captured()).isEmpty();
    }
  }

  @Test
  @DisplayName("handle PonderHit sur session non-ponder : no-op silencieux")
  void handlePonderHitOnNonPonderSession() throws Exception {
    try (var h = new Harness()) {
      h.state.setLastPosition(new GameState(), new int[0]);
      var goArgs =
          new GoArgs(
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalInt.empty(),
              true,
              false,
              java.util.List.of());
      UciCommandHandler.handle(new UciCommand.Go(goArgs), h.state);
      // Session non-ponder créée (infinite). ponderhit doit être ignored.
      UciCommandHandler.handle(new UciCommand.PonderHit(), h.state);
      // Cleanup
      UciCommandHandler.handle(new UciCommand.Stop(), h.state);
      awaitCondition(() -> h.state.currentSession().isBestmoveEmitted(), DEFAULT_TIMEOUT_MS);
    }
  }

  @Test
  @DisplayName("handle Go ponder avec Ponder option = false : no-op silencieux (SPEC §6.3)")
  void handleGoPonderWithPonderOptionFalseIsNoop() {
    try (var h = new Harness()) {
      UciCommandHandler.handle(new UciCommand.SetOption("Ponder", "false"), h.state);
      assertThat(h.state.options().ponder()).isFalse();
      h.state.setLastPosition(new GameState(), new int[0]);
      var goPonder =
          new GoArgs(
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalInt.empty(),
              false,
              true, /* ponder */
              java.util.List.of());
      var action = UciCommandHandler.handle(new UciCommand.Go(goPonder), h.state);
      assertThat(action).isEqualTo(UciCommandHandler.HandleAction.CONTINUE);
      assertThat(h.state.engine().state()).isEqualTo(EngineState.IDLE);
      assertThat(h.state.currentSession()).isNull();
    }
  }

  @Test
  @DisplayName("handle Go ponder sans moves préalables : fallback startSearch+UNLIMITED")
  void handleGoPonderWithoutMovesFallback() throws Exception {
    try (var h = new Harness()) {
      h.state.setLastPosition(new GameState(), new int[0]); // playedMoves vide
      h.state.setDebugMode(true);
      var goPonder =
          new GoArgs(
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalInt.empty(),
              false,
              true, /* ponder */
              java.util.List.of());
      UciCommandHandler.handle(new UciCommand.Go(goPonder), h.state);
      // Session ponder créée mais en fallback (startSearch UNLIMITED, pas startPonder).
      assertThat(h.state.currentSession()).isNotNull();
      assertThat(h.state.currentSession().isPonder()).isTrue();
      // Cleanup : stop pour libérer
      Thread.sleep(50);
      UciCommandHandler.handle(new UciCommand.Stop(), h.state);
      awaitCondition(() -> h.state.currentSession().isBestmoveEmitted(), DEFAULT_TIMEOUT_MS);
    }
  }

  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Après handle Go : engine.state != IDLE")
  void engineStateNotIdleAfterGo() throws Exception {
    try (var h = new Harness()) {
      h.state.setLastPosition(new GameState(), new int[0]);
      var goInfinite =
          new GoArgs(
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalLong.empty(),
              java.util.OptionalInt.empty(),
              java.util.OptionalInt.empty(),
              true,
              false,
              java.util.List.of());
      UciCommandHandler.handle(new UciCommand.Go(goInfinite), h.state);
      // Attendre la transition vers SEARCHING
      awaitCondition(() -> h.engine.state() == EngineState.SEARCHING, 1000);
      UciCommandHandler.handle(new UciCommand.Stop(), h.state);
      awaitCondition(() -> h.state.currentSession().isBestmoveEmitted(), DEFAULT_TIMEOUT_MS);
    }
  }
}
