package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.engine.Engine;
import org.nanozero.engine.EngineConfig;
import org.nanozero.engine.SearchBudget;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/** Tests unitaires de {@link UciAdapterState} (cf. SPEC §3.4, §12 phase 7). */
class UciAdapterStateTest {

  private static Network sharedNetwork;

  @BeforeAll
  static void loadNetwork() throws IOException {
    var url = UciAdapterStateTest.class.getResource("/npz/parity-model.npz");
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

  private static UciResponseWriter writerOver(ByteArrayOutputStream baos) {
    PrintStream ps = new PrintStream(baos, true, StandardCharsets.UTF_8);
    return new UciResponseWriter(ps);
  }

  private static Engine newEngine() {
    return new Engine(sharedNetwork, EngineConfig.defaults());
  }

  @Test
  @DisplayName("Constructor : args valides")
  void constructorValidArgs() {
    Engine e = newEngine();
    var state =
        new UciAdapterState(e, writerOver(new ByteArrayOutputStream()), new UciOptionsState());
    assertThat(state.engine()).isSameAs(e);
    assertThat(state.options()).isNotNull();
    assertThat(state.debugMode()).isFalse();
    assertThat(state.currentSession()).isNull();
    assertThat(state.lastPosition()).isNull();
    assertThat(state.lastPlayedMoves()).isEmpty();
    state.close();
  }

  @Test
  @DisplayName("Constructor null engine → NPE")
  void constructorNullEngineThrowsNpe() {
    assertThatThrownBy(
            () ->
                new UciAdapterState(
                    (Engine) null, writerOver(new ByteArrayOutputStream()), new UciOptionsState()))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("engine");
  }

  @Test
  @DisplayName("Constructor null writer → NPE")
  void constructorNullWriterThrowsNpe() {
    try (Engine e = newEngine()) {
      assertThatThrownBy(() -> new UciAdapterState(e, null, new UciOptionsState()))
          .isInstanceOf(NullPointerException.class)
          .hasMessageContaining("writer");
    }
  }

  @Test
  @DisplayName("Constructor null options → NPE")
  void constructorNullOptionsThrowsNpe() {
    try (Engine e = newEngine()) {
      assertThatThrownBy(
              () -> new UciAdapterState(e, writerOver(new ByteArrayOutputStream()), null))
          .isInstanceOf(NullPointerException.class)
          .hasMessageContaining("options");
    }
  }

  @Test
  @DisplayName("setLastPosition + getter cohérents, défensif sur null playedMoves")
  void setLastPosition() {
    Engine e = newEngine();
    var state =
        new UciAdapterState(e, writerOver(new ByteArrayOutputStream()), new UciOptionsState());
    GameState pos = new GameState();
    int[] moves = {1, 2, 3};
    state.setLastPosition(pos, moves);
    assertThat(state.lastPosition()).isSameAs(pos);
    assertThat(state.lastPlayedMoves()).containsExactly(1, 2, 3);

    // null moves → empty array
    state.setLastPosition(pos, null);
    assertThat(state.lastPlayedMoves()).isEmpty();
    state.close();
  }

  @Test
  @DisplayName("setDebugMode toggle")
  void setDebugMode() {
    Engine e = newEngine();
    var state =
        new UciAdapterState(e, writerOver(new ByteArrayOutputStream()), new UciOptionsState());
    assertThat(state.debugMode()).isFalse();
    state.setDebugMode(true);
    assertThat(state.debugMode()).isTrue();
    state.setDebugMode(false);
    assertThat(state.debugMode()).isFalse();
    state.close();
  }

  @Test
  @DisplayName("close idempotent")
  void closeIdempotent() {
    Engine e = newEngine();
    var state =
        new UciAdapterState(e, writerOver(new ByteArrayOutputStream()), new UciOptionsState());
    state.close();
    state.close(); // 2e appel : no-op via Engine.close idempotent
  }

  @Test
  @DisplayName("close arrête une session active (bestmove émis)")
  void closeStopsActiveSession() throws Exception {
    Engine e = newEngine();
    var baos = new ByteArrayOutputStream();
    var state = new UciAdapterState(e, writerOver(baos), new UciOptionsState());

    AtomicBoolean stop = new AtomicBoolean(false);
    e.startSearch(new GameState(), SearchBudget.untilStopped(stop));
    var session = new UciSession(e, state.writer(), false, GoArgs.empty());
    session.startInfoReporter();
    state.setCurrentSession(session);

    // Attendre un peu pour que des sims soient effectivement faites avant close.
    Thread.sleep(200);

    state.close();
    assertThat(session.isBestmoveEmitted()).isTrue();
  }
}
