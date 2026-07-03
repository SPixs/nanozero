package org.nanozero.worker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;
import java.time.Duration;
import org.junit.jupiter.api.Test;

class WorkerMainTest {

  @Test
  void parsesMinimalRequiredArgs() {
    WorkerConfig cfg = WorkerMain.parseArgs(new String[] {"--server", "http://devsrv:8090"});
    assertThat(cfg.serverUrl()).isEqualTo("http://devsrv:8090");
    assertThat(cfg.apiKey()).isEmpty();
    assertThat(cfg.modelsDir()).isEqualTo(Path.of("./model-cache"));
    assertThat(cfg.requestTimeout()).isEqualTo(Duration.ofSeconds(30));
    assertThat(cfg.pollIdleSleep()).isEqualTo(Duration.ofSeconds(5));
  }

  @Test
  void parsesAllFlags() {
    String[] args = {
      "--server",
      "http://w3090:8090",
      "--key",
      "secret",
      "--worker-id",
      "w3090-gpu-0",
      "--models-dir",
      "/var/cache/nanozero",
      "--request-timeout-seconds",
      "60",
      "--poll-idle-seconds",
      "10",
      "--cuda",
      "--concurrent-games",
      "128",
      "--flush-micros",
      "6000",
      "--min-batch",
      "120",
      "--nncache",
      "131072",
    };
    WorkerConfig cfg = WorkerMain.parseArgs(args);

    assertThat(cfg.serverUrl()).isEqualTo("http://w3090:8090");
    assertThat(cfg.apiKey()).isEqualTo("secret");
    assertThat(cfg.workerId()).isEqualTo("w3090-gpu-0");
    assertThat(cfg.modelsDir()).isEqualTo(Path.of("/var/cache/nanozero"));
    assertThat(cfg.requestTimeout()).isEqualTo(Duration.ofSeconds(60));
    assertThat(cfg.pollIdleSleep()).isEqualTo(Duration.ofSeconds(10));
    assertThat(cfg.cuda()).isTrue();
    assertThat(cfg.concurrentGames()).isEqualTo(128);
    assertThat(cfg.flushMicros()).isEqualTo(6000);
    assertThat(cfg.minBatch()).isEqualTo(120);
    assertThat(cfg.nnCacheSize()).isEqualTo(131072);
  }

  @Test
  void cudaAndConcurrentGamesDefaultToCpuSingleGame() {
    WorkerConfig cfg = WorkerMain.parseArgs(new String[] {"--server", "http://devsrv:8090"});
    assertThat(cfg.cuda()).isFalse();
    assertThat(cfg.concurrentGames()).isEqualTo(1);
    assertThat(cfg.flushMicros()).isEqualTo(org.nanozero.nn.BatchingNetwork.DEFAULT_FLUSH_MICROS);
  }

  @Test
  void rejectsNonPositiveConcurrentGames() {
    assertThatThrownBy(
            () ->
                WorkerMain.parseArgs(
                    new String[] {"--server", "http://b:1", "--concurrent-games", "0"}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void rejectsNegativeFlushMicros() {
    assertThatThrownBy(
            () ->
                WorkerMain.parseArgs(
                    new String[] {"--server", "http://b:1", "--flush-micros", "-1"}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void missingServerFlagRaises() {
    assertThatThrownBy(() -> WorkerMain.parseArgs(new String[] {"--key", "x"}))
        .isInstanceOf(WorkerMain.CliError.class)
        .hasMessageContaining("--server");
  }

  @Test
  void flagWithoutValueRaises() {
    assertThatThrownBy(() -> WorkerMain.parseArgs(new String[] {"--server"}))
        .isInstanceOf(WorkerMain.CliError.class)
        .hasMessageContaining("--server expects a value");
  }

  @Test
  void unknownFlagRaises() {
    assertThatThrownBy(
            () -> WorkerMain.parseArgs(new String[] {"--server", "http://x", "--bogus", "1"}))
        .isInstanceOf(WorkerMain.CliError.class)
        .hasMessageContaining("Unknown flag");
  }

  @Test
  void nonIntegerNumericRaises() {
    assertThatThrownBy(
            () ->
                WorkerMain.parseArgs(
                    new String[] {"--server", "http://x", "--poll-idle-seconds", "not-a-number"}))
        .isInstanceOf(WorkerMain.CliError.class)
        .hasMessageContaining("Expected integer");
  }

  @Test
  void workerIdDefaultsToHostnameOrAnonymous() {
    WorkerConfig cfg = WorkerMain.parseArgs(new String[] {"--server", "http://x"});
    // Either $HOSTNAME from env, or fallback "anonymous-worker" — both valid.
    assertThat(cfg.workerId()).isNotBlank();
  }

  @Test
  void parsesRequestTimeoutFlag() {
    WorkerConfig cfg =
        WorkerMain.parseArgs(
            new String[] {"--server", "http://x", "--request-timeout-seconds", "45"});
    assertThat(cfg.requestTimeout()).isEqualTo(Duration.ofSeconds(45));
  }

  @Test
  void parsesModelsDirAndKeyFlags() {
    WorkerConfig cfg =
        WorkerMain.parseArgs(
            new String[] {
              "--server", "http://x", "--key", "topsecret", "--models-dir", "/data/cache"
            });
    assertThat(cfg.apiKey()).isEqualTo("topsecret");
    assertThat(cfg.modelsDir()).isEqualTo(Path.of("/data/cache"));
  }

  @Test
  void requestTimeoutFlagWithoutValueRaises() {
    assertThatThrownBy(
            () ->
                WorkerMain.parseArgs(
                    new String[] {"--server", "http://x", "--request-timeout-seconds"}))
        .isInstanceOf(WorkerMain.CliError.class)
        .hasMessageContaining("expects a value");
  }

  @Test
  void nonIntegerRequestTimeoutRaises() {
    assertThatThrownBy(
            () ->
                WorkerMain.parseArgs(
                    new String[] {"--server", "http://x", "--request-timeout-seconds", "abc"}))
        .isInstanceOf(WorkerMain.CliError.class)
        .hasMessageContaining("Expected integer");
  }

  @Test
  void cliErrorCarriesMessage() {
    WorkerMain.CliError err = new WorkerMain.CliError("boom");
    assertThat(err).hasMessage("boom");
  }

  @Test
  void dirichletDefaultsToCanonicalChessValues() {
    WorkerConfig cfg = WorkerMain.parseArgs(new String[] {"--server", "http://devsrv:8090"});
    assertThat(cfg.dirichletAlpha()).isEqualTo(0.3f);
    assertThat(cfg.dirichletEpsilon()).isEqualTo(0.25f);
  }

  @Test
  void parsesDirichletFlags() {
    WorkerConfig cfg =
        WorkerMain.parseArgs(
            new String[] {
              "--server", "http://x", "--dirichlet-alpha", "0.5", "--dirichlet-epsilon", "0"
            });
    assertThat(cfg.dirichletAlpha()).isEqualTo(0.5f);
    assertThat(cfg.dirichletEpsilon()).isZero();
  }

  @Test
  void rejectsEpsilonAboveOneViaCli() {
    assertThatThrownBy(
            () ->
                WorkerMain.parseArgs(
                    new String[] {"--server", "http://x", "--dirichlet-epsilon", "2"}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void nonNumericDirichletAlphaRaises() {
    assertThatThrownBy(
            () ->
                WorkerMain.parseArgs(
                    new String[] {"--server", "http://x", "--dirichlet-alpha", "abc"}))
        .isInstanceOf(WorkerMain.CliError.class)
        .hasMessageContaining("Expected number");
  }

  // ---------------------------------------------------------------------------------------
  // --nncache (ADR-018) — optional NN-eval cache size.
  // ---------------------------------------------------------------------------------------

  @Test
  void nnCacheDefaultsToOff() {
    WorkerConfig cfg = WorkerMain.parseArgs(new String[] {"--server", "http://devsrv:8090"});
    assertThat(cfg.nnCacheSize()).isZero();
  }

  @Test
  void parsesNnCacheFlag() {
    WorkerConfig cfg =
        WorkerMain.parseArgs(new String[] {"--server", "http://x", "--nncache", "65536"});
    assertThat(cfg.nnCacheSize()).isEqualTo(65536);
  }

  @Test
  void rejectsNegativeNnCache() {
    assertThatThrownBy(
            () -> WorkerMain.parseArgs(new String[] {"--server", "http://x", "--nncache", "-1"}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("nnCacheSize");
  }

  @Test
  void nonIntegerNnCacheRaises() {
    assertThatThrownBy(
            () -> WorkerMain.parseArgs(new String[] {"--server", "http://x", "--nncache", "big"}))
        .isInstanceOf(WorkerMain.CliError.class)
        .hasMessageContaining("Expected integer");
  }
}
