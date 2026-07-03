package org.nanozero.worker;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;
import java.time.Duration;
import org.junit.jupiter.api.Test;

class WorkerConfigTest {

  private WorkerConfig validConfig() {
    return new WorkerConfig(
        "http://devsrv:8090",
        "secret-key",
        "worker-1",
        Path.of("/tmp/models"),
        Duration.ofSeconds(30),
        Duration.ofSeconds(5));
  }

  @Test
  void buildsWithValidArgs() {
    WorkerConfig cfg = validConfig();
    assertThat(cfg.serverUrl()).isEqualTo("http://devsrv:8090");
    assertThat(cfg.apiKey()).isEqualTo("secret-key");
    assertThat(cfg.workerId()).isEqualTo("worker-1");
  }

  @Test
  void stripsTrailingSlashInServerUrl() {
    WorkerConfig cfg =
        new WorkerConfig(
            "http://devsrv:8090/",
            "k",
            "w",
            Path.of("/m"),
            Duration.ofSeconds(30),
            Duration.ofSeconds(5));
    assertThat(cfg.serverUrl()).isEqualTo("http://devsrv:8090");
  }

  @Test
  void rejectsBlankServerUrl() {
    assertThatThrownBy(
            () ->
                new WorkerConfig(
                    "", "k", "w", Path.of("/m"), Duration.ofSeconds(30), Duration.ofSeconds(5)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("serverUrl");
  }

  @Test
  void rejectsBlankWorkerId() {
    assertThatThrownBy(
            () ->
                new WorkerConfig(
                    "http://x",
                    "k",
                    "",
                    Path.of("/m"),
                    Duration.ofSeconds(30),
                    Duration.ofSeconds(5)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("workerId");
  }

  @Test
  void rejectsNonPositiveRequestTimeout() {
    assertThatThrownBy(
            () ->
                new WorkerConfig(
                    "http://x", "k", "w", Path.of("/m"), Duration.ZERO, Duration.ofSeconds(5)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("requestTimeout");
    assertThatThrownBy(
            () ->
                new WorkerConfig(
                    "http://x",
                    "k",
                    "w",
                    Path.of("/m"),
                    Duration.ofSeconds(-1),
                    Duration.ofSeconds(5)))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void rejectsNegativePollIdleSleep() {
    assertThatThrownBy(
            () ->
                new WorkerConfig(
                    "http://x",
                    "k",
                    "w",
                    Path.of("/m"),
                    Duration.ofSeconds(30),
                    Duration.ofSeconds(-1)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("pollIdleSleep");
  }

  @Test
  void zeroPollIdleSleepIsAllowed() {
    // Useful for tests that want no idle sleep.
    WorkerConfig cfg =
        new WorkerConfig(
            "http://x", "k", "w", Path.of("/m"), Duration.ofSeconds(30), Duration.ZERO);
    assertThat(cfg.pollIdleSleep()).isZero();
  }

  @Test
  void emptyApiKeyAllowedForDevMode() {
    WorkerConfig cfg =
        new WorkerConfig(
            "http://x", "", "w", Path.of("/m"), Duration.ofSeconds(30), Duration.ofSeconds(5));
    assertThat(cfg.apiKey()).isEmpty();
  }

  @Test
  void defaultsEnableCanonicalDirichletNoise() {
    WorkerConfig cfg = validConfig();
    assertThat(cfg.dirichletAlpha()).isEqualTo(0.3f);
    assertThat(cfg.dirichletEpsilon()).isEqualTo(0.25f);
  }

  @Test
  void rejectsEpsilonAboveOne() {
    assertThatThrownBy(
            () ->
                new WorkerConfig(
                    "http://x",
                    "k",
                    "w",
                    Path.of("/m"),
                    Duration.ofSeconds(30),
                    Duration.ofSeconds(5),
                    false,
                    1,
                    1000,
                    1,
                    0.3f,
                    1.5f,
                    0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dirichletEpsilon");
  }

  @Test
  void rejectsNegativeAlpha() {
    assertThatThrownBy(
            () ->
                new WorkerConfig(
                    "http://x",
                    "k",
                    "w",
                    Path.of("/m"),
                    Duration.ofSeconds(30),
                    Duration.ofSeconds(5),
                    false,
                    1,
                    1000,
                    1,
                    -0.1f,
                    0.25f,
                    0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dirichletAlpha");
  }

  @Test
  void rejectsEpsilonWithoutAlpha() {
    assertThatThrownBy(
            () ->
                new WorkerConfig(
                    "http://x",
                    "k",
                    "w",
                    Path.of("/m"),
                    Duration.ofSeconds(30),
                    Duration.ofSeconds(5),
                    false,
                    1,
                    1000,
                    1,
                    0.0f,
                    0.25f,
                    0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dirichletAlpha");
  }

  // ---------------------------------------------------------------------------------------
  // nnCacheSize (ADR-018) — optional NN-eval cache flowed into EngineConfig.
  // ---------------------------------------------------------------------------------------

  @Test
  void defaultsDisableNnCache() {
    // All convenience constructors must default the cache OFF (behaviour bit-for-bit unchanged).
    assertThat(validConfig().nnCacheSize()).isZero();
    WorkerConfig eightArg =
        new WorkerConfig(
            "http://x",
            "k",
            "w",
            Path.of("/m"),
            Duration.ofSeconds(30),
            Duration.ofSeconds(5),
            false,
            4);
    assertThat(eightArg.nnCacheSize()).isZero();
    WorkerConfig nineArg =
        new WorkerConfig(
            "http://x",
            "k",
            "w",
            Path.of("/m"),
            Duration.ofSeconds(30),
            Duration.ofSeconds(5),
            false,
            4,
            2000);
    assertThat(nineArg.nnCacheSize()).isZero();
  }

  @Test
  void acceptsZeroAndPositiveNnCacheSize() {
    assertThat(withNnCache(0).nnCacheSize()).isZero();
    assertThat(withNnCache(131072).nnCacheSize()).isEqualTo(131072);
  }

  @Test
  void rejectsNegativeNnCacheSize() {
    assertThatThrownBy(() -> withNnCache(-1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("nnCacheSize");
  }

  /** Build a config via the canonical 13-arg constructor with a custom nnCacheSize. */
  private static WorkerConfig withNnCache(int nnCacheSize) {
    return new WorkerConfig(
        "http://x",
        "k",
        "w",
        Path.of("/m"),
        Duration.ofSeconds(30),
        Duration.ofSeconds(5),
        false,
        1,
        1000,
        1,
        0.3f,
        0.25f,
        nnCacheSize);
  }
}
