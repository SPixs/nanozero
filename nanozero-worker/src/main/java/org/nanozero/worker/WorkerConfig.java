package org.nanozero.worker;

import java.nio.file.Path;
import java.time.Duration;
import java.util.Objects;

/**
 * Immutable config for one worker instance. Parsed once from CLI args at startup.
 *
 * @param serverUrl jobserver base URL (e.g., "http://devsrv:8090"). Trailing slash stripped.
 * @param apiKey value of the {@code X-API-Key} header. Empty string ok for dev mode.
 * @param workerId free-form identifier sent in {@code X-Worker-Id} (typically hostname).
 * @param modelsDir local directory where downloaded {@code .onnx} files are cached. Created on
 *     demand.
 * @param requestTimeout per-HTTP-request timeout.
 * @param pollIdleSleep how long to sleep when the server returns 204 (empty queue).
 * @param cuda load models on the CUDA EP wrapped in a {@code BatchingNetwork} (GPU worker, v1.6.0).
 * @param concurrentGames number of self-play games played concurrently in this process (≥ 1). On a
 *     GPU worker, this is what fills the evaluation batches — target 64-256.
 * @param flushMicros collect window of the {@code BatchingNetwork} (µs, ≥ 0) : how long the
 *     assembler waits for more evaluations before shipping a partial batch. With the pipelined
 *     BatchingNetwork the window is hidden by the in-flight GPU run, so a larger window yields
 *     fuller batches (better SM efficiency, less padding) at no throughput cost — tune toward the
 *     GPU service time of the target bucket.
 * @param minBatch quorum of the {@code BatchingNetwork} (≥ 1) : a silence only ships the batch if
 *     it holds at least this many positions (self-healing cohort merge — set to ~60 % of
 *     concurrentGames on GPU workers). {@code 1} disables the quorum.
 * @param dirichletAlpha Dirichlet root-noise concentration for self-play exploration (≥ 0 ;
 *     canonical chess value 0.3, cf. ADR-012). With {@code dirichletEpsilon=0} noise is disabled
 *     (legacy ε=0 behavior).
 * @param dirichletEpsilon mix factor between NN priors and Dirichlet noise at the root (in [0, 1] ;
 *     canonical 0.25). {@code 0} disables root noise ; if {@code > 0} then {@code dirichletAlpha}
 *     must be {@code > 0}.
 * @param nnCacheSize size (in entries) of the OPTIONAL NN-evaluation cache flowed into {@link
 *     org.nanozero.engine.EngineConfig#nnCacheSize} (ADR-018). {@code 0} (default) disables the
 *     cache : behaviour is bit-for-bit unchanged (no key computation, no overhead) and the ADR-010
 *     determinism guarantee is preserved. {@code > 0} enables a bounded Lc0-style {@code NNCache}
 *     that mutualizes a leaf's NN evaluation (~+25 % on CPU self-play at fixed sims) ; keep it
 *     modest under {@code -Xmx256m} (e.g. {@code 131072} ≈ 26 MB). Must be {@code >= 0}.
 */
public record WorkerConfig(
    String serverUrl,
    String apiKey,
    String workerId,
    Path modelsDir,
    Duration requestTimeout,
    Duration pollIdleSleep,
    boolean cuda,
    int concurrentGames,
    int flushMicros,
    int minBatch,
    float dirichletAlpha,
    float dirichletEpsilon,
    int nnCacheSize) {

  /** Canonical AlphaZero chess Dirichlet concentration (cf. ADR-012). */
  public static final float DEFAULT_DIRICHLET_ALPHA = 0.3f;

  /** Canonical AlphaZero Dirichlet mix factor (25 % noise / 75 % prior). */
  public static final float DEFAULT_DIRICHLET_EPSILON = 0.25f;

  /** Convenience constructor — historical single-game CPU worker (cuda=false, 1 game). */
  public WorkerConfig(
      String serverUrl,
      String apiKey,
      String workerId,
      Path modelsDir,
      Duration requestTimeout,
      Duration pollIdleSleep) {
    this(serverUrl, apiKey, workerId, modelsDir, requestTimeout, pollIdleSleep, false, 1);
  }

  /** Convenience constructor — default collect window (v1.6.0 compat). */
  public WorkerConfig(
      String serverUrl,
      String apiKey,
      String workerId,
      Path modelsDir,
      Duration requestTimeout,
      Duration pollIdleSleep,
      boolean cuda,
      int concurrentGames) {
    this(
        serverUrl,
        apiKey,
        workerId,
        modelsDir,
        requestTimeout,
        pollIdleSleep,
        cuda,
        concurrentGames,
        org.nanozero.nn.BatchingNetwork.DEFAULT_FLUSH_MICROS,
        1,
        DEFAULT_DIRICHLET_ALPHA,
        DEFAULT_DIRICHLET_EPSILON,
        0);
  }

  /** Convenience constructor — no batch quorum (compat). */
  public WorkerConfig(
      String serverUrl,
      String apiKey,
      String workerId,
      Path modelsDir,
      Duration requestTimeout,
      Duration pollIdleSleep,
      boolean cuda,
      int concurrentGames,
      int flushMicros) {
    this(
        serverUrl,
        apiKey,
        workerId,
        modelsDir,
        requestTimeout,
        pollIdleSleep,
        cuda,
        concurrentGames,
        flushMicros,
        1,
        DEFAULT_DIRICHLET_ALPHA,
        DEFAULT_DIRICHLET_EPSILON,
        0);
  }

  public WorkerConfig {
    Objects.requireNonNull(serverUrl, "serverUrl");
    Objects.requireNonNull(apiKey, "apiKey");
    Objects.requireNonNull(workerId, "workerId");
    Objects.requireNonNull(modelsDir, "modelsDir");
    Objects.requireNonNull(requestTimeout, "requestTimeout");
    Objects.requireNonNull(pollIdleSleep, "pollIdleSleep");

    if (serverUrl.isBlank()) {
      throw new IllegalArgumentException("serverUrl must not be blank");
    }
    if (workerId.isBlank()) {
      throw new IllegalArgumentException("workerId must not be blank");
    }
    if (requestTimeout.isNegative() || requestTimeout.isZero()) {
      throw new IllegalArgumentException("requestTimeout must be > 0");
    }
    if (pollIdleSleep.isNegative()) {
      throw new IllegalArgumentException("pollIdleSleep must be >= 0");
    }
    if (concurrentGames < 1) {
      throw new IllegalArgumentException("concurrentGames must be >= 1, got " + concurrentGames);
    }
    if (flushMicros < 0) {
      throw new IllegalArgumentException("flushMicros must be >= 0, got " + flushMicros);
    }
    if (minBatch < 1) {
      throw new IllegalArgumentException("minBatch must be >= 1, got " + minBatch);
    }
    // !(x >= 0) also rejects NaN.
    if (!(dirichletAlpha >= 0f)) {
      throw new IllegalArgumentException("dirichletAlpha must be >= 0, got " + dirichletAlpha);
    }
    if (!(dirichletEpsilon >= 0f) || dirichletEpsilon > 1f) {
      throw new IllegalArgumentException(
          "dirichletEpsilon must be in [0, 1], got " + dirichletEpsilon);
    }
    if (dirichletEpsilon > 0f && dirichletAlpha == 0f) {
      throw new IllegalArgumentException("dirichletAlpha must be > 0 when dirichletEpsilon > 0");
    }
    if (nnCacheSize < 0) {
      throw new IllegalArgumentException("nnCacheSize must be >= 0, got " + nnCacheSize);
    }

    // Strip trailing slash for consistent URL building.
    if (serverUrl.endsWith("/")) {
      serverUrl = serverUrl.substring(0, serverUrl.length() - 1);
    }
  }
}
