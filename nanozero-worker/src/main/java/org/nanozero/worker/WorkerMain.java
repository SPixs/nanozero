package org.nanozero.worker;

import java.nio.file.Path;
import java.time.Duration;

/**
 * CLI entry point for the worker JAR.
 *
 * <pre>
 * java --add-modules jdk.incubator.vector -jar nanozero-worker-X.Y.jar \
 *      --server http://devsrv:8090 \
 *      --key SECRET \
 *      --worker-id $(hostname) \
 *      --models-dir ./model-cache
 * </pre>
 *
 * <p>Phase 13.4b.1 scope : argument parsing + config wire-up only. The actual selfplay loop
 * (WorkerLoop calling {@code Engine} directly + {@code Sample} generation) is wired in 13.4b.2.
 */
public final class WorkerMain {

  private WorkerMain() {}

  public static void main(String[] args) {
    try {
      WorkerConfig cfg = parseArgs(args);
      System.err.printf(
          "[worker] starting %s → server=%s models-dir=%s backend=%s concurrent-games=%d%n",
          cfg.workerId(),
          cfg.serverUrl(),
          cfg.modelsDir(),
          cfg.cuda() ? "cuda-batched" : "cpu",
          cfg.concurrentGames());

      JobserverClient client = new JobserverClient(cfg);
      // Mode GPU : CUDA EP + paliers (ADR-013-nn), enveloppé dans un BatchingNetwork partagé qui
      // agrège les évaluations des N threads de jeu (le BatchingNetwork possède son délégué :
      // close() du cache ferme les deux).
      ModelCache.NetworkLoader loader =
          cfg.cuda()
              ? path ->
                  new org.nanozero.nn.BatchingNetwork(
                      org.nanozero.nn.NetworkOnnx.loadCuda(path),
                      cfg.flushMicros(),
                      /* closeDelegate= */ true,
                      cfg.minBatch())
              : org.nanozero.nn.NetworkOnnx::load;
      try (ModelCache modelCache = new ModelCache(cfg.modelsDir(), client, loader)) {
        GamePlayer player = new GamePlayer();
        WorkerLoop loop = new WorkerLoop(cfg, client, modelCache, player);
        int submitted = loop.run(/* maxJobs= */ -1);
        System.err.println("[worker] exiting after " + submitted + " jobs submitted.");
      }
    } catch (CliError e) {
      System.err.println("ERROR: " + e.getMessage());
      printUsage();
      System.exit(2);
    } catch (Exception e) {
      System.err.println("FATAL: " + e.getMessage());
      e.printStackTrace(System.err);
      System.exit(1);
    }
  }

  // ---------------------------------------------------------------------------------------
  // Argument parsing
  // ---------------------------------------------------------------------------------------

  /** Parses argv into a {@link WorkerConfig}. Visible for testing. */
  static WorkerConfig parseArgs(String[] args) {
    String serverUrl = null;
    String apiKey = "";
    String workerId = null;
    Path modelsDir = Path.of("./model-cache");
    Duration requestTimeout = Duration.ofSeconds(30);
    Duration pollIdleSleep = Duration.ofSeconds(5);
    boolean cuda = false;
    int concurrentGames = 1;
    int flushMicros = org.nanozero.nn.BatchingNetwork.DEFAULT_FLUSH_MICROS;
    int minBatch = 1;
    float dirichletAlpha = WorkerConfig.DEFAULT_DIRICHLET_ALPHA;
    float dirichletEpsilon = WorkerConfig.DEFAULT_DIRICHLET_EPSILON;
    int nnCacheSize = 0; // OFF by default — behaviour bit-for-bit unchanged (ADR-018).

    int i = 0;
    while (i < args.length) {
      String arg = args[i];
      switch (arg) {
        case "--server" -> serverUrl = requireValue(args, ++i, "--server");
        case "--key" -> apiKey = requireValue(args, ++i, "--key");
        case "--worker-id" -> workerId = requireValue(args, ++i, "--worker-id");
        case "--models-dir" -> modelsDir = Path.of(requireValue(args, ++i, "--models-dir"));
        case "--request-timeout-seconds" ->
            requestTimeout =
                Duration.ofSeconds(parseInt(requireValue(args, ++i, "--request-timeout-seconds")));
        case "--poll-idle-seconds" ->
            pollIdleSleep =
                Duration.ofSeconds(parseInt(requireValue(args, ++i, "--poll-idle-seconds")));
        case "--cuda" -> cuda = true;
        case "--concurrent-games" ->
            concurrentGames = parseInt(requireValue(args, ++i, "--concurrent-games"));
        case "--flush-micros" -> flushMicros = parseInt(requireValue(args, ++i, "--flush-micros"));
        case "--min-batch" -> minBatch = parseInt(requireValue(args, ++i, "--min-batch"));
        case "--dirichlet-alpha" ->
            dirichletAlpha = parseFloat(requireValue(args, ++i, "--dirichlet-alpha"));
        case "--dirichlet-epsilon" ->
            dirichletEpsilon = parseFloat(requireValue(args, ++i, "--dirichlet-epsilon"));
        case "--nncache" -> nnCacheSize = parseInt(requireValue(args, ++i, "--nncache"));
        case "--help", "-h" -> {
          printUsage();
          System.exit(0);
        }
        default -> throw new CliError("Unknown flag: " + arg);
      }
      i++;
    }

    if (serverUrl == null) throw new CliError("Missing required --server URL");
    if (workerId == null) {
      // Default to hostname if not set explicitly.
      workerId = System.getenv().getOrDefault("HOSTNAME", "anonymous-worker");
    }
    return new WorkerConfig(
        serverUrl,
        apiKey,
        workerId,
        modelsDir,
        requestTimeout,
        pollIdleSleep,
        cuda,
        concurrentGames,
        flushMicros,
        minBatch,
        dirichletAlpha,
        dirichletEpsilon,
        nnCacheSize);
  }

  private static String requireValue(String[] args, int idx, String flag) {
    if (idx >= args.length) throw new CliError(flag + " expects a value");
    return args[idx];
  }

  private static int parseInt(String s) {
    try {
      return Integer.parseInt(s);
    } catch (NumberFormatException e) {
      throw new CliError("Expected integer, got " + s);
    }
  }

  private static float parseFloat(String s) {
    try {
      return Float.parseFloat(s);
    } catch (NumberFormatException e) {
      throw new CliError("Expected number, got " + s);
    }
  }

  private static void printUsage() {
    System.err.println(
        """

        Usage:
          java [--add-modules jdk.incubator.vector] -jar nanozero-worker-X.Y.jar \\
               --server URL --key SECRET [--worker-id ID] [--models-dir DIR] \\
               [--request-timeout-seconds N] [--poll-idle-seconds N]

        Required:
          --server URL          jobserver base URL (e.g., http://devsrv:8090)

        Optional:
          --key SECRET          X-API-Key header value (default: empty / dev mode)
          --worker-id ID        unique identifier sent in X-Worker-Id (default: $HOSTNAME)
          --models-dir DIR      local .onnx cache (default: ./model-cache)
          --request-timeout-seconds N  per-request HTTP timeout (default: 30)
          --poll-idle-seconds N        sleep between empty-queue claims (default: 5)
          --cuda                load models on the CUDA EP (requires GPU build + CUDA 12/cuDNN 9
                                DLLs in PATH); evaluations are batched across games
          --concurrent-games N  self-play games played concurrently (default: 1). On GPU,
                                this fills the evaluation batches — target 64-128.
                                Heap sizing: >8 MB per game measured (MCTS trees dominate
                                the engine buffers) — -Xmx2g holds 128 games; 256 games
                                OOMed at 2g (w1080 2026-06-10), use -Xmx4g+ AND enough
                                CPU cores to feed the GPU between evaluations.
          --flush-micros N      BatchingNetwork collect window in µs (default: 1000).
                                With the pipelined BatchingNetwork the window is hidden by
                                the in-flight GPU run : a larger window (≈ the GPU service
                                time, e.g. 5000-10000) yields fuller batches without
                                throughput loss. 0 = opportunistic batches.
          --min-batch N         BatchingNetwork quorum (default: 1 = off) : a silence only
                                ships the batch when it holds >= N positions (capped wait
                                ~150 ms). Set to ~60 percent of --concurrent-games on GPU
                                workers to make cohort fusion self-healing (fragmented
                                cohorts otherwise lock into the slow back-to-back regime).
          --dirichlet-alpha A   Dirichlet root-noise concentration for self-play (default: 0.3,
                                canonical chess). Must be >= 0.
          --dirichlet-epsilon E noise/prior mix at the MCTS root (default: 0.25). 0 disables root
                                noise (legacy behavior) ; must be in [0, 1].
          --nncache N           NN-eval cache size in entries (0=off). Speeds CPU self-play ~+25%
                                at fixed sims; keep modest under -Xmx256m (e.g. 131072 ≈ 26MB).
          --help                show this message
        """);
  }

  /** Marker exception for CLI-level errors that should result in a non-zero exit + usage. */
  static final class CliError extends RuntimeException {
    CliError(String msg) {
      super(msg);
    }
  }
}
