package org.nanozero.worker;

import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Random;
import java.util.UUID;
import java.util.logging.Level;
import java.util.logging.Logger;
import org.nanozero.engine.EngineConfig;
import org.nanozero.nn.Network;

/**
 * Main worker loop : claim → acquire model → play game → submit positions, forever.
 *
 * <p>Mode historique ({@code concurrentGames == 1}) : un seul thread, un {@link GamePlayer}
 * réutilisé. Mode multi-parties (v1.6.0, ADR-001-worker Phase 3) : {@code concurrentGames} threads
 * de jeu indépendants (claim au fil de l'eau), chacun avec son {@link GamePlayer} et son RNG — tous
 * partagent le {@link ModelCache} (refcounté) et donc, en mode GPU, le {@code BatchingNetwork} qui
 * agrège leurs évaluations.
 *
 * <p>Failures are logged at WARNING and the loop continues : the server's {@code
 * requeue_stale_jobs} watchdog will revive abandoned jobs after their timeout.
 */
public final class WorkerLoop {

  private static final Logger LOG = Logger.getLogger(WorkerLoop.class.getName());

  private final WorkerConfig config;
  private final JobserverClient client;
  private final ModelCache modelCache;
  private final GamePlayer player;
  private final Random rng;

  // Self-play Engine config : Dirichlet root noise enabled by default (canonical AlphaZero, cf.
  // ADR-012), alpha/epsilon sourced from the worker CLI. The per-game random seed is injected in
  // GamePlayer.play() — a mono-thread engine reuses randomSeed verbatim, so without it every game
  // (and machine) would sample an identical noise stream.
  private final EngineConfig engineConfig;

  public WorkerLoop(
      WorkerConfig config, JobserverClient client, ModelCache modelCache, GamePlayer player) {
    this(config, client, modelCache, player, new Random());
  }

  /** Constructor variant for tests : inject a seeded RNG for determinism. */
  WorkerLoop(
      WorkerConfig config,
      JobserverClient client,
      ModelCache modelCache,
      GamePlayer player,
      Random rng) {
    this.config = config;
    this.client = client;
    this.modelCache = modelCache;
    this.player = player;
    this.rng = rng;
    this.engineConfig = selfPlayEngineConfig(config);
  }

  /**
   * Builds the self-play engine config : canonical engine defaults except the Dirichlet root noise,
   * whose alpha/epsilon come from {@code cfg}. {@code randomSeed} stays at the engine default here
   * — a mono-thread engine reuses it verbatim across games, so {@link GamePlayer#play} re-seeds per
   * game to give concurrent games (and machines) independent noise streams.
   *
   * <p>The optional NN-evaluation cache (ADR-018) is GATED on CPU mode : on a GPU worker ({@code
   * cfg.cuda()}, search Mode B / batched queue) it is forced to {@code 0} because the batched path
   * bypasses {@code LeafEvaluator} — the cache would be inert and only waste heap. On CPU ({@code
   * !cuda}, Mode A) {@code cfg.nnCacheSize()} flows through ; {@code 0} (default) keeps it OFF,
   * preserving the bit-for-bit determinism guarantee. Package-private for testing.
   */
  static EngineConfig selfPlayEngineConfig(WorkerConfig cfg) {
    EngineConfig d = EngineConfig.defaults();
    // GPU (Mode B batched) bypasses LeafEvaluator → the cache is inert there ; force it OFF so a
    // GPU
    // worker never allocates it. On CPU (Mode A) the configured size flows through.
    int nnCache = cfg.cuda() ? 0 : cfg.nnCacheSize();
    return new EngineConfig(
        d.cPuct(),
        d.fpuValue(),
        d.treeInitialCapacity(),
        cfg.dirichletAlpha(),
        cfg.dirichletEpsilon(),
        d.randomSeed(),
        d.searchThreads(),
        d.batchSize(),
        d.virtualLoss(),
        nnCache);
  }

  /**
   * Run the loop, returning the count of successfully-submitted jobs.
   *
   * <p>Si {@code config.concurrentGames() > 1}, lance autant de threads de jeu indépendants ; le
   * compte {@code maxJobs} est partagé entre eux (léger dépassement possible si plusieurs parties
   * se terminent simultanément sur la dernière unité — sans conséquence : le serveur compte ce
   * qu'il reçoit).
   *
   * @param maxJobs if non-negative, stop after this many successful submits ({@code 0} = exit
   *     immediately, {@code -1} = forever).
   * @return number of successfully submitted jobs.
   */
  public int run(int maxJobs) {
    int games = config.concurrentGames();
    java.util.concurrent.atomic.AtomicInteger submitted =
        new java.util.concurrent.atomic.AtomicInteger();
    if (games <= 1) {
      runGameLoop(maxJobs, player, rng, submitted);
      return submitted.get();
    }
    Thread[] threads = new Thread[games];
    for (int g = 0; g < games; g++) {
      // RNG par thread, seedé depuis le RNG injecté (déterminisme des tests préservé).
      Random gameRng;
      synchronized (rng) {
        gameRng = new Random(rng.nextLong());
      }
      final Random threadRng = gameRng;
      threads[g] =
          new Thread(
              () -> runGameLoop(maxJobs, new GamePlayer(), threadRng, submitted),
              "selfplay-game-" + g);
      threads[g].start();
    }
    for (Thread t : threads) {
      try {
        t.join();
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        for (Thread other : threads) {
          other.interrupt();
        }
      }
    }
    return submitted.get();
  }

  /** Boucle d'un thread de jeu : claim → play → submit jusqu'à épuisement de {@code maxJobs}. */
  private void runGameLoop(
      int maxJobs,
      GamePlayer gamePlayer,
      Random gameRng,
      java.util.concurrent.atomic.AtomicInteger submitted) {
    while (maxJobs < 0 || submitted.get() < maxJobs) {
      boolean processed;
      try {
        processed = runOneIteration(gamePlayer, gameRng);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        LOG.info("Worker interrupted — exiting cleanly.");
        return;
      }
      if (processed) {
        submitted.incrementAndGet();
        continue;
      }
      // Empty queue — sleep before retrying.
      try {
        Thread.sleep(config.pollIdleSleep().toMillis());
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        return;
      }
    }
  }

  /**
   * One iteration : claim → (if job) → play → submit. Returns {@code true} if a job was completed.
   */
  boolean runOneIteration() throws InterruptedException {
    return runOneIteration(player, rng);
  }

  private boolean runOneIteration(GamePlayer gamePlayer, Random gameRng)
      throws InterruptedException {
    Optional<Map<String, Object>> maybeJob;
    try {
      maybeJob = client.claim();
    } catch (IOException e) {
      LOG.log(Level.WARNING, "claim failed (will retry): " + e.getMessage());
      return false;
    }
    if (maybeJob.isEmpty()) {
      return false;
    }
    Map<String, Object> job = maybeJob.get();
    String jobId = (String) job.get("job_id");
    int modelVersion = ((Number) job.get("model_version")).intValue();
    int numSims = ((Number) job.get("num_sims")).intValue();

    try {
      Network network = modelCache.acquire(modelVersion);
      try {
        List<Sample> samples =
            gamePlayer.play(
                network,
                engineConfig,
                numSims,
                GamePlayer.DEFAULT_MAX_PLIES,
                GamePlayer.DEFAULT_TEMPERATURE,
                GamePlayer.DEFAULT_TEMPERATURE_SWITCH_PLY,
                gameRng);
        submitSamples(jobId, modelVersion, samples);
        LOG.info(() -> "Job " + jobId + " completed : " + samples.size() + " positions submitted.");
      } finally {
        modelCache.release(modelVersion);
      }
      return true;
    } catch (IOException | RuntimeException e) {
      LOG.log(
          Level.WARNING, "Job " + jobId + " failed (server will requeue): " + e.getMessage(), e);
      return false;
    }
  }

  private void submitSamples(String jobId, int modelVersion, List<Sample> samples)
      throws IOException, InterruptedException {
    Map<String, Object> body = new LinkedHashMap<>();
    body.put("game_id", UUID.randomUUID().toString());
    body.put("model_version", modelVersion);

    var positions = new java.util.ArrayList<Map<String, Object>>(samples.size());
    for (Sample s : samples) {
      Map<String, Object> p = new LinkedHashMap<>();
      p.put("ply", s.ply);
      p.put("fen", ""); // empty — trainer doesn't consume it
      p.put("input_planes_b64", s.inputPlanesBase64());
      p.put("policy_target_b64", s.policyTargetBase64());
      p.put("outcome", (double) s.valueTarget);
      positions.add(p);
    }
    body.put("positions", positions);

    client.submit(jobId, body);
  }
}
