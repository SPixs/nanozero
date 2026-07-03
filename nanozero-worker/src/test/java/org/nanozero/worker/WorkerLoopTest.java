package org.nanozero.worker;

import static org.assertj.core.api.Assertions.assertThat;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Random;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.nanozero.engine.EngineConfig;

/**
 * End-to-end {@link WorkerLoop} tests : real {@link JobserverClient} against an in-process {@link
 * HttpServer}, real {@link ModelCache} (fake loader), real {@link GamePlayer} playing a short game
 * with {@link FakeNetwork}.
 */
class WorkerLoopTest {

  private HttpServer server;
  private WorkerConfig config;
  private JobserverClient client;

  @TempDir Path modelsDir;

  // Job-claim behavior is swapped per test.
  private volatile boolean claimErrors = false;
  private volatile int submitStatus = 200;
  private volatile long claimSleepMillis = 0;
  private final AtomicInteger claimsServed = new AtomicInteger();
  private final AtomicInteger jobsAvailable = new AtomicInteger(0);
  private final AtomicInteger submitsReceived = new AtomicInteger();

  @BeforeEach
  void startServer() throws IOException {
    server = HttpServer.create(new InetSocketAddress("localhost", 0), 0);

    server.createContext(
        "/jobs/claim",
        ex -> {
          claimsServed.incrementAndGet();
          if (claimSleepMillis > 0) {
            try {
              Thread.sleep(claimSleepMillis);
            } catch (InterruptedException ignored) {
              Thread.currentThread().interrupt();
            }
          }
          if (claimErrors) {
            respond(ex, 503, "unavailable");
            return;
          }
          if (jobsAvailable.getAndUpdate(n -> n > 0 ? n - 1 : 0) > 0) {
            respond(
                ex,
                200,
                "{\"job_id\":\"job-"
                    + claimsServed.get()
                    + "\",\"model_version\":1,\"num_sims\":2}");
          } else {
            respond(ex, 204, "");
          }
        });

    server.createContext(
        "/models/1/download",
        ex -> {
          byte[] bytes = "fake-onnx".getBytes(StandardCharsets.UTF_8);
          ex.sendResponseHeaders(200, bytes.length);
          ex.getResponseBody().write(bytes);
          ex.close();
        });

    // Matches any /jobs/{id}/submit (the claim context is more specific, so it wins for /jobs/claim
    // only if registered ; here /jobs/ catches submit paths).
    server.createContext(
        "/jobs/job-",
        ex -> {
          ex.getRequestBody().readAllBytes();
          submitsReceived.incrementAndGet();
          respond(ex, submitStatus, submitStatus == 200 ? "{\"status\":\"ok\"}" : "rejected");
        });

    server.start();
    int port = server.getAddress().getPort();
    config =
        new WorkerConfig(
            "http://localhost:" + port,
            "k",
            "w",
            modelsDir,
            Duration.ofSeconds(10),
            Duration.ZERO); // no idle sleep → fast tests
    client = new JobserverClient(config);
  }

  @AfterEach
  void stopServer() {
    server.stop(0);
  }

  private static void respond(HttpExchange ex, int status, String body) throws IOException {
    byte[] bytes = body.getBytes(StandardCharsets.UTF_8);
    if (status == 204) {
      ex.sendResponseHeaders(204, -1);
    } else {
      ex.sendResponseHeaders(status, bytes.length);
      ex.getResponseBody().write(bytes);
    }
    ex.close();
  }

  private WorkerLoop newLoop(ModelCache.NetworkLoader loader, ModelCache[] outCache) {
    ModelCache cache = new ModelCache(modelsDir, client, loader);
    if (outCache != null) {
      outCache[0] = cache;
    }
    return new WorkerLoop(config, client, cache, new GamePlayer(), new Random(123));
  }

  // ---------------------------------------------------------------------------------------
  // run(maxJobs)
  // ---------------------------------------------------------------------------------------

  @Test
  void runZeroExitsImmediately() {
    ModelCache[] holder = new ModelCache[1];
    WorkerLoop loop = newLoop(p -> new FakeNetwork(), holder);
    int submitted = loop.run(0);
    assertThat(submitted).isZero();
    assertThat(claimsServed.get()).isZero(); // never even claimed
    holder[0].close();
  }

  @Test
  void runOneCompletesASingleJob() {
    jobsAvailable.set(1);
    ModelCache[] holder = new ModelCache[1];
    WorkerLoop loop = newLoop(p -> new FakeNetwork(), holder);

    int submitted = loop.run(1);

    assertThat(submitted).isEqualTo(1);
    assertThat(submitsReceived.get()).isEqualTo(1);
    // The model file was downloaded + cached.
    assertThat(Files.exists(modelsDir.resolve("model-v0001.onnx"))).isTrue();
    holder[0].close();
  }

  @Test
  void runMultiGameCompletesAllJobsAcrossThreads() {
    jobsAvailable.set(8);
    WorkerConfig multiConfig =
        new WorkerConfig(
            config.serverUrl(),
            config.apiKey(),
            config.workerId(),
            config.modelsDir(),
            config.requestTimeout(),
            Duration.ZERO,
            /* cuda= */ false,
            /* concurrentGames= */ 4);
    JobserverClient multiClient = new JobserverClient(multiConfig);
    AtomicInteger loadCount = new AtomicInteger();
    try (ModelCache cache =
        new ModelCache(
            modelsDir,
            multiClient,
            p -> {
              loadCount.incrementAndGet();
              return new FakeNetwork();
            })) {
      WorkerLoop loop = new WorkerLoop(multiConfig, multiClient, cache, new GamePlayer());

      int submitted = loop.run(8);

      assertThat(submitted).isEqualTo(8);
      assertThat(submitsReceived.get()).isEqualTo(8);
      // Un seul chargement réseau : les 4 threads partagent l'instance refcountée.
      assertThat(loadCount.get()).isEqualTo(1);
    }
  }

  @Test
  void runTwoCompletesTwoJobsReusingCachedModel() {
    jobsAvailable.set(2);
    AtomicInteger loadCount = new AtomicInteger();
    ModelCache[] holder = new ModelCache[1];
    WorkerLoop loop =
        newLoop(
            p -> {
              loadCount.incrementAndGet();
              return new FakeNetwork();
            },
            holder);

    int submitted = loop.run(2);

    assertThat(submitted).isEqualTo(2);
    assertThat(submitsReceived.get()).isEqualTo(2);
    // Same model version across both jobs → loaded exactly once.
    assertThat(loadCount.get()).isEqualTo(1);
    holder[0].close();
  }

  // ---------------------------------------------------------------------------------------
  // runOneIteration : individual branches
  // ---------------------------------------------------------------------------------------

  @Test
  void iterationReturnsFalseWhenQueueEmpty() throws Exception {
    jobsAvailable.set(0);
    ModelCache[] holder = new ModelCache[1];
    WorkerLoop loop = newLoop(p -> new FakeNetwork(), holder);
    assertThat(loop.runOneIteration()).isFalse();
    holder[0].close();
  }

  @Test
  void iterationReturnsFalseOnClaimError() throws Exception {
    claimErrors = true;
    ModelCache[] holder = new ModelCache[1];
    WorkerLoop loop = newLoop(p -> new FakeNetwork(), holder);
    assertThat(loop.runOneIteration()).isFalse();
    holder[0].close();
  }

  @Test
  void iterationReturnsTrueOnSuccessfulJob() throws Exception {
    jobsAvailable.set(1);
    ModelCache[] holder = new ModelCache[1];
    WorkerLoop loop = newLoop(p -> new FakeNetwork(), holder);
    assertThat(loop.runOneIteration()).isTrue();
    assertThat(submitsReceived.get()).isEqualTo(1);
    holder[0].close();
  }

  @Test
  void iterationReturnsFalseWhenSubmitFails() throws Exception {
    // Job plays fine, but the submit endpoint rejects → IOException caught → false.
    jobsAvailable.set(1);
    submitStatus = 410;
    ModelCache[] holder = new ModelCache[1];
    WorkerLoop loop = newLoop(p -> new FakeNetwork(), holder);
    assertThat(loop.runOneIteration()).isFalse();
    assertThat(submitsReceived.get()).isEqualTo(1); // server saw the attempt
    holder[0].close();
  }

  @Test
  void iterationReturnsFalseWhenModelLoadThrows() throws Exception {
    // RuntimeException during play (model load) → caught → false.
    jobsAvailable.set(1);
    ModelCache[] holder = new ModelCache[1];
    WorkerLoop loop =
        newLoop(
            p -> {
              throw new IllegalStateException("loader boom");
            },
            holder);
    assertThat(loop.runOneIteration()).isFalse();
    holder[0].close();
  }

  // ---------------------------------------------------------------------------------------
  // run(-1) terminating via thread interrupt
  // ---------------------------------------------------------------------------------------

  @Test
  void runForeverStopsOnInterrupt() throws Exception {
    // No jobs → loop sleeps between empty claims. Interrupt to break out cleanly. Use a non-zero
    // idle sleep so the interrupt reliably lands in Thread.sleep (which throws
    // InterruptedException).
    jobsAvailable.set(0);
    WorkerConfig slowConfig =
        new WorkerConfig(
            config.serverUrl(),
            config.apiKey(),
            config.workerId(),
            modelsDir,
            Duration.ofSeconds(10),
            Duration.ofMillis(200));
    ModelCache cache = new ModelCache(modelsDir, client, p -> new FakeNetwork());
    WorkerLoop loop = new WorkerLoop(slowConfig, client, cache, new GamePlayer(), new Random(123));

    Thread t = new Thread(() -> loop.run(-1));
    t.start();
    // Let it spin at least one empty-claim iteration, then interrupt during the idle sleep.
    while (claimsServed.get() < 1) {
      Thread.onSpinWait();
    }
    t.interrupt();
    t.join(5_000);
    assertThat(t.isAlive()).isFalse();
    cache.close();
  }

  @Test
  void runForeverStopsWhenInterruptedDuringClaim() throws Exception {
    // Server holds the claim open ; interrupting the worker thread makes http.send throw
    // InterruptedException, which run() catches at the runOneIteration level → clean exit.
    claimSleepMillis = 3_000;
    ModelCache[] holder = new ModelCache[1];
    WorkerLoop loop = newLoop(p -> new FakeNetwork(), holder);

    Thread t = new Thread(() -> loop.run(-1));
    t.start();
    while (claimsServed.get() < 1) {
      Thread.onSpinWait();
    }
    // The claim is now in-flight (server sleeping). Interrupt the blocked http.send.
    t.interrupt();
    t.join(5_000);
    assertThat(t.isAlive()).isFalse();
    holder[0].close();
  }

  // ---------------------------------------------------------------------------------------
  // public production constructor smoke (default RNG path)
  // ---------------------------------------------------------------------------------------

  @Test
  void publicConstructorRunZero() {
    ModelCache cache = new ModelCache(modelsDir, client, p -> new FakeNetwork());
    WorkerLoop loop = new WorkerLoop(config, client, cache, new GamePlayer());
    assertThat(loop.run(0)).isZero();
    cache.close();
  }

  // ---------------------------------------------------------------------------------------
  // selfPlayEngineConfig : the worker CLI's --nncache flows into EngineConfig (ADR-018).
  // ---------------------------------------------------------------------------------------

  @Test
  void selfPlayEngineConfigPropagatesNonzeroNnCacheSize() {
    WorkerConfig cfg =
        new WorkerConfig(
            "http://x",
            "k",
            "w",
            modelsDir,
            Duration.ofSeconds(30),
            Duration.ofSeconds(5),
            false,
            1,
            1000,
            1,
            0.3f,
            0.25f,
            131072);
    EngineConfig engineConfig = WorkerLoop.selfPlayEngineConfig(cfg);
    assertThat(engineConfig.nnCacheSize()).isEqualTo(131072);
    // Dirichlet still flows through from the worker config.
    assertThat(engineConfig.dirichletAlpha()).isEqualTo(0.3f);
    assertThat(engineConfig.dirichletEpsilon()).isEqualTo(0.25f);
  }

  @Test
  void selfPlayEngineConfigForcesCacheOffUnderCuda() {
    // GPU worker (cuda=true, Mode B) : cache gated OFF even with --nncache set, because the batched
    // path bypasses LeafEvaluator (the cache would be inert + waste heap). cf. ADR-018.
    WorkerConfig cfg =
        new WorkerConfig(
            "http://x",
            "k",
            "w",
            modelsDir,
            Duration.ofSeconds(30),
            Duration.ofSeconds(5),
            /* cuda= */ true,
            1,
            1000,
            1,
            0.3f,
            0.25f,
            131072);
    assertThat(WorkerLoop.selfPlayEngineConfig(cfg).nnCacheSize()).isZero();
  }

  @Test
  void selfPlayEngineConfigDefaultsCacheOff() {
    // The @BeforeEach config uses the 6-arg convenience constructor → nnCacheSize defaults to 0.
    assertThat(WorkerLoop.selfPlayEngineConfig(config).nnCacheSize()).isZero();
  }
}
