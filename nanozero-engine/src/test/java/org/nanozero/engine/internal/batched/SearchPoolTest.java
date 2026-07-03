package org.nanozero.engine.internal.batched;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.nn.Network;

class SearchPoolTest {

  private static final Network NOOP_NETWORK = TestNetworks.noop();

  @Test
  @DisplayName("numSearchThreads < 1 → IAE")
  void invalidNumThreads() {
    assertThatThrownBy(() -> new SearchPool(0, 1, NOOP_NETWORK))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("construct sans start → pas de thread spawned")
  void noThreadsBeforeStart() {
    try (SearchPool pool = new SearchPool(4, 8, NOOP_NETWORK)) {
      assertThat(pool.isStarted()).isFalse();
      assertThat(pool.numSearchThreads()).isEqualTo(4);
      assertThat(pool.queue().capacity()).isEqualTo(16); // 2 × batchSize
    }
  }

  @Test
  @DisplayName("start idempotent : double appel OK")
  void startIdempotent() {
    try (SearchPool pool = new SearchPool(2, 4, NOOP_NETWORK)) {
      pool.start();
      pool.start(); // doit pas crasher
      assertThat(pool.isStarted()).isTrue();
    }
  }

  @Test
  @DisplayName("Shutdown graceful : threads exit avant timeout")
  void shutdownGraceful() {
    SearchPool pool = new SearchPool(2, 4, NOOP_NETWORK);
    pool.start();
    boolean clean = pool.shutdown();
    assertThat(clean).as("shutdown propre attendu").isTrue();
    // Re-shutdown OK (idempotent).
    pool.shutdown();
  }

  @Test
  @DisplayName("Try-with-resources : ferme proprement")
  void tryWithResourcesShutdown() {
    SearchPool poolRef;
    try (SearchPool pool = new SearchPool(2, 4, NOOP_NETWORK)) {
      pool.start();
      poolRef = pool;
    }
    // Après close, le pool doit être shutdown.
    assertThat(poolRef.isShutdown()).isTrue();
  }

  @Test
  @DisplayName("Bout en bout : 4 search tasks soumettent leaves → NN-eval thread les évalue")
  void endToEndSearchTasksEvalLeaves() throws Exception {
    int n = 4;
    int batchSize = 8;
    // (v1.3.0 fix M1-review) Les threads producteurs ne sont plus fournis par SearchPool : on
    // utilise un executor local au test pour simuler les N search threads (rôle tenu par
    // ThreadController.executeSearchBatched en prod).
    ExecutorService producers = Executors.newFixedThreadPool(n);
    try (SearchPool pool = new SearchPool(n, batchSize, NOOP_NETWORK)) {
      pool.start();
      CountDownLatch done = new CountDownLatch(n);
      for (int i = 0; i < n; i++) {
        producers.submit(
            () -> {
              // 1 search thread soumet 1 leaf et attend le résultat.
              CompletableFuture<LeafSubmission.Result> future = new CompletableFuture<>();
              LeafSubmission leaf = new LeafSubmission(new float[119 * 64], future);
              try {
                pool.queue().submit(leaf);
                LeafSubmission.Result result = future.get(5, TimeUnit.SECONDS);
                assertThat(result).isNotNull();
              } catch (Exception e) {
                throw new RuntimeException(e);
              } finally {
                done.countDown();
              }
            });
      }
      assertThat(done.await(10, TimeUnit.SECONDS)).isTrue();
    } finally {
      producers.shutdownNow();
    }
  }
}
