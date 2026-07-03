package org.nanozero.nn;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Test;
import org.nanozero.board.BitboardPlaneEncoder;
import org.nanozero.board.GameState;

/**
 * Tests du décorateur {@link BatchingNetwork} (Phase 2 worker GPU, ADR-001-worker). Délégué fake
 * déterministe : {@code value[b] = planes[b][0]} (marqueur) et {@code logits[b][i] = marqueur + i}
 * — vérifie le routage exact des positions et des résultats à travers le batching.
 */
final class BatchingNetworkTest {

  private static final int FLOATS = NetworkOnnx.INPUT_FLOATS_PER_POS;
  private static final int LOGITS = MoveEncoding.POLICY_INDICES;

  /** Délégué fake : déterministe par marqueur, enregistre les tailles de batch reçues. */
  private static class RecordingDelegate implements Network {
    final ConcurrentLinkedQueue<Integer> batchSizes = new ConcurrentLinkedQueue<>();
    final AtomicBoolean failNext = new AtomicBoolean();
    volatile CountDownLatch gate; // si non-null : forward bloque dessus (interruptible)
    private final int maxBatch;

    RecordingDelegate(int maxBatch) {
      this.maxBatch = maxBatch;
    }

    @Override
    public int maxBatch() {
      return maxBatch;
    }

    @Override
    public void forward(float[] planes, int batchSize, NNOutput output) {
      batchSizes.add(batchSize);
      CountDownLatch g = gate;
      if (g != null) {
        try {
          if (!g.await(5, TimeUnit.SECONDS)) {
            throw new IllegalStateException("gate timeout");
          }
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new IllegalStateException("interrupted in delegate", e);
        }
      }
      if (failNext.getAndSet(false)) {
        throw new IllegalStateException("simulated delegate failure");
      }
      for (int b = 0; b < batchSize; b++) {
        float marker = planes[b * FLOATS];
        output.values[b] = marker;
        for (int i = 0; i < LOGITS; i++) {
          output.logits[b * LOGITS + i] = marker + i;
        }
      }
    }

    @Override
    public NNSingleResult forwardSingle(GameState state) {
      throw new UnsupportedOperationException();
    }

    @Override
    public NetworkMetadata metadata() {
      return new NetworkMetadata("fake", "n/a", 0, "n/a", "test");
    }

    @Override
    public BitboardPlaneEncoder planeEncoder() {
      return BitboardPlaneEncoderVector.INSTANCE;
    }
  }

  /** Planes d'une position marquée : premier float = marqueur, reste à zéro. */
  private static float[] markedPlanes(int positions, float firstMarker) {
    float[] planes = new float[positions * FLOATS];
    for (int b = 0; b < positions; b++) {
      planes[b * FLOATS] = firstMarker + b;
    }
    return planes;
  }

  @Test
  void roundTripSingleCaller() {
    RecordingDelegate delegate = new RecordingDelegate(64);
    try (BatchingNetwork net = new BatchingNetwork(delegate)) {
      NNOutput out = new NNOutput();
      net.forward(markedPlanes(1, 7.0f), 1, out);
      assertThat(out.valueOf(0)).isEqualTo(7.0f);
      assertThat(out.logitsOf(0)[0]).isEqualTo(7.0f);
      assertThat(out.logitsOf(0)[LOGITS - 1]).isEqualTo(7.0f + LOGITS - 1);
      assertThat(net.maxBatch()).isEqualTo(64);
    }
  }

  @Test
  void multiPositionRequestRoutesEveryRow() {
    RecordingDelegate delegate = new RecordingDelegate(64);
    try (BatchingNetwork net = new BatchingNetwork(delegate)) {
      NNOutput out = new NNOutput();
      net.forward(markedPlanes(5, 100.0f), 5, out);
      for (int b = 0; b < 5; b++) {
        assertThat(out.valueOf(b)).isEqualTo(100.0f + b);
        assertThat(out.logitsOf(b)[1]).isEqualTo(100.0f + b + 1);
      }
    }
  }

  @Test
  void concurrentCallersGetTheirOwnResultsBack() throws Exception {
    RecordingDelegate delegate = new RecordingDelegate(64);
    int threads = 16;
    int evalsPerThread = 50;
    try (BatchingNetwork net = new BatchingNetwork(delegate);
        ExecutorService pool = Executors.newFixedThreadPool(threads)) {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < threads; t++) {
        final int threadId = t;
        futures.add(
            pool.submit(
                () -> {
                  NNOutput out = new NNOutput();
                  float[] planes = new float[FLOATS];
                  for (int i = 0; i < evalsPerThread; i++) {
                    float marker = threadId * 1000.0f + i;
                    planes[0] = marker;
                    net.forward(planes, 1, out);
                    assertThat(out.valueOf(0)).isEqualTo(marker);
                    assertThat(out.logitsOf(0)[3]).isEqualTo(marker + 3);
                  }
                }));
      }
      for (Future<?> f : futures) {
        f.get(30, TimeUnit.SECONDS); // rethrow toute assertion d'un thread
      }
    }
  }

  @Test
  void concurrentSubmissionsAreActuallyBatched() throws Exception {
    RecordingDelegate delegate = new RecordingDelegate(64);
    int threads = 16;
    // fenêtre généreuse (50 ms) : les 16 soumissions partent dans la même fenêtre de collecte
    try (BatchingNetwork net = new BatchingNetwork(delegate, 50_000);
        ExecutorService pool = Executors.newFixedThreadPool(threads)) {
      CyclicBarrier barrier = new CyclicBarrier(threads);
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < threads; t++) {
        final float marker = t;
        futures.add(
            pool.submit(
                () -> {
                  barrier.await();
                  NNOutput out = new NNOutput();
                  float[] planes = new float[FLOATS];
                  planes[0] = marker;
                  net.forward(planes, 1, out);
                  assertThat(out.valueOf(0)).isEqualTo(marker);
                  return null;
                }));
      }
      for (Future<?> f : futures) {
        f.get(30, TimeUnit.SECONDS);
      }
      int maxBatch = delegate.batchSizes.stream().mapToInt(Integer::intValue).max().orElse(0);
      int totalPositions = delegate.batchSizes.stream().mapToInt(Integer::intValue).sum();
      assertThat(totalPositions).isEqualTo(threads);
      assertThat(maxBatch).isGreaterThan(1); // au moins une vraie agrégation
    }
  }

  @Test
  void requestsOverflowingCapacitySplitAcrossBatches() throws Exception {
    RecordingDelegate delegate = new RecordingDelegate(8);
    try (BatchingNetwork net = new BatchingNetwork(delegate, 20_000);
        ExecutorService pool = Executors.newFixedThreadPool(3)) {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < 3; t++) {
        final float marker = 500.0f + t * 10;
        futures.add(
            pool.submit(
                () -> {
                  NNOutput out = new NNOutput();
                  net.forward(markedPlanes(5, marker), 5, out);
                  for (int b = 0; b < 5; b++) {
                    assertThat(out.valueOf(b)).isEqualTo(marker + b);
                  }
                  return null;
                }));
      }
      for (Future<?> f : futures) {
        f.get(30, TimeUnit.SECONDS);
      }
      // 3 requêtes de 5 positions, capacité 8 : aucun batch ne peut dépasser 8
      assertThat(delegate.batchSizes).allSatisfy(size -> assertThat(size).isLessThanOrEqualTo(8));
      assertThat(delegate.batchSizes.stream().mapToInt(Integer::intValue).sum()).isEqualTo(15);
    }
  }

  @Test
  void delegateFailurePropagatesAndCollectorSurvives() {
    RecordingDelegate delegate = new RecordingDelegate(64);
    try (BatchingNetwork net = new BatchingNetwork(delegate)) {
      NNOutput out = new NNOutput();
      delegate.failNext.set(true);
      assertThatThrownBy(() -> net.forward(markedPlanes(1, 1.0f), 1, out))
          .isInstanceOf(IllegalStateException.class)
          .hasRootCauseMessage("simulated delegate failure");
      // le collecteur survit : l'appel suivant fonctionne
      net.forward(markedPlanes(1, 2.0f), 1, out);
      assertThat(out.valueOf(0)).isEqualTo(2.0f);
    }
  }

  @Test
  void forwardAfterCloseThrows() {
    RecordingDelegate delegate = new RecordingDelegate(64);
    BatchingNetwork net = new BatchingNetwork(delegate);
    net.close();
    assertThatThrownBy(() -> net.forward(markedPlanes(1, 1.0f), 1, new NNOutput()))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("closed");
    net.close(); // idempotent
  }

  @Test
  void closeUnblocksPendingCallers() throws Exception {
    RecordingDelegate delegate = new RecordingDelegate(64);
    delegate.gate = new CountDownLatch(1); // le délégué bloque le collecteur
    BatchingNetwork net = new BatchingNetwork(delegate);
    try (ExecutorService pool = Executors.newFixedThreadPool(2)) {
      Future<?> stuckInDelegate =
          pool.submit(() -> net.forward(markedPlanes(1, 1.0f), 1, new NNOutput()));
      Thread.sleep(100); // laisse le collecteur entrer dans le délégué (bloqué sur gate)
      Future<?> stuckInQueue =
          pool.submit(() -> net.forward(markedPlanes(1, 2.0f), 1, new NNOutput()));
      Thread.sleep(100);

      net.close(); // interrompt le collecteur (gate interruptible) + draine la queue

      assertThatThrownBy(() -> stuckInDelegate.get(10, TimeUnit.SECONDS))
          .hasCauseInstanceOf(IllegalStateException.class);
      assertThatThrownBy(() -> stuckInQueue.get(10, TimeUnit.SECONDS))
          .hasCauseInstanceOf(IllegalStateException.class);
    }
  }

  @Test
  void rejectsInvalidArguments() {
    RecordingDelegate delegate = new RecordingDelegate(8);
    try (BatchingNetwork net = new BatchingNetwork(delegate)) {
      NNOutput big = new NNOutput(8);
      assertThatThrownBy(() -> net.forward(markedPlanes(1, 0f), 0, big))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> net.forward(markedPlanes(9, 0f), 9, big))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> net.forward(new float[FLOATS], 2, big))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> net.forward(markedPlanes(4, 0f), 4, new NNOutput(2)))
          .isInstanceOf(IllegalArgumentException.class);
      assertThatThrownBy(() -> new BatchingNetwork(delegate, -1))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }

  @Test
  void closesDelegateOnlyWhenOwned() {
    final class CloseableDelegate extends RecordingDelegate implements AutoCloseable {
      final AtomicBoolean closedFlag = new AtomicBoolean();

      CloseableDelegate() {
        super(64);
      }

      @Override
      public void close() {
        closedFlag.set(true);
      }
    }
    CloseableDelegate notOwned = new CloseableDelegate();
    new BatchingNetwork(notOwned, 1000, false).close();
    assertThat(notOwned.closedFlag).isFalse();

    CloseableDelegate owned = new CloseableDelegate();
    new BatchingNetwork(owned, 1000, true).close();
    assertThat(owned.closedFlag).isTrue();
  }

  @Test
  void forwardSingleRoutesThroughBatchPathAndDelegatesMetadata() {
    RecordingDelegate delegate = new RecordingDelegate(64);
    try (BatchingNetwork net = new BatchingNetwork(delegate)) {
      NNSingleResult result = net.forwardSingle(new GameState());
      // startpos : le marqueur (premier float des planes) dépend de l'encodage — on vérifie
      // simplement la cohérence value/logits du délégué fake : logits[i] = value + i.
      assertThat(result.logits()[5]).isEqualTo(result.value() + 5);
      assertThat(net.metadata().architectureVersion()).isEqualTo("fake");
      assertThat(net.planeEncoder()).isSameAs(BitboardPlaneEncoderVector.INSTANCE);
    }
  }

  /**
   * Cœur de la v2 (pipeline 3 étages) : pendant que le délégué exécute le batch N, l'assembler doit
   * avoir construit et remis le batch N+1 au runner — c'est ce recouvrement qui supprime
   * l'alternance stricte CPU/GPU mesurée au JFR (GPU plafonné à ~48 % en v1).
   */
  @Test
  void pipelineAssemblesNextBatchWhileDelegateRuns() throws Exception {
    RecordingDelegate delegate = new RecordingDelegate(64);
    delegate.gate = new CountDownLatch(1); // bloque le runner DANS le délégué (batch 1)
    try (BatchingNetwork net = new BatchingNetwork(delegate, 0);
        ExecutorService pool = Executors.newFixedThreadPool(5)) {
      List<Future<?>> futures = new ArrayList<>();
      futures.add(
          pool.submit(
              () -> {
                NNOutput out = new NNOutput();
                net.forward(markedPlanes(1, 1.0f), 1, out);
                assertThat(out.valueOf(0)).isEqualTo(1.0f);
                return null;
              }));
      waitUntil(() -> delegate.batchSizes.size() == 1); // batch 1 est dans le délégué
      for (int t = 0; t < 4; t++) {
        final float marker = 10.0f + t;
        futures.add(
            pool.submit(
                () -> {
                  NNOutput out = new NNOutput();
                  net.forward(markedPlanes(1, marker), 1, out);
                  assertThat(out.valueOf(0)).isEqualTo(marker);
                  return null;
                }));
      }
      // le batch 2 doit être assemblé et remis au runner PENDANT que le batch 1 est en vol —
      // et rien ne peut être complété tant que le délégué est bloqué.
      waitUntil(() -> net.stats().assembled() >= 2);
      assertThat(net.stats().completed()).isZero();

      delegate.gate.countDown();
      for (Future<?> f : futures) {
        f.get(30, TimeUnit.SECONDS);
      }
      assertThat(net.stats().completed()).isGreaterThanOrEqualTo(2);
      assertThat(net.stats().positions()).isEqualTo(5);
    }
  }

  /** Traverse le seuil de log périodique ({@code STATS_EVERY_BATCHES}) sans perturber les flux. */
  @Test
  void periodicStatsLoggingSurvivesManyBatches() {
    RecordingDelegate delegate = new RecordingDelegate(8);
    try (BatchingNetwork net = new BatchingNetwork(delegate, 0)) {
      NNOutput out = new NNOutput();
      float[] planes = new float[FLOATS];
      int evals = BatchingNetwork.STATS_EVERY_BATCHES + 50;
      for (int i = 0; i < evals; i++) {
        planes[0] = i;
        net.forward(planes, 1, out);
        assertThat(out.valueOf(0)).isEqualTo((float) i);
      }
      assertThat(net.stats().completed()).isEqualTo(evals);
      assertThat(net.stats().positions()).isEqualTo(evals);
    }
  }

  /** Close avec des lots engagés dans TOUS les étages : chaque appelant doit être débloqué. */
  @Test
  void closeFailsBatchesThroughoutThePipeline() throws Exception {
    RecordingDelegate delegate = new RecordingDelegate(4); // petite capacité → plusieurs lots
    delegate.gate = new CountDownLatch(1);
    BatchingNetwork net = new BatchingNetwork(delegate, 0);
    try (ExecutorService pool = Executors.newFixedThreadPool(6)) {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < 6; t++) {
        final float marker = t;
        futures.add(pool.submit(() -> net.forward(markedPlanes(1, marker), 1, new NNOutput())));
      }
      waitUntil(() -> delegate.batchSizes.size() >= 1); // un lot en vol, d'autres derrière
      net.close();
      for (Future<?> f : futures) {
        assertThatThrownBy(() -> f.get(10, TimeUnit.SECONDS))
            .hasCauseInstanceOf(IllegalStateException.class);
      }
    }
  }

  /**
   * Quorum {@code minBatch} : un silence n'expédie pas un batch sous le quorum — la collecte
   * persiste et agrège les arrivées tardives (auto-réparation de la fusion des cohortes).
   */
  @Test
  void minBatchQuorumHoldsPartialBatchesUntilQuorum() throws Exception {
    RecordingDelegate delegate = new RecordingDelegate(64);
    // idle-gap court (1 ms) : sans quorum, chaque soumission partirait quasi seule.
    try (BatchingNetwork net = new BatchingNetwork(delegate, 1000, false, 4);
        ExecutorService pool = Executors.newFixedThreadPool(4)) {
      List<Future<?>> futures = new ArrayList<>();
      for (int t = 0; t < 2; t++) {
        final float marker = t;
        futures.add(pool.submit(() -> net.forward(markedPlanes(1, marker), 1, new NNOutput())));
      }
      Thread.sleep(40); // >> idle-gap : sans quorum le batch de 2 serait déjà parti
      assertThat(delegate.batchSizes).isEmpty(); // tenu par le quorum
      for (int t = 2; t < 4; t++) {
        final float marker = t;
        futures.add(pool.submit(() -> net.forward(markedPlanes(1, marker), 1, new NNOutput())));
      }
      for (Future<?> f : futures) {
        f.get(10, TimeUnit.SECONDS);
      }
      // le quorum atteint, tout part dans UN batch de 4
      assertThat(delegate.batchSizes).containsExactly(4);
    }
  }

  /** Sous le quorum, le plafond d'attente (~150 ms) expédie quand même — pas de blocage. */
  @Test
  void minBatchQuorumHardCapShipsUndersizedBatch() throws Exception {
    RecordingDelegate delegate = new RecordingDelegate(64);
    try (BatchingNetwork net = new BatchingNetwork(delegate, 1000, false, 32)) {
      long start = System.nanoTime();
      NNOutput out = new NNOutput();
      net.forward(markedPlanes(1, 9.0f), 1, out); // seul → quorum jamais atteint
      long elapsedMs = (System.nanoTime() - start) / 1_000_000;
      assertThat(out.valueOf(0)).isEqualTo(9.0f);
      assertThat(elapsedMs).isBetween(100L, 5_000L); // a attendu ~150 ms puis expédié
      assertThat(delegate.batchSizes).containsExactly(1);
    }
  }

  @Test
  void rejectsInvalidMinBatch() {
    RecordingDelegate delegate = new RecordingDelegate(8);
    assertThatThrownBy(() -> new BatchingNetwork(delegate, 1000, false, 0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new BatchingNetwork(delegate, 1000, false, 9))
        .isInstanceOf(IllegalArgumentException.class);
  }

  /** Attente active bornée d'une condition (tests de concurrence). */
  private static void waitUntil(java.util.function.BooleanSupplier condition) throws Exception {
    long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(5);
    while (!condition.getAsBoolean()) {
      if (System.nanoTime() > deadline) {
        throw new AssertionError("condition not met within 5s");
      }
      Thread.sleep(2);
    }
  }
}
