package org.nanozero.engine.internal.batched;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class LeafSubmissionQueueTest {

  private static LeafSubmission newLeaf() {
    return new LeafSubmission(new float[8], new CompletableFuture<>());
  }

  @Test
  @DisplayName("capacity = 2 × batchSize")
  void capacityIsTwiceBatchSize() {
    LeafSubmissionQueue q = new LeafSubmissionQueue(16);
    assertThat(q.capacity()).isEqualTo(32);
    assertThat(q.size()).isZero();
  }

  @Test
  @DisplayName("batchSize < 1 → IAE")
  void invalidBatchSize() {
    assertThatThrownBy(() -> new LeafSubmissionQueue(0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("batchSize");
  }

  @Test
  @DisplayName("submit + drainBatch retourne les éléments dans l'ordre FIFO")
  void submitDrainBatchFifo() throws InterruptedException {
    LeafSubmissionQueue q = new LeafSubmissionQueue(4);
    LeafSubmission a = newLeaf();
    LeafSubmission b = newLeaf();
    LeafSubmission c = newLeaf();
    q.submit(a);
    q.submit(b);
    q.submit(c);
    List<LeafSubmission> batch = q.drainBatch(4, 100, TimeUnit.MILLISECONDS);
    assertThat(batch).containsExactly(a, b, c);
    assertThat(q.size()).isZero();
  }

  @Test
  @DisplayName("drainBatch respecte maxBatch")
  void drainBatchRespectsMax() throws InterruptedException {
    LeafSubmissionQueue q = new LeafSubmissionQueue(8);
    for (int i = 0; i < 10; i++) {
      q.submit(newLeaf());
    }
    List<LeafSubmission> batch = q.drainBatch(5, 100, TimeUnit.MILLISECONDS);
    assertThat(batch).hasSize(5);
    assertThat(q.size()).isEqualTo(5);
  }

  @Test
  @DisplayName("drainBatch timeout sans submission → liste vide")
  void drainBatchTimeoutEmpty() throws InterruptedException {
    LeafSubmissionQueue q = new LeafSubmissionQueue(4);
    List<LeafSubmission> batch = q.drainBatch(4, 50, TimeUnit.MILLISECONDS);
    assertThat(batch).isEmpty();
  }

  @Test
  @DisplayName("drainBatch bloque jusqu'à 1 leaf (rapide), draine le reste non-bloquant")
  void drainBatchBlocksForFirstThenDrainsNonBlocking() throws InterruptedException {
    LeafSubmissionQueue q = new LeafSubmissionQueue(4);
    // Soumettre 3 puis appeler drain → doit retourner les 3 immédiatement.
    q.submit(newLeaf());
    q.submit(newLeaf());
    q.submit(newLeaf());
    long t0 = System.nanoTime();
    List<LeafSubmission> batch = q.drainBatch(8, 1000, TimeUnit.MILLISECONDS);
    long elapsedMs = (System.nanoTime() - t0) / 1_000_000;
    assertThat(batch).hasSize(3);
    assertThat(elapsedMs).as("drain doit être ~immédiat").isLessThan(200);
  }

  @Test
  @DisplayName("submitPoisonPill : la POISON apparaît dans le drain suivant")
  void submitPoisonPillVisibleInDrain() throws InterruptedException {
    LeafSubmissionQueue q = new LeafSubmissionQueue(4);
    LeafSubmission normal = newLeaf();
    q.submit(normal);
    q.submitPoisonPill();
    List<LeafSubmission> batch = q.drainBatch(8, 100, TimeUnit.MILLISECONDS);
    assertThat(batch).hasSize(2);
    assertThat(batch.get(0)).isEqualTo(normal);
    assertThat(batch.get(1)).isSameAs(LeafSubmissionQueue.POISON_PILL);
  }

  @Test
  @DisplayName("maxBatch < 1 → IAE")
  void drainBatchInvalidMax() {
    LeafSubmissionQueue q = new LeafSubmissionQueue(4);
    assertThatThrownBy(() -> q.drainBatch(0, 100, TimeUnit.MILLISECONDS))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("maxBatch");
  }
}
