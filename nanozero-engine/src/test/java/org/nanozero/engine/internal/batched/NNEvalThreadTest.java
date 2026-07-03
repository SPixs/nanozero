package org.nanozero.engine.internal.batched;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.nn.Network;

class NNEvalThreadTest {

  private static final int PLANES_PER_POSITION = 119 * 64;

  private static LeafSubmission makeLeaf() {
    return new LeafSubmission(new float[PLANES_PER_POSITION], new CompletableFuture<>());
  }

  @Test
  @DisplayName("batchSize hors bornes → IAE")
  void invalidBatchSize() {
    LeafSubmissionQueue q = new LeafSubmissionQueue(4);
    Network nn = TestNetworks.noop();
    assertThatThrownBy(() -> new NNEvalThread(q, nn, 0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new NNEvalThread(q, nn, Network.MAX_BATCH + 1))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("Eval thread : 4 submissions → forward call(s) + 4 futures complétés")
  void multipleSubmissionsCompletedAfterForward() throws Exception {
    LeafSubmissionQueue q = new LeafSubmissionQueue(8);
    AtomicInteger forwardCalls = new AtomicInteger(0);
    AtomicInteger totalBatchSize = new AtomicInteger(0);
    Network nn =
        TestNetworks.forwardOnly(
            (planes, batchSize, output) -> {
              forwardCalls.incrementAndGet();
              totalBatchSize.addAndGet(batchSize);
            });

    NNEvalThread evalRunnable = new NNEvalThread(q, nn, 8);
    Thread evalThread = new Thread(evalRunnable, "test-eval");
    evalThread.setDaemon(true);

    LeafSubmission a = makeLeaf();
    LeafSubmission b = makeLeaf();
    LeafSubmission c = makeLeaf();
    LeafSubmission d = makeLeaf();
    q.submit(a);
    q.submit(b);
    q.submit(c);
    q.submit(d);

    evalThread.start();

    LeafSubmission.Result rA = a.future().get(2, TimeUnit.SECONDS);
    LeafSubmission.Result rB = b.future().get(2, TimeUnit.SECONDS);
    LeafSubmission.Result rC = c.future().get(2, TimeUnit.SECONDS);
    LeafSubmission.Result rD = d.future().get(2, TimeUnit.SECONDS);

    evalRunnable.requestStop();
    q.submitPoisonPill();
    evalThread.join(2000);

    // Chaque future doit recevoir un Result non-null (peu importe le contenu — pas testable sans
    // accès package-private à NNOutput).
    assertThat(rA).isNotNull();
    assertThat(rB).isNotNull();
    assertThat(rC).isNotNull();
    assertThat(rD).isNotNull();
    assertThat(rA.policyLogits()).hasSize(org.nanozero.nn.MoveEncoding.POLICY_INDICES);

    // 1 forward call (ou plusieurs si timing crée des batches plus petits), mais batchSize cumulé
    // = exactement 4 (toutes les submissions traitées).
    assertThat(forwardCalls.get()).as("au moins 1 forward call").isGreaterThanOrEqualTo(1);
    assertThat(totalBatchSize.get()).as("4 submissions traitées au total").isEqualTo(4);
  }

  @Test
  @DisplayName("Eval thread : poison-pill arrête la boucle")
  void poisonPillStopsThread() throws Exception {
    LeafSubmissionQueue q = new LeafSubmissionQueue(4);
    Network nn = TestNetworks.noop();
    NNEvalThread evalRunnable = new NNEvalThread(q, nn, 4);
    Thread evalThread = new Thread(evalRunnable, "test-eval");
    evalThread.setDaemon(true);
    evalThread.start();

    q.submitPoisonPill();
    evalThread.join(2000);
    assertThat(evalThread.isAlive()).as("thread doit avoir exit après poison-pill").isFalse();
  }

  @Test
  @DisplayName(
      "Eval thread : forward throws → tous les futures du batch sont completedExceptionally")
  void forwardExceptionPropagated() throws Exception {
    LeafSubmissionQueue q = new LeafSubmissionQueue(4);
    Network throwingNetwork =
        TestNetworks.forwardOnly(
            (planes, batchSize, output) -> {
              throw new RuntimeException("simulated NN failure");
            });
    NNEvalThread evalRunnable = new NNEvalThread(q, throwingNetwork, 4);
    Thread evalThread = new Thread(evalRunnable, "test-eval");
    evalThread.setDaemon(true);
    evalThread.start();

    LeafSubmission a = makeLeaf();
    LeafSubmission b = makeLeaf();
    q.submit(a);
    q.submit(b);

    assertThatThrownBy(() -> a.future().get(2, TimeUnit.SECONDS))
        .hasRootCauseInstanceOf(RuntimeException.class)
        .hasMessageContaining("simulated NN failure");
    assertThatThrownBy(() -> b.future().get(2, TimeUnit.SECONDS))
        .hasRootCauseInstanceOf(RuntimeException.class);

    evalRunnable.requestStop();
    q.submitPoisonPill();
    evalThread.join(2000);
  }

  @Test
  @DisplayName("Eval thread : shutdown avec leaves en attente → futures completedExceptionally")
  void shutdownDrainsRemainingAsCancelled() throws Exception {
    LeafSubmissionQueue q = new LeafSubmissionQueue(8);
    // Network qui prend du temps : la première forward bloque 200ms.
    Network slowNetwork =
        TestNetworks.forwardOnly(
            (planes, batchSize, output) -> {
              try {
                Thread.sleep(200);
              } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
              }
            });
    NNEvalThread evalRunnable = new NNEvalThread(q, slowNetwork, 1);
    Thread evalThread = new Thread(evalRunnable, "test-eval-slow");
    evalThread.setDaemon(true);
    evalThread.start();

    LeafSubmission first = makeLeaf();
    LeafSubmission queued = makeLeaf();
    q.submit(first);
    q.submit(queued);

    Thread.sleep(50); // laisser le temps au thread de pop first
    evalRunnable.requestStop();
    q.submitPoisonPill();
    evalThread.join(3000);

    assertThat(evalThread.isAlive()).isFalse();
    assertThat(first.future().isDone()).isTrue();
    // queued doit être completedExceptionally (jamais traité par forward).
    assertThat(queued.future().isDone()).isTrue();
    assertThat(queued.future().isCompletedExceptionally()).isTrue();
  }
}
