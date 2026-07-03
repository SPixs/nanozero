package org.nanozero.engine.internal.batched;

import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import org.nanozero.nn.MoveEncoding;
import org.nanozero.nn.NNOutput;
import org.nanozero.nn.Network;

/**
 * Thread dédié qui draine la {@link LeafSubmissionQueue} par batches et appelle {@code
 * Network.forward(planes, K, output)} pour évaluer K feuilles en un seul appel NN (cf. ADR-013
 * §15.3, ADR-015).
 *
 * <p>Pattern d'exécution :
 *
 * <ol>
 *   <li>Boucle principale : {@code queue.drainBatch(K, timeout)} → liste de 1..K submissions.
 *   <li>Copie les planes de chaque submission dans le buffer aplati attendu par {@code forward}.
 *   <li>Appelle {@code network.forward(planes, batchSize, output)} une seule fois pour tout le
 *       batch.
 *   <li>Pour chaque submission, extrait son slot de l'output et complete sa future.
 *   <li>Exit propre si la poison-pill apparaît dans le batch.
 * </ol>
 *
 * <p>Garanties :
 *
 * <ul>
 *   <li>Le NN-eval thread est <strong>le seul à invoquer {@link Network#forward}</strong> (cf. SPEC
 *       §15.4). Ça isole ORT EP (CPU ou GPU) de toute concurrence non-maîtrisée.
 *   <li>Les futures sont complétées dans l'ordre du batch, mais le caller n'a pas vocation à
 *       compter sur cet ordre — chaque thread reçoit son propre Result.
 *   <li>Exceptions du forward sont propagées via {@code future.completeExceptionally} pour ne pas
 *       crasher le NN-eval thread (autres futures restent en attente jusqu'au shutdown).
 * </ul>
 *
 * @apiNote Internal — instancié par {@code SearchPool} en mode batched uniquement.
 */
public final class NNEvalThread implements Runnable {

  /** Timeout par défaut pour {@link LeafSubmissionQueue#drainBatch} (ms). */
  public static final long DRAIN_TIMEOUT_MS = 50L;

  private final LeafSubmissionQueue queue;
  private final Network network;
  private final int batchSize;
  private final AtomicBoolean stopRequested = new AtomicBoolean(false);

  // Buffer pre-alloué pour le forward : MAX_BATCH × planes par position.
  private static final int PLANES_PER_POSITION = 119 * 64;
  private final float[] planesBuffer;
  private final NNOutput nnOutput = new NNOutput();

  /**
   * Construit le thread NN-eval.
   *
   * @param queue queue d'où drainer les submissions
   * @param network instance Network (typiquement {@code NetworkOnnx} ou {@code NetworkVectorApi}) —
   *     sera invoquée exclusivement par ce thread
   * @param batchSize K = taille du batch (config.batchSize)
   */
  public NNEvalThread(LeafSubmissionQueue queue, Network network, int batchSize) {
    if (batchSize < 1 || batchSize > Network.MAX_BATCH) {
      throw new IllegalArgumentException(
          "batchSize must be in [1, " + Network.MAX_BATCH + "], got " + batchSize);
    }
    this.queue = queue;
    this.network = network;
    this.batchSize = batchSize;
    this.planesBuffer = new float[Network.MAX_BATCH * PLANES_PER_POSITION];
  }

  /** Signale au thread d'exit à la prochaine itération (idempotent). */
  public void requestStop() {
    stopRequested.set(true);
  }

  @Override
  public void run() {
    try {
      while (!stopRequested.get() && !Thread.currentThread().isInterrupted()) {
        List<LeafSubmission> batch;
        try {
          batch = queue.drainBatch(batchSize, DRAIN_TIMEOUT_MS, TimeUnit.MILLISECONDS);
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          break;
        }
        if (batch.isEmpty()) {
          continue; // timeout sans leaf, on re-poll
        }
        // Détection poison-pill : si présente, on traite ce qui est avant elle et on exit.
        boolean foundPoison = false;
        int effectiveSize = 0;
        for (LeafSubmission s : batch) {
          if (s == LeafSubmissionQueue.POISON_PILL) {
            foundPoison = true;
            break;
          }
          effectiveSize++;
        }
        if (effectiveSize > 0) {
          evaluateBatch(batch, effectiveSize);
        }
        if (foundPoison) {
          break;
        }
      }
    } finally {
      // Si on exit avec des futures en attente dans la queue, on les complete avec exception
      // pour ne pas bloquer indéfiniment les search threads.
      drainRemainingAsCancelled();
    }
  }

  private void evaluateBatch(List<LeafSubmission> batch, int batchSize) {
    try {
      // Copie chaque planes dans le buffer aplati. Layout : [batch_idx * PLANES_PER_POSITION].
      for (int i = 0; i < batchSize; i++) {
        float[] src = batch.get(i).planes();
        System.arraycopy(src, 0, planesBuffer, i * PLANES_PER_POSITION, PLANES_PER_POSITION);
      }
      network.forward(planesBuffer, batchSize, nnOutput);
      // Complete chaque future avec son slot.
      for (int i = 0; i < batchSize; i++) {
        float[] logits = new float[MoveEncoding.POLICY_INDICES];
        nnOutput.copyLogitsTo(i, logits);
        float value = nnOutput.valueOf(i);
        batch.get(i).future().complete(new LeafSubmission.Result(logits, value));
      }
    } catch (RuntimeException e) {
      // Propagate à tous les futures pour ne pas bloquer les search threads.
      for (int i = 0; i < batchSize; i++) {
        batch.get(i).future().completeExceptionally(e);
      }
    }
  }

  private void drainRemainingAsCancelled() {
    List<LeafSubmission> remaining;
    try {
      remaining = queue.drainBatch(Network.MAX_BATCH, 0, TimeUnit.MILLISECONDS);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      return;
    }
    for (LeafSubmission s : remaining) {
      if (s != LeafSubmissionQueue.POISON_PILL) {
        s.future().completeExceptionally(new InterruptedException("NN-eval thread shutdown"));
      }
    }
  }
}
