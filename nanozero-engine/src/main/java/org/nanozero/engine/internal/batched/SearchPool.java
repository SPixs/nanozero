package org.nanozero.engine.internal.batched;

import org.nanozero.nn.Network;

/**
 * Conteneur du NN-eval thread dédié + queue partagée pour le mode batched (cf. ADR-015, SPEC
 * §15.4).
 *
 * <p>Composé de :
 *
 * <ul>
 *   <li>un {@link Thread} dédié {@code nanozero-nn-eval} hébergeant le {@link NNEvalThread} ;
 *   <li>une {@link LeafSubmissionQueue} partagée entre les search threads (producers, gérés par
 *       {@code ThreadController}) et le NN-eval thread (consumer).
 * </ul>
 *
 * <p><strong>(v1.3.0 fix M1-review)</strong> : les search threads MCTS sont gérés directement par
 * {@code ThreadController.executeSearchBatched} (une recherche = un lot de threads daemon joints
 * via {@code CountDownLatch}), PAS par ce pool. L'{@code ExecutorService} interne de la v1.2.0
 * n'était jamais utilisé pour exécuter des simulations (infra morte) et a été retiré. La
 * responsabilité de {@code SearchPool} est réduite à : héberger la queue + le NN-eval thread, et
 * orchestrer leur lifecycle.
 *
 * <p>Lifecycle (cf. ADR-015 §5) :
 *
 * <ol>
 *   <li>Constructor : alloue la queue + le NN-eval thread sans le démarrer.
 *   <li>{@link #start()} : démarre le NN-eval thread.
 *   <li>{@link #shutdown()} : graceful — requestStop + poison-pill, join borné puis interrupt.
 * </ol>
 *
 * @apiNote Internal — un seul {@code SearchPool} par {@code Engine} en mode batched.
 */
public final class SearchPool implements AutoCloseable {

  /** Timeout par défaut pour le shutdown graceful (cf. ADR-015 §5). */
  public static final long SHUTDOWN_TIMEOUT_SEC = 5L;

  private final int numSearchThreads;
  private final LeafSubmissionQueue queue;
  private final NNEvalThread nnEvalRunnable;
  private final Thread nnEvalThread;
  private volatile boolean started = false;
  private volatile boolean shutdownInitiated = false;

  /**
   * Construit le pool sans le démarrer. Appeler {@link #start()} pour activer le NN-eval thread.
   *
   * @param numSearchThreads N = nombre de search threads attendus (informatif ; les threads sont
   *     gérés par {@code ThreadController}, pas par ce pool)
   * @param batchSize K = taille du batch NN (= {@code config.batchSize})
   * @param network instance Network qui sera invoquée par le NN-eval thread
   */
  public SearchPool(int numSearchThreads, int batchSize, Network network) {
    if (numSearchThreads < 1) {
      throw new IllegalArgumentException("numSearchThreads must be >= 1, got " + numSearchThreads);
    }
    this.numSearchThreads = numSearchThreads;
    this.queue = new LeafSubmissionQueue(batchSize);
    this.nnEvalRunnable = new NNEvalThread(queue, network, batchSize);

    this.nnEvalThread = new Thread(nnEvalRunnable, "nanozero-nn-eval");
    this.nnEvalThread.setDaemon(true);
    // Priorité légèrement plus élevée pour ne pas être affamé par les search threads.
    this.nnEvalThread.setPriority(Thread.NORM_PRIORITY + 1);
  }

  /** Nombre de search threads attendus (informatif). */
  public int numSearchThreads() {
    return numSearchThreads;
  }

  /** Queue partagée (search threads y soumettent leurs leaves). */
  public LeafSubmissionQueue queue() {
    return queue;
  }

  /** Démarre le NN-eval thread. Idempotent. */
  public synchronized void start() {
    if (started) {
      return;
    }
    nnEvalThread.start();
    started = true;
  }

  /** {@code true} si {@link #start()} a été appelé. */
  public boolean isStarted() {
    return started;
  }

  /** {@code true} si {@link #shutdown()} a été initié. */
  public boolean isShutdown() {
    return shutdownInitiated;
  }

  /**
   * Graceful shutdown (cf. ADR-015 §5). Signale l'arrêt au NN-eval thread (requestStop +
   * poison-pill), le join jusqu'à {@link #SHUTDOWN_TIMEOUT_SEC}, puis {@code interrupt()} en
   * dernier recours. Idempotent.
   *
   * @return {@code true} si le NN-eval thread s'est arrêté proprement avant le timeout
   */
  public synchronized boolean shutdown() {
    if (shutdownInitiated) {
      return true;
    }
    shutdownInitiated = true;

    // Signal NN-eval thread + poison-pill.
    nnEvalRunnable.requestStop();
    try {
      queue.submitPoisonPill();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }

    // Wait graceful (join borné).
    try {
      nnEvalThread.join(SHUTDOWN_TIMEOUT_SEC * 1000L);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }

    if (nnEvalThread.isAlive()) {
      nnEvalThread.interrupt();
    }
    return !nnEvalThread.isAlive();
  }

  /** Alias de {@link #shutdown()} pour utilisation try-with-resources. */
  @Override
  public void close() {
    shutdown();
  }
}
