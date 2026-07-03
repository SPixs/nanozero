package org.nanozero.engine.internal.batched;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Queue bornée producer-consumer entre les {@code N} search threads (producers) et l'unique {@link
 * NNEvalThread} (consumer, cf. ADR-013 §15.3, ADR-015).
 *
 * <p>Capacité bornée à {@code 2 × batchSize} : backpressure naturelle quand le NN-eval thread est
 * saturé (les search threads bloquent en {@code put()} au lieu de remplir indéfiniment la queue →
 * pas d'OOM).
 *
 * <p>Le NN-eval thread appelle {@link #drainBatch(int, long, TimeUnit)} pour récupérer jusqu'à K
 * leaves disponibles. Si moins de K leaves arrivent dans le timeout, retourne ce qui est dispo
 * (peut être vide) — permet d'éviter le starvation en fin de recherche quand le budget est presque
 * épuisé.
 *
 * @apiNote Internal — thread-safe, exclusivement réservé à l'orchestration batched.
 */
public final class LeafSubmissionQueue {

  /** Sentinelle "poison-pill" pour signaler au NN-eval thread qu'il doit s'arrêter. */
  public static final LeafSubmission POISON_PILL = new LeafSubmission(null, null);

  private final LinkedBlockingQueue<LeafSubmission> queue;
  private final int capacity;

  /**
   * Construit la queue avec une capacité de {@code 2 × batchSize}.
   *
   * @param batchSize taille K du batch NN (cf. {@link org.nanozero.engine.EngineConfig#batchSize})
   * @throws IllegalArgumentException si {@code batchSize < 1}
   */
  public LeafSubmissionQueue(int batchSize) {
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be >= 1, got " + batchSize);
    }
    this.capacity = 2 * batchSize;
    this.queue = new LinkedBlockingQueue<>(this.capacity);
  }

  /** Capacité maximale (= {@code 2 × batchSize}). */
  public int capacity() {
    return capacity;
  }

  /** Nombre de submissions actuellement en attente d'eval. */
  public int size() {
    return queue.size();
  }

  /**
   * Soumet une feuille à évaluer. Bloque si la queue est pleine (backpressure → le search thread
   * attend que le NN-eval thread consomme avant de continuer).
   *
   * @param submission leaf à évaluer
   * @throws InterruptedException si le thread est interrompu pendant l'attente
   */
  public void submit(LeafSubmission submission) throws InterruptedException {
    queue.put(submission);
  }

  /**
   * Soumet la poison-pill pour signaler au NN-eval thread d'exit. Non-bloquant (offer + retry sur 1
   * seconde max) : si la queue est pleine au moment du shutdown, on peut perdre des soumissions
   * mais c'est OK car on shutdown.
   */
  public void submitPoisonPill() throws InterruptedException {
    if (!queue.offer(POISON_PILL, 1, TimeUnit.SECONDS)) {
      // Queue pleine pendant > 1s : on force-clear et on push (best effort shutdown).
      queue.clear();
      queue.put(POISON_PILL);
    }
  }

  /**
   * Draine jusqu'à {@code maxBatch} leaves dans une nouvelle {@link List}. Attend au moins {@code
   * 1} leaf (bloquant) puis draine en non-bloquant le reste. Si {@code timeout} expire sans aucune
   * leaf, retourne une liste vide.
   *
   * <p>Si la poison-pill apparaît dans le batch, elle est incluse en dernier (le caller doit
   * détecter et exit son main loop).
   *
   * @param maxBatch capacité max du batch retourné (typiquement K = batchSize)
   * @param timeout délai max pour la première leaf (≥ 0)
   * @param unit unité du timeout
   * @return liste de 0 à {@code maxBatch} leaves (peut être vide en cas de timeout)
   * @throws InterruptedException si le thread est interrompu pendant l'attente
   */
  public List<LeafSubmission> drainBatch(int maxBatch, long timeout, TimeUnit unit)
      throws InterruptedException {
    if (maxBatch < 1) {
      throw new IllegalArgumentException("maxBatch must be >= 1, got " + maxBatch);
    }
    List<LeafSubmission> batch = new ArrayList<>(maxBatch);
    LeafSubmission first = queue.poll(timeout, unit);
    if (first == null) {
      return batch; // timeout, rien à drainer
    }
    batch.add(first);
    queue.drainTo(batch, maxBatch - 1);
    return batch;
  }
}
