package org.nanozero.nn;

import java.util.Arrays;

/**
 * Conteneur pré-alloué pour les sorties d'un {@link Network#forward(float[], int, NNOutput)} (cf.
 * SPEC §4.2.3, §5.3).
 *
 * <p>Allocation paquetée à la capacité de batch : les buffers internes sont dimensionnés pour le
 * batch maximal ({@link Network#MAX_BATCH} par défaut, ou la capacité explicite passée au
 * constructeur — cf. {@link Network#maxBatch()}, amendement v1.6.0), jamais réalloués. Une instance
 * par thread d'inférence (invariant {@code I-Out-1}, non thread-safe).
 *
 * <p>Invariants :
 *
 * <ul>
 *   <li>{@code I-Out-1} : non thread-safe. Chaque thread d'inférence DOIT avoir sa propre instance.
 *   <li>{@code I-Out-2} : buffers pré-alloués à la capacité ; aucune allocation par appel à {@code
 *       forward()}.
 * </ul>
 *
 * <p>Deux APIs de lecture des logits selon le contexte :
 *
 * <ul>
 *   <li>{@link #logitsOf(int)} : commodité allouante (copie via {@link Arrays#copyOfRange}). Hors
 *       hot path. Sûre vis-à-vis des buffers internes (le caller ne peut pas les corrompre).
 *   <li>{@link #copyLogitsTo(int, float[])} : zéro alloc. Pour hot path MCTS où le caller gère son
 *       buffer.
 * </ul>
 */
public final class NNOutput {

  /**
   * Logits bruts paquetés : {@code float[capacity * POLICY_INDICES]}, layout linéaire par batch.
   */
  final float[] logits;

  /** Values bruts (post-tanh) : {@code float[capacity]}. */
  final float[] values;

  /**
   * Construit un {@code NNOutput} pré-alloué à la capacité {@link Network#MAX_BATCH}. Aucune
   * allocation supplémentaire par appel à {@code forward()}.
   */
  public NNOutput() {
    this(Network.MAX_BATCH);
  }

  /**
   * Construit un {@code NNOutput} pré-alloué à une capacité explicite (amendement v1.6.0 — chemins
   * batchés au-delà de {@link Network#MAX_BATCH}, cf. {@link Network#maxBatch()} et ADR-013).
   *
   * @param capacity capacité maximale de batch ({@code >= 1})
   * @throws IllegalArgumentException si {@code capacity < 1}
   */
  public NNOutput(int capacity) {
    if (capacity < 1) {
      throw new IllegalArgumentException("capacity must be >= 1, got " + capacity);
    }
    this.logits = new float[capacity * MoveEncoding.POLICY_INDICES];
    this.values = new float[capacity];
  }

  /**
   * Retourne la capacité maximale de batch de ce conteneur (taille de pré-allocation).
   *
   * @return capacité en positions
   */
  public int capacity() {
    return values.length;
  }

  /**
   * Retourne les logits de la position {@code batchIndex} sous forme d'un nouveau {@code float[]}
   * de longueur {@link MoveEncoding#POLICY_INDICES}. Allouante par appel.
   *
   * @param batchIndex index dans {@code [0, capacity())}
   * @return copie indépendante des logits (le caller peut la muter sans affecter l'état interne)
   * @throws IndexOutOfBoundsException si {@code batchIndex} hors plage
   */
  public float[] logitsOf(int batchIndex) {
    checkBatchIndex(batchIndex);
    int from = batchIndex * MoveEncoding.POLICY_INDICES;
    return Arrays.copyOfRange(logits, from, from + MoveEncoding.POLICY_INDICES);
  }

  /**
   * Copie les logits de la position {@code batchIndex} dans {@code dest} (zéro allocation). Pour
   * hot path MCTS où le caller maintient un buffer pré-alloué.
   *
   * @param batchIndex index dans {@code [0, capacity())}
   * @param dest tableau destination de longueur &geq; {@link MoveEncoding#POLICY_INDICES}
   * @throws IndexOutOfBoundsException si {@code batchIndex} hors plage
   * @throws IllegalArgumentException si {@code dest.length < POLICY_INDICES}
   */
  public void copyLogitsTo(int batchIndex, float[] dest) {
    checkBatchIndex(batchIndex);
    if (dest.length < MoveEncoding.POLICY_INDICES) {
      throw new IllegalArgumentException(
          "dest length must be >= " + MoveEncoding.POLICY_INDICES + ", got " + dest.length);
    }
    System.arraycopy(
        logits, batchIndex * MoveEncoding.POLICY_INDICES, dest, 0, MoveEncoding.POLICY_INDICES);
  }

  /**
   * Retourne le value scalaire pour la position {@code batchIndex}.
   *
   * @param batchIndex index dans {@code [0, capacity())}
   * @return value scalaire dans {@code [-1, 1]} (post-tanh)
   * @throws IndexOutOfBoundsException si {@code batchIndex} hors plage
   */
  public float valueOf(int batchIndex) {
    checkBatchIndex(batchIndex);
    return values[batchIndex];
  }

  private void checkBatchIndex(int batchIndex) {
    if (batchIndex < 0 || batchIndex >= values.length) {
      throw new IndexOutOfBoundsException(
          "batchIndex " + batchIndex + " hors [0, " + values.length + ")");
    }
  }
}
