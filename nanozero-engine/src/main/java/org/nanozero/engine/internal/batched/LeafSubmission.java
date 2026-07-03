package org.nanozero.engine.internal.batched;

import java.util.concurrent.CompletableFuture;

/**
 * Submission d'une feuille MCTS à évaluer par le {@link NNEvalThread} (cf. ADR-013 §15.3).
 *
 * <p>Un search thread qui atteint une feuille pendant son SELECT-EXPAND construit un {@code
 * LeafSubmission} avec :
 *
 * <ul>
 *   <li>les planes d'entrée du NN, encodés à partir de la position de la feuille ;
 *   <li>une {@link CompletableFuture} sur laquelle il bloque ({@code future.join()}) jusqu'à ce que
 *       le NN-eval thread lui retourne le {@link Result} (priors + value).
 * </ul>
 *
 * <p>Le NN-eval thread pop la queue, batche K submissions, appelle {@code Network.forward(planes,
 * K, output)} une seule fois, et complete chaque future avec son slot du résultat.
 *
 * <p>Le record est immutable (les champs sont des références mais le caller n'a pas vocation à les
 * muter après submission — convention coopérative interne).
 *
 * @param planes buffer aplati au format {@code Network.forward} (119 × 64 = 7616 floats par
 *     position).
 * @param future future complétée par le NN-eval thread avec les priors masqués + value.
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
public record LeafSubmission(float[] planes, CompletableFuture<Result> future) {

  /**
   * Résultat d'une évaluation NN pour une feuille unique (slot du batch).
   *
   * @param policyLogits logits non-normalisés pour les {@code POLICY_INDICES = 4672} indices. Le
   *     consumer (SearcherCore.expandLeaf) effectue lui-même le masquage softmax sur les coups
   *     légaux.
   * @param value valeur scalaire post-tanh dans {@code [-1, +1]}, du POV side-to-move de la
   *     position évaluée.
   */
  public record Result(float[] policyLogits, float value) {}
}
