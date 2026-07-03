/**
 * Sous-package contenant l'orchestration multi-thread du mode batched MCTS (cf. ADR-013, ADR-015,
 * SPEC §15).
 *
 * <p>Classes :
 *
 * <ul>
 *   <li>{@link org.nanozero.engine.internal.batched.LeafSubmission} : record représentant une
 *       feuille MCTS à évaluer (planes encodés + future pour récupérer le résultat).
 *   <li>{@link org.nanozero.engine.internal.batched.LeafSubmissionQueue} : queue bornée
 *       producer-consumer où les search threads déposent leurs leaves et le NN-eval thread les
 *       draine par batches.
 *   <li>{@link org.nanozero.engine.internal.batched.NNEvalThread} : thread dédié qui pop la queue
 *       par batches de K leaves, appelle {@code Network.forward(planes, K, output)}, distribue les
 *       résultats aux search threads via les futures.
 *   <li>{@link org.nanozero.engine.internal.batched.SearchPool} : gestionnaire de N search threads
 *       + lifecycle (lazy spawn, graceful shutdown, poison-pill).
 * </ul>
 *
 * <p>Visibilité : package-private sauf pour {@code internal} qui peut utiliser ces classes. Tout
 * est interne au module ; pas d'API publique.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
package org.nanozero.engine.internal.batched;
