package org.nanozero.engine;

/**
 * État courant d'une instance {@code Engine} (cf. SPEC §4.3).
 *
 * <p>La machine à états et ses transitions sont implémentées dans {@code EngineStateMachine} en
 * phase 9. Cet enum n'est qu'un porteur de valeur ; il ne contient aucune logique de transition.
 *
 * <p>Transitions valides (cf. SPEC §4.3) :
 *
 * <pre>
 * IDLE       --startSearch-->  SEARCHING
 * IDLE       --startPonder-->  PONDERING
 * SEARCHING  --budget exhausted--> DONE
 * SEARCHING  --stop()-->       STOPPING
 * PONDERING  --ponderhit()-->  SEARCHING (avec nouveau budget)
 * PONDERING  --stop()-->       STOPPING
 * STOPPING   --thread join -->  DONE
 * DONE       --stop()-->       IDLE  (retourne le résultat déjà calculé)
 * DONE       --startSearch ou startPonder--> SEARCHING ou PONDERING (avec re-rooting)
 * ANY        --close()-->      CLOSED
 * CLOSED     --any operation--> IllegalStateException
 * </pre>
 *
 * <p>{@code stop()} depuis {@code DONE} est légal : il retourne le résultat déjà calculé sans
 * bloquer puis transitionne vers {@code IDLE}.
 */
public enum EngineState {

  /**
   * Pas de recherche en cours. État initial après construction et après {@code stop} depuis DONE.
   */
  IDLE,

  /** Recherche normale (budget actif, lance la boucle MCTS). */
  SEARCHING,

  /**
   * Recherche en mode ponder (budget {@code UNLIMITED}, attente d'un {@code ponderhit} ou d'un
   * {@code stop}).
   */
  PONDERING,

  /** {@code stop()} a été demandé : attente de l'arrêt propre du thread interne. */
  STOPPING,

  /** Recherche terminée (budget épuisé ou {@code stop} traité), résultat disponible. */
  DONE,

  /** {@code close()} a été appelé : instance non réutilisable, toute opération lèvera. */
  CLOSED
}
