package org.nanozero.engine.internal;

import org.nanozero.engine.EngineState;

/**
 * État {@link EngineState#STOPPING} (cf. SPEC §4.3) — stop demandé, en attente de l'arrêt propre du
 * worker. Singleton ({@link #INSTANCE}).
 *
 * <ul>
 *   <li>{@code onStopBegin} : self (déjà en arrêt).
 *   <li>{@code stopShouldBlock} : true (attendre la transition vers DONE par le worker).
 *   <li>{@code onCloseBegin} : self (déjà en arrêt — close finalisera après wait).
 *   <li>{@code closeShouldBlock} : true.
 *   <li>{@code awaitDoneShouldBlock} : true.
 *   <li>{@code onWorkerCheck} : {@code ACK_STOP} — race où stop()/close() a flippé STOPPING avant
 *       même que le worker ait démarré la recherche, le worker doit acquitter en passant à DONE.
 *   <li>{@code keepExecutingSearch} : false (le worker doit sortir de la boucle MCTS).
 *   <li>{@code onWorkerSearchCompleted} : transition vers {@link DoneState} (défaut).
 * </ul>
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
final class StoppingState extends EngineStateBehavior {

  static final StoppingState INSTANCE = new StoppingState();

  private StoppingState() {}

  @Override
  EngineState publicValue() {
    return EngineState.STOPPING;
  }

  @Override
  EngineStateBehavior onStopBegin(ThreadController ctx) {
    return this;
  }

  @Override
  boolean stopShouldBlock() {
    return true;
  }

  @Override
  EngineStateBehavior onCloseBegin(ThreadController ctx) {
    return this;
  }

  @Override
  boolean closeShouldBlock() {
    return true;
  }

  @Override
  boolean awaitDoneShouldBlock() {
    return true;
  }

  @Override
  WorkerDecision onWorkerCheck(ThreadController ctx) {
    return WorkerDecision.ACK_STOP;
  }

  @Override
  boolean keepExecutingSearch() {
    return false;
  }
}
