package org.nanozero.engine.internal;

import org.nanozero.engine.EngineState;

/**
 * État {@link EngineState#SEARCHING} (cf. SPEC §4.3) — recherche active dans le worker thread.
 * Singleton ({@link #INSTANCE}).
 *
 * <ul>
 *   <li>{@code onSubmit} : refus (déjà en recherche) — défaut {@code IllegalStateException}.
 *   <li>{@code onStopBegin} : transition vers {@link StoppingState} (signal au worker via flag
 *       d'état observable lock-free entre simulations).
 *   <li>{@code stopShouldBlock} : true (attendre la transition vers DONE par le worker).
 *   <li>{@code onCloseBegin} : transition vers {@link StoppingState} (même signal que stop).
 *   <li>{@code closeShouldBlock} : true.
 *   <li>{@code awaitDoneShouldBlock} : true.
 *   <li>{@code onWorkerCheck} : {@code EXECUTE_SEARCH}.
 *   <li>{@code keepExecutingSearch} : true (le worker continue à boucler tant que SEARCHING).
 *   <li>{@code onWorkerSearchCompleted} : transition vers {@link DoneState} (défaut).
 * </ul>
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
final class SearchingState extends EngineStateBehavior {

  static final SearchingState INSTANCE = new SearchingState();

  private SearchingState() {}

  @Override
  EngineState publicValue() {
    return EngineState.SEARCHING;
  }

  @Override
  EngineStateBehavior onStopBegin(ThreadController ctx) {
    return StoppingState.INSTANCE;
  }

  @Override
  boolean stopShouldBlock() {
    return true;
  }

  @Override
  EngineStateBehavior onCloseBegin(ThreadController ctx) {
    return StoppingState.INSTANCE;
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
    return WorkerDecision.EXECUTE_SEARCH;
  }

  @Override
  boolean keepExecutingSearch() {
    return true;
  }
}
