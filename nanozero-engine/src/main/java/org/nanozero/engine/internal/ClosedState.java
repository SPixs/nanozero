package org.nanozero.engine.internal;

import org.nanozero.board.GameState;
import org.nanozero.engine.EngineState;
import org.nanozero.engine.SearchBudget;

/**
 * État {@link EngineState#CLOSED} (cf. SPEC §4.3) — engine définitivement clos, instance non
 * réutilisable. Singleton ({@link #INSTANCE}).
 *
 * <ul>
 *   <li>{@code onSubmit} : throw {@link IllegalStateException} (« engine is CLOSED »).
 *   <li>{@code onStopBegin} : throw.
 *   <li>{@code onCloseBegin} / {@code onCloseFinalize} : self (idempotent — close() est idempotent
 *       par contrat SPEC §4.4).
 *   <li>{@code onWorkerCheck} : {@code EXIT_THREAD} (le worker doit sortir).
 *   <li>{@code onWorkerSearchCompleted} : self (race close-during-search — le CLOSED set par le
 *       caller ne doit pas être écrasé en DONE).
 * </ul>
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
final class ClosedState extends EngineStateBehavior {

  static final ClosedState INSTANCE = new ClosedState();

  private ClosedState() {}

  @Override
  EngineState publicValue() {
    return EngineState.CLOSED;
  }

  @Override
  EngineStateBehavior onSubmit(ThreadController ctx, GameState position, SearchBudget budget) {
    throw new IllegalStateException("engine is CLOSED");
  }

  @Override
  EngineStateBehavior onStopBegin(ThreadController ctx) {
    throw new IllegalStateException("engine is CLOSED");
  }

  @Override
  EngineStateBehavior onCloseBegin(ThreadController ctx) {
    return this;
  }

  @Override
  EngineStateBehavior onCloseFinalize(ThreadController ctx) {
    return this;
  }

  @Override
  WorkerDecision onWorkerCheck(ThreadController ctx) {
    return WorkerDecision.EXIT_THREAD;
  }

  @Override
  EngineStateBehavior onWorkerSearchCompleted(ThreadController ctx) {
    return this;
  }
}
