package org.nanozero.engine.internal;

import org.nanozero.engine.EngineState;

/**
 * État {@link EngineState#DONE} (cf. SPEC §4.3) — recherche terminée (budget épuisé ou stop
 * acquitté), résultat disponible dans {@code currentSnapshot}. Singleton ({@link #INSTANCE}).
 *
 * <ul>
 *   <li>{@code onStopBegin} : self (pas d'attente nécessaire).
 *   <li>{@code stopShouldBlock} : false.
 *   <li>{@code onStopComplete} : (snapshot, IDLE) + clear pendingPosition/pendingBudget. C'est le
 *       point d'extraction du résultat utilisé tant après une terminaison naturelle qu'après un
 *       stop signalé.
 *   <li>{@code onCloseBegin} : self (transition finale en {@code onCloseFinalize}).
 *   <li>{@code closeShouldBlock} : false.
 *   <li>{@code awaitDoneShouldBlock} : false.
 *   <li>{@code onWorkerCheck} : {@code WAIT}.
 * </ul>
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
final class DoneState extends EngineStateBehavior {

  static final DoneState INSTANCE = new DoneState();

  private DoneState() {}

  @Override
  EngineState publicValue() {
    return EngineState.DONE;
  }

  @Override
  EngineStateBehavior onStopBegin(ThreadController ctx) {
    return this;
  }

  @Override
  StopOutcome onStopComplete(ThreadController ctx) {
    var result = ctx.currentSnapshot;
    ctx.pendingBudget = null;
    return new StopOutcome(result, IdleState.INSTANCE);
  }

  @Override
  EngineStateBehavior onCloseBegin(ThreadController ctx) {
    return this;
  }

  @Override
  WorkerDecision onWorkerCheck(ThreadController ctx) {
    return WorkerDecision.WAIT;
  }
}
