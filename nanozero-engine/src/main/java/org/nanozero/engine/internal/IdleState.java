package org.nanozero.engine.internal;

import org.nanozero.board.GameState;
import org.nanozero.engine.EngineState;
import org.nanozero.engine.SearchBudget;

/**
 * État {@link EngineState#IDLE} (cf. SPEC §4.3) — pas de recherche en cours, prêt à accepter un
 * {@code startSearch} ou un {@code startPonder}. Singleton ({@link #INSTANCE}).
 *
 * <ul>
 *   <li>{@code onSubmit} : transition vers {@link SearchingState}. Tente le re-rooting du tree
 *       courant via {@link ThreadController#tryReroot} (0/1/2 plies), sinon fresh tree. Le budget
 *       est enveloppé dans un {@link MutableSearchBudget} pour permettre un futur {@code
 *       ponderhit}-style swap (uniformité d'interface entre searches normaux et ponder).
 *   <li>{@code onSubmitPonder} : transition vers {@link PonderingState}. Clone la position,
 *       applique {@code predictedOpponentMove}, prépare le tree sur la position pondérée. Budget
 *       interne = {@code MutableSearchBudget(SearchBudget.UNLIMITED)} jusqu'à {@code ponderhit} ou
 *       {@code stop}.
 *   <li>{@code onStopBegin} : self (pas de transition).
 *   <li>{@code stopShouldBlock} : false (pas d'attente).
 *   <li>{@code onStopComplete} : ({@code null}, IDLE) — caller mappera null vers EMPTY_RESULT.
 *   <li>{@code onCloseBegin} : transition immédiate vers {@link ClosedState}.
 *   <li>{@code onWorkerCheck} : {@code WAIT}.
 * </ul>
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
final class IdleState extends EngineStateBehavior {

  static final IdleState INSTANCE = new IdleState();

  private IdleState() {}

  @Override
  EngineState publicValue() {
    return EngineState.IDLE;
  }

  @Override
  EngineStateBehavior onSubmit(ThreadController ctx, GameState position, SearchBudget budget) {
    GameState target = ThreadController.cloneState(position);
    if (!ctx.tryReroot(target)) {
      ctx.tree = new SearchTree(target);
    }
    // v1.1.2 — Clear Dirichlet noise on new search (cf. SPEC §5.5 mise à jour v1.1.2, ADR-012
    // "Isolation cross-search"). Garantit qu'un re-search sur exactement la même position
    // (0-ply tree reuse) re-sample le noise au lieu de réutiliser celui de la search précédente.
    // Inconditionnel (coût négligeable, 1 affectation) ; pas appliqué à onSubmitPonder ni à
    // PonderingState.onPonderhit (cf. §5.5 distinctions documentées).
    ctx.tree.root().dirichletNoise = null;
    ctx.pendingBudget = new MutableSearchBudget(budget);
    ctx.currentSnapshot = ThreadController.EMPTY_RESULT;
    return SearchingState.INSTANCE;
  }

  @Override
  EngineStateBehavior onSubmitPonder(
      ThreadController ctx, GameState position, int predictedOpponentMove) {
    GameState ponderState = ThreadController.cloneState(position);
    ponderState.applyMove(predictedOpponentMove);
    if (!ctx.tryReroot(ponderState)) {
      ctx.tree = new SearchTree(ponderState);
    }
    ctx.pendingBudget = new MutableSearchBudget(SearchBudget.UNLIMITED);
    ctx.currentSnapshot = ThreadController.EMPTY_RESULT;
    return PonderingState.INSTANCE;
  }

  @Override
  StopOutcome onStopComplete(ThreadController ctx) {
    return new StopOutcome(null, this);
  }

  @Override
  WorkerDecision onWorkerCheck(ThreadController ctx) {
    return WorkerDecision.WAIT;
  }
}
