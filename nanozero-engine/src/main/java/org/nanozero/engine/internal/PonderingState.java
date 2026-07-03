package org.nanozero.engine.internal;

import org.nanozero.engine.EngineState;
import org.nanozero.engine.SearchBudget;

/**
 * État {@link EngineState#PONDERING} (cf. SPEC §4.3, §12 phase 11) — réflexion sur le temps de
 * l'adversaire. Le worker boucle MCTS sur {@code ctx.tree} avec un budget {@link
 * MutableSearchBudget} initialisé sur {@link SearchBudget#UNLIMITED}. Singleton ({@link
 * #INSTANCE}).
 *
 * <p>Comportement quasi-identique à {@link SearchingState} (mêmes décisions de stop / close /
 * worker), à deux différences près :
 *
 * <ul>
 *   <li>{@code onPonderhit(ctx, realBudget)} : <strong>nouveau</strong>. Mute le délégué du {@code
 *       MutableSearchBudget} courant via {@code ctx.pendingBudget.replace(realBudget)}, sans
 *       interrompre le worker. Au prochain check {@code shouldStop}, le worker observe le nouveau
 *       budget et peut sortir naturellement. Transitionne vers {@link SearchingState}.
 *   <li>{@code onSubmit} : refus (par défaut hérité de {@link EngineStateBehavior} — message « got
 *       PONDERING » indique au caller d'appeler {@code ponderhit} ou {@code stop} d'abord).
 * </ul>
 *
 * <p>L'arbre pondéré est <strong>conservé tel quel</strong> sur {@code onPonderhit} (cf. SPEC §5.5)
 * : la racine du tree pondéré est exactement la position que le caller veut chercher (= la position
 * après le coup adverse prédit, qui s'est avéré joué). Aucun re-rooting nécessaire.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
final class PonderingState extends EngineStateBehavior {

  static final PonderingState INSTANCE = new PonderingState();

  private PonderingState() {}

  @Override
  EngineState publicValue() {
    return EngineState.PONDERING;
  }

  @Override
  EngineStateBehavior onPonderhit(ThreadController ctx, SearchBudget realBudget) {
    ctx.pendingBudget.replace(realBudget);
    return SearchingState.INSTANCE;
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
