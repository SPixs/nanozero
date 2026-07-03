package org.nanozero.engine.internal;

import org.nanozero.board.GameState;
import org.nanozero.engine.EngineState;
import org.nanozero.engine.SearchBudget;
import org.nanozero.engine.SearchResult;

/**
 * Base d'un dispatcher polymorphique encodant la machine à états d'{@code Engine} (cf. SPEC §4.3,
 * §12 phase 10.5).
 *
 * <p>Refactor anticipation phase 11 (cf. {@code AGENTS.md} engine, phase courante 10.5). Élimine la
 * classe de bugs « missed-case in conditional state check » dans {@link ThreadController} via
 * dispatch via méthodes abstraites — l'omission d'un événement sur un état nouveau provoque un
 * échec de compilation, pas un comportement à l'exécution non spécifié.
 *
 * <p>Choix : <em>OPTION X — abstract sealed class + 6 singletons GoF</em>. Les sous-classes
 * permises sont les 6 états du SPEC §4.3 ({@link IdleState}, {@link SearchingState}, {@link
 * PonderingState}, {@link StoppingState}, {@link DoneState}, {@link ClosedState}). Chaque
 * sous-classe est un singleton — instance unique {@code INSTANCE}, constructor privé.
 *
 * <p><strong>Méthodes événement</strong> :
 *
 * <ul>
 *   <li>{@link #onSubmit} — caller appelle {@code startSearch}.
 *   <li>{@link #onStopBegin} — caller appelle {@code stop} (transition pré-attente).
 *   <li>{@link #stopShouldBlock} — pendant {@code stop}, faut-il continuer à attendre ?
 *   <li>{@link #onStopComplete} — après attente, extraction du {@link SearchResult} + transition.
 *   <li>{@link #onCloseBegin} — caller appelle {@code close} (transition pré-attente).
 *   <li>{@link #closeShouldBlock} — pendant {@code close}, faut-il continuer à attendre ?
 *   <li>{@link #onCloseFinalize} — après attente, transition finale vers {@code CLOSED}.
 *   <li>{@link #awaitDoneShouldBlock} — pendant {@code awaitDone}, faut-il continuer à attendre ?
 *   <li>{@link #onWorkerCheck} — décision du worker thread (WAIT / EXECUTE_SEARCH / ACK_STOP /
 *       EXIT_THREAD).
 *   <li>{@link #keepExecutingSearch} — pendant {@code executeSearch}, doit-on continuer ?
 *   <li>{@link #onWorkerSearchCompleted} — worker termine la recherche, transition vers {@code
 *       DONE} (sauf race {@code CLOSED} déjà set).
 *   <li>{@link #publicValue} — mapping vers l'enum public {@link EngineState}.
 * </ul>
 *
 * <p>Les transitions sont retournées par valeur (style fonctionnel) ; {@link ThreadController}
 * applique l'affectation à son champ {@code currentState}. Les états ne mutent pas {@code
 * currentState} eux-mêmes — la mutation est centralisée pour clarifier la coordination
 * cross-thread.
 *
 * <p><strong>Coordination cross-thread</strong> : les méthodes événement sont appelées par {@code
 * ThreadController} <em>sous le verrou</em> ({@code synchronized(lock)}) sauf indication contraire
 * (lock-free read). La {@code volatile} de {@code currentState}, les {@code wait}/{@code
 * notifyAll}, et la barrière happens-before associée sont orthogonaux au pattern de dispatch.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
abstract sealed class EngineStateBehavior
    permits IdleState, SearchingState, PonderingState, StoppingState, DoneState, ClosedState {

  /** Décision retournée par {@link #onWorkerCheck} pour orienter la boucle du worker thread. */
  enum WorkerDecision {
    /** État passif (IDLE / DONE) : worker doit attendre via {@code lock.wait()}. */
    WAIT,
    /** État actif SEARCHING (et plus tard PONDERING) : worker exécute la boucle MCTS. */
    EXECUTE_SEARCH,
    /**
     * Race STOPPING avant exécution effective : worker doit acquitter en transitionnant vers DONE
     * sans avoir effectué de simulation.
     */
    ACK_STOP,
    /** CLOSED : worker doit sortir de sa boucle. */
    EXIT_THREAD
  }

  /**
   * Sortie de {@link #onStopComplete} : couple ({@link SearchResult} retourné au caller, prochain
   * état). Le résultat peut être {@code null} (cas IDLE entrant : le caller mappera vers son propre
   * EMPTY_RESULT).
   */
  record StopOutcome(SearchResult result, EngineStateBehavior nextState) {}

  /** Mapping vers l'enum public {@link EngineState} (cf. SPEC §4.3). */
  abstract EngineState publicValue();

  // -------------------------------------------------------------------------------------------
  // Événements caller (appelés sous lock)
  // -------------------------------------------------------------------------------------------

  /**
   * Caller a appelé {@code startSearch}. Par défaut : refus avec {@link IllegalStateException}
   * (seul {@link IdleState} accepte). Lorsque accepté, l'implémentation prépare {@code ctx.tree}
   * (via {@link ThreadController#tryReroot} ou fresh tree), wrap le budget dans un {@link
   * MutableSearchBudget}, réinitialise {@code currentSnapshot} et retourne {@link
   * SearchingState#INSTANCE}.
   */
  EngineStateBehavior onSubmit(ThreadController ctx, GameState position, SearchBudget budget) {
    throw new IllegalStateException("submitSearch requires IDLE state, got " + publicValue());
  }

  /**
   * Caller a appelé {@code startPonder}. Par défaut : refus avec {@link IllegalStateException}
   * (seul {@link IdleState} accepte, cf. SPEC §12 phase 11). Lorsque accepté, l'implémentation
   * clone {@code position}, applique {@code predictedOpponentMove}, prépare {@code ctx.tree} (via
   * tryReroot ou fresh) sur la position pondérée, wrap un {@link MutableSearchBudget} initialisé
   * sur {@link SearchBudget#UNLIMITED}, et retourne {@link PonderingState#INSTANCE}.
   */
  EngineStateBehavior onSubmitPonder(
      ThreadController ctx, GameState position, int predictedOpponentMove) {
    throw new IllegalStateException("startPonder requires IDLE state, got " + publicValue());
  }

  /**
   * Caller a appelé {@code ponderhit}. Par défaut : refus avec {@link IllegalStateException} (seul
   * {@link PonderingState} accepte, cf. SPEC §12 phase 11). Lorsque accepté, l'implémentation
   * appelle {@link MutableSearchBudget#replace} sur {@code ctx.pendingBudget} pour basculer du
   * budget {@code UNLIMITED} initial vers {@code realBudget}, sans interrompre le worker. Retourne
   * {@link SearchingState#INSTANCE}.
   */
  EngineStateBehavior onPonderhit(ThreadController ctx, SearchBudget realBudget) {
    throw new IllegalStateException(
        "ponderhit only valid in PONDERING state, got " + publicValue());
  }

  /**
   * Caller a appelé {@code stop}. Phase 1/3 (pré-attente) : transition d'état avant la boucle
   * {@code wait}. Selon l'état :
   *
   * <ul>
   *   <li>{@link IdleState} / {@link DoneState} : self (pas de transition).
   *   <li>{@link SearchingState} : transition vers {@link StoppingState} (signal au worker) +
   *       notifyAll pour réveiller le worker s'il attend.
   *   <li>{@link StoppingState} : self (déjà en phase d'arrêt).
   *   <li>{@link ClosedState} : throw.
   * </ul>
   */
  EngineStateBehavior onStopBegin(ThreadController ctx) {
    return this;
  }

  /**
   * Phase 2/3 de {@code stop} : prédicat lock-respecté (relu après chaque {@code wait}). Vrai si
   * {@link ThreadController} doit continuer à attendre une transition. Faux pour IDLE/DONE/CLOSED ;
   * vrai pour SEARCHING/STOPPING/PONDERING (en attente de la transition vers DONE par le worker).
   */
  boolean stopShouldBlock() {
    return false;
  }

  /**
   * Phase 3/3 de {@code stop} : extraction du {@link SearchResult} après sortie de la boucle
   * d'attente. Selon l'état :
   *
   * <ul>
   *   <li>{@link IdleState} : ({@code null}, IDLE) — caller mappera null vers EMPTY_RESULT.
   *   <li>{@link DoneState} : (snapshot, IDLE) + clear pendingPosition/pendingBudget.
   *   <li>{@link ClosedState} : never reached (throw caught en phase 1).
   * </ul>
   */
  StopOutcome onStopComplete(ThreadController ctx) {
    return new StopOutcome(null, this);
  }

  /**
   * Caller a appelé {@code close}. Phase 1/3 (pré-attente). Selon l'état :
   *
   * <ul>
   *   <li>IDLE / DONE : transition directe vers {@link ClosedState} (pas d'attente nécessaire).
   *   <li>SEARCHING / PONDERING : transition vers {@link StoppingState} + notifyAll.
   *   <li>STOPPING : self (déjà en arrêt).
   *   <li>CLOSED : self (idempotent).
   * </ul>
   */
  EngineStateBehavior onCloseBegin(ThreadController ctx) {
    return ClosedState.INSTANCE;
  }

  /**
   * Phase 2/3 de {@code close} : prédicat. Vrai pour SEARCHING/STOPPING/PONDERING (worker doit
   * acquitter en passant à DONE) ; faux pour IDLE/DONE/CLOSED.
   */
  boolean closeShouldBlock() {
    return false;
  }

  /**
   * Phase 3/3 de {@code close} : finalisation après wait. Transitionne vers {@link ClosedState} et
   * libère les références mutables ({@code pendingBudget}, {@code tree}). Idempotent sur
   * ClosedState.
   */
  EngineStateBehavior onCloseFinalize(ThreadController ctx) {
    ctx.pendingBudget = null;
    ctx.tree = null;
    return ClosedState.INSTANCE;
  }

  /**
   * Prédicat pour la boucle d'attente d'{@code awaitDone} (sémantique {@code searchSync} = submit +
   * wait + retrieve). Vrai pour SEARCHING/STOPPING/PONDERING ; faux pour IDLE/DONE/CLOSED.
   */
  boolean awaitDoneShouldBlock() {
    return false;
  }

  // -------------------------------------------------------------------------------------------
  // Événements worker (appelés depuis le worker thread, sous lock pour onWorkerCheck/Completed,
  // lock-free pour keepExecutingSearch)
  // -------------------------------------------------------------------------------------------

  /**
   * Décision du worker thread sous lock (cf. {@link WorkerDecision}). Toutes les sous-classes
   * spécifient leur dispatch — pas de défaut pour forcer l'exhaustivité.
   */
  abstract WorkerDecision onWorkerCheck(ThreadController ctx);

  /**
   * Prédicat lock-free relu entre simulations dans {@code executeSearch}. Vrai si le worker doit
   * continuer à boucler. Faux dès que STOPPING ou CLOSED. Override sur SearchingState
   * (PonderingState aussi à terme phase 11).
   */
  boolean keepExecutingSearch() {
    return false;
  }

  /**
   * Worker a fini sa recherche (budget épuisé ou STOPPING/CLOSED détecté). Transition par défaut
   * vers {@link DoneState}. Override sur ClosedState pour préserver CLOSED en cas de race
   * close-during-search.
   */
  EngineStateBehavior onWorkerSearchCompleted(ThreadController ctx) {
    return DoneState.INSTANCE;
  }
}
