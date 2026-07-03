package org.nanozero.engine.internal;

import java.util.Arrays;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import org.nanozero.board.GameState;
import org.nanozero.engine.EngineConfig;
import org.nanozero.engine.EngineState;
import org.nanozero.engine.SearchBudget;
import org.nanozero.engine.SearchResult;
import org.nanozero.engine.internal.EngineStateBehavior.StopOutcome;
import org.nanozero.engine.internal.EngineStateBehavior.WorkerDecision;

/**
 * Thread interne de contrôle servant l'API asynchrone d'{@code Engine} (cf. SPEC §7.2, §4.4, §12
 * phase 10, §12 phase 11).
 *
 * <p>Spawn EAGER : un thread daemon dédié, créé au constructor, attend sur un objet de lock jusqu'à
 * ce qu'une recherche soit soumise. Exécute la boucle MCTS avec check d'état inter-simulations
 * (réactivité bornée à ~10-20 ms = durée d'une simulation typique).
 *
 * <p><strong>Machine à états</strong> (cf. SPEC §4.3) :
 *
 * <pre>
 *   IDLE      ── submitSearch ──▶ SEARCHING
 *   IDLE      ── submitPonder ──▶ PONDERING
 *   SEARCHING ── budget exhausted ─▶ DONE
 *   SEARCHING ── stop() ──────────▶ STOPPING
 *   PONDERING ── ponderhit() ─────▶ SEARCHING (avec nouveau budget)
 *   PONDERING ── stop() ──────────▶ STOPPING
 *   STOPPING  ── sim courante finie ─▶ DONE
 *   DONE      ── stop() retourne result, transition ─▶ IDLE
 *   ANY       ── close() ────────▶ CLOSED
 * </pre>
 *
 * <p><strong>Phase 10.5 — refactor State pattern</strong> : la machine à états est encodée par
 * {@link EngineStateBehavior} (abstract sealed class) + 6 singletons {@link IdleState}, {@link
 * SearchingState}, {@link PonderingState}, {@link StoppingState}, {@link DoneState}, {@link
 * ClosedState}.
 *
 * <p><strong>Phase 11 — Ponder + tree reuse</strong> : ajout de {@link #submitPonder} et {@link
 * #ponderhit}. Le budget courant est toujours enveloppé dans un {@link MutableSearchBudget} pour
 * permettre à {@code ponderhit} de muter le délégué (de {@code UNLIMITED} à un budget réel) sans
 * interrompre le worker. Le re-rooting effectif (0/1/2 plies) est tenté via {@link #tryReroot} au
 * moment du {@code submit*} pour économiser le travail des recherches précédentes.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
public final class ThreadController implements AutoCloseable {

  private static final AtomicInteger COUNTER = new AtomicInteger();

  /** Snapshot tous les K simulations pour limiter le coût de buildSnapshot. */
  private static final int SNAPSHOT_INTERVAL = 16;

  /**
   * Profondeur maximale de la PV extraite par descente max-visits (cf. SPEC §3.3 I-Result-2, §14
   * annexe). 64 plies = 32 coups complets ; au-delà, le coût d'extraction et la mémoire occupée par
   * {@code int[]} ne se justifient pas. Tronqué silencieusement si l'arbre est plus profond.
   */
  static final int MAX_PV = 64;

  /**
   * Snapshot vide réutilisable. Visibilité {@code package-private} pour que {@link IdleState}
   * puisse réinitialiser {@code currentSnapshot} sur {@code submitSearch} (phase 10.5 state
   * pattern).
   */
  static final SearchResult EMPTY_RESULT =
      new SearchResult(0, new int[0], Float.NaN, 0, 0L, new int[0], new int[0], false);

  private final SearcherCore searcher;
  private final Thread workerThread;
  private final Object lock = new Object();

  /**
   * (v1.2.0) Factory pour produire des {@link SearcherCore} per-thread en mode batched. {@code
   * null} si mode mono-thread strict ({@code config.searchThreads() == 1}, comportement v1.1.2).
   */
  private final Supplier<SearcherCore> searcherFactory;

  /**
   * (v1.2.0) Configuration courante (lue pour {@code searchThreads}, {@code batchSize}, {@code
   * virtualLoss}). En mode mono-thread, peut être {@code null} (compat constructor 1-arg).
   */
  private final EngineConfig config;

  /**
   * (v1.2.0) ThreadLocal de {@link SearcherCore} : chaque thread du pool batched alloue son propre
   * searcher la première fois qu'il en a besoin, puis le réutilise pour toutes ses sims. {@code
   * null} si mode mono-thread.
   */
  private final ThreadLocal<SearcherCore> threadLocalSearcher;

  /**
   * (v1.2.0) ThreadLocal de {@link GameState} working copy : chaque thread du pool alloue son
   * propre state la première fois et le réutilise. À chaque sim, on copyFrom(rootState) pour reset
   * (équivalent fonctionnel à un nouveau copy() mais zéro-alloc en hot path).
   */
  private final ThreadLocal<GameState> threadLocalState;

  /**
   * État courant. {@code volatile} pour permettre lecture sans verrou (state(), worker
   * keepExecutingSearch). Mutations exclusivement sous {@code synchronized(lock)}.
   */
  private volatile EngineStateBehavior currentState = IdleState.INSTANCE;

  /**
   * Budget courant, toujours enveloppé dans un {@link MutableSearchBudget} pour permettre le swap
   * via {@code ponderhit} (cf. phase 11). Set sous lock par {@link IdleState}, lu par le worker.
   * Visibilité {@code package-private} pour mutation par les classes {@link EngineStateBehavior}.
   */
  MutableSearchBudget pendingBudget;

  /**
   * Snapshot lock-free du résultat courant, mis à jour par le worker. Visibilité {@code
   * package-private} pour mutation/lecture par les classes {@link EngineStateBehavior}.
   */
  volatile SearchResult currentSnapshot = EMPTY_RESULT;

  /**
   * Arbre courant. Survit entre soumissions successives pour permettre le re-rooting effectif (cf.
   * phase 11, {@link #tryReroot}). Visibilité {@code package-private} pour mutation par les classes
   * {@link EngineStateBehavior}. Accédé exclusivement sous lock par {@code IdleState} et par le
   * worker pendant {@code executeSearch} (mutuellement exclusifs par construction de la machine à
   * états).
   */
  SearchTree tree;

  /**
   * Construit un {@code ThreadController} en mode mono-thread (v1.1.2 compat). Démarre le worker
   * daemon en mode IDLE.
   *
   * @param searcher cœur MCTS pré-configuré, non null
   */
  public ThreadController(SearcherCore searcher) {
    this(searcher, null, null);
  }

  /**
   * (v1.2.0) Construit un {@code ThreadController} avec support du mode batched. Si {@code
   * config.searchThreads() > 1}, l'exécution de la recherche utilise {@code searcherFactory} pour
   * allouer un {@link SearcherCore} par thread du pool, chacun travaillant sur une copie
   * indépendante du root {@link GameState}.
   *
   * <p>En mode mono-thread (defaults : {@code searchThreads=1}), {@code searcherFactory} et {@code
   * config} sont ignorés ; le comportement est strictement v1.1.2.
   *
   * @param searcher cœur MCTS principal (utilisé en mode mono-thread)
   * @param searcherFactory factory pour produire des {@link SearcherCore} per-thread en mode
   *     batched. Peut être {@code null} en mode mono-thread.
   * @param config configuration de l'engine. Peut être {@code null} en mode mono-thread.
   */
  public ThreadController(
      SearcherCore searcher, Supplier<SearcherCore> searcherFactory, EngineConfig config) {
    this.searcher = searcher;
    this.searcherFactory = searcherFactory;
    this.config = config;
    if (searcherFactory != null) {
      this.threadLocalSearcher = ThreadLocal.withInitial(searcherFactory);
      this.threadLocalState = ThreadLocal.withInitial(GameState::new);
    } else {
      this.threadLocalSearcher = null;
      this.threadLocalState = null;
    }
    this.workerThread =
        new Thread(this::workerLoop, "nanozero-engine-worker-" + COUNTER.incrementAndGet());
    this.workerThread.setDaemon(true);
    this.workerThread.start();
  }

  // -------------------------------------------------------------------------------------------
  // API package-private utilisée par Engine
  // -------------------------------------------------------------------------------------------

  /** État courant (lecture {@code volatile}, sans verrou). */
  public EngineState state() {
    return currentState.publicValue();
  }

  /**
   * Soumet une recherche au worker. Méthode non bloquante : le worker prend la requête et démarre
   * la boucle MCTS de façon asynchrone.
   *
   * @throws IllegalStateException si l'état n'est pas IDLE (rejet via {@link
   *     EngineStateBehavior#onSubmit})
   */
  public void submitSearch(GameState position, SearchBudget budget) {
    synchronized (lock) {
      currentState = currentState.onSubmit(this, position, budget);
      lock.notifyAll();
    }
  }

  /**
   * Soumet une recherche en mode ponder (cf. SPEC §12 phase 11). Position de recherche = clone de
   * {@code position} puis {@code applyMove(predictedOpponentMove)}. Méthode non bloquante.
   *
   * @throws IllegalStateException si l'état n'est pas IDLE (rejet via {@link
   *     EngineStateBehavior#onSubmitPonder})
   */
  public void submitPonder(GameState position, int predictedOpponentMove) {
    synchronized (lock) {
      currentState = currentState.onSubmitPonder(this, position, predictedOpponentMove);
      lock.notifyAll();
    }
  }

  /**
   * Convertit un ponder en cours en recherche réelle (cf. SPEC §12 phase 11). L'arbre pondéré est
   * conservé tel quel ; le budget interne est muté de {@code UNLIMITED} vers {@code realBudget} via
   * {@link MutableSearchBudget#replace}. Le worker observe le nouveau budget au prochain check
   * {@code shouldStop} et peut alors sortir naturellement (transition vers DONE).
   *
   * @throws IllegalStateException si l'état n'est pas PONDERING (rejet via {@link
   *     EngineStateBehavior#onPonderhit})
   */
  public void ponderhit(SearchBudget realBudget) {
    synchronized (lock) {
      currentState = currentState.onPonderhit(this, realBudget);
      // Pas de notifyAll : le worker n'attend pas, il itère.
    }
  }

  /**
   * Demande l'arrêt de la recherche en cours et bloque jusqu'à arrêt propre. Retourne le {@link
   * SearchResult} final.
   *
   * <p>Sémantique selon état entrant (cf. méthodes {@link EngineStateBehavior#onStopBegin}, {@link
   * EngineStateBehavior#stopShouldBlock}, {@link EngineStateBehavior#onStopComplete}) :
   *
   * <ul>
   *   <li>{@code IDLE} : retourne {@code null}.
   *   <li>{@code SEARCHING} / {@code PONDERING} : signal {@code STOPPING}, attend que le worker
   *       termine la sim courante et transitionne en {@code DONE}, retourne le snapshot final.
   *   <li>{@code DONE} : retourne le snapshot, transitionne en {@code IDLE}.
   *   <li>{@code CLOSED} : lève {@link IllegalStateException}.
   * </ul>
   *
   * @return résultat (ou {@code null} si IDLE entrant)
   * @throws IllegalStateException si l'engine est CLOSED
   */
  public SearchResult stop() {
    synchronized (lock) {
      currentState = currentState.onStopBegin(this);
      lock.notifyAll();
      while (currentState.stopShouldBlock()) {
        try {
          lock.wait();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new IllegalStateException("stop() interrupted", e);
        }
      }
      if (currentState == ClosedState.INSTANCE) {
        throw new IllegalStateException("engine became CLOSED during stop()");
      }
      StopOutcome out = currentState.onStopComplete(this);
      currentState = out.nextState();
      return out.result();
    }
  }

  /**
   * Retourne un snapshot lock-free du résultat courant. Peut être appelé concurremment au worker.
   * Le snapshot est immutable (record), publié atomiquement via {@code volatile}.
   *
   * @return snapshot non null (résultat vide avant la première mise à jour)
   */
  public SearchResult currentBest() {
    return currentSnapshot;
  }

  /**
   * Bloque jusqu'à ce que la recherche en cours atteigne {@link EngineState#DONE} (budget
   * naturellement épuisé) ou {@link EngineState#CLOSED}. Ne signal PAS de stop — utile pour
   * attendre la complétion sans interrompre. Sémantique de {@code searchSync} (= submit + wait +
   * retrieve).
   *
   * <p>Si l'état est {@code IDLE} ou déjà {@code DONE}, retourne immédiatement.
   *
   * @throws IllegalStateException si l'engine est ou devient CLOSED pendant l'attente
   */
  public void awaitDone() {
    synchronized (lock) {
      while (currentState.awaitDoneShouldBlock()) {
        try {
          lock.wait();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new IllegalStateException("awaitDone interrupted", e);
        }
      }
      if (currentState == ClosedState.INSTANCE) {
        throw new IllegalStateException("engine became CLOSED during awaitDone()");
      }
    }
  }

  /**
   * Variante avec timeout de {@link #awaitDone()}. Retourne dès que l'état n'est plus "blocking"
   * (typiquement passage à {@link EngineState#DONE} via {@code notifyAll} du worker), OU lorsque
   * {@code timeoutMs} est écoulé. Utile pour l'InfoReporter UCI qui veut réveiller immédiatement
   * sur fin de recherche mais émettre un snapshot régulier sinon.
   *
   * <p>Si l'engine est déjà {@code IDLE}/{@code DONE} (non-blocking), retourne immédiatement.
   *
   * @param timeoutMs durée maximale d'attente en millisecondes (≥ 0). 0 = poll immédiat.
   * @throws IllegalStateException si l'engine devient {@link EngineState#CLOSED} pendant l'attente.
   * @throws IllegalArgumentException si {@code timeoutMs} est négatif.
   */
  public void awaitDone(long timeoutMs) throws InterruptedException {
    if (timeoutMs < 0) {
      throw new IllegalArgumentException("timeoutMs must be >= 0, got " + timeoutMs);
    }
    long deadlineNanos = System.nanoTime() + timeoutMs * 1_000_000L;
    synchronized (lock) {
      while (currentState.awaitDoneShouldBlock()) {
        long remainingNanos = deadlineNanos - System.nanoTime();
        if (remainingNanos <= 0) {
          return;
        }
        long remainingMs = remainingNanos / 1_000_000L;
        int remainingNanosFrac = (int) (remainingNanos % 1_000_000L);
        // Object.wait(long, int) traite (0, 0) comme wait indéfini : on garantit > 0.
        if (remainingMs == 0 && remainingNanosFrac == 0) {
          return;
        }
        lock.wait(remainingMs, remainingNanosFrac);
      }
      if (currentState == ClosedState.INSTANCE) {
        throw new IllegalStateException("engine became CLOSED during awaitDone(timeout)");
      }
    }
  }

  @Override
  public void close() {
    synchronized (lock) {
      currentState = currentState.onCloseBegin(this);
      lock.notifyAll();
      while (currentState.closeShouldBlock()) {
        try {
          lock.wait();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          break;
        }
      }
      currentState = currentState.onCloseFinalize(this);
      lock.notifyAll();
    }
    // Join le worker (max 1 s).
    try {
      workerThread.join(1000);
      if (workerThread.isAlive()) {
        workerThread.interrupt();
        workerThread.join(1000);
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }

  // -------------------------------------------------------------------------------------------
  // Helpers package-private (utilisés par les classes EngineStateBehavior)
  // -------------------------------------------------------------------------------------------

  /**
   * Clone d'un {@code GameState} via FEN parse. Utilisé par {@link IdleState} pour s'assurer que le
   * {@code rootState} stocké dans le tree est une instance privée que le caller ne peut pas muter
   * accidentellement (cf. SPEC §3.2 invariant I-Tree-1).
   *
   * <p><strong>Note</strong> : le clone perd l'historique (positions précédentes pour la détection
   * de répétition triple). Acceptable en phase 11 — la répétition pendant la recherche est encore
   * détectée via l'historique reconstruit pendant les simulations. Une amélioration future
   * consisterait à exposer un constructeur de copie complet sur {@code GameState}.
   */
  static GameState cloneState(GameState src) {
    return new GameState(src.toFen());
  }

  /**
   * Tente de re-rooter le tree courant sur {@code target} via match par hash Zobrist 0/1/2 plies
   * (cf. SPEC §5.5). Retourne {@code true} si re-rooting réussi (le tree pointe alors sur target),
   * {@code false} sinon (caller doit créer un fresh tree).
   *
   * <ul>
   *   <li>0 plie : si {@code tree.rootState()} a déjà le même hash Zobrist que {@code target},
   *       conserve l'arbre tel quel (pas de mutation, pas de re-root).
   *   <li>1 plie : pour chaque child instancié, applique le coup, compare le hash. Si match :
   *       {@code tree.rerootOnMove}.
   *   <li>2 plies : pour chaque (child, grandchild) instancié, applique les deux coups, compare. Si
   *       match : {@code tree.rerootOnMoves}.
   * </ul>
   *
   * <p>Mute temporairement {@code tree.rootState()} via make-undo (apply/unapply strict). Toute
   * sortie de la méthode laisse {@code rootState} dans son état initial sauf en cas de re-root
   * réussi où {@code SearchTree} le remplace par {@code target}.
   *
   * <p>Probabilité de faux match par collision Zobrist : ~600 comparaisons × 2^-64 ≈ 3 × 10^-17.
   * Négligeable pour MCTS — au pire un sous-arbre incorrect promu, MCTS recalibre rapidement.
   */
  boolean tryReroot(GameState target) {
    if (tree == null) {
      return false;
    }
    Node root = tree.root();
    if (!root.expanded || root.terminal) {
      return false;
    }
    if (root.childMoves == null || root.children == null) {
      return false;
    }
    GameState rs = tree.rootState();
    long targetHash = target.currentPosition().zobristHash();

    // 0 plie : déjà à la cible.
    if (rs.currentPosition().zobristHash() == targetHash) {
      return true;
    }

    for (int i = 0; i < root.childMoves.length; i++) {
      Node child = root.children[i];
      if (child == null) {
        continue;
      }
      int firstMove = root.childMoves[i];
      rs.applyMove(firstMove);
      if (rs.currentPosition().zobristHash() == targetHash) {
        rs.unapplyLastMove();
        tree.rerootOnMove(firstMove, target);
        return true;
      }
      // 2 plies via ce child.
      if (child.expanded && !child.terminal && child.childMoves != null && child.children != null) {
        for (int j = 0; j < child.childMoves.length; j++) {
          Node grand = child.children[j];
          if (grand == null) {
            continue;
          }
          int secondMove = child.childMoves[j];
          rs.applyMove(secondMove);
          if (rs.currentPosition().zobristHash() == targetHash) {
            rs.unapplyLastMove();
            rs.unapplyLastMove();
            tree.rerootOnMoves(firstMove, secondMove, target);
            return true;
          }
          rs.unapplyLastMove();
        }
      }
      rs.unapplyLastMove();
    }
    return false;
  }

  // -------------------------------------------------------------------------------------------
  // Worker loop
  // -------------------------------------------------------------------------------------------

  private void workerLoop() {
    while (true) {
      MutableSearchBudget budget;
      synchronized (lock) {
        WorkerDecision decision;
        while ((decision = currentState.onWorkerCheck(this)) == WorkerDecision.WAIT) {
          try {
            lock.wait();
          } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return;
          }
        }
        if (decision == WorkerDecision.EXIT_THREAD) {
          return;
        }
        if (decision == WorkerDecision.ACK_STOP) {
          // Cas race : close() ou stop() a basculé l'état à STOPPING avant que le worker ait eu
          // le temps de démarrer la recherche (entre submit*.notifyAll et le wakeup du worker).
          // Aucun travail à faire ; transition directe vers DONE pour notifier le caller.
          currentState = currentState.onWorkerSearchCompleted(this);
          lock.notifyAll();
          continue;
        }
        // EXECUTE_SEARCH : récupère le budget. Le tree est déjà préparé par IdleState
        // (tryReroot ou fresh) sous lock dans submitSearch / submitPonder.
        budget = pendingBudget;
      }
      executeSearch(budget);
    }
  }

  private void executeSearch(MutableSearchBudget budget) {
    long startNanos = System.nanoTime();
    int simulations;

    // (v1.2.0, ADR-016) Branchement mode : mono-thread strict si searchThreads=1 (= défaut,
    // comportement v1.1.2 bit-pour-bit) ou batched multi-thread.
    if (config != null && config.searchThreads() > 1 && searcherFactory != null) {
      simulations = executeSearchBatched(budget, startNanos);
    } else {
      simulations = executeSearchMono(budget, startNanos);
    }

    // Snapshot final + transition vers DONE (ou CLOSED si race close-during-search).
    long finalElapsed = System.nanoTime() - startNanos;
    currentSnapshot = buildSnapshot(simulations, finalElapsed);
    synchronized (lock) {
      currentState = currentState.onWorkerSearchCompleted(this);
      lock.notifyAll();
    }
  }

  /** Boucle MCTS mono-thread strict (v1.1.2). */
  private int executeSearchMono(MutableSearchBudget budget, long startNanos) {
    int simulations = 0;
    while (currentState.keepExecutingSearch()) {
      long elapsed = System.nanoTime() - startNanos;
      if (budget.shouldStop(simulations, elapsed)) {
        break;
      }
      searcher.runOneSimulation(tree, tree.rootState());
      simulations++;
      if (simulations % SNAPSHOT_INTERVAL == 0) {
        currentSnapshot = buildSnapshot(simulations, System.nanoTime() - startNanos);
      }
    }
    return simulations;
  }

  /**
   * (v1.2.0) Boucle MCTS batched multi-thread (cf. ADR-013/015/016). Spawn N tasks dans un pool
   * temporaire ; chaque task descend des sims avec son propre {@link SearcherCore} (via
   * ThreadLocal) et sa propre copie de {@link GameState}. Tous partagent le {@code tree} (atomics).
   *
   * <p>Le compteur {@code simulations} est partagé via {@link AtomicInteger} ; budget vérifié
   * collectivement (toutes les sims contribuent au même budget).
   */
  private int executeSearchBatched(MutableSearchBudget budget, long startNanos) {
    int n = config.searchThreads();
    AtomicInteger sharedSimulations = new AtomicInteger(0);
    CountDownLatch done = new CountDownLatch(n);
    Thread[] workers = new Thread[n];

    for (int i = 0; i < n; i++) {
      final int wid = i;
      workers[i] =
          new Thread(
              () -> {
                try {
                  // ThreadLocal init : 1 SearcherCore + 1 GameState par thread.
                  SearcherCore localSearcher = threadLocalSearcher.get();
                  GameState localState = threadLocalState.get();
                  while (currentState.keepExecutingSearch()) {
                    long elapsed = System.nanoTime() - startNanos;
                    int simsSoFar = sharedSimulations.get();
                    if (budget.shouldStop(simsSoFar, elapsed)) {
                      break;
                    }
                    // Reset state à rootState (zéro-alloc) avant chaque sim.
                    localState.copyFrom(tree.rootState());
                    localSearcher.runOneSimulation(tree, localState);
                    int s = sharedSimulations.incrementAndGet();
                    // Snapshot périodique (seul un thread écrit ; race tolérée — snapshot est
                    // best-effort, pas critique).
                    if (s % SNAPSHOT_INTERVAL == 0) {
                      currentSnapshot = buildSnapshot(s, System.nanoTime() - startNanos);
                    }
                  }
                } finally {
                  done.countDown();
                }
              },
              "nanozero-search-" + wid);
      workers[i].setDaemon(true);
      workers[i].start();
    }

    try {
      done.await();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
    return sharedSimulations.get();
  }

  /**
   * Construit un {@link SearchResult} immutable à partir de l'arbre courant. {@code value} est
   * calculé du POV parent à la racine (négation cf. SPEC §5.2 convention zero-sum). {@code
   * principalVariation} est extraite par descente max-visits via {@link
   * #extractPrincipalVariation}.
   */
  private SearchResult buildSnapshot(int simulations, long elapsedNanos) {
    if (tree == null) {
      return EMPTY_RESULT;
    }
    Node root = tree.root();
    if (simulations == 0 || !root.expanded || root.childMoves == null) {
      return new SearchResult(
          0, new int[0], Float.NaN, 0, elapsedNanos, new int[0], new int[0], false);
    }

    int n = root.childMoves.length;
    int[] childVisits = new int[n];
    int bestIdx = 0;
    int bestVisits = -1;
    for (int i = 0; i < n; i++) {
      Node child = root.children[i];
      // (v1.2.0) Lecture atomique via .get() — overhead négligeable.
      int v = child == null ? 0 : child.totalVisits.get();
      childVisits[i] = v;
      if (v > bestVisits) {
        bestVisits = v;
        bestIdx = i;
      }
    }

    int bestMove = n == 0 ? 0 : root.childMoves[bestIdx];

    float value;
    Node best = (n == 0) ? null : root.children[bestIdx];
    int bestVisitsForValue = (best == null) ? 0 : best.totalVisits.get();
    if (bestVisitsForValue == 0) {
      value = Float.NaN;
    } else {
      // Négation pour POV parent (cf. SPEC §5.2 convention zero-sum).
      value = -best.totalValueSum.get() / bestVisitsForValue;
      // (v1.3.0 fix M5) W (totalValueSum) et N (totalVisits) sont des atomics DISTINCTS lus à des
      // instants différents ; sous backup concurrent (mode batched), le ratio peut transitoirement
      // sortir de [-1, 1] (N lu avant un add de W). Clamp défensif : la value exposée via
      // currentBest() reste dans le domaine zero-sum valide. Snapshot best-effort (non corruptif).
      value = Math.max(-1f, Math.min(1f, value));
    }

    int[] pv = n == 0 ? new int[0] : extractPrincipalVariation(root, bestIdx, bestMove);
    int[] childMovesCopy = Arrays.copyOf(root.childMoves, n);
    boolean terminated = simulations > 0;

    return new SearchResult(
        bestMove, pv, value, simulations, elapsedNanos, childVisits, childMovesCopy, terminated);
  }

  /**
   * Extrait la principal variation par descente récursive max-visits depuis {@code root} (cf. SPEC
   * §3.3 I-Result-2, §14 annexe {@code buildSearchResult}). Sémantique stricte : à chaque niveau,
   * on suit le child instancié de plus grand {@code totalVisits} (tie-break sur l'ordre
   * d'apparition dans {@code childMoves}).
   *
   * <p>Cette descente n'est PAS une « ligne principale alpha-beta » au sens des moteurs classiques
   * : elle suit le coup le plus exploré, pas celui de plus haut score Q. La cohérence avec {@code
   * bestMove} (lui-même argmax visites au root, cf. ADR-007) est garantie par construction.
   *
   * <p>Conditions d'arrêt :
   *
   * <ul>
   *   <li>Profondeur {@link #MAX_PV} atteinte.
   *   <li>Nœud courant non expandé (pas de descente possible).
   *   <li>Nœud courant terminal (mat / nul, pas de coup à jouer).
   *   <li>Aucun child instancié dans le nœud courant (cas pathologique : le seul child apparu dans
   *       childMoves n'a jamais été visité).
   * </ul>
   *
   * <p>Retourne un tableau tronqué à la profondeur réellement extraite, garantissant {@code pv[0]
   * == bestMove}.
   */
  static int[] extractPrincipalVariation(Node root, int bestIdx, int bestMove) {
    int[] pv = new int[MAX_PV];
    pv[0] = bestMove;
    int length = 1;

    Node current = root.children[bestIdx];
    while (current != null
        && current.expanded
        && !current.terminal
        && length < MAX_PV
        && current.childMoves != null
        && current.children != null) {
      int bestNextIdx = -1;
      int bestNextVisits = -1;
      for (int i = 0; i < current.childMoves.length; i++) {
        Node child = current.children[i];
        if (child == null) {
          continue;
        }
        int visits = child.totalVisits.get();
        if (visits > bestNextVisits) {
          bestNextVisits = visits;
          bestNextIdx = i;
        }
      }
      if (bestNextIdx == -1) {
        break;
      }
      pv[length++] = current.childMoves[bestNextIdx];
      current = current.children[bestNextIdx];
    }

    return Arrays.copyOf(pv, length);
  }

  // -------------------------------------------------------------------------------------------
  // Accesseur réservé aux tests
  // -------------------------------------------------------------------------------------------

  /** Référence au worker thread, exposée aux tests pour vérifier qu'il est daemon. */
  public Thread workerThreadForTest() {
    return workerThread;
  }
}
