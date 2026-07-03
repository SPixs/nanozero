package org.nanozero.engine;

import java.util.Objects;
import org.nanozero.board.GameState;
import org.nanozero.board.MoveGen;
import org.nanozero.engine.internal.NNEvalCache;
import org.nanozero.engine.internal.NetworkProviderImpl;
import org.nanozero.engine.internal.SearcherCore;
import org.nanozero.engine.internal.ThreadController;
import org.nanozero.engine.internal.batched.SearchPool;
import org.nanozero.nn.Network;

/**
 * Point d'entrée principal du module {@code nanozero-engine} (cf. SPEC §4.2, §4.3, §4.4).
 *
 * <p>Encapsule un {@link Network} pré-chargé, un {@link EngineConfig} d'hyperparamètres et un
 * {@code ThreadController} interne (worker thread daemon eager spawn). Expose une API
 * <strong>asynchrone</strong> ({@link #startSearch}, {@link #stop}, {@link #currentBest}) plus une
 * méthode de commodité synchrone {@link #searchSync}.
 *
 * <p><strong>Threading</strong> :
 *
 * <ul>
 *   <li>Toutes les méthodes publiques sont thread-safe (synchronisation interne via le {@code
 *       ThreadController}).
 *   <li>{@code startSearch} retourne immédiatement (la recherche s'exécute dans le worker thread).
 *   <li>{@code stop} bloque jusqu'à arrêt propre.
 *   <li>{@code currentBest} est lock-free (volatile read).
 *   <li>Le worker thread est daemon ({@code setDaemon(true)}) : oublier {@code close()} n'empêche
 *       pas la terminaison de la JVM.
 *   <li>{@code close()} interrompt la recherche en cours et joine le worker (timeout 1 s).
 * </ul>
 *
 * <p><strong>Machine à états</strong> (cf. SPEC §4.3) : transitions gérées par {@code
 * ThreadController} :
 *
 * <pre>
 *   IDLE      ── startSearch ─▶ SEARCHING
 *   IDLE      ── startPonder ─▶ PONDERING
 *   SEARCHING ── budget exhausted ─▶ DONE
 *   SEARCHING ── stop() ───────▶ STOPPING
 *   PONDERING ── ponderhit() ──▶ SEARCHING (avec nouveau budget)
 *   PONDERING ── stop() ───────▶ STOPPING
 *   STOPPING  ── sim courante finie ─▶ DONE
 *   DONE      ── stop() retourne result ─▶ IDLE
 *   ANY       ── close() ──────▶ CLOSED
 * </pre>
 *
 * <p><strong>Re-rooting</strong> : tentative automatique au début de chaque {@code startSearch} /
 * {@code startPonder} via match par hash Zobrist 0/1/2 plies depuis la racine courante (cf. SPEC
 * §3.2, §5.5). Si la nouvelle position est atteignable, le sous-arbre correspondant est promu en
 * nouvelle racine (réutilisation des statistiques Q/N). Sinon, fresh tree.
 *
 * <p><strong>SearchResult</strong> : {@code bestMove} = argmax visites racine (cf. ADR-007), {@code
 * value} = Q du POV parent à la racine (négation cf. SPEC §5.2 convention zero-sum), {@code
 * principalVariation} = descente max-visits depuis le bestChild bornée à 64 plies (cf. SPEC §3.3
 * I-Result-2, §14 annexe).
 *
 * <p>Usage UCI typique :
 *
 * <pre>{@code
 * try (Engine engine = new Engine(network, EngineConfig.defaults())) {
 *     engine.startSearch(position, SearchBudget.duration(Duration.ofSeconds(5)));
 *     // ... périodiquement, lire currentBest() pour info UCI ...
 *     SearchResult result = engine.stop();  // bloque jusqu'à arrêt propre
 *     int bestMove = result.bestMove();
 * }
 * }</pre>
 */
public final class Engine implements AutoCloseable {

  private static final SearchResult EMPTY_RESULT =
      new SearchResult(0, new int[0], Float.NaN, 0, 0L, new int[0], new int[0], false);

  private final ThreadController controller;

  /**
   * (v1.2.0-fix2) Pool batched NN inference. {@code null} en mode mono ({@code searchThreads=1}).
   * Démarré au constructor, shutdown à {@link #close()}. Héberge le {@code NNEvalThread} dédié qui
   * batche les soumissions des N threads MCTS et appelle {@code Network.forward(planes, K, output)}
   * une seule fois — c'est cette infra qui transforme le parallélisme CPU en vrai gain Elo.
   */
  private final SearchPool searchPool;

  /**
   * Construit un engine sur un réseau pré-chargé et une configuration. Démarre le worker thread
   * daemon en mode IDLE ; oublier {@link #close()} n'empêche pas la JVM de terminer mais peut
   * laisser une recherche en cours indéfiniment (préfère {@code try-with-resources}).
   *
   * @param network réseau pré-chargé via {@code NetworkLoader} (cf. SPEC-nn §5.2), non null
   * @param config hyperparamètres ({@link EngineConfig#cPuct}, {@link EngineConfig#fpuValue}), non
   *     null
   * @throws NullPointerException si {@code network} ou {@code config} est null
   */
  public Engine(Network network, EngineConfig config) {
    Objects.requireNonNull(network, "network must not be null");
    Objects.requireNonNull(config, "config must not be null");
    NetworkProviderImpl provider = new NetworkProviderImpl(network);
    // (ADR-018) Cache d'évaluation NN OPTIONNEL, créé UNE fois et partagé par TOUS les SearcherCore
    // (primary + factory mode A/B). null ⇒ désactivé (nnCacheSize=0) : comportement bit-pour-bit.
    NNEvalCache cache = config.nnCacheSize() > 0 ? new NNEvalCache(config.nnCacheSize()) : null;
    SearcherCore primarySearcher = new SearcherCore(provider, config, null, cache);
    // (v1.3.0, cf. ADR-017 + SPEC §16) Critère d'activation du path batched-queue (mode B) :
    // `batchSize > 1 ET searchThreads > 1`. Les DEUX sont requis : le mode B n'a de sens qu'avec
    // plusieurs threads producteurs alimentant le NNEvalThread — avec un seul thread, il
    // sérialiserait ses propres soumissions (submit 1, await, submit 1...) et le NNEvalThread
    // tournerait à vide (fix M4-review : NNEvalThread parasite si batchSize>1 + threads=1).
    // En mode A (`batchSize=1`) : forward direct par thread MCTS — recommandé pour CPU SIMD /
    // ORT CPU EP, gain SPRT +180 Elo confirmé (cf. memory engine-v1.2.0-batched-lessons).
    // En mode B (`batchSize>1 && threads>1`) : `SearchPool` héberge NNEvalThread dédié → 1
    // forward(K) batché — recommandé pour `NetworkOnnx.loadCuda` (amortit l'overhead PCIe).
    // ⚠️ Si caller direct configure `batchSize=1 + threads>1 + CUDA`, `session.run()` concurrent
    // n'est PAS thread-safe en ORT CUDA EP → garde-fou explicite dans NetworkOnnx (CRASH avec
    // IllegalStateException > UB silencieux). Cf. SPEC-engine §16.4.1.
    if (config.batchSize() > 1 && config.searchThreads() > 1) {
      // Mode B : path batched-queue avec NNEvalThread dédié.
      System.err.println(
          "[engine] batchSize="
              + config.batchSize()
              + " > 1 — path NNEvalThread actif. Si le backend Network est CPU SIMD ou ORT CPU EP,"
              + " ce mode SÉRIALISE et est contre-productif (cf. SPEC §16.3, ADR-017)."
              + " batchSize > 1 recommandé UNIQUEMENT avec NetworkOnnx.loadCuda (GPU).");
      this.searchPool = new SearchPool(config.searchThreads(), config.batchSize(), network);
      this.searchPool.start();
      // Factory : chaque SearcherCore per-thread reçoit la queue du pool → submit/await/decode.
      java.util.function.Supplier<SearcherCore> factory =
          () -> new SearcherCore(provider, config, this.searchPool.queue(), cache);
      this.controller = new ThreadController(primarySearcher, factory, config);
    } else {
      this.searchPool = null;
      // (v1.3.0 fix M4-review) Config dégénérée batchSize>1 + searchThreads=1 : batching impossible
      // (1 seul producteur). On retombe en mode A direct SANS allouer de SearchPool/NNEvalThread,
      // et
      // on avertit que le batching demandé est inactif.
      if (config.batchSize() > 1) {
        System.err.println(
            "[engine] batchSize="
                + config.batchSize()
                + " > 1 mais searchThreads=1 : batching INACTIF (pas de NNEvalThread)."
                + " Configurer searchThreads > 1 pour activer le mode B.");
      }
      // Mode A : pas de queue, forward direct par thread MCTS.
      // Si searchThreads=1 : comportement v1.1.2 bit-pour-bit strict.
      // Si searchThreads>1 : N threads en parallèle avec virtual loss wired (Fix #1, cf. ADR-017
      // commit 7c8245d). Gain SPRT +180 Elo vs mono à TC 10+0.1 (CPU Vector API SIMD).
      java.util.function.Supplier<SearcherCore> factory =
          () -> new SearcherCore(provider, config, null, cache);
      this.controller = new ThreadController(primarySearcher, factory, config);
    }
  }

  /**
   * Démarre une recherche asynchrone depuis la position donnée. Retourne immédiatement (typique
   * &lt; 1 ms, dominé par le {@code notify} du worker). La recherche s'exécute dans le thread
   * interne ; le caller récupère le résultat via {@link #stop()} ou consulte la progression via
   * {@link #currentBest()}.
   *
   * <p>Tree reuse effectif (cf. SPEC §3.2, §5.5) : si la nouvelle position est atteignable depuis
   * la racine courante en 0, 1 ou 2 plies (match par hash Zobrist 64-bit), le sous-arbre
   * correspondant est promu en nouvelle racine. Sinon, fresh tree.
   *
   * @param position position d'échecs depuis laquelle chercher, non null
   * @param budget critère d'arrêt, non null
   * @throws NullPointerException si {@code position} ou {@code budget} est null
   * @throws IllegalStateException si l'engine n'est pas {@link EngineState#IDLE}
   */
  public void startSearch(GameState position, SearchBudget budget) {
    Objects.requireNonNull(position, "position must not be null");
    Objects.requireNonNull(budget, "budget must not be null");
    controller.submitSearch(position, budget);
  }

  /**
   * Démarre une recherche en mode ponder (cf. SPEC §4.2, §12 phase 11). L'engine clone {@code
   * position}, applique {@code predictedOpponentMove} sur le clone, et démarre la boucle MCTS sur
   * la position résultante avec un budget interne {@code UNLIMITED}. Le caller devra appeler {@link
   * #ponderhit} (si l'adversaire joue effectivement le coup prédit) ou {@link #stop} (sinon) pour
   * clôturer le ponder.
   *
   * <p>Retourne immédiatement (asynchrone). La recherche s'exécute dans le worker thread interne.
   *
   * @param position position courante <em>avant</em> le coup adverse supposé, non null
   * @param predictedOpponentMove coup que l'adversaire est supposé jouer (encodage Move 16-bit) ;
   *     doit être légal depuis {@code position}
   * @throws NullPointerException si {@code position} est null
   * @throws IllegalArgumentException si {@code predictedOpponentMove} n'est pas un coup légal
   *     depuis {@code position}
   * @throws IllegalStateException si l'engine n'est pas {@link EngineState#IDLE}
   */
  public void startPonder(GameState position, int predictedOpponentMove) {
    Objects.requireNonNull(position, "position must not be null");
    validateLegalMove(position, predictedOpponentMove);
    controller.submitPonder(position, predictedOpponentMove);
  }

  /**
   * Convertit un ponder en cours en recherche réelle (cf. SPEC §4.2, §12 phase 11). À appeler quand
   * l'adversaire a effectivement joué le coup prédit lors du {@link #startPonder}. L'arbre pondéré
   * est conservé tel quel — la racine du tree pondéré est exactement la position que le caller veut
   * chercher. Le budget interne (initialement {@code UNLIMITED}) est remplacé par {@code
   * realBudget} sans interruption du worker.
   *
   * <p>Retourne immédiatement (le worker continue à itérer ; le nouveau budget peut déclencher
   * l'arrêt à la prochaine vérification {@code shouldStop}).
   *
   * @param realBudget budget réel pour la recherche, non null
   * @throws NullPointerException si {@code realBudget} est null
   * @throws IllegalStateException si l'engine n'est pas {@link EngineState#PONDERING}
   */
  public void ponderhit(SearchBudget realBudget) {
    Objects.requireNonNull(realBudget, "realBudget must not be null");
    controller.ponderhit(realBudget);
  }

  /**
   * Vérifie que {@code move} est un coup légal depuis {@code position}. Coût : un appel à {@link
   * MoveGen#generateMoves} + scan linéaire (O(n) sur n ≤ 218 coups légaux). Acceptable à la
   * frontière API publique, hors hot path.
   */
  private static void validateLegalMove(GameState position, int move) {
    int[] buffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = position.generateMoves(buffer, 0);
    for (int i = 0; i < n; i++) {
      if (buffer[i] == move) {
        return;
      }
    }
    throw new IllegalArgumentException(
        "predictedOpponentMove is not a legal move from the given position: 0x"
            + Integer.toHexString(move & 0xFFFF));
  }

  /**
   * Stoppe la recherche en cours et retourne le résultat. Sémantique selon état entrant :
   *
   * <ul>
   *   <li>{@link EngineState#IDLE} : retourne un {@link SearchResult} vide ({@code
   *       simulationsCount=0}, {@code bestMove=0}, {@code value=NaN}). Pas d'exception.
   *   <li>{@link EngineState#SEARCHING} : signal {@code STOPPING}, bloque jusqu'à arrêt propre,
   *       retourne le snapshot final.
   *   <li>{@link EngineState#DONE} (recherche terminée naturellement) : retourne le snapshot,
   *       transitionne en IDLE.
   * </ul>
   *
   * @return résultat (jamais null)
   * @throws IllegalStateException si l'engine est {@link EngineState#CLOSED}
   */
  public SearchResult stop() {
    SearchResult result = controller.stop();
    return result == null ? EMPTY_RESULT : result;
  }

  /**
   * Snapshot lock-free du résultat courant sans interrompre la recherche. Utile pour reporting UCI
   * {@code info} périodique. Thread-safe : peut être appelé concurremment au worker. Le snapshot
   * retourné est immutable et reflète l'état au moment de l'appel (potentiellement stale d'au plus
   * quelques simulations selon l'intervalle de rafraîchissement interne).
   *
   * @return snapshot non null (résultat vide avant la première mise à jour)
   */
  public SearchResult currentBest() {
    return controller.currentBest();
  }

  /** État courant. Lecture {@code volatile} via le {@code ThreadController}. */
  public EngineState state() {
    return controller.state();
  }

  /**
   * Bloque jusqu'à ce que la recherche atteigne {@link EngineState#DONE} (budget épuisé) OU jusqu'à
   * écoulement de {@code timeoutMs}. Réveil immédiat sur transition DONE via {@code notifyAll}
   * interne — pas de polling. Idempotent : si déjà IDLE/DONE, retour immédiat.
   *
   * <p>Cas d'usage principal : InfoReporter UCI ({@code UciSession.infoReporterLoop}) qui veut
   * émettre {@code bestmove} dès que la recherche se termine, sans payer la latence d'un polling
   * périodique (évite ~400ms gaspillés à TC=3+0.03 — cf. bug Phase 12 hotfix-005).
   *
   * @param timeoutMs timeout en ms (≥ 0).
   * @throws InterruptedException si le thread courant est interrompu pendant l'attente. Le caller
   *     (ex. infoReporter thread) est responsable de restaurer l'interrupt flag et de sortir
   *     proprement. Contrairement à {@link #stop()} ou {@link #close()}, on ne wrappe pas en
   *     IllegalStateException : les callers veulent typiquement traiter l'interrupt explicitement.
   * @throws IllegalArgumentException si {@code timeoutMs} est négatif.
   * @throws IllegalStateException si l'engine est ou devient {@link EngineState#CLOSED}.
   */
  public void awaitDone(long timeoutMs) throws InterruptedException {
    controller.awaitDone(timeoutMs);
  }

  /**
   * Méthode de commodité synchrone : {@code startSearch} + attente de la complétion naturelle du
   * budget + {@code stop} pour récupérer le résultat. Sémantique synchrone simple : bloque jusqu'à
   * épuisement du budget, sans interrompre prématurément.
   *
   * <p>Pas dans SPEC §4.2 (qui décrit l'API async pure). Conservée comme commodité pour les tests
   * et les usages simples qui ne ont pas besoin du pattern UCI start/stop.
   *
   * @param position position depuis laquelle chercher, non null
   * @param budget critère d'arrêt, non null
   * @return résultat
   * @throws NullPointerException si {@code position} ou {@code budget} est null
   * @throws IllegalStateException si l'engine n'est pas IDLE ou est CLOSED
   */
  public SearchResult searchSync(GameState position, SearchBudget budget) {
    startSearch(position, budget);
    controller.awaitDone();
    return stop();
  }

  @Override
  public void close() {
    // (v1.3.0) ORDRE CRITIQUE : shutdown searchPool AVANT controller.close().
    // Raison : si on close controller en premier, il interrupt les search threads qui peuvent
    // être bloqués dans future.join() (path batched). future.join() N'EST PAS interruptible
    // → deadlock. En shutdownant searchPool d'abord, NNEvalThread.drainRemainingAsCancelled()
    // complete les futures pending avec InterruptedException → future.join() lève
    // CompletionException → les search threads sortent proprement → controller.close() peut
    // alors joindre les threads MCTS sans deadlock.
    if (searchPool != null) {
      searchPool.shutdown();
    }
    controller.close();
  }
}
