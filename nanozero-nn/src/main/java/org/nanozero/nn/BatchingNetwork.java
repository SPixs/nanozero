package org.nanozero.nn;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.locks.LockSupport;
import java.util.logging.Logger;
import org.nanozero.board.BitboardPlaneEncoder;
import org.nanozero.board.GameState;

/**
 * Décorateur {@link Network} thread-safe qui agrège les forwards de N threads appelants en batchs
 * pour un délégué à chemin batché réel (amendement v1.6.0, ADR-013-nn / ADR-001-worker).
 *
 * <p>Usage cible : worker self-play GPU multi-parties — N threads de jeu (un {@code Engine} Mode A
 * chacun) partagent UNE instance, dont le délégué est un {@link NetworkOnnx#loadCuda(Path) CUDA}.
 * Chaque {@code forward} appelant est bloquant : la requête est mise en queue et complétée quand
 * son batch a été évalué par le délégué.
 *
 * <p><b>Pipeline 3 étages (v2)</b> — le profil JFR du worker GPU en production (2026-06-11, w1080
 * 1080 Ti) a montré que la v1 mono-thread (collecte → forward → dispatch séquentiels) plafonnait le
 * GPU à ~48 % d'utilisation : pendant la moitié CPU du cycle, le GPU dormait, et réciproquement
 * (alternance stricte, un seul batch en vol). La v2 recouvre les trois phases :
 *
 * <pre>
 *   [assembler]  drain de la queue + recopie des planes dans un slot libre
 *   [runner]     delegate.forward(K) — SEUL thread à toucher le délégué (garde-fou CUDA EP intact)
 *   [completer]  recopie des sorties vers les NNOutput des appelants + déblocage (unpark ciblé)
 * </pre>
 *
 * <p>Pendant que le GPU exécute le batch N, l'assembler construit N+1 et le completer livre N−1
 * ({@value #PIPELINE_SLOTS} slots pré-alloués en rotation). Le déblocage des appelants utilise
 * {@link LockSupport} par requête (la v1 réveillait via {@code notifyAll}, mesuré à 11 % du CPU du
 * worker au JFR).
 *
 * <p>Garanties :
 *
 * <ul>
 *   <li><b>Thread-safe côté appelants</b> ({@code forward} / {@code forwardSingle} concurrents) ;
 *       le délégué n'est invoqué QUE par le thread runner (satisfait le garde-fou single-thread du
 *       CUDA EP, cf. {@code NetworkOnnx.cudaForwardInFlight}).
 *   <li><b>Backpressure</b> : queue bornée à {@code 2 × delegate.maxBatch()} positions — les
 *       appelants bloquent à la soumission quand le pipeline est saturé.
 *   <li><b>Propagation d'erreur</b> : si le délégué lève, tous les appelants du batch reçoivent une
 *       {@link IllegalStateException} (cause attachée) ; le pipeline survit.
 *   <li><b>Arrêt propre</b> : {@link #close()} débloque les appelants en attente (exception) et
 *       arrête les trois étages. Tout {@code forward} ultérieur lève.
 *   <li><b>Observabilité</b> : un résumé périodique (K moyen, pos/s, taux d'occupation du délégué,
 *       coûts par étage) est logué toutes les {@value #STATS_EVERY_BATCHES} évaluations de batch.
 * </ul>
 *
 * <p>Note de conception : l'engine possède une mécanique voisine ({@code LeafSubmissionQueue} +
 * {@code NNEvalThread}, Mode B) mais typée sur ses feuilles MCTS internes (virtual loss,
 * poison-pill, futures par feuille) — la réutiliser ici aurait exigé de publiciser des internals
 * engine pour ~100 lignes de logique. Cette classe est l'équivalent générique au niveau {@link
 * Network}, sans dépendance engine.
 */
public final class BatchingNetwork implements Network, AutoCloseable {

  /** Fenêtre de collecte par défaut après la première requête d'un batch (microsecondes). */
  public static final int DEFAULT_FLUSH_MICROS = 1000;

  /**
   * Plafond d'attente du quorum {@code minBatch} (nanosecondes) : si le quorum n'est pas atteint
   * dans ce délai (fin de lot, claims en pause, moins de parties vivantes que le quorum), le batch
   * part quand même — borne la latence, évite tout blocage.
   */
  private static final long MIN_BATCH_HARD_WAIT_NANOS = TimeUnit.MILLISECONDS.toNanos(150);

  /** Période du log de stats pipeline, en batchs complétés. */
  static final int STATS_EVERY_BATCHES = 512;

  /**
   * Slots de batch pré-alloués en rotation dans le pipeline : un par étage (assembler / runner /
   * completer) pour un recouvrement complet sans allocation par batch.
   */
  private static final int PIPELINE_SLOTS = 3;

  private static final Logger LOG = Logger.getLogger(BatchingNetwork.class.getName());

  private static final int FLOATS_PER_POS = NetworkOnnx.INPUT_FLOATS_PER_POS;

  /** Requête en attente d'évaluation. Réutilisée par thread appelant (zéro alloc par forward). */
  private static final class Request {
    float[] planes;
    int count;
    NNOutput output;
    Thread waiter;
    Throwable error;
    volatile boolean done;

    void reset(float[] planes, int count, NNOutput output) {
      this.planes = planes;
      this.count = count;
      this.output = output;
      this.error = null;
      this.waiter = Thread.currentThread();
      this.done = false;
    }

    void awaitDone() {
      boolean interrupted = false;
      while (!done) {
        LockSupport.park(this);
        if (Thread.interrupted()) {
          interrupted = true; // on attend quand même la complétion (résultats en vol)
        }
      }
      if (interrupted) {
        Thread.currentThread().interrupt();
      }
      if (error != null) {
        throw new IllegalStateException("BatchingNetwork delegate forward failed", error);
      }
    }

    /**
     * Marque la requête complétée et réveille SON appelant ({@code unpark} ciblé — pas de moniteur
     * partagé). L'écriture volatile de {@code done} publie {@code error} et les résultats recopiés
     * dans le {@code NNOutput} de l'appelant avant l'appel.
     */
    void complete(Throwable error) {
      this.error = error;
      this.planes = null;
      this.output = null;
      Thread w = this.waiter;
      this.done = true;
      LockSupport.unpark(w);
    }
  }

  /** Lot en vol dans le pipeline : requêtes drainées + planes assemblés + sorties du délégué. */
  private static final class Slot {
    final List<Request> requests;
    final float[] planes;
    final NNOutput output;
    int positions;
    Throwable failure;

    Slot(int capacity) {
      this.requests = new ArrayList<>(capacity);
      this.planes = new float[capacity * FLOATS_PER_POS];
      this.output = new NNOutput(capacity);
    }
  }

  /** Photo des compteurs du pipeline (tests + diagnostic). */
  record PipelineStats(long assembled, long completed, long positions, long runNanos) {}

  private final Network delegate;
  private final boolean closeDelegate;
  private final int capacity;
  private final long flushNanos;
  private final int minBatch;
  private final LinkedBlockingQueue<Request> queue;
  private final ArrayBlockingQueue<Slot> freeSlots = new ArrayBlockingQueue<>(PIPELINE_SLOTS);
  private final ArrayBlockingQueue<Slot> toRun = new ArrayBlockingQueue<>(PIPELINE_SLOTS);
  private final ArrayBlockingQueue<Slot> toComplete = new ArrayBlockingQueue<>(PIPELINE_SLOTS);
  private final Thread assembler;
  private final Thread runner;
  private final Thread completer;
  private final ThreadLocal<Request> localRequest = ThreadLocal.withInitial(Request::new);

  /** Requête multi-positions qui ne tenait plus dans le batch courant (assembler uniquement). */
  private Request pendingCarry;

  private volatile boolean closed;

  // Instrumentation pipeline — un seul thread écrivain par champ (volatile pour la lecture
  // cross-thread du log périodique ; un écrivain unique rend le read-modify-write sûr).
  private volatile long batchesAssembled;
  private volatile long batchesCompleted;
  private volatile long positionsCompleted;
  private volatile long assembleNanos;
  private volatile long slotWaitNanos;
  private volatile long runNanos;
  private volatile long scatterNanos;
  private volatile int statsMinK = Integer.MAX_VALUE;
  private volatile int statsMaxK;

  // État du log périodique (completer uniquement).
  private long lastStatsNanos;
  private long lastStatsBatches;
  private long lastStatsPositions;
  private long lastStatsRunNanos;
  private long lastStatsAssembleNanos;
  private long lastStatsScatterNanos;
  private long lastStatsSlotWaitNanos;

  /**
   * Construit le décorateur avec la fenêtre de collecte par défaut ({@value DEFAULT_FLUSH_MICROS}
   * µs).
   *
   * @param delegate réseau cible (typiquement {@code NetworkOnnx} CUDA EP)
   */
  public BatchingNetwork(Network delegate) {
    this(delegate, DEFAULT_FLUSH_MICROS);
  }

  /**
   * Construit le décorateur.
   *
   * @param delegate réseau cible — ses forwards seront tous exécutés par le thread runner
   * @param flushMicros fenêtre de collecte après la première requête d'un batch : l'assembler
   *     attend au plus ce délai que d'autres requêtes arrivent avant de lancer un batch partiel
   *     ({@code >= 0} ; 0 = batchs opportunistes sans attente)
   */
  public BatchingNetwork(Network delegate, int flushMicros) {
    this(delegate, flushMicros, false);
  }

  /**
   * Construit le décorateur avec contrôle du cycle de vie du délégué.
   *
   * @param delegate réseau cible — ses forwards seront tous exécutés par le thread runner
   * @param flushMicros fenêtre de collecte ({@code >= 0})
   * @param closeDelegate si {@code true}, {@link #close()} ferme aussi le délégué (s'il est {@link
   *     AutoCloseable}) — utile quand le caller ne gère qu'une seule référence (ex. {@code
   *     ModelCache} du worker)
   */
  public BatchingNetwork(Network delegate, int flushMicros, boolean closeDelegate) {
    this(delegate, flushMicros, closeDelegate, 1);
  }

  /**
   * Construit le décorateur avec quorum de batch (auto-réparation de la fusion).
   *
   * @param delegate réseau cible — ses forwards seront tous exécutés par le thread runner
   * @param flushMicros fenêtre de silence (idle-gap, {@code >= 0})
   * @param closeDelegate si {@code true}, {@link #close()} ferme aussi le délégué
   * @param minBatch quorum : un batch n'est expédié à l'expiration de l'idle-gap que s'il contient
   *     au moins ce nombre de positions ; sinon la collecte continue (au plus ~150 ms, cf. {@link
   *     #MIN_BATCH_HARD_WAIT_NANOS}). Rend le régime fusion AUTO-RÉPARANT : mesuré 2026-06-11/12,
   *     la fusion (toutes les parties dans un batch) est métastable — le bruit (fins de partie,
   *     claims, GC) finit par fragmenter les cohortes, qui retombent alors dans le plancher WDDM
   *     (~33 ms/run back-to-back) sans pouvoir re-fusionner. Avec un quorum à ~60 % des parties
   *     concurrentes, toute fragmentation se résorbe en un batch. {@code 1} = désactivé
   *     (comportement historique).
   */
  public BatchingNetwork(Network delegate, int flushMicros, boolean closeDelegate, int minBatch) {
    if (flushMicros < 0) {
      throw new IllegalArgumentException("flushMicros must be >= 0, got " + flushMicros);
    }
    if (minBatch < 1 || minBatch > delegate.maxBatch()) {
      throw new IllegalArgumentException(
          "minBatch " + minBatch + " out of [1, " + delegate.maxBatch() + "]");
    }
    this.minBatch = minBatch;
    this.delegate = delegate;
    this.closeDelegate = closeDelegate;
    this.capacity = delegate.maxBatch();
    this.flushNanos = TimeUnit.MICROSECONDS.toNanos(flushMicros);
    this.queue = new LinkedBlockingQueue<>(2 * capacity);
    for (int i = 0; i < PIPELINE_SLOTS; i++) {
      freeSlots.add(new Slot(capacity));
    }
    this.lastStatsNanos = System.nanoTime();
    this.assembler = new Thread(this::assemblerLoop, "batching-network-assembler");
    this.runner = new Thread(this::runnerLoop, "batching-network-runner");
    this.completer = new Thread(this::completerLoop, "batching-network-completer");
    // Priorités hautes sur le chemin critique : quand un batch se complète, N threads de jeu se
    // réveillent SIMULTANÉMENT (storm) — sans priorité, le runner (qui a besoin de CPU pour
    // lancer chaque kernel) se fait préempter par quantums Windows (~15 ms) et un run() de ~5 ms
    // s'étire à 30-35 ms (mesuré 2026-06-11 sur w1080 ET w3090, indépendant de K et des clocks
    // GPU). MAX_PRIORITY sur le runner inverse la préemption : c'est lui qui passe devant.
    runner.setPriority(Thread.MAX_PRIORITY);
    assembler.setPriority(Thread.NORM_PRIORITY + 2);
    completer.setPriority(Thread.NORM_PRIORITY + 2);
    for (Thread t : new Thread[] {assembler, runner, completer}) {
      t.setDaemon(true);
      t.start();
    }
  }

  @Override
  public int maxBatch() {
    return capacity;
  }

  @Override
  public void forward(float[] planes, int batchSize, NNOutput output) {
    if (batchSize < MIN_BATCH || batchSize > capacity) {
      throw new IllegalArgumentException(
          "batchSize " + batchSize + " out of [" + MIN_BATCH + ", " + capacity + "]");
    }
    if (planes.length < batchSize * FLOATS_PER_POS) {
      throw new IllegalArgumentException(
          "planes length " + planes.length + " < batchSize × " + FLOATS_PER_POS);
    }
    if (output.capacity() < batchSize) {
      throw new IllegalArgumentException(
          "output capacity " + output.capacity() + " < batchSize " + batchSize);
    }
    ensureOpen();
    Request request = localRequest.get();
    request.reset(planes, batchSize, output);
    submit(request);
    request.awaitDone();
  }

  /** Commodité allouante (hors hot path) — route par le chemin batché. */
  @Override
  public NNSingleResult forwardSingle(GameState state) {
    float[] planes = new float[FLOATS_PER_POS];
    state.toPlanes(planes, 0, planeEncoder());
    NNOutput output = new NNOutput(1);
    forward(planes, 1, output);
    return new NNSingleResult(output.logitsOf(0), output.valueOf(0));
  }

  @Override
  public NetworkMetadata metadata() {
    return delegate.metadata();
  }

  @Override
  public BitboardPlaneEncoder planeEncoder() {
    return delegate.planeEncoder();
  }

  /** Photo instantanée des compteurs du pipeline (tests + diagnostic). */
  PipelineStats stats() {
    return new PipelineStats(batchesAssembled, batchesCompleted, positionsCompleted, runNanos);
  }

  /**
   * Arrête les trois étages et débloque les requêtes en attente (elles reçoivent une {@link
   * IllegalStateException}). Idempotent. Ne ferme le délégué que si {@code closeDelegate} a été
   * demandé au constructeur (sinon il reste la propriété du caller).
   */
  @Override
  public void close() {
    closed = true;
    assembler.interrupt();
    runner.interrupt();
    completer.interrupt();
    joinQuietly(assembler);
    joinQuietly(runner);
    joinQuietly(completer);
    IllegalStateException cause = closedException();
    drainAndFail(cause);
    Slot slot;
    while ((slot = toRun.poll()) != null) {
      failRequests(slot.requests, cause);
    }
    while ((slot = toComplete.poll()) != null) {
      failRequests(slot.requests, cause);
    }
    if (closeDelegate && delegate instanceof AutoCloseable c) {
      try {
        c.close();
      } catch (Exception e) {
        throw new IllegalStateException("Failed to close BatchingNetwork delegate", e);
      }
    }
  }

  private static IllegalStateException closedException() {
    return new IllegalStateException("BatchingNetwork closed");
  }

  private static void joinQuietly(Thread thread) {
    try {
      thread.join(TimeUnit.SECONDS.toMillis(5));
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }

  private void ensureOpen() {
    if (closed) {
      throw new IllegalStateException("BatchingNetwork is closed");
    }
  }

  private void submit(Request request) {
    try {
      // offer avec timeout court en boucle plutôt que put() : un close() concurrent doit pouvoir
      // débloquer un appelant coincé sur une queue pleine.
      while (!queue.offer(request, 50, TimeUnit.MILLISECONDS)) {
        ensureOpen();
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException("Interrupted while submitting to BatchingNetwork", e);
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Étage 1 — assembler : drain de la queue + recopie des planes dans un slot libre.
  // ---------------------------------------------------------------------------------------------

  private void assemblerLoop() {
    while (!closed) {
      Slot slot;
      long waitStart = System.nanoTime();
      try {
        // Prend le slot AVANT de drainer : si le pipeline aval est plein, la queue continue de se
        // remplir pendant l'attente et le batch suivant part d'autant plus garni.
        slot = freeSlots.take();
      } catch (InterruptedException e) {
        break;
      }
      slotWaitNanos += System.nanoTime() - waitStart;
      slot.requests.clear();
      slot.failure = null;
      try {
        slot.positions = collectBatch(slot.requests);
      } catch (InterruptedException e) {
        // close() pendant la collecte : les requêtes déjà draînées doivent être débloquées.
        failRequests(slot.requests, closedException());
        if (pendingCarry != null) {
          pendingCarry.complete(closedException());
          pendingCarry = null;
        }
        break;
      }
      long assembleStart = System.nanoTime();
      gather(slot);
      assembleNanos += System.nanoTime() - assembleStart;
      batchesAssembled++;
      if (slot.positions < statsMinK) {
        statsMinK = slot.positions;
      }
      if (slot.positions > statsMaxK) {
        statsMaxK = slot.positions;
      }
      try {
        toRun.put(slot);
      } catch (InterruptedException e) {
        failRequests(slot.requests, closedException());
        break;
      }
    }
  }

  /**
   * Collecte un batch : bloque jusqu'à la première requête, puis agrège ce qui arrive pendant la
   * fenêtre {@code flushNanos} (ou jusqu'à remplir {@code capacity} positions). Une requête
   * multi-positions qui ne tient plus dans le batch courant est portée ({@link #pendingCarry}) et
   * ouvre le batch suivant.
   *
   * @return nombre total de positions collectées
   */
  private int collectBatch(List<Request> batch) throws InterruptedException {
    Request first;
    if (pendingCarry != null) {
      first = pendingCarry;
      pendingCarry = null;
    } else {
      first = queue.take();
    }
    batch.add(first);
    int positions = first.count;
    // Sémantique "idle-gap" (2026-06-11) : le deadline se réarme à CHAQUE requête collectée — le
    // batch part après flushMicros de SILENCE, pas à durée fixe depuis la première requête. Tant
    // que le storm de re-soumissions coule, on agrège (fusion des cohortes automatique, zéro
    // gaspillage de fenêtre) ; dès qu'il s'interrompt, on expédie. C'était la vertu accidentelle
    // de la v1 (le dispatch absorbait le storm avant la collecte suivante) — ici explicite.
    // Quorum minBatch (2026-06-12) : un silence n'expédie le batch que si le quorum est atteint ;
    // sinon on continue d'attendre (au plus MIN_BATCH_HARD_WAIT_NANOS) — re-fusionne les cohortes
    // fragmentées au lieu de les laisser s'installer dans le plancher WDDM.
    long gapDeadline = System.nanoTime() + flushNanos;
    long hardDeadline = System.nanoTime() + MIN_BATCH_HARD_WAIT_NANOS;
    while (positions < capacity) {
      Request next = queue.peek();
      if (next != null) {
        if (positions + next.count > capacity) {
          break; // ne tient plus — laissé pour le batch suivant
        }
        queue.poll();
        batch.add(next);
        positions += next.count;
        gapDeadline = System.nanoTime() + flushNanos;
        continue;
      }
      long now = System.nanoTime();
      long waitUntil = positions >= minBatch ? gapDeadline : hardDeadline;
      long remaining = waitUntil - now;
      if (remaining <= 0) {
        break; // silence avec quorum, ou plafond d'attente atteint
      }
      next = queue.poll(remaining, TimeUnit.NANOSECONDS);
      if (next == null) {
        if (positions >= minBatch || System.nanoTime() >= hardDeadline) {
          break; // le flux s'est tari (quorum ok) ou plafond atteint
        }
        continue; // pas de quorum — on persiste jusqu'au plafond
      }
      if (positions + next.count > capacity) {
        pendingCarry = next; // ouvrira le batch suivant
        break;
      }
      batch.add(next);
      positions += next.count;
      gapDeadline = System.nanoTime() + flushNanos;
    }
    return positions;
  }

  /** Recopie les planes des requêtes du slot dans son buffer contigu. */
  private void gather(Slot slot) {
    List<Request> batch = slot.requests;
    int offset = 0;
    for (int i = 0; i < batch.size(); i++) {
      Request request = batch.get(i);
      System.arraycopy(
          request.planes, 0, slot.planes, offset * FLOATS_PER_POS, request.count * FLOATS_PER_POS);
      offset += request.count;
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Étage 2 — runner : SEUL thread à invoquer le délégué (garde-fou CUDA EP).
  // ---------------------------------------------------------------------------------------------

  private void runnerLoop() {
    while (true) {
      Slot slot;
      try {
        slot = toRun.take();
      } catch (InterruptedException e) {
        break;
      }
      long runStart = System.nanoTime();
      try {
        delegate.forward(slot.planes, slot.positions, slot.output);
        slot.failure = null;
      } catch (Throwable t) {
        slot.failure = t;
      }
      runNanos += System.nanoTime() - runStart;
      try {
        toComplete.put(slot);
      } catch (InterruptedException e) {
        // close() pendant le handoff : débloque les appelants du lot directement.
        failRequests(slot.requests, slot.failure != null ? slot.failure : closedException());
        break;
      }
    }
  }

  // ---------------------------------------------------------------------------------------------
  // Étage 3 — completer : recopie des sorties vers les appelants + déblocage + stats.
  // ---------------------------------------------------------------------------------------------

  private void completerLoop() {
    while (true) {
      Slot slot;
      try {
        slot = toComplete.take();
      } catch (InterruptedException e) {
        break;
      }
      // Compteurs incrémentés AVANT le dispatch : un appelant débloqué par dispatch() doit voir
      // son batch déjà compté s'il lit stats() immédiatement après son forward().
      batchesCompleted++;
      positionsCompleted += slot.positions;
      long scatterStart = System.nanoTime();
      dispatch(slot);
      scatterNanos += System.nanoTime() - scatterStart;
      slot.requests.clear(); // ne pas retenir les références Request entre deux batchs
      if (batchesCompleted % STATS_EVERY_BATCHES == 0) {
        logStats();
      }
      try {
        freeSlots.put(slot);
      } catch (InterruptedException e) {
        break;
      }
    }
  }

  /** Recopie les sorties du slot vers les {@code NNOutput} des appelants, puis les débloque. */
  private void dispatch(Slot slot) {
    List<Request> batch = slot.requests;
    Throwable failure = slot.failure;
    int offset = 0;
    for (int i = 0; i < batch.size(); i++) {
      Request request = batch.get(i);
      if (failure == null) {
        System.arraycopy(
            slot.output.logits,
            offset * MoveEncoding.POLICY_INDICES,
            request.output.logits,
            0,
            request.count * MoveEncoding.POLICY_INDICES);
        System.arraycopy(slot.output.values, offset, request.output.values, 0, request.count);
      }
      offset += request.count;
      request.complete(failure);
    }
  }

  /** Résumé périodique : cadence, K, taux d'occupation du délégué, coûts par étage. */
  private void logStats() {
    long now = System.nanoTime();
    long batches = batchesCompleted - lastStatsBatches;
    long positions = positionsCompleted - lastStatsPositions;
    long run = runNanos - lastStatsRunNanos;
    long assemble = assembleNanos - lastStatsAssembleNanos;
    long scatter = scatterNanos - lastStatsScatterNanos;
    long slotWait = slotWaitNanos - lastStatsSlotWaitNanos;
    long elapsed = now - lastStatsNanos;
    int minK = statsMinK;
    int maxK = statsMaxK;
    statsMinK = Integer.MAX_VALUE;
    statsMaxK = 0;
    lastStatsNanos = now;
    lastStatsBatches = batchesCompleted;
    lastStatsPositions = positionsCompleted;
    lastStatsRunNanos = runNanos;
    lastStatsAssembleNanos = assembleNanos;
    lastStatsScatterNanos = scatterNanos;
    lastStatsSlotWaitNanos = slotWaitNanos;
    if (batches <= 0 || elapsed <= 0) {
      return;
    }
    final long fBatches = batches;
    final long fPositions = positions;
    final double avgK = positions / (double) batches;
    final double posPerSec = positions * 1e9 / elapsed;
    final double delegateBusyPct = 100.0 * run / elapsed;
    final double assembleUs = assemble / 1e3 / batches;
    final double scatterUs = scatter / 1e3 / batches;
    final double slotWaitUs = slotWait / 1e3 / batches;
    LOG.info(
        () ->
            String.format(
                "pipeline: %d batchs, K=%.1f (min %d, max %d), %d pos (%.0f pos/s) | delegate busy"
                    + " %.1f%% | par batch: assemble %.0f µs, scatter %.0f µs, slot-wait %.0f µs",
                fBatches,
                avgK,
                minK == Integer.MAX_VALUE ? 0 : minK,
                maxK,
                fPositions,
                posPerSec,
                delegateBusyPct,
                assembleUs,
                scatterUs,
                slotWaitUs));
  }

  /** Complète en erreur toutes les requêtes d'un lot. */
  private static void failRequests(List<Request> requests, Throwable cause) {
    for (int i = 0; i < requests.size(); i++) {
      requests.get(i).complete(cause);
    }
    requests.clear();
  }

  /** Débloque (en erreur) toutes les requêtes encore en queue. */
  private void drainAndFail(IllegalStateException cause) {
    Request pending;
    while ((pending = queue.poll()) != null) {
      pending.complete(cause);
    }
  }
}
