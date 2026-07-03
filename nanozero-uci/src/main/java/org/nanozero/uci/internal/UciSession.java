package org.nanozero.uci.internal;

import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;
import java.util.concurrent.atomic.AtomicBoolean;
import org.nanozero.engine.Engine;
import org.nanozero.engine.EngineState;
import org.nanozero.engine.SearchResult;

/**
 * État d'une session de recherche UCI entre {@code go} et {@code bestmove} (cf. SPEC §3.4, §5.6,
 * §5.7, §7, §12 phase 6).
 *
 * <p>Encapsule trois responsabilités liées au lifecycle d'une recherche :
 *
 * <ol>
 *   <li><strong>Émission périodique d'{@code info}</strong> via un thread daemon nommé {@code
 *       uci-info-reporter} qui poll {@code engine.currentBest()} toutes les 500 ms.
 *   <li><strong>Détection de fin naturelle</strong> : le même thread surveille {@code
 *       engine.state()} et appelle {@code engine.stop()} sur transition vers {@code DONE}.
 *   <li><strong>Émission unique de {@code bestmove}</strong> via {@link AtomicBoolean} CAS partagé
 *       entre {@code InfoReporter} et le main thread (qui peut recevoir un {@code stop} UCI).
 * </ol>
 *
 * <p><strong>Protocole CAS</strong> (cf. SPEC §5.7) :
 *
 * <ul>
 *   <li>{@link #emitBestmove(SearchResult)} prend un {@link SearchResult} qui peut être {@code
 *       null} (cas où l'autre thread a déjà consommé via {@code engine.stop()} qui retourne {@code
 *       null} sur état IDLE — cf. SPEC-engine §4.2).
 *   <li>Sur {@code result == null} : no-op silencieux.
 *   <li>Sur {@code result != null} : {@code bestmoveEmitted.compareAndSet(false, true)} ; le
 *       gagnant émet, le perdant ignore.
 * </ul>
 *
 * <p>Cette double-protection (null check + CAS) garantit qu'<strong>exactement une</strong> ligne
 * {@code bestmove} est émise par session, indépendamment de l'ordre d'exécution des threads.
 *
 * <p><strong>Lifecycle attendu</strong> (côté caller, typiquement {@code UciCommandHandler} en
 * phase 7) :
 *
 * <pre>
 *   engine.startSearch(...)  // ou engine.startPonder(...)
 *   var session = new UciSession(engine, writer, isPonder);
 *   session.startInfoReporter();
 *   // ... attendre fin naturelle ou réception stop UCI ...
 *   // Cas A : InfoReporter détecte DONE et émet bestmove. session.stopInfoReporter() est
 *   //         appelé en interne (idempotent).
 *   // Cas B : main thread reçoit "stop" → session.onUciStop() qui bloque, émet, et stoppe
 *   //         InfoReporter.
 * </pre>
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-uci} module.
 */
public final class UciSession {

  /** Période de polling de l'InfoReporter (cf. SPEC §5.6). */
  static final long INFO_POLL_MS = 500;

  /** Timeout de join sur {@link #stopInfoReporter} après interrupt. */
  static final long INFO_REPORTER_JOIN_TIMEOUT_MS = 1000;

  /** Nom du thread daemon InfoReporter (utilisé par les tests). */
  static final String INFO_REPORTER_THREAD_NAME = "uci-info-reporter";

  /**
   * Seuil heuristique pour distinguer {@code score mate} de {@code score cp}. Si {@code |value| >
   * 0.95} ET {@code pv.length < 20}, on émet {@code score mate}. Sinon {@code score cp}.
   * Best-effort ; à reconsidérer v1.1.0+ avec modèle entraîné (cf. SPEC §5.6).
   */
  private static final float MATE_VALUE_THRESHOLD = 0.95f;

  /** Borne supérieure de la PV pour considérer un mate à émettre (anti-faux-positif). */
  private static final int MATE_PV_LENGTH_LIMIT = 20;

  /**
   * Coefficients du mapping value → centipawns UCI : {@code cp = K * tan(A * v)}, où {@code v} est
   * la value scalaire MCTS ∈ [-1, +1] (depuis v1.5.0 = {@code P(W) - P(L)} issu du value head WDL,
   * équivalent au Q = W - L de Lc0 ; ≤ v1.4.0 c'était la sortie tanh). Étale la value sur une plage
   * cp réaliste : v=0 → 0 cp, v=0.5 → 111 cp, v=0.95 → 1283 cp, v=0.99 → 4587 cp. Remplace le
   * mapping naïf {@code cp = round(v * 100)} qui cappait à ±100 cp.
   *
   * <p><strong>Source des constantes</strong> : c'est la formule Q→cp <em>classique</em> de Lc0
   * (constantes exactes {@code K = 111.714640912}, {@code A = 1.5620688421}), confirmée par <a
   * href="https://talkchess.com/viewtopic.php?p=971568">TalkChess</a> et <a
   * href="https://github.com/LeelaChessZero/lc0/pull/841">Lc0 PR #841</a>. <strong>Ce n'est PAS le
   * WDL-rescale de Lc0 v0.30</strong> (lequel utilise la distribution W/D/L complète + un paramètre
   * de contempt/Elo — non implémenté ici : on collapse en Q = W - L AVANT le cp). Le <a
   * href="https://lczero.org/blog/2023/07/the-lc0-v0.30.0-wdl-rescale/contempt-implementation/">blog
   * v0.30</a> ne justifie que le <em>choix de {@code tan()}</em> (queues plus plates que logit,
   * plus de disparité au-delà de 90 % de winrate). Lc0 a depuis recalibré à {@code K =
   * 290.680623072}, {@code A = 1.548090806} (pour atteindre 12800 cp à Q=1) ; on conserve
   * volontairement le jeu de constantes Lc0 historique (111.714640912 / 1.5620688421).
   */
  private static final double CP_K = 111.714640912;

  private static final double CP_A = 1.5620688421;

  private final Engine engine;
  private final UciResponseWriter writer;
  private final AtomicBoolean bestmoveEmitted = new AtomicBoolean(false);
  private final boolean isPonder;
  private final GoArgs originalGoArgs;
  private volatile Thread infoReporterThread;

  /**
   * Construit une nouvelle session associée à un {@code go} ou {@code go ponder}.
   *
   * <p><strong>Pré-requis</strong> : le caller a déjà appelé {@code engine.startSearch} ou {@code
   * engine.startPonder} avant l'instanciation. {@code UciSession} ne déclenche pas la recherche
   * elle-même ; il orchestre l'observation et l'émission.
   *
   * @param engine moteur déjà en cours de recherche, non null
   * @param writer writer UCI cible, non null
   * @param isPonder {@code true} si la session correspond à un {@code go ponder} (l'émission de
   *     {@code bestmove} attend {@code ponderhit} ou {@code stop} ; pas d'émission spontanée sur
   *     fin naturelle car {@code UNLIMITED} budget ne se termine pas)
   * @throws NullPointerException si {@code engine} ou {@code writer} est null
   */
  public UciSession(Engine engine, UciResponseWriter writer, boolean isPonder) {
    this(engine, writer, isPonder, GoArgs.empty());
  }

  /**
   * Construit une session avec les {@link GoArgs} originels du {@code go} qui a déclenché la
   * recherche. Utilisé par {@link UciCommandHandler} (phase 7) pour permettre à {@code ponderhit}
   * de recalculer le budget réel via {@link TimeManagementPolicy}.
   *
   * @param engine moteur déjà en cours de recherche, non null
   * @param writer writer UCI cible, non null
   * @param isPonder {@code true} si la session correspond à un {@code go ponder}
   * @param originalGoArgs arguments du {@code go} initial (pour recalcul budget sur ponderhit), non
   *     null
   * @throws NullPointerException si l'un des arguments référence est null
   */
  public UciSession(
      Engine engine, UciResponseWriter writer, boolean isPonder, GoArgs originalGoArgs) {
    this.engine = Objects.requireNonNull(engine, "engine must not be null");
    this.writer = Objects.requireNonNull(writer, "writer must not be null");
    this.isPonder = isPonder;
    this.originalGoArgs = Objects.requireNonNull(originalGoArgs, "originalGoArgs must not be null");
  }

  /** {@link GoArgs} originels du {@code go} qui a déclenché cette session (cf. constructor). */
  public GoArgs originalGoArgs() {
    return originalGoArgs;
  }

  /**
   * Démarre le thread daemon InfoReporter. Idempotent au sens « rejet d'un second appel sur la même
   * session » : le second appel lève {@link IllegalStateException} pour signaler une erreur de
   * logique côté caller.
   *
   * @throws IllegalStateException si déjà démarré sur cette session
   */
  public synchronized void startInfoReporter() {
    if (infoReporterThread != null) {
      throw new IllegalStateException("InfoReporter already started for this session");
    }
    Thread t = new Thread(this::infoReporterLoop, INFO_REPORTER_THREAD_NAME);
    t.setDaemon(true);
    infoReporterThread = t;
    t.start();
  }

  /**
   * Boucle principale du thread InfoReporter (cf. SPEC §5.6, §5.7). Poll toutes les {@link
   * #INFO_POLL_MS} ms ; émet une ligne {@code info} si la recherche a progressé ; détecte la
   * transition vers {@code DONE} et déclenche l'émission de {@code bestmove}.
   *
   * <p>Sortie sur :
   *
   * <ul>
   *   <li>Émission de {@code bestmove} (réussite CAS ou détection {@code bestmoveEmitted=true} au
   *       prochain cycle).
   *   <li>Interruption explicite ({@link Thread#interrupt}, typiquement par {@link
   *       #stopInfoReporter}).
   *   <li>État inattendu ({@code IDLE} ou {@code CLOSED}).
   * </ul>
   */
  private void infoReporterLoop() {
    try {
      while (!Thread.currentThread().isInterrupted()) {
        // Phase 12 hotfix-005 — au lieu de Thread.sleep(INFO_POLL_MS) qui bloque 500ms même si le
        // search se termine à t+100ms (latence wasted → time forfeits massifs à TC=3+0.03),
        // engine.awaitDone(timeout) se réveille immédiatement quand le worker passe en DONE
        // (lock.notifyAll interne). Sinon retourne après INFO_POLL_MS pour émettre un snapshot
        // régulier pendant SEARCHING.
        engine.awaitDone(INFO_POLL_MS);
        if (bestmoveEmitted.get()) {
          return;
        }
        EngineState state = engine.state();
        switch (state) {
          case DONE -> {
            // Race possible : main thread a peut-être déjà appelé engine.stop().
            SearchResult result = engine.stop();
            if (result != null) {
              emitInfoFromResult(result);
              emitBestmove(result);
            }
            // Sinon : autre thread a consommé. bestmoveEmitted sera true dès qu'il aura émis.
            return;
          }
          case SEARCHING, PONDERING -> {
            SearchResult snap = engine.currentBest();
            if (snap.simulationsCount() > 0) {
              emitInfoFromResult(snap);
            }
          }
          case STOPPING -> {
            // Main thread est en charge (a appelé engine.stop() bloquant). On poll passivement.
          }
          case IDLE, CLOSED -> {
            // État inattendu (probable race avec close ou reset). Sortie propre.
            return;
          }
        }
      }
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      // Pas d'émission supplémentaire ; le main thread a vraisemblablement pris la main.
    }
  }

  /**
   * Émet {@code bestmove} via protocole CAS (cf. SPEC §5.7). Appelable depuis n'importe quel
   * thread.
   *
   * <p>Double-protection :
   *
   * <ol>
   *   <li><strong>Pas de bestmove valide à émettre</strong> : si {@code result == null} OU si
   *       {@code result.simulationsCount() == 0}, no-op silencieux. Le second cas correspond au
   *       contrat de {@link Engine#stop()} qui retourne un {@code EMPTY_RESULT} ({@code
   *       simulationsCount=0}, {@code bestMove=0}) sur état IDLE entrant — c'est-à-dire quand
   *       l'autre thread a déjà consommé via {@code engine.stop()} (race) ou que la session n'a
   *       jamais démarré effectivement.
   *   <li>{@code bestmoveEmitted.compareAndSet(false, true)} : le gagnant émet, le perdant ignore.
   * </ol>
   *
   * <p>Sur émission réussie, l'InfoReporter est arrêté (interrupt + join borné) — le caller peut
   * appeler depuis le thread InfoReporter lui-même, dans ce cas {@link #stopInfoReporter} ne tente
   * pas de se joindre.
   *
   * @param result résultat de la recherche, peut être {@code null} ou {@code EMPTY_RESULT}
   */
  public void emitBestmove(SearchResult result) {
    if (result == null || result.simulationsCount() == 0) {
      return;
    }
    if (bestmoveEmitted.compareAndSet(false, true)) {
      // v1.2.0 — Émission "info string visits" AVANT bestmove (cf. SPEC §6.5, ADR-003 mise à
      // jour v1.2.0 "hidden behaviors pattern"). Permet aux clients training de lire la
      // distribution N(a) au root pour appliquer la temperature sampling.
      writer.emit(
          new UciResponse.InfoString(
              formatVisitsContent(result.childMoves(), result.childVisits())));
      OptionalInt ponderOpt = extractPonderMove(result);
      writer.emit(new UciResponse.BestMove(result.bestMove(), ponderOpt));
      stopInfoReporter();
    }
  }

  /**
   * (v1.2.0) Formate le contenu de la ligne {@code info string visits ...} : préfixe {@code
   * "visits"} suivi des pairs {@code "<move> <count>"} space-separated, triés par {@code count}
   * décroissant. Skip les coups avec {@code count == 0}. Position terminale ({@code moves.length ==
   * 0}) retourne {@code "visits"} seul.
   *
   * <p>Visibilité package-private pour permettre les tests unitaires sans accès à l'émission
   * stdout.
   */
  static String formatVisitsContent(int[] moves, int[] visits) {
    int n = moves.length;
    // Indices triés par visits décroissant (Arrays.sort sur Integer[] est stable → tie-break
    // naturel par index original = ordre MoveGen).
    Integer[] indices = new Integer[n];
    for (int i = 0; i < n; i++) {
      indices[i] = i;
    }
    java.util.Arrays.sort(indices, (a, b) -> Integer.compare(visits[b], visits[a]));
    StringBuilder sb = new StringBuilder("visits");
    for (int idx : indices) {
      if (visits[idx] <= 0) {
        // Skip silencieusement (déjà à la fin du tri descendant ; break dès qu'on en voit un).
        break;
      }
      sb.append(' ').append(UciMoveCodec.encode(moves[idx])).append(' ').append(visits[idx]);
    }
    return sb.toString();
  }

  /**
   * Réception d'un {@code stop} UCI côté main thread. Bloque sur {@link Engine#stop()} jusqu'à
   * arrêt propre (transition vers {@code DONE}), puis émet {@code bestmove} via {@link
   * #emitBestmove}.
   *
   * <p>Sur {@code state == IDLE} entrant, {@code engine.stop()} retourne {@code null} et {@link
   * #emitBestmove} no-op silencieusement. C'est le comportement attendu pour un {@code stop} reçu
   * hors session active.
   */
  public void onUciStop() {
    SearchResult result = engine.stop();
    emitBestmove(result);
  }

  /**
   * Termine le thread InfoReporter proprement. Idempotent ; appelable depuis n'importe quel thread,
   * y compris depuis InfoReporter lui-même (auquel cas on évite l'auto-join).
   */
  public synchronized void stopInfoReporter() {
    Thread t = infoReporterThread;
    if (t == null || !t.isAlive()) {
      return;
    }
    if (t == Thread.currentThread()) {
      // Auto-stop depuis l'InfoReporter lui-même : pas de join, le thread va sortir naturellement.
      t.interrupt();
      return;
    }
    t.interrupt();
    try {
      t.join(INFO_REPORTER_JOIN_TIMEOUT_MS);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }

  /** Indique si {@code bestmove} a déjà été émis pour cette session. */
  public boolean isBestmoveEmitted() {
    return bestmoveEmitted.get();
  }

  /** Indique si la session est en mode ponder (cf. constructor). */
  public boolean isPonder() {
    return isPonder;
  }

  /**
   * Référence au thread InfoReporter, exposée aux tests pour inspection (état daemon, terminaison).
   * {@code null} si {@link #startInfoReporter} n'a pas encore été appelé.
   */
  Thread infoReporterThreadForTest() {
    return infoReporterThread;
  }

  // -------------------------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------------------------

  private void emitInfoFromResult(SearchResult r) {
    writer.emit(new UciResponse.Info(buildInfoFields(r)));
  }

  /**
   * Construit un {@link InfoFields} depuis un {@link SearchResult} (cf. SPEC §5.6).
   *
   * <p>Visibilité package-private pour les tests unitaires directs (cohérent avec {@code
   * UciResponseWriter.format} en phase 4).
   */
  InfoFields buildInfoFields(SearchResult r) {
    long elapsedNanos = r.elapsedNanos();
    int sims = r.simulationsCount();
    int pvLen = r.principalVariation().length;

    OptionalInt depth = (pvLen > 0) ? OptionalInt.of(pvLen) : OptionalInt.empty();
    OptionalInt nodes = (sims > 0) ? OptionalInt.of(sims) : OptionalInt.empty();
    OptionalInt nps = OptionalInt.empty();
    OptionalLong timeMs = OptionalLong.empty();
    if (elapsedNanos > 0) {
      timeMs = OptionalLong.of(elapsedNanos / 1_000_000L);
      if (sims > 0) {
        long npsVal = (long) sims * 1_000_000_000L / elapsedNanos;
        nps = OptionalInt.of((int) Math.min((long) Integer.MAX_VALUE, npsVal));
      }
    }

    OptionalInt scoreCp = OptionalInt.empty();
    OptionalInt scoreMate = OptionalInt.empty();
    if (!Float.isNaN(r.value())) {
      float v = r.value();
      if (v > MATE_VALUE_THRESHOLD && pvLen > 0 && pvLen < MATE_PV_LENGTH_LIMIT) {
        scoreMate = OptionalInt.of((pvLen + 1) / 2);
      } else if (v < -MATE_VALUE_THRESHOLD && pvLen > 0 && pvLen < MATE_PV_LENGTH_LIMIT) {
        scoreMate = OptionalInt.of(-((pvLen + 1) / 2));
      } else {
        scoreCp = OptionalInt.of(valueToCentipawns(v));
      }
    }

    return new InfoFields(
        depth,
        depth, // seldepth = depth (MCTS sans extensions)
        nodes,
        nps,
        timeMs,
        scoreCp,
        scoreMate,
        r.principalVariation(),
        OptionalInt.of(1), // multipv toujours 1 en v1.0.0 (pas de MultiPV)
        Optional.empty()); // string générique non utilisé
  }

  /**
   * Extrait le coup à pondérer depuis la PV : {@code pv[1]} si {@code pv.length >= 2}, sinon {@link
   * OptionalInt#empty()}.
   */
  private static OptionalInt extractPonderMove(SearchResult r) {
    int[] pv = r.principalVariation();
    return (pv.length >= 2) ? OptionalInt.of(pv[1]) : OptionalInt.empty();
  }

  /**
   * Convertit la value MCTS ∈ [-1, +1] (= {@code P(W) - P(L)} = Q de Lc0 depuis v1.5.0) en
   * centipawns UCI via la formule Q→cp classique de Lc0 : {@code cp = K * tan(A * v)}. Cf. {@link
   * #CP_K} pour la source des constantes et la distinction avec le WDL-rescale v0.30 (non
   * implémenté).
   */
  static int valueToCentipawns(float v) {
    return (int) Math.round(CP_K * Math.tan(CP_A * v));
  }
}
