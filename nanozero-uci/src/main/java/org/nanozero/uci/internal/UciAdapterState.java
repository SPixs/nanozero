package org.nanozero.uci.internal;

import java.util.Objects;
import org.nanozero.board.GameState;
import org.nanozero.engine.Engine;
import org.nanozero.engine.EngineConfig;
import org.nanozero.nn.Network;

/**
 * État process-wide de l'adapter UCI (cf. SPEC §3.4 lifecycle, §5.1, §5.7, §12 phase 7).
 *
 * <p>Holder des références partagées entre la boucle principale (`UciMain`) et le dispatcher
 * (`UciCommandHandler`) :
 *
 * <ul>
 *   <li>{@link #engine()} : moteur de recherche (1 instance par process v1.0.0, cf. ADR-004).
 *   <li>{@link #writer()} : writer UCI sur stdout.
 *   <li>{@link #options()} : holder mutable des options UCI.
 *   <li>{@link #currentSession()} / {@link #setCurrentSession} : session active (nullable),
 *       associée au {@code go} courant.
 *   <li>{@link #lastPosition()} / {@link #lastPlayedMoves()} : dernière position reçue via {@code
 *       position}, utilisée par {@code go} comme position de recherche. Initialement null ;
 *       fallback {@code GameState.start()} si {@code go} arrive sans {@code position} préalable.
 *   <li>{@link #debugMode()} : flag tolérant pour logs stderr (cf. ADR-007).
 * </ul>
 *
 * <p><strong>AutoCloseable</strong> : {@link #close()} arrête une session active si nécessaire,
 * puis {@code engine.close()}. Idempotent. Le main loop wrap dans try-with-resources pour cleanup
 * garanti sur EOF / quit / exception.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-uci} module.
 */
public final class UciAdapterState implements AutoCloseable {

  /**
   * Engine pré-créé OU lazy-créé au premier appel à {@link #engine()} si {@link #network} non-null.
   * v1.0.0 : constructor {@code (Engine, ...)} fournit l'instance directement. v1.1.0 : constructor
   * {@code (Network, ...)} laisse {@code engine == null}, lazy creation permet de lire les hidden
   * options Dirichlet courantes au moment de la construction (cf. SPEC §6.4).
   */
  private volatile Engine engine;

  /**
   * (v1.1.0) Réseau utilisé pour la lazy creation d'Engine. Non-null ssi constructor lazy utilisé.
   */
  private final Network network;

  private final UciResponseWriter writer;
  private final UciOptionsState options;

  /**
   * (v1.3.0) Flag passé par {@code UciMain --cuda} : intention utilisateur d'utiliser ORT CUDA EP.
   * Reste {@code true} même après fallback CPU (l'auto-config {@code batchSize=threads} dans {@link
   * #buildEngineConfigFromOptions} se base sur cette intention, pas sur le backend effectif —
   * fallback ⇒ engine warning [engine] couvre le cas). Cf. ADR-008-uci.
   */
  private final boolean useCuda;

  private volatile UciSession currentSession;
  private volatile GameState lastPosition;
  private volatile int[] lastPlayedMoves = new int[0];
  private volatile boolean debugMode;

  /**
   * Construit le holder process-wide avec un {@link Engine} pré-créé (constructor v1.0.0). Ne
   * permet pas l'application runtime des hidden options Dirichlet (cf. SPEC §6.4 amendée v1.1.0) :
   * les options Dirichlet courantes ne seront pas relues pour l'engine fourni. Pour ce nouveau
   * usage, utiliser le constructor {@link #UciAdapterState(Network, UciResponseWriter,
   * UciOptionsState)}.
   *
   * @throws NullPointerException si l'un des arguments est null
   */
  public UciAdapterState(Engine engine, UciResponseWriter writer, UciOptionsState options) {
    this.engine = Objects.requireNonNull(engine, "engine must not be null");
    this.network = null;
    this.writer = Objects.requireNonNull(writer, "writer must not be null");
    this.options = Objects.requireNonNull(options, "options must not be null");
    this.useCuda = false;
  }

  /**
   * (v1.1.0) Construit le holder avec un {@link Network}. L'{@link Engine} est lazy-créé au premier
   * appel à {@link #engine()}, lisant à ce moment les hidden options Dirichlet courantes (cf. SPEC
   * §6.4). Permet aux clients training de configurer Dirichlet via {@code setoption} avant le
   * premier {@code go}.
   *
   * @throws NullPointerException si l'un des arguments est null
   */
  public UciAdapterState(Network network, UciResponseWriter writer, UciOptionsState options) {
    this(network, writer, options, false);
  }

  /**
   * (v1.3.0) Construit le holder avec un {@link Network} et le flag {@code useCuda} (cf. ADR-008
   * uci). Si {@code useCuda=true}, l'auto-config {@code batchSize = threads} est activée dans
   * {@link #buildEngineConfigFromOptions} (path engine batched-queue Mode B, cf. engine ADR-017).
   *
   * @throws NullPointerException si {@code network}, {@code writer} ou {@code options} est null
   */
  public UciAdapterState(
      Network network, UciResponseWriter writer, UciOptionsState options, boolean useCuda) {
    this.engine = null;
    this.network = Objects.requireNonNull(network, "network must not be null");
    this.writer = Objects.requireNonNull(writer, "writer must not be null");
    this.options = Objects.requireNonNull(options, "options must not be null");
    this.useCuda = useCuda;
  }

  /**
   * Retourne l'{@link Engine}. Si le constructor lazy a été utilisé, l'engine est créé à la
   * première invocation à partir du {@link Network} et des hidden options Dirichlet courantes.
   * Idempotent ; instance unique après création.
   *
   * <p>Pattern volatile-only cohérent avec le reste de la classe (mono-thread main UCI ; pas de
   * concurrent access attendu sur l'engine creation).
   */
  public Engine engine() {
    if (engine == null) {
      engine = new Engine(network, buildEngineConfigFromOptions(options, useCuda));
    }
    return engine;
  }

  /**
   * (v1.1.0) Construit un {@link EngineConfig} à partir de {@code EngineConfig.defaults()} (champs
   * v1.0.0 conservés) et des hidden options Dirichlet courantes (cf. SPEC §6.4) : {@code
   * DirichletAlpha} et {@code DirichletEpsilon} sont divisés par 1000 (convention spin x1000 →
   * float), {@code DirichletSeed} est parsé comme long (défaut 0 si parse échoue).
   *
   * <p>(v1.2.0+) Lit aussi l'option UCI standard {@code Threads} (cf. {@link
   * UciOptionsState#threads()}). Si {@code threads == 1}, configure mono-thread strict (bit-for-bit
   * v1.1.2). Si {@code threads > 1}, active le mode MCTS multi-thread (tree parallelization +
   * virtual loss=1.0 par défaut, aligné sur KataGo {@code numVirtualLossesPerThread=1} et ELF
   * OpenGo. Valeur précédemment 3.0 — folklore non sourcé dans les papiers AlphaZero primaires).
   *
   * <p>Variante v1.2.0 (1-arg) : surcharge legacy. Délègue à la variante 2-arg avec {@code
   * useCuda=false}. Préserve la compat avec les tests pré-v1.3.0.
   */
  static EngineConfig buildEngineConfigFromOptions(UciOptionsState opts) {
    return buildEngineConfigFromOptions(opts, false);
  }

  /**
   * (v1.3.0) Construit un {@link EngineConfig} en prenant en compte le flag {@code useCuda} pour
   * l'auto-config {@code batchSize} (cf. ADR-008-uci) :
   *
   * <ul>
   *   <li>{@code useCuda=true && threads>1} : auto-config {@code batchSize = threads} (active path
   *       engine batched-queue Mode B, cf. engine ADR-017 + SPEC §16).
   *   <li>Sinon : {@code batchSize = 1} (Mode A, forward direct par thread, recommandé CPU).
   * </ul>
   *
   * <p>L'option UCI {@code BatchSize} override l'auto-config quand set explicitement (cf. {@link
   * UciOptionsState#batchSizeOverride()} — valeur sentinelle {@code 0} = non set).
   *
   * @param opts options UCI courantes
   * @param useCuda intention utilisateur d'utiliser CUDA EP (cf. {@code UciMain --cuda}). Reste
   *     {@code true} même après fallback CPU côté NetworkOnnx (le warning Engine boot avertit du
   *     mode B contre-productif sur CPU SIMD).
   */
  static EngineConfig buildEngineConfigFromOptions(UciOptionsState opts, boolean useCuda) {
    EngineConfig d = EngineConfig.defaults();
    int threads = opts.threads();
    float virtualLoss = threads > 1 ? 1.0f : 0.0f;
    int autoBatchSize = (useCuda && threads > 1) ? threads : 1;
    int batchSize = opts.batchSizeOverride() > 0 ? opts.batchSizeOverride() : autoBatchSize;
    return new EngineConfig(
        d.cPuct(),
        d.fpuValue(),
        d.treeInitialCapacity(),
        opts.dirichletAlpha() / 1000f,
        opts.dirichletEpsilon() / 1000f,
        parseLongOrZero(opts.dirichletSeed()),
        threads,
        batchSize,
        virtualLoss,
        // (ADR-018) Cache d'évaluation NN : pré-amorcé au boot par le flag CLI --nncache, puis
        // surchargeable runtime via setoption NNCacheSize. 0 = désactivé (défaut).
        opts.nnCacheSize());
  }

  private static long parseLongOrZero(String s) {
    try {
      return Long.parseLong(s.trim());
    } catch (NumberFormatException ignored) {
      return 0L;
    }
  }

  public UciResponseWriter writer() {
    return writer;
  }

  public UciOptionsState options() {
    return options;
  }

  /** Session courante associée au {@code go} actuel. {@code null} si aucune session active. */
  public UciSession currentSession() {
    return currentSession;
  }

  /**
   * Set par {@code UciCommandHandler.handleGo} ; clearé par {@code close()} ou nouvelle session.
   */
  public void setCurrentSession(UciSession session) {
    this.currentSession = session;
  }

  /**
   * Dernière position reçue via la commande {@code position}. {@code null} si aucune {@code
   * position} n'a été émise depuis le boot.
   */
  public GameState lastPosition() {
    return lastPosition;
  }

  /**
   * Coups joués pour atteindre {@link #lastPosition()} depuis la base ({@code startpos} ou FEN).
   * Tableau vide si {@code position startpos} sans {@code moves}, ou si aucune {@code position} n'a
   * été émise.
   */
  public int[] lastPlayedMoves() {
    return lastPlayedMoves;
  }

  public void setLastPosition(GameState position, int[] playedMoves) {
    this.lastPosition = position;
    this.lastPlayedMoves = playedMoves != null ? playedMoves : new int[0];
  }

  /** Flag de mode debug. Lu par les handlers pour conditionner les logs stderr. */
  public boolean debugMode() {
    return debugMode;
  }

  public void setDebugMode(boolean enabled) {
    this.debugMode = enabled;
  }

  /**
   * Cleanup : arrête une session active (best-effort), puis ferme l'engine. Idempotent.
   *
   * <p>L'arrêt de session est wrap dans try/catch silencieux (sauf debug mode où l'erreur est
   * loggée sur stderr) pour ne jamais bloquer le shutdown.
   */
  @Override
  public void close() {
    UciSession session = currentSession;
    if (session != null && !session.isBestmoveEmitted()) {
      try {
        session.onUciStop();
      } catch (RuntimeException e) {
        if (debugMode) {
          System.err.println(
              "[debug] error stopping session in close: "
                  + e.getClass().getSimpleName()
                  + ": "
                  + e.getMessage());
        }
      }
    }
    Engine e = engine;
    if (e != null) {
      e.close();
    }
  }
}
