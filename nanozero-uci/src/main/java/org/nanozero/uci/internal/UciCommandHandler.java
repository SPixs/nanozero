package org.nanozero.uci.internal;

import org.nanozero.board.GameState;
import org.nanozero.engine.EngineState;
import org.nanozero.engine.SearchBudget;

/**
 * Dispatcher UCI : transforme une {@link UciCommand} parsée en action sur l'{@link UciAdapterState}
 * (cf. SPEC §5.1, §5.7, §12 phase 7).
 *
 * <p>API minimaliste : une méthode statique {@link #handle} qui dispatch via sealed switch
 * exhaustif sur les 11 sous-types de {@link UciCommand} et retourne {@link HandleAction#CONTINUE}
 * ou {@link HandleAction#QUIT}. Le main loop ({@code UciMain}) interprète le retour pour continuer
 * ou sortir.
 *
 * <p><strong>Tolérance</strong> (cf. SPEC §5.1) : toute exception levée par les sous-handlers
 * (engine state invalide, IO error sur le writer, etc.) est attrapée par le {@code try/catch}
 * global et loggée sur stderr en debug mode. La boucle continue. Seul {@code Quit} retourne {@code
 * QUIT}.
 *
 * <p><strong>Lifecycle session</strong> :
 *
 * <ul>
 *   <li>{@code go} (non-ponder) : {@code engine.startSearch} + nouvelle {@link UciSession} + {@code
 *       startInfoReporter}.
 *   <li>{@code go ponder} : si l'option UCI {@code Ponder} est {@code false}, no-op silencieux (cf.
 *       SPEC §6.3, ADR-003). Sinon, {@code engine.startPonder(prePondPos, predictedMove)} où
 *       prePondPos est reconstruit via {@code unapplyLastMove} sur {@link
 *       UciAdapterState#lastPosition()}. Si {@code lastPlayedMoves} est vide (cas dégénéré), debug
 *       log + fallback {@code startSearch + UNLIMITED}.
 *   <li>{@code stop} : {@link UciSession#onUciStop} (no-op si pas de session active).
 *   <li>{@code ponderhit} : recalcule budget via {@link TimeManagementPolicy} avec les {@link
 *       UciSession#originalGoArgs} (mode ponder=false implicite) et appelle {@code
 *       engine.ponderhit(realBudget)}.
 * </ul>
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-uci} module.
 */
public final class UciCommandHandler {

  /** Action retournée au main loop par {@link #handle}. */
  public enum HandleAction {
    /** Continue la boucle UCI (lire la prochaine commande). */
    CONTINUE,
    /** Quitter la boucle UCI proprement (cleanup via try-with-resources sur UciAdapterState). */
    QUIT
  }

  /** Identité du moteur émise sur {@code uci}. */
  private static final String ENGINE_NAME = "NanoZero 1.0.0";

  /** Auteur du moteur émis sur {@code uci}. */
  private static final String ENGINE_AUTHOR = "Mametz";

  private UciCommandHandler() {
    throw new AssertionError("Non-instantiable");
  }

  /**
   * Dispatche une {@link UciCommand} parsée vers son handler. Sealed switch exhaustif. Tolérant :
   * toute {@link RuntimeException} interne est attrapée et loggée en debug mode ; la boucle
   * continue.
   *
   * @param cmd commande parsée, non null
   * @param state holder process-wide, non null
   * @return {@link HandleAction#CONTINUE} ou {@link HandleAction#QUIT}
   */
  public static HandleAction handle(UciCommand cmd, UciAdapterState state) {
    try {
      return switch (cmd) {
        case UciCommand.Uci ignored -> handleUci(state);
        case UciCommand.IsReady ignored -> handleIsReady(state);
        case UciCommand.UciNewGame ignored -> handleUciNewGame(state);
        case UciCommand.Position p -> handlePosition(p, state);
        case UciCommand.Go g -> handleGo(g, state);
        case UciCommand.Stop ignored -> handleStop(state);
        case UciCommand.PonderHit ignored -> handlePonderHit(state);
        case UciCommand.SetOption so -> handleSetOption(so, state);
        case UciCommand.Debug d -> handleDebug(d, state);
        case UciCommand.Unknown u -> handleUnknown(u, state);
        case UciCommand.Quit ignored -> HandleAction.QUIT;
      };
    } catch (RuntimeException e) {
      if (state.debugMode()) {
        System.err.println(
            "[error] "
                + e.getClass().getSimpleName()
                + " handling "
                + cmd.getClass().getSimpleName()
                + ": "
                + e.getMessage());
      }
      return HandleAction.CONTINUE;
    }
  }

  // -------------------------------------------------------------------------------------------
  // uci : handshake initial
  // -------------------------------------------------------------------------------------------

  private static HandleAction handleUci(UciAdapterState state) {
    state.writer().emit(new UciResponse.Id(ENGINE_NAME, ENGINE_AUTHOR));
    for (UciOption opt : UciOptionsState.declaredOptions()) {
      state.writer().emit(new UciResponse.Option(opt));
    }
    state.writer().emit(new UciResponse.UciOk());
    return HandleAction.CONTINUE;
  }

  // -------------------------------------------------------------------------------------------
  // isready : ping
  // -------------------------------------------------------------------------------------------

  private static HandleAction handleIsReady(UciAdapterState state) {
    state.writer().emit(new UciResponse.ReadyOk());
    return HandleAction.CONTINUE;
  }

  // -------------------------------------------------------------------------------------------
  // ucinewgame : no-op (cf. ADR-004)
  // -------------------------------------------------------------------------------------------

  private static HandleAction handleUciNewGame(UciAdapterState state) {
    if (state.debugMode()) {
      System.err.println("[debug] ucinewgame received (no-op, tree reuse via Zobrist)");
    }
    return HandleAction.CONTINUE;
  }

  // -------------------------------------------------------------------------------------------
  // position : stocke la position courante pour le prochain go
  // -------------------------------------------------------------------------------------------

  private static HandleAction handlePosition(UciCommand.Position p, UciAdapterState state) {
    state.setLastPosition(p.position(), p.playedMoves());
    return HandleAction.CONTINUE;
  }

  // -------------------------------------------------------------------------------------------
  // go : démarre une recherche (normale ou ponder)
  // -------------------------------------------------------------------------------------------

  private static HandleAction handleGo(UciCommand.Go g, UciAdapterState state) {
    if (state.engine().state() != EngineState.IDLE) {
      // Tolérance : un nouveau go pendant qu'une recherche est active est un bug du caller.
      // On l'ignore silencieusement (debug log si activé) plutôt que de crasher.
      if (state.debugMode()) {
        System.err.println(
            "[debug] go received but engine state="
                + state.engine().state()
                + " (expected IDLE) — ignored");
      }
      return HandleAction.CONTINUE;
    }
    GameState pos = state.lastPosition();
    int[] moves = state.lastPlayedMoves();
    if (pos == null) {
      pos = new GameState();
      moves = new int[0];
      if (state.debugMode()) {
        System.err.println("[debug] go without prior position, using startpos");
      }
    }

    if (g.args().ponder()) {
      if (!state.options().ponder()) {
        // Ponder désactivé par l'option UCI : no-op silencieux (cf. SPEC §6.3, ADR-003).
        if (state.debugMode()) {
          System.err.println("[debug] go ponder ignored (Ponder option = false)");
        }
        return HandleAction.CONTINUE;
      }
      startPonderSearch(g.args(), pos, moves, state);
    } else {
      SearchBudget budget =
          TimeManagementPolicy.computeBudget(
              g.args(), pos.currentPosition().sideToMove(), state.options());
      state.engine().startSearch(pos, budget);
      UciSession session = new UciSession(state.engine(), state.writer(), false, g.args());
      session.startInfoReporter();
      state.setCurrentSession(session);
    }
    return HandleAction.CONTINUE;
  }

  /**
   * Démarre une recherche ponder. Reconstruit la position pré-pondermove via {@code
   * unapplyLastMove} sur la {@link GameState} qui possède l'historique des coups (set par {@code
   * UciCommandParser.parsePosition}). Si {@code playedMoves} est vide (cas dégénéré "go ponder" sur
   * startpos sans moves), fallback debug log + {@code startSearch + UNLIMITED} (sachant que {@code
   * ponderhit} échouera ensuite, à la charge du caller GUI buggy).
   */
  private static void startPonderSearch(
      GoArgs args, GameState pos, int[] moves, UciAdapterState state) {
    if (moves.length == 0) {
      if (state.debugMode()) {
        System.err.println(
            "[debug] go ponder without played moves, fallback startSearch UNLIMITED "
                + "(ponderhit will fail in this degenerate case)");
      }
      state.engine().startSearch(pos, SearchBudget.UNLIMITED);
    } else {
      int predictedMove = moves[moves.length - 1];
      // Mute pos pour obtenir prePondPos : la GameState possède l'historique car les moves ont
      // été appliqués via applyMove dans le parser (phase 3). engine.startPonder clone ensuite
      // via FEN, donc notre mutation locale n'affecte pas l'engine après l'appel.
      pos.unapplyLastMove();
      state.engine().startPonder(pos, predictedMove);
      // Restaure pos pour cohérence des reads ultérieurs de state.lastPosition()
      // (l'engine a déjà cloné, mais d'autres handlers peuvent encore lire la position post-coup).
      pos.applyMove(predictedMove);
    }
    UciSession session = new UciSession(state.engine(), state.writer(), true, args);
    session.startInfoReporter();
    state.setCurrentSession(session);
  }

  // -------------------------------------------------------------------------------------------
  // stop : arrêt de la recherche en cours
  // -------------------------------------------------------------------------------------------

  private static HandleAction handleStop(UciAdapterState state) {
    UciSession session = state.currentSession();
    if (session == null || session.isBestmoveEmitted()) {
      // Tolérance : stop sans session active ou déjà émis → silently ignore (cf. SPEC §8.5
      // testStopWithoutActiveSearch).
      return HandleAction.CONTINUE;
    }
    session.onUciStop();
    return HandleAction.CONTINUE;
  }

  // -------------------------------------------------------------------------------------------
  // ponderhit : transition ponder → search réelle
  // -------------------------------------------------------------------------------------------

  private static HandleAction handlePonderHit(UciAdapterState state) {
    UciSession session = state.currentSession();
    if (session == null || !session.isPonder() || session.isBestmoveEmitted()) {
      // Tolérance : ponderhit hors contexte ponder → silently ignore.
      if (state.debugMode()) {
        System.err.println("[debug] ponderhit ignored (no ponder session active)");
      }
      return HandleAction.CONTINUE;
    }
    GameState pos = state.lastPosition();
    int sideToMove =
        (pos != null) ? pos.currentPosition().sideToMove() : org.nanozero.board.Color.WHITE;
    SearchBudget realBudget =
        TimeManagementPolicy.computeBudget(session.originalGoArgs(), sideToMove, state.options());
    state.engine().ponderhit(realBudget);
    return HandleAction.CONTINUE;
  }

  // -------------------------------------------------------------------------------------------
  // setoption : mute UciOptionsState
  // -------------------------------------------------------------------------------------------

  private static HandleAction handleSetOption(UciCommand.SetOption so, UciAdapterState state) {
    state.options().set(so.name(), so.value());
    if (state.debugMode()) {
      System.err.println("[debug] setoption " + so.name() + " = " + so.value());
    }
    return HandleAction.CONTINUE;
  }

  // -------------------------------------------------------------------------------------------
  // debug on/off : flip le flag debug
  // -------------------------------------------------------------------------------------------

  private static HandleAction handleDebug(UciCommand.Debug d, UciAdapterState state) {
    state.setDebugMode(d.enabled());
    if (d.enabled()) {
      System.err.println("[debug] mode enabled");
    }
    return HandleAction.CONTINUE;
  }

  // -------------------------------------------------------------------------------------------
  // unknown : silently ignored (cf. SPEC §1.2 tolérance UCI)
  // -------------------------------------------------------------------------------------------

  private static HandleAction handleUnknown(UciCommand.Unknown u, UciAdapterState state) {
    if (state.debugMode()) {
      System.err.println("[debug] unknown command ignored: \"" + u.rawLine() + "\"");
    }
    return HandleAction.CONTINUE;
  }
}
