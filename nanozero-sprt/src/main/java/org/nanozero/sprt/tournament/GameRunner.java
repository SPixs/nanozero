package org.nanozero.sprt.tournament;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.nanozero.board.GameState;
import org.nanozero.board.MoveGen;
import org.nanozero.board.Result;
import org.nanozero.engine.Engine;
import org.nanozero.engine.SearchBudget;
import org.nanozero.engine.SearchResult;
import org.nanozero.sprt.stats.GameOutcome;

/**
 * Joue une game complète entre deux engines (challenger et baseline) et retourne le {@link
 * GameResult}.
 *
 * <p>Cf. SPEC-sprt §3.1 (Game class).
 *
 * <p><strong>Pattern d'usage</strong> :
 *
 * <pre>{@code
 * try (Engine challenger = new Engine(challengerNet, config);
 *      Engine baseline = new Engine(baselineNet, config)) {
 *   GameRunner runner = new GameRunner(challenger, baseline, "startpos",
 *       challengerWhite, timeControl, maxPlies);
 *   GameResult result = runner.play();
 * }
 * }</pre>
 *
 * <p><strong>Threading</strong> : non thread-safe. Chaque instance gère 1 game, 1 thread.
 *
 * <p><strong>Crash handling</strong> : si un engine throws RuntimeException pendant {@code
 * searchSync}, la game est abandonnée avec {@link GameResult.Termination#ENGINE_CRASH} et le
 * challenger est déclaré perdant (convention défensive : on rejette tout réseau qui crash).
 */
public final class GameRunner {

  /**
   * Limite plies par défaut (200 plies = 100 full moves). Au-delà, la game est abandonnée. Évite
   * les games infinies dans les cas pathologiques (engines évitent les pat, etc.).
   */
  public static final int DEFAULT_MAX_PLIES = 400;

  private final Engine challenger;
  private final Engine baseline;
  private final boolean challengerWhite;
  private final TimeControl timeControl;
  private final int maxPlies;
  private final GameState gameState;
  private final String initialFen;

  /**
   * Construit un GameRunner.
   *
   * @param challenger engine challenger (peut jouer blanc ou noir selon {@code challengerWhite})
   * @param baseline engine baseline
   * @param initialFen FEN de la position initiale (typiquement "startpos" ou opening book sample)
   * @param challengerWhite si {@code true}, challenger joue les blancs
   * @param timeControl time control par engine
   * @param maxPlies limite de plies pour éviter games infinies (default {@link #DEFAULT_MAX_PLIES})
   * @throws NullPointerException si un argument requis est null
   * @throws IllegalArgumentException si initialFen invalide
   */
  public GameRunner(
      Engine challenger,
      Engine baseline,
      String initialFen,
      boolean challengerWhite,
      TimeControl timeControl,
      int maxPlies) {
    this.challenger = Objects.requireNonNull(challenger, "challenger must not be null");
    this.baseline = Objects.requireNonNull(baseline, "baseline must not be null");
    this.initialFen = Objects.requireNonNull(initialFen, "initialFen must not be null");
    this.challengerWhite = challengerWhite;
    this.timeControl = Objects.requireNonNull(timeControl, "timeControl must not be null");
    if (maxPlies <= 0) {
      throw new IllegalArgumentException("maxPlies must be > 0, got " + maxPlies);
    }
    this.maxPlies = maxPlies;
    this.gameState = new GameState();
    if (!"startpos".equals(initialFen)) {
      this.gameState.setFromFen(initialFen);
    }
  }

  /**
   * Joue la game jusqu'à termination et retourne le résultat.
   *
   * @return résultat de la game (outcome POV challenger, termination, moves, etc.)
   */
  public GameResult play() {
    EngineClock challengerClock = new EngineClock(timeControl);
    EngineClock baselineClock = new EngineClock(timeControl);
    List<Integer> movesPlayed = new ArrayList<>(maxPlies);

    int ply = 0;
    GameOutcome outcome = null;
    GameResult.Termination termination = null;

    // Boucle game.
    while (ply < maxPlies) {
      // Check terminal AVANT de tenter un coup.
      Result currentResult = gameState.getResult();
      if (currentResult != Result.IN_PROGRESS) {
        termination = mapTermination(currentResult);
        outcome = mapOutcome(currentResult);
        break;
      }

      // Détermine qui joue maintenant.
      boolean whiteToMove = gameState.currentPosition().sideToMove() == 0; // Color.WHITE = 0
      boolean isChallengerTurn = whiteToMove == challengerWhite;
      Engine currentEngine = isChallengerTurn ? challenger : baseline;
      EngineClock currentClock = isChallengerTurn ? challengerClock : baselineClock;

      // Vérif temps suffisant.
      if (!currentClock.hasEnoughTime()) {
        currentClock.forceTimeForfeit();
        termination = GameResult.Termination.TIME_FORFEIT;
        outcome = isChallengerTurn ? GameOutcome.LOSS : GameOutcome.WIN;
        break;
      }

      // Search budget = budget restant de l'engine (ou unlimited).
      SearchBudget budget = budgetFromClock(currentClock);

      // Lance le search synchrone + capture crash.
      currentClock.startMove();
      SearchResult result;
      try {
        result = currentEngine.searchSync(gameState, budget);
      } catch (RuntimeException e) {
        // Engine crash — convention défensive : challenger perd.
        currentClock.endMove(); // discard timing
        termination = GameResult.Termination.ENGINE_CRASH;
        outcome = isChallengerTurn ? GameOutcome.LOSS : GameOutcome.WIN;
        break;
      }
      boolean forfeit = currentClock.endMove();
      if (forfeit) {
        termination = GameResult.Termination.TIME_FORFEIT;
        outcome = isChallengerTurn ? GameOutcome.LOSS : GameOutcome.WIN;
        break;
      }

      int move = result.bestMove();
      // bestMove == 0 = engine n'a complété aucune simulation (timeout cold start).
      // Convention défensive : équivalent à un time forfeit (engine n'a pas pu jouer).
      if (move == 0) {
        termination = GameResult.Termination.TIME_FORFEIT;
        outcome = isChallengerTurn ? GameOutcome.LOSS : GameOutcome.WIN;
        break;
      }
      if (!isLegalMove(move)) {
        // Coup illégal — engine fautif perd.
        termination = GameResult.Termination.ILLEGAL_MOVE;
        outcome = isChallengerTurn ? GameOutcome.LOSS : GameOutcome.WIN;
        break;
      }

      // Apply move.
      gameState.applyMove(move);
      movesPlayed.add(move);
      ply++;
    }

    // Si on a atteint maxPlies sans terminer.
    if (outcome == null) {
      termination = GameResult.Termination.ABORTED;
      outcome = GameOutcome.DRAW;
    }

    return new GameResult(
        outcome,
        termination,
        movesPlayed,
        initialFen,
        gameState.toFen(),
        challengerWhite,
        wallClock(challengerClock),
        wallClock(baselineClock),
        ply);
  }

  // ============================ Helpers internes ============================

  /** Time used = baseTime initial - remaining (ou Duration.ZERO si unlimited). */
  private Duration wallClock(EngineClock clock) {
    if (clock.config().unlimited()) {
      return Duration.ZERO;
    }
    return clock.config().baseTime().minus(clock.remaining());
  }

  /**
   * Min simulations garanties par coup. Évite que la deadline coupe avant qu'un coup légal soit
   * trouvé (cold start NN + JIT warmup). Stratégie : on accepte de dépasser légèrement le budget
   * temps pour garantir au moins {@link #MIN_SIMS_PER_MOVE} sims = bestMove légal. Si l'engine
   * prend trop longtemps pour faire ces sims, le forfeit time est détecté ensuite via {@link
   * EngineClock#endMove()}.
   */
  static final int MIN_SIMS_PER_MOVE = 16;

  /**
   * Convertit le budget restant de l'horloge en SearchBudget hybride : continue tant que (sims &lt;
   * {@link #MIN_SIMS_PER_MOVE}) OU (nanotime &lt; deadline). Stop dès que les deux conditions sont
   * fausses.
   *
   * <p>Sémantique : garantit au moins {@code MIN_SIMS_PER_MOVE} simulations même si la deadline est
   * dans le passé. Une fois ce minimum atteint, respecte la deadline pour les sims supplémentaires.
   */
  private static SearchBudget budgetFromClock(EngineClock clock) {
    if (clock.config().unlimited()) {
      return SearchBudget.nodes(Integer.MAX_VALUE);
    }
    long deadlineNanos = System.nanoTime() + clock.remaining().toNanos();
    return (sims, elapsedNanos) -> sims >= MIN_SIMS_PER_MOVE && System.nanoTime() >= deadlineNanos;
  }

  /** Mappe le {@link Result} board vers {@link GameOutcome} POV challenger. */
  private GameOutcome mapOutcome(Result result) {
    return switch (result) {
      case WIN_WHITE -> challengerWhite ? GameOutcome.WIN : GameOutcome.LOSS;
      case WIN_BLACK -> challengerWhite ? GameOutcome.LOSS : GameOutcome.WIN;
      case DRAW -> GameOutcome.DRAW;
      case IN_PROGRESS ->
          throw new IllegalStateException("IN_PROGRESS not a valid terminal result");
    };
  }

  /** Mappe le {@link Result} vers la raison de termination. */
  private GameResult.Termination mapTermination(Result result) {
    return switch (result) {
      case WIN_WHITE, WIN_BLACK -> GameResult.Termination.CHECKMATE;
      case DRAW -> classifyDrawTermination();
      case IN_PROGRESS ->
          throw new IllegalStateException("IN_PROGRESS not a valid terminal result");
    };
  }

  /** Sub-classify pourquoi la game est nulle (50-move/3-fold/insufficient material/stalemate). */
  private GameResult.Termination classifyDrawTermination() {
    if (gameState.isFiftyMoveRule()) return GameResult.Termination.FIFTY_MOVE_RULE;
    if (gameState.isInsufficientMaterial()) return GameResult.Termination.INSUFFICIENT_MATERIAL;
    if (gameState.isRepetition(3)) return GameResult.Termination.THREEFOLD_REPETITION;
    if (gameState.isStalemate()) return GameResult.Termination.STALEMATE;
    return GameResult.Termination.ABORTED; // fallback theoretical
  }

  /** Vérifie qu'un coup est légal dans la position courante. */
  private boolean isLegalMove(int move) {
    int[] buffer = new int[256];
    int count = MoveGen.generateMoves(gameState.currentPosition(), buffer, 0);
    for (int i = 0; i < count; i++) {
      if (buffer[i] == move) return true;
    }
    return false;
  }
}
