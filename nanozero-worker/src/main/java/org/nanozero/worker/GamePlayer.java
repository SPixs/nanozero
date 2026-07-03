package org.nanozero.worker;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.nanozero.board.Color;
import org.nanozero.board.GameState;
import org.nanozero.board.Result;
import org.nanozero.engine.Engine;
import org.nanozero.engine.EngineConfig;
import org.nanozero.engine.SearchBudget;
import org.nanozero.engine.SearchResult;
import org.nanozero.nn.MoveEncoding;
import org.nanozero.nn.Network;

/**
 * Plays one self-play game using the embedded {@link Engine}. No UCI subprocess.
 *
 * <p>For each ply:
 *
 * <ol>
 *   <li>Encode the current position into 119 NN planes ({@link GameState#toPlanes}).
 *   <li>Run MCTS for {@code numSims} simulations via {@link Engine#searchSync}.
 *   <li>Convert {@code SearchResult.childVisits} → {@code policy_target[4672]} via {@link
 *       MoveEncoding#encode}.
 *   <li>Pick a move (temperature sampling before {@code temperatureSwitchPly}, argmax after).
 *   <li>Apply the move and repeat.
 * </ol>
 *
 * <p>When the game ends (terminal position or {@code maxPlies}), back-fill {@code
 * Sample.valueTarget} from the side-to-move's POV.
 */
public final class GamePlayer {

  /** Default MCTS simulations per move. Overridden per job by {@link #play}. */
  public static final int DEFAULT_NUM_SIMS = 200;

  /** Default game-length cap (plies). Aborts the game as a draw if hit. */
  public static final int DEFAULT_MAX_PLIES = 512;

  /** Default temperature in the opening (stochastic move sampling). */
  public static final float DEFAULT_TEMPERATURE = 1.0f;

  /** After this ply, switch to argmax (greedy) move selection. */
  public static final int DEFAULT_TEMPERATURE_SWITCH_PLY = 30;

  /** Pre-allocated buffer for {@code GameState.generateMoves}. */
  private final int[] legalMovesBuf = new int[256];

  /**
   * Play a single complete game and return the collected samples (value_target backfilled).
   *
   * @param network the loaded NN to use for MCTS evaluation.
   * @param engineConfig MCTS hyperparameters (cPuct, Dirichlet, ...).
   * @param numSims MCTS simulations per move (≥ 1).
   * @param maxPlies hard cap on plies — game is declared a draw if hit.
   * @param temperature initial temperature for move sampling (≥ 0 ; ≤ 0.01 means argmax).
   * @param temperatureSwitchPly after this ply, switch to argmax regardless of temperature.
   * @param rng RNG for temperature-based move sampling.
   * @return list of samples (length = plies played), with {@code valueTarget} backfilled.
   */
  public List<Sample> play(
      Network network,
      EngineConfig engineConfig,
      int numSims,
      int maxPlies,
      float temperature,
      int temperatureSwitchPly,
      Random rng) {

    if (numSims < 1) throw new IllegalArgumentException("numSims must be >= 1, got " + numSims);
    if (maxPlies < 1) throw new IllegalArgumentException("maxPlies must be >= 1, got " + maxPlies);
    if (rng == null) throw new IllegalArgumentException("rng must not be null");

    GameState state = new GameState();
    List<Sample> samples = new ArrayList<>();
    SearchBudget budget = SearchBudget.nodes(numSims);

    EngineConfig gameConfig = perGameConfig(engineConfig, rng);

    try (Engine engine = new Engine(network, gameConfig)) {
      while (samples.size() < maxPlies && !state.isTerminal()) {
        int turn = state.currentPosition().sideToMove();
        int ply = samples.size();

        // 1. Encode the current position into 119 planes (~30 KB float32).
        float[] inputPlanes = new float[GameState.NN_PLANES * 64];
        state.toPlanes(inputPlanes, 0);

        // 2. Run MCTS for numSims simulations.
        SearchResult result = engine.searchSync(state, budget);

        // 3. Build policy_target[4672] from the visit distribution.
        float[] policyTarget = buildPolicyTarget(result.childMoves(), result.childVisits(), turn);

        // 4. Save the sample (value_target backfilled later).
        samples.add(new Sample(inputPlanes, policyTarget, turn, ply));

        // 5. Pick the next move.
        int chosenMove =
            pickMove(
                result.childMoves(),
                result.childVisits(),
                ply,
                temperature,
                temperatureSwitchPly,
                rng);

        state.applyMove(chosenMove);
      }
    }

    // Backfill value_target from this side-to-move's POV.
    backfillValueTargets(samples, gameOutcomeWhite(state));
    return samples;
  }

  /**
   * Derive the per-game engine config from the shared self-play config.
   *
   * <p>Per-game Dirichlet seed : a mono-thread engine reuses {@code engineConfig.randomSeed()}
   * verbatim, so without a fresh seed every game (and machine) would sample an identical noise
   * stream. When noise is off ({@code dirichletEpsilon == 0}) the config is returned untouched — no
   * {@code rng} draw — to preserve the bit-for-bit determinism guarantee (ADR-010).
   *
   * <p>The optional NN-evaluation cache size ({@code engineConfig.nnCacheSize()}, ADR-018) is
   * PRESERVED across the rebuild : it would otherwise silently drop back to {@code 0} (cache off)
   * whenever Dirichlet noise is enabled. Package-private for testing.
   *
   * @param engineConfig the shared self-play engine config.
   * @param rng RNG used to draw the per-game Dirichlet seed (only consulted when noise is on).
   * @return a per-game config (re-seeded when noise is on, else {@code engineConfig} unchanged).
   */
  static EngineConfig perGameConfig(EngineConfig engineConfig, Random rng) {
    return engineConfig.dirichletEpsilon() > 0f
        ? new EngineConfig(
            engineConfig.cPuct(),
            engineConfig.fpuValue(),
            engineConfig.treeInitialCapacity(),
            engineConfig.dirichletAlpha(),
            engineConfig.dirichletEpsilon(),
            rng.nextLong(),
            engineConfig.searchThreads(),
            engineConfig.batchSize(),
            engineConfig.virtualLoss(),
            engineConfig.nnCacheSize())
        : engineConfig;
  }

  /**
   * Convert {@code SearchResult.childMoves + childVisits} into a 4 672-float policy target
   * normalized to sum to 1.
   *
   * <p>{@link MoveEncoding#encode} maps each move (16-bit encoding) + side-to-move to the policy
   * index in {@code [0, 4671]}.
   *
   * @param moves legal moves at the root.
   * @param visits visit count per legal move (parallel array).
   * @param sideToMove WHITE (0) or BLACK (1).
   * @return normalized {@code float[4672]} policy target.
   */
  static float[] buildPolicyTarget(int[] moves, int[] visits, int sideToMove) {
    int total = 0;
    for (int v : visits) total += v;
    float[] policy = new float[MoveEncoding.POLICY_INDICES];
    if (total == 0) return policy; // pathological — return all zeros
    float invTotal = 1.0f / total;
    for (int i = 0; i < moves.length; i++) {
      int idx = MoveEncoding.encode(moves[i], sideToMove);
      policy[idx] = visits[i] * invTotal;
    }
    return policy;
  }

  /**
   * Pick a move : argmax visits if {@code ply >= temperatureSwitchPly} or {@code temperature <=
   * 0.01}, otherwise sample with weights {@code visits^(1/temperature)}.
   */
  static int pickMove(
      int[] moves, int[] visits, int ply, float temperature, int temperatureSwitchPly, Random rng) {
    if (moves.length == 0) {
      throw new IllegalStateException("pickMove on empty move list (caller missed terminal?)");
    }
    if (moves.length == 1) {
      return moves[0];
    }
    if (ply >= temperatureSwitchPly || temperature <= 0.01f) {
      return argmax(moves, visits);
    }
    return sampleWeighted(moves, visits, temperature, rng);
  }

  private static int argmax(int[] moves, int[] visits) {
    int bestIdx = 0;
    for (int i = 1; i < visits.length; i++) {
      if (visits[i] > visits[bestIdx]) bestIdx = i;
    }
    return moves[bestIdx];
  }

  private static int sampleWeighted(int[] moves, int[] visits, float temperature, Random rng) {
    double invT = 1.0 / temperature;
    double[] weights = new double[visits.length];
    double total = 0.0;
    for (int i = 0; i < visits.length; i++) {
      weights[i] = Math.pow(Math.max(visits[i], 0), invT);
      total += weights[i];
    }
    if (total <= 0.0) {
      // All visits == 0 (shouldn't happen with numSims ≥ 1, but be defensive).
      return moves[rng.nextInt(moves.length)];
    }
    double r = rng.nextDouble() * total;
    double cum = 0.0;
    for (int i = 0; i < weights.length; i++) {
      cum += weights[i];
      if (r <= cum) return moves[i];
    }
    return moves[moves.length - 1];
  }

  /**
   * Returns the game outcome from WHITE's POV : {@code +1} if white won, {@code -1} if black won,
   * {@code 0} otherwise (stalemate / draw / max-plies / non-terminal mid-game).
   */
  private static float gameOutcomeWhite(GameState state) {
    if (!state.isTerminal()) {
      return 0.0f; // hit max_plies without natural termination
    }
    Result r = state.getResult();
    return switch (r) {
      case WIN_WHITE -> 1.0f;
      case WIN_BLACK -> -1.0f;
      case DRAW -> 0.0f;
      case IN_PROGRESS -> 0.0f; // defensive — isTerminal() guard should prevent this
    };
  }

  /**
   * For each sample, set {@code valueTarget} to the outcome FROM THE SAMPLE'S SIDE-TO-MOVE POV.
   *
   * <p>If turn==WHITE at this ply, value = outcomeWhite ; if turn==BLACK, value = -outcomeWhite.
   */
  static void backfillValueTargets(List<Sample> samples, float outcomeWhite) {
    for (Sample s : samples) {
      s.valueTarget = (s.turn == Color.WHITE) ? outcomeWhite : -outcomeWhite;
    }
  }
}
