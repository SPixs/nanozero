package org.nanozero.sprt.tournament;

import java.time.Duration;
import java.util.List;
import java.util.Objects;
import org.nanozero.sprt.stats.GameOutcome;

/**
 * Résultat d'une game SPRT individuelle.
 *
 * <p>Encapsule l'outcome (du POV challenger), la raison de termination, la liste des coups joués
 * (encodage Move 16-bit), la position finale FEN, et les temps utilisés.
 *
 * <p>Cf. SPEC-sprt §3.1, §7 (PGN output).
 *
 * @param outcome résultat depuis le POV du challenger (WIN/LOSS/DRAW)
 * @param termination raison de la fin de game (cf. {@link Termination})
 * @param movesPlayed liste ordonnée des coups joués (encodage 16-bit Move ; vide si game abort
 *     avant move 1)
 * @param initialFen FEN de la position initiale (typiquement startpos ou opening book sample)
 * @param finalFen FEN de la position finale
 * @param challengerWhite {@code true} si challenger jouait blanc dans cette game
 * @param challengerTimeUsed total wall-clock challenger pendant les coups
 * @param baselineTimeUsed total wall-clock baseline pendant les coups
 * @param totalMoves nombre total de plies joués
 */
public record GameResult(
    GameOutcome outcome,
    Termination termination,
    List<Integer> movesPlayed,
    String initialFen,
    String finalFen,
    boolean challengerWhite,
    Duration challengerTimeUsed,
    Duration baselineTimeUsed,
    int totalMoves) {

  /** Raison de la termination d'une game. */
  public enum Termination {
    /** Mat infligé à l'un des camps. */
    CHECKMATE,
    /** Pat (stalemate). */
    STALEMATE,
    /** Règle des 50 coups. */
    FIFTY_MOVE_RULE,
    /** Répétition triple. */
    THREEFOLD_REPETITION,
    /** Matériel insuffisant pour le mat. */
    INSUFFICIENT_MATERIAL,
    /** Forfait au temps : un engine a dépassé son budget. */
    TIME_FORFEIT,
    /** Coup illégal joué par un engine (rare, cf. SPEC §5.3 edge cases). */
    ILLEGAL_MOVE,
    /** Crash engine (RuntimeException pendant search) — challenger perd. */
    ENGINE_CRASH,
    /** Game abandonnée pour autre raison (max plies atteint, par exemple). */
    ABORTED
  }

  /** Validation des arguments au compact constructor. */
  public GameResult {
    Objects.requireNonNull(outcome, "outcome must not be null");
    Objects.requireNonNull(termination, "termination must not be null");
    Objects.requireNonNull(movesPlayed, "movesPlayed must not be null");
    Objects.requireNonNull(initialFen, "initialFen must not be null");
    Objects.requireNonNull(finalFen, "finalFen must not be null");
    Objects.requireNonNull(challengerTimeUsed, "challengerTimeUsed must not be null");
    Objects.requireNonNull(baselineTimeUsed, "baselineTimeUsed must not be null");
    if (totalMoves < 0) {
      throw new IllegalArgumentException("totalMoves must be ≥ 0, got " + totalMoves);
    }
    movesPlayed = List.copyOf(movesPlayed); // defensive copy + immutable
  }
}
