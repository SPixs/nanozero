package org.nanozero.sprt.tournament;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Duration;
import java.util.List;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.sprt.stats.GameOutcome;

class GameResultTest {

  @Test
  @DisplayName("constructor stores all fields + immutable moves list")
  void constructorImmutable() {
    List<Integer> moves = new java.util.ArrayList<>();
    moves.add(123);
    moves.add(456);
    GameResult r =
        new GameResult(
            GameOutcome.WIN,
            GameResult.Termination.CHECKMATE,
            moves,
            "startpos",
            "8/8/8/8/8/8/8/K1k5 w - - 0 1",
            true,
            Duration.ofSeconds(5),
            Duration.ofSeconds(4),
            2);
    assertThat(r.outcome()).isEqualTo(GameOutcome.WIN);
    assertThat(r.termination()).isEqualTo(GameResult.Termination.CHECKMATE);
    assertThat(r.movesPlayed()).containsExactly(123, 456);
    assertThat(r.totalMoves()).isEqualTo(2);
    assertThat(r.challengerWhite()).isTrue();
    assertThat(r.challengerTimeUsed()).isEqualTo(Duration.ofSeconds(5));
    assertThat(r.baselineTimeUsed()).isEqualTo(Duration.ofSeconds(4));
    // List is immutable (defensive copy).
    assertThatThrownBy(() -> r.movesPlayed().add(789))
        .isInstanceOf(UnsupportedOperationException.class);
    // External modif doesn't affect record.
    moves.add(789);
    assertThat(r.movesPlayed()).hasSize(2);
  }

  @Test
  @DisplayName("null outcome throws")
  void nullOutcomeThrows() {
    assertThatThrownBy(
            () ->
                new GameResult(
                    null,
                    GameResult.Termination.CHECKMATE,
                    List.of(),
                    "startpos",
                    "startpos",
                    true,
                    Duration.ZERO,
                    Duration.ZERO,
                    0))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("outcome");
  }

  @Test
  @DisplayName("null termination throws")
  void nullTerminationThrows() {
    assertThatThrownBy(
            () ->
                new GameResult(
                    GameOutcome.DRAW,
                    null,
                    List.of(),
                    "startpos",
                    "startpos",
                    true,
                    Duration.ZERO,
                    Duration.ZERO,
                    0))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("termination");
  }

  @Test
  @DisplayName("negative totalMoves throws")
  void negativeTotalMovesThrows() {
    assertThatThrownBy(
            () ->
                new GameResult(
                    GameOutcome.DRAW,
                    GameResult.Termination.STALEMATE,
                    List.of(),
                    "startpos",
                    "startpos",
                    true,
                    Duration.ZERO,
                    Duration.ZERO,
                    -1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("totalMoves");
  }

  @Test
  @DisplayName("empty moves list ok (game abort before move 1)")
  void emptyMovesOk() {
    GameResult r =
        new GameResult(
            GameOutcome.DRAW,
            GameResult.Termination.ABORTED,
            List.of(),
            "startpos",
            "startpos",
            true,
            Duration.ZERO,
            Duration.ZERO,
            0);
    assertThat(r.movesPlayed()).isEmpty();
    assertThat(r.totalMoves()).isEqualTo(0);
  }

  @Test
  @DisplayName("all 9 termination values defined")
  void allTerminationValues() {
    GameResult.Termination[] values = GameResult.Termination.values();
    assertThat(values).hasSize(9);
    assertThat(values)
        .containsExactlyInAnyOrder(
            GameResult.Termination.CHECKMATE,
            GameResult.Termination.STALEMATE,
            GameResult.Termination.FIFTY_MOVE_RULE,
            GameResult.Termination.THREEFOLD_REPETITION,
            GameResult.Termination.INSUFFICIENT_MATERIAL,
            GameResult.Termination.TIME_FORFEIT,
            GameResult.Termination.ILLEGAL_MOVE,
            GameResult.Termination.ENGINE_CRASH,
            GameResult.Termination.ABORTED);
  }
}
