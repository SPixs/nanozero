package org.nanozero.sprt.tournament;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Duration;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class EngineClockTest {

  @Test
  @DisplayName("clock starts with baseTime as remaining")
  void clockStartsWithBaseTime() {
    TimeControl tc = TimeControl.fischer(Duration.ofSeconds(10), Duration.ofMillis(100));
    EngineClock clock = new EngineClock(tc);
    assertThat(clock.remaining()).isEqualTo(Duration.ofSeconds(10));
    assertThat(clock.movesPlayed()).isEqualTo(0);
    assertThat(clock.isForfeit()).isFalse();
    assertThat(clock.config()).isEqualTo(tc);
  }

  @Test
  @DisplayName("move applies elapsed + adds Fischer increment")
  void moveAppliesIncrement() throws InterruptedException {
    TimeControl tc = TimeControl.fischer(Duration.ofSeconds(10), Duration.ofMillis(100));
    EngineClock clock = new EngineClock(tc);
    clock.startMove();
    Thread.sleep(50); // ~50ms move
    boolean forfeit = clock.endMove();
    assertThat(forfeit).isFalse();
    assertThat(clock.movesPlayed()).isEqualTo(1);
    // remaining should be 10s - ~50ms + 100ms ≈ 10.05s
    assertThat(clock.remaining()).isGreaterThan(Duration.ofMillis(9_990));
    assertThat(clock.remaining()).isLessThan(Duration.ofMillis(10_110));
  }

  @Test
  @DisplayName("forfeit when remaining hits zero")
  void forfeitOnTimeOut() throws InterruptedException {
    TimeControl tc = TimeControl.sudden(Duration.ofMillis(80));
    EngineClock clock = new EngineClock(tc);
    clock.startMove();
    Thread.sleep(100); // exceeds 80ms
    boolean forfeit = clock.endMove();
    assertThat(forfeit).isTrue();
    assertThat(clock.isForfeit()).isTrue();
    assertThat(clock.remaining()).isLessThanOrEqualTo(Duration.ZERO);
  }

  @Test
  @DisplayName("startMove twice without endMove throws")
  void doubleStartMoveThrows() {
    EngineClock clock = new EngineClock(TimeControl.fischer(Duration.ofSeconds(1), Duration.ZERO));
    clock.startMove();
    assertThatThrownBy(clock::startMove)
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("startMove");
  }

  @Test
  @DisplayName("endMove without startMove throws")
  void endMoveWithoutStartThrows() {
    EngineClock clock = new EngineClock(TimeControl.fischer(Duration.ofSeconds(1), Duration.ZERO));
    assertThatThrownBy(clock::endMove)
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("endMove");
  }

  @Test
  @DisplayName("startMove on already forfeitted clock throws")
  void startOnForfeitClock() {
    EngineClock clock = new EngineClock(TimeControl.sudden(Duration.ofMillis(10)));
    clock.forceTimeForfeit();
    assertThatThrownBy(clock::startMove)
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("forfeit");
  }

  @Test
  @DisplayName("budgetForNextMove returns remaining (Fischer/sudden)")
  void budgetReturnsRemaining() {
    TimeControl tc = TimeControl.fischer(Duration.ofSeconds(5), Duration.ofMillis(50));
    EngineClock clock = new EngineClock(tc);
    assertThat(clock.budgetForNextMove()).isEqualTo(Duration.ofSeconds(5));
  }

  @Test
  @DisplayName("budgetForNextMove returns ZERO for unlimited mode")
  void budgetUnlimited() {
    EngineClock clock = new EngineClock(TimeControl.infinite());
    assertThat(clock.budgetForNextMove()).isEqualTo(Duration.ZERO);
  }

  @Test
  @DisplayName("budgetForNextMove on forfeit throws")
  void budgetOnForfeitThrows() {
    EngineClock clock = new EngineClock(TimeControl.sudden(Duration.ofSeconds(1)));
    clock.forceTimeForfeit();
    assertThatThrownBy(clock::budgetForNextMove)
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("forfeit");
  }

  @Test
  @DisplayName("hasEnoughTime returns false below MIN_BUDGET")
  void hasEnoughTimeBelowMin() {
    EngineClock clock = new EngineClock(TimeControl.sudden(Duration.ofMillis(30))); // < 50ms MIN
    assertThat(clock.hasEnoughTime()).isFalse();
  }

  @Test
  @DisplayName("hasEnoughTime returns true above MIN_BUDGET")
  void hasEnoughTimeAboveMin() {
    EngineClock clock = new EngineClock(TimeControl.sudden(Duration.ofMillis(200)));
    assertThat(clock.hasEnoughTime()).isTrue();
  }

  @Test
  @DisplayName("hasEnoughTime always true for unlimited")
  void hasEnoughTimeUnlimited() {
    EngineClock clock = new EngineClock(TimeControl.infinite());
    assertThat(clock.hasEnoughTime()).isTrue();
  }

  @Test
  @DisplayName("unlimited mode never forfeits")
  void unlimitedNeverForfeits() throws InterruptedException {
    EngineClock clock = new EngineClock(TimeControl.infinite());
    clock.startMove();
    Thread.sleep(100);
    assertThat(clock.endMove()).isFalse();
    assertThat(clock.isForfeit()).isFalse();
  }

  @Test
  @DisplayName("classical TC resets baseTime every movesPerControl moves")
  void classicalReset() {
    // 2 moves per 100ms — after 2nd move, +100ms reset bonus
    TimeControl tc = new TimeControl(Duration.ofMillis(500), Duration.ZERO, 2, false);
    EngineClock clock = new EngineClock(tc);
    // Move 1 : 50ms, no reset
    clock.startMove();
    sleepUninterruptibly(50);
    clock.endMove();
    Duration afterMove1 = clock.remaining();
    // Move 2 : 50ms, reset triggered (move 2 % 2 == 0)
    clock.startMove();
    sleepUninterruptibly(50);
    clock.endMove();
    Duration afterMove2 = clock.remaining();
    // afterMove2 should be ~ afterMove1 - 50ms + 500ms (reset) = afterMove1 + 450ms
    assertThat(afterMove2).isGreaterThan(afterMove1.plusMillis(300));
  }

  @Test
  @DisplayName("forceTimeForfeit marks forfeit + clears move start")
  void forceTimeForfeit() {
    EngineClock clock = new EngineClock(TimeControl.sudden(Duration.ofSeconds(1)));
    clock.startMove(); // simulate in-progress move
    clock.forceTimeForfeit();
    assertThat(clock.isForfeit()).isTrue();
  }

  // ============================ Helpers ============================

  private static void sleepUninterruptibly(long millis) {
    try {
      Thread.sleep(millis);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }
}
