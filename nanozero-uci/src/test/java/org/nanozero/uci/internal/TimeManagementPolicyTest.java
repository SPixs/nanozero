package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import java.util.OptionalInt;
import java.util.OptionalLong;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.Color;
import org.nanozero.engine.SearchBudget;

/**
 * Tests unitaires de {@link TimeManagementPolicy} (cf. SPEC §5.5, §12 phase 5).
 *
 * <p>Tests par mode (priorité décroissante) + cumuls (priorité explicite vérifiée) + garde-fous
 * (floor / cap / moveOverhead) + validation arguments.
 *
 * <p>Stratégie de test : on observe le {@link SearchBudget} produit via son comportement {@link
 * SearchBudget#shouldStop} (pas de getters internes — c'est l'unique surface fonctionnelle). Pour
 * les budgets {@code duration}, on teste avec {@code shouldStop(0, thresholdNanos)} pour vérifier
 * la borne. Pour {@code nodes}, {@code shouldStop(N-1, 0)} et {@code shouldStop(N, 0)}. Pour {@code
 * UNLIMITED}, {@code shouldStop(MAX, MAX)} reste false.
 */
class TimeManagementPolicyTest {

  // -------------------------------------------------------------------------------------------
  // Helpers : construction d'un GoArgs avec 1 ou 2 paramètres
  // -------------------------------------------------------------------------------------------

  /** Builder utilitaire pour construire des GoArgs partiels avec quelques params seulement. */
  private static final class GoArgsBuilder {
    OptionalLong wtime = OptionalLong.empty();
    OptionalLong btime = OptionalLong.empty();
    OptionalLong winc = OptionalLong.empty();
    OptionalLong binc = OptionalLong.empty();
    OptionalInt movestogo = OptionalInt.empty();
    OptionalLong movetime = OptionalLong.empty();
    OptionalInt nodes = OptionalInt.empty();
    OptionalInt depth = OptionalInt.empty();
    boolean infinite;
    boolean ponder;

    GoArgsBuilder wtime(long ms) {
      this.wtime = OptionalLong.of(ms);
      return this;
    }

    GoArgsBuilder btime(long ms) {
      this.btime = OptionalLong.of(ms);
      return this;
    }

    GoArgsBuilder winc(long ms) {
      this.winc = OptionalLong.of(ms);
      return this;
    }

    GoArgsBuilder binc(long ms) {
      this.binc = OptionalLong.of(ms);
      return this;
    }

    GoArgsBuilder movestogo(int n) {
      this.movestogo = OptionalInt.of(n);
      return this;
    }

    GoArgsBuilder movetime(long ms) {
      this.movetime = OptionalLong.of(ms);
      return this;
    }

    GoArgsBuilder nodes(int n) {
      this.nodes = OptionalInt.of(n);
      return this;
    }

    GoArgsBuilder depth(int n) {
      this.depth = OptionalInt.of(n);
      return this;
    }

    GoArgsBuilder infinite() {
      this.infinite = true;
      return this;
    }

    GoArgsBuilder ponder() {
      this.ponder = true;
      return this;
    }

    GoArgs build() {
      return new GoArgs(
          wtime, btime, winc, binc, movestogo, movetime, nodes, depth, infinite, ponder, List.of());
    }
  }

  /** Vérifie que {@code budget} est un duration de {@code expectedMs} via shouldStop. */
  private static void assertDurationBudget(SearchBudget budget, long expectedMs) {
    long limit = expectedMs * 1_000_000L;
    assertThat(budget.shouldStop(0, limit - 1)).as("avant limite").isFalse();
    assertThat(budget.shouldStop(0, limit)).as("à limite").isTrue();
  }

  /** Vérifie que {@code budget} est un nodes de {@code expected} via shouldStop. */
  private static void assertNodesBudget(SearchBudget budget, int expected) {
    if (expected > 0) {
      assertThat(budget.shouldStop(expected - 1, 0L)).as("avant limite").isFalse();
    }
    assertThat(budget.shouldStop(expected, 0L)).as("à limite").isTrue();
  }

  // -------------------------------------------------------------------------------------------
  // Modes par priorité (1-3)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("movetime 5000 → durée 5000ms")
  void movetime() {
    var args = new GoArgsBuilder().movetime(5_000).build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    assertDurationBudget(budget, 5_000);
  }

  @Test
  @DisplayName("nodes 1000 → SearchBudget.nodes(1000)")
  void nodes() {
    var args = new GoArgsBuilder().nodes(1_000).build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    assertNodesBudget(budget, 1_000);
  }

  @Test
  @DisplayName("infinite → UNLIMITED, jamais shouldStop")
  void infinite() {
    var args = new GoArgsBuilder().infinite().build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    assertThat(budget.shouldStop(Integer.MAX_VALUE, Long.MAX_VALUE)).isFalse();
  }

  @Test
  @DisplayName("ponder → UNLIMITED (idem infinite)")
  void ponder() {
    var args = new GoArgsBuilder().ponder().build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    assertThat(budget.shouldStop(Integer.MAX_VALUE, Long.MAX_VALUE)).isFalse();
  }

  // -------------------------------------------------------------------------------------------
  // Priorités cumulées
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("infinite + wtime → UNLIMITED (priorité 3 sur 4)")
  void infinitePrimesOverTimeControl() {
    var args = new GoArgsBuilder().infinite().wtime(10_000).build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    assertThat(budget.shouldStop(Integer.MAX_VALUE, Long.MAX_VALUE)).isFalse();
  }

  @Test
  @DisplayName("movetime + infinite → durée movetime (priorité 1 sur 3)")
  void movetimePrimesOverInfinite() {
    var args = new GoArgsBuilder().movetime(1_000).infinite().build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    assertDurationBudget(budget, 1_000);
  }

  @Test
  @DisplayName("nodes + wtime → nodes (priorité 2 sur 4)")
  void nodesPrimesOverTimeControl() {
    var args = new GoArgsBuilder().nodes(500).wtime(60_000).build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    assertNodesBudget(budget, 500);
  }

  // -------------------------------------------------------------------------------------------
  // Time control standard (formule allocate)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Time control white sans movestogo : formule avec divisor=30")
  void timeControlWhiteNoMovestogo() {
    var args = new GoArgsBuilder().wtime(60_000).winc(1_000).build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    // allocated = 60000/30 + 1000 - 30 = 2970ms ; cap = 54000 ; floor = 50.
    assertDurationBudget(budget, 2_970);
  }

  @Test
  @DisplayName("Time control black : utilise btime/binc, pas wtime/winc")
  void timeControlBlackUsesBlackTime() {
    var args =
        new GoArgsBuilder()
            .wtime(999_999) // ne doit PAS être utilisé
            .winc(999_999)
            .btime(60_000)
            .binc(1_000)
            .build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.BLACK, new UciOptionsState());
    // allocated = 60000/30 + 1000 - 30 = 2970ms (basé sur btime/binc)
    assertDurationBudget(budget, 2_970);
  }

  @Test
  @DisplayName("Time control avec movestogo : divisor explicit")
  void timeControlWithMovestogo() {
    var args = new GoArgsBuilder().wtime(60_000).winc(500).movestogo(10).build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    // allocated = 60000/10 + 500 - 30 = 6470ms
    assertDurationBudget(budget, 6_470);
  }

  @Test
  @DisplayName("Time control sans winc : winc absent traité comme 0")
  void timeControlNoIncrement() {
    var args = new GoArgsBuilder().wtime(60_000).movestogo(30).build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    // allocated = 60000/30 + 0 - 30 = 1970ms
    assertDurationBudget(budget, 1_970);
  }

  // -------------------------------------------------------------------------------------------
  // Garde-fous : floor / cap / moveOverhead
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Floor 50ms appliqué sur allocated négatif (low-time)")
  void floorAppliedOnNegativeAllocated() {
    var args = new GoArgsBuilder().wtime(100).build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    // allocated = 100/30 + 0 - 30 = 3 - 30 = -27 ; cap = 90 ; floor = 50.
    // bounded = max(50, min(-27, 90)) = max(50, -27) = 50.
    assertDurationBudget(budget, 50);
  }

  @Test
  @DisplayName("Cap 0.9×myTime appliqué sur allocated trop large (winc énorme)")
  void capAppliedOnLargeAllocated() {
    var args = new GoArgsBuilder().wtime(1_000).winc(2_000).movestogo(1).build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    // allocated = 1000/1 + 2000 - 30 = 2970 ; cap = 900 ; floor = 50.
    // bounded = max(50, min(2970, 900)) = max(50, 900) = 900.
    assertDurationBudget(budget, 900);
  }

  @Test
  @DisplayName("Floor prime sur cap quand low-time avec inc (cap < floor)")
  void floorPrimesOverCapInLowTime() {
    var args = new GoArgsBuilder().wtime(10).winc(200).movestogo(30).build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    // allocated = 10/30 + 200 - 30 = 0 + 200 - 30 = 170 ; cap = (long)(10*0.9) = 9 ; floor = 50.
    // bounded = max(50, min(170, 9)) = max(50, 9) = 50.
    assertDurationBudget(budget, 50);
  }

  @Test
  @DisplayName("moveOverhead UCI option soustrait dans la formule")
  void moveOverheadHonored() {
    var options = new UciOptionsState();
    options.set("Move Overhead", "100");
    var args = new GoArgsBuilder().wtime(60_000).movestogo(30).build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, options);
    // allocated = 60000/30 + 0 - 100 = 1900ms
    assertDurationBudget(budget, 1_900);
  }

  @Test
  @DisplayName("moveOverhead large + low-time : floor absorbe la valeur négative")
  void largeMoveOverheadWithLowTime() {
    var options = new UciOptionsState();
    options.set("Move Overhead", "5000"); // max
    var args = new GoArgsBuilder().wtime(30).build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, options);
    // allocated = 30/30 + 0 - 5000 = 1 - 5000 = -4999 ; cap = 27 ; floor = 50.
    // bounded = max(50, min(-4999, 27)) = max(50, -4999) = 50.
    assertDurationBudget(budget, 50);
  }

  // -------------------------------------------------------------------------------------------
  // Modes 5 et 6 : depth et fallback
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("depth 5 → best-effort SearchBudget.nodes(500)")
  void depthBestEffort() {
    var args = new GoArgsBuilder().depth(5).build();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    assertNodesBudget(budget, 500);
  }

  @Test
  @DisplayName("Aucun param → fallback durée 5000ms")
  void fallbackNoArgs() {
    var args = GoArgs.empty();
    var budget = TimeManagementPolicy.computeBudget(args, Color.WHITE, new UciOptionsState());
    assertDurationBudget(budget, 5_000);
  }

  // -------------------------------------------------------------------------------------------
  // Validation arguments
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("args null → NPE")
  void nullArgsThrowsNpe() {
    assertThatThrownBy(
            () -> TimeManagementPolicy.computeBudget(null, Color.WHITE, new UciOptionsState()))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("args");
  }

  @Test
  @DisplayName("options null → NPE")
  void nullOptionsThrowsNpe() {
    assertThatThrownBy(() -> TimeManagementPolicy.computeBudget(GoArgs.empty(), Color.WHITE, null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("options");
  }

  @Test
  @DisplayName("sideToMove invalide (≠ WHITE/BLACK) → IAE")
  void invalidSideToMoveThrowsIae() {
    assertThatThrownBy(
            () -> TimeManagementPolicy.computeBudget(GoArgs.empty(), 42, new UciOptionsState()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("sideToMove");
  }
}
