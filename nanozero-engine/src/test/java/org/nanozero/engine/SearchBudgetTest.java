package org.nanozero.engine;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Duration;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

class SearchBudgetTest {

  // ---------------------------------------------------------------------------------------------
  // nodes(N)
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("nodes(maxSimulations)")
  class NodesFactory {

    @Test
    @DisplayName("nodes(0) : stop immédiat (sims >= 0 trivialement)")
    void zeroMaxStopsImmediately() {
      SearchBudget b = SearchBudget.nodes(0);
      assertThat(b.shouldStop(0, 0L)).isTrue();
      assertThat(b.shouldStop(0, Long.MAX_VALUE)).isTrue();
    }

    @Test
    @DisplayName("nodes(100) : false avant 100, true à 100, ignore elapsedNanos")
    void typicalThreshold() {
      SearchBudget b = SearchBudget.nodes(100);
      assertThat(b.shouldStop(0, 0L)).isFalse();
      assertThat(b.shouldStop(99, Long.MAX_VALUE)).isFalse();
      assertThat(b.shouldStop(100, 0L)).isTrue();
      assertThat(b.shouldStop(101, 0L)).isTrue();
      assertThat(b.shouldStop(Integer.MAX_VALUE, 0L)).isTrue();
    }

    @Test
    @DisplayName("nodes(-1) → IAE")
    void negativeMaxRejected() {
      assertThatThrownBy(() -> SearchBudget.nodes(-1))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("maxSimulations")
          .hasMessageContaining(">= 0");
    }
  }

  // ---------------------------------------------------------------------------------------------
  // deadline(D)
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("deadline(deadlineNanos)")
  class DeadlineFactory {

    @Test
    @DisplayName("Avant la deadline : false ; après wallclock sleep : true")
    void wallClockComparison() throws InterruptedException {
      long deadline = System.nanoTime() + 200_000_000L; // +200ms
      SearchBudget b = SearchBudget.deadline(deadline);
      assertThat(b.shouldStop(0, 0L)).isFalse();
      Thread.sleep(250L);
      assertThat(b.shouldStop(0, 0L)).isTrue();
    }

    @Test
    @DisplayName("Deadline dans le passé : true immédiatement (ignore elapsedNanos)")
    void pastDeadlineStops() {
      SearchBudget b = SearchBudget.deadline(System.nanoTime() - 1_000_000_000L);
      assertThat(b.shouldStop(0, 0L)).isTrue();
      assertThat(b.shouldStop(Integer.MAX_VALUE, 0L)).isTrue();
    }
  }

  // ---------------------------------------------------------------------------------------------
  // duration(d)
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("duration(d)")
  class DurationFactory {

    @Test
    @DisplayName("duration(500ms) : seuil exact à elapsedNanos = toNanos()")
    void exactBoundary() {
      SearchBudget b = SearchBudget.duration(Duration.ofMillis(500));
      long limit = Duration.ofMillis(500).toNanos();
      assertThat(b.shouldStop(0, 0L)).isFalse();
      assertThat(b.shouldStop(0, limit - 1)).isFalse();
      assertThat(b.shouldStop(0, limit)).isTrue();
      assertThat(b.shouldStop(0, Long.MAX_VALUE)).isTrue();
    }

    @Test
    @DisplayName("duration(0) : stop immédiat")
    void zeroDurationStopsImmediately() {
      SearchBudget b = SearchBudget.duration(Duration.ZERO);
      assertThat(b.shouldStop(0, 0L)).isTrue();
    }

    @Test
    @DisplayName("duration(null) → NPE")
    void nullRejected() {
      assertThatThrownBy(() -> SearchBudget.duration(null))
          .isInstanceOf(NullPointerException.class)
          .hasMessageContaining("duration");
    }

    @Test
    @DisplayName("duration négative → IAE")
    void negativeRejected() {
      assertThatThrownBy(() -> SearchBudget.duration(Duration.ofMillis(-1)))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("duration");
    }
  }

  // ---------------------------------------------------------------------------------------------
  // untilStopped(flag)
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("untilStopped(stopFlag)")
  class UntilStoppedFactory {

    @Test
    @DisplayName("Reflète l'état du flag (set/clear)")
    void reflectsFlagState() {
      AtomicBoolean flag = new AtomicBoolean(false);
      SearchBudget b = SearchBudget.untilStopped(flag);
      assertThat(b.shouldStop(0, 0L)).isFalse();
      flag.set(true);
      assertThat(b.shouldStop(0, 0L)).isTrue();
      // Note : la factory ne mémorise pas le passage à true ; c'est l'engine qui matérialise
      // I-Budget-2 (irréversibilité). La factory elle-même reste sans état.
      flag.set(false);
      assertThat(b.shouldStop(0, 0L)).isFalse();
    }

    @Test
    @DisplayName("untilStopped(null) → NPE")
    void nullRejected() {
      assertThatThrownBy(() -> SearchBudget.untilStopped(null))
          .isInstanceOf(NullPointerException.class)
          .hasMessageContaining("stopFlag");
    }
  }

  // ---------------------------------------------------------------------------------------------
  // firstOf(...)
  // ---------------------------------------------------------------------------------------------

  @Nested
  @DisplayName("firstOf(budgets...)")
  class FirstOfFactory {

    @Test
    @DisplayName("firstOf() vide : équivalent UNLIMITED (toujours false)")
    void emptyEquivalentToUnlimited() {
      SearchBudget b = SearchBudget.firstOf();
      assertThat(b.shouldStop(0, 0L)).isFalse();
      assertThat(b.shouldStop(Integer.MAX_VALUE, Long.MAX_VALUE)).isFalse();
    }

    @Test
    @DisplayName("firstOf(nodes(100)) : se comporte comme nodes(100)")
    void singleBudgetWraps() {
      SearchBudget b = SearchBudget.firstOf(SearchBudget.nodes(100));
      assertThat(b.shouldStop(99, 0L)).isFalse();
      assertThat(b.shouldStop(100, 0L)).isTrue();
    }

    @Test
    @DisplayName("firstOf(nodes(100), duration(500ms)) : le premier déclencheur l'emporte")
    void shortCircuitOnFirstTrigger() {
      SearchBudget b =
          SearchBudget.firstOf(
              SearchBudget.nodes(100), SearchBudget.duration(Duration.ofMillis(500)));
      // Aucun déclenché.
      assertThat(b.shouldStop(99, 100_000_000L)).isFalse();
      // Premier déclenché (nodes).
      assertThat(b.shouldStop(100, 100_000_000L)).isTrue();
      // Deuxième déclenché (duration).
      assertThat(b.shouldStop(50, 600_000_000L)).isTrue();
    }

    @Test
    @DisplayName("firstOf(null) → NPE")
    void nullArrayRejected() {
      assertThatThrownBy(() -> SearchBudget.firstOf((SearchBudget[]) null))
          .isInstanceOf(NullPointerException.class)
          .hasMessageContaining("budgets");
    }

    @Test
    @DisplayName("firstOf(b1, null, b3) → NPE localisée par index")
    void nullElementRejected() {
      assertThatThrownBy(
              () ->
                  SearchBudget.firstOf(
                      SearchBudget.nodes(100), null, SearchBudget.duration(Duration.ofMillis(1))))
          .isInstanceOf(NullPointerException.class)
          .hasMessageContaining("budgets[1]");
    }

    @Test
    @DisplayName("Copie défensive : mutation externe de l'array source n'affecte pas le composite")
    void defensiveCopy() {
      SearchBudget[] arr = {SearchBudget.nodes(100)};
      SearchBudget composite = SearchBudget.firstOf(arr);
      // Mutation externe : remplace par UNLIMITED. Le composite doit garder le nodes(100) original.
      arr[0] = SearchBudget.UNLIMITED;
      assertThat(composite.shouldStop(100, 0L)).isTrue();
    }
  }

  // ---------------------------------------------------------------------------------------------
  // UNLIMITED
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("UNLIMITED : toujours false (mode ponder)")
  void unlimitedAlwaysFalse() {
    assertThat(SearchBudget.UNLIMITED.shouldStop(0, 0L)).isFalse();
    assertThat(SearchBudget.UNLIMITED.shouldStop(Integer.MAX_VALUE, Long.MAX_VALUE)).isFalse();
  }

  // ---------------------------------------------------------------------------------------------
  // Scénarios composés (intégration plusieurs factories)
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Scénario UCI : firstOf(nodes(N), duration(d), untilStopped(flag))")
  void uciTypicalComposition() {
    AtomicBoolean stop = new AtomicBoolean(false);
    SearchBudget b =
        SearchBudget.firstOf(
            SearchBudget.nodes(10_000),
            SearchBudget.duration(Duration.ofSeconds(5)),
            SearchBudget.untilStopped(stop));

    // Aucun critère atteint.
    assertThat(b.shouldStop(5_000, 1_000_000_000L)).isFalse();

    // Flag flippé : déclenche immédiatement, peu importe nodes/duration.
    stop.set(true);
    assertThat(b.shouldStop(5_000, 1_000_000_000L)).isTrue();
    stop.set(false);

    // Sans flag, nodes atteint en premier.
    assertThat(b.shouldStop(10_000, 1_000_000_000L)).isTrue();

    // Sans flag, duration atteinte en premier.
    assertThat(b.shouldStop(5_000, 5_000_000_000L)).isTrue();
  }
}
