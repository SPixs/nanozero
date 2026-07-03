package org.nanozero.engine.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.engine.SearchBudget;

/** Tests unitaires de {@link MutableSearchBudget} (cf. SPEC §12 phase 11). */
class MutableSearchBudgetTest {

  @Test
  @DisplayName("Construction OK avec délégué non null")
  void constructionAcceptsNonNull() {
    var budget = new MutableSearchBudget(SearchBudget.UNLIMITED);
    assertThat(budget.shouldStop(0, 0L)).isFalse();
  }

  @Test
  @DisplayName("Construction throws sur délégué null")
  void constructionRejectsNull() {
    assertThatThrownBy(() -> new MutableSearchBudget(null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("initial");
  }

  @Test
  @DisplayName("shouldStop délègue au délégué initial (UNLIMITED → false)")
  void shouldStopDelegatesUnlimited() {
    var budget = new MutableSearchBudget(SearchBudget.UNLIMITED);
    assertThat(budget.shouldStop(0, 0L)).isFalse();
    assertThat(budget.shouldStop(1_000_000, Long.MAX_VALUE / 2)).isFalse();
  }

  @Test
  @DisplayName("shouldStop délègue au délégué initial (nodes(N) → true à N sims)")
  void shouldStopDelegatesNodesBudget() {
    var budget = new MutableSearchBudget(SearchBudget.nodes(5));
    assertThat(budget.shouldStop(0, 0L)).isFalse();
    assertThat(budget.shouldStop(4, 0L)).isFalse();
    assertThat(budget.shouldStop(5, 0L)).isTrue();
    assertThat(budget.shouldStop(100, 0L)).isTrue();
  }

  @Test
  @DisplayName("replace change le délégué, visible immédiatement par shouldStop suivant")
  void replaceSwapsDelegate() {
    var budget = new MutableSearchBudget(SearchBudget.UNLIMITED);
    assertThat(budget.shouldStop(1000, 0L)).isFalse();

    budget.replace(SearchBudget.nodes(10));
    assertThat(budget.shouldStop(5, 0L)).isFalse();
    assertThat(budget.shouldStop(10, 0L)).isTrue();
  }

  @Test
  @DisplayName("replace throws sur délégué null")
  void replaceRejectsNull() {
    var budget = new MutableSearchBudget(SearchBudget.UNLIMITED);
    assertThatThrownBy(() -> budget.replace(null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("newDelegate");
  }

  @Test
  @DisplayName("replace successifs : seul le dernier délégué est consulté")
  void replaceMultipleTimes() {
    var budget = new MutableSearchBudget(SearchBudget.UNLIMITED);
    budget.replace(SearchBudget.nodes(100));
    budget.replace(SearchBudget.nodes(5));
    assertThat(budget.shouldStop(4, 0L)).isFalse();
    assertThat(budget.shouldStop(5, 0L)).isTrue();
  }

  @Test
  @DisplayName("Cohérence I-Budget-2 préservée par délégué : true reste true (pour le délégué)")
  void delegateStickyTrueIsRespected() {
    // I-Budget-2 garantit qu'un délégué qui retourne true ne reprend pas. Le wrapper ne
    // garantit pas cette monotonie cross-replace (c'est par construction : le caller peut
    // remplacer un délégué « fini » par un autre « non fini »). Test informatif.
    var n5 = SearchBudget.nodes(5);
    var budget = new MutableSearchBudget(n5);
    assertThat(budget.shouldStop(5, 0L)).isTrue();
    budget.replace(SearchBudget.UNLIMITED);
    assertThat(budget.shouldStop(5, 0L)).isFalse();
  }
}
