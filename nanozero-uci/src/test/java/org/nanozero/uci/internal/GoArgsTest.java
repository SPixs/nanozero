package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;
import java.util.OptionalLong;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Tests unitaires de {@link GoArgs} (cf. SPEC §3.1, §12 phase 1). */
class GoArgsTest {

  @Test
  @DisplayName("empty() : tous Optional vides, infinite/ponder false, searchMoves vide")
  void emptyDefault() {
    var a = GoArgs.empty();
    assertThat(a.wtimeMs()).isEmpty();
    assertThat(a.btimeMs()).isEmpty();
    assertThat(a.wincMs()).isEmpty();
    assertThat(a.bincMs()).isEmpty();
    assertThat(a.movestogo()).isEmpty();
    assertThat(a.movetimeMs()).isEmpty();
    assertThat(a.nodes()).isEmpty();
    assertThat(a.depth()).isEmpty();
    assertThat(a.infinite()).isFalse();
    assertThat(a.ponder()).isFalse();
    assertThat(a.searchMoves()).isEmpty();
  }

  @Test
  @DisplayName("Construction valide avec tous les champs renseignés")
  void fullConstruction() {
    var a =
        new GoArgs(
            OptionalLong.of(60_000L),
            OptionalLong.of(60_000L),
            OptionalLong.of(1_000L),
            OptionalLong.of(1_000L),
            OptionalInt.of(40),
            OptionalLong.of(5_000L),
            OptionalInt.of(1000),
            OptionalInt.of(20),
            true,
            true,
            List.of(new int[] {1, 2}, new int[] {3}));
    assertThat(a.wtimeMs()).hasValue(60_000L);
    assertThat(a.movestogo()).hasValue(40);
    assertThat(a.infinite()).isTrue();
    assertThat(a.searchMoves()).hasSize(2);
  }

  @Test
  @DisplayName("searchMoves : copie immutable via List.copyOf (mutation externe sans effet)")
  void searchMovesImmutableCopy() {
    List<int[]> external = new ArrayList<>();
    external.add(new int[] {1});
    var a =
        new GoArgs(
            OptionalLong.empty(),
            OptionalLong.empty(),
            OptionalLong.empty(),
            OptionalLong.empty(),
            OptionalInt.empty(),
            OptionalLong.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            false,
            false,
            external);
    assertThat(a.searchMoves()).hasSize(1);

    external.add(new int[] {2});
    assertThat(a.searchMoves()).as("copie défensive : l'ajout externe ne propage pas").hasSize(1);
  }

  @Test
  @DisplayName("searchMoves : List immutable, modification directe → UnsupportedOperationException")
  void searchMovesListIsImmutable() {
    var a = GoArgs.empty();
    assertThatThrownBy(() -> a.searchMoves().add(new int[] {1}))
        .isInstanceOf(UnsupportedOperationException.class);
  }

  @Test
  @DisplayName("Compact constructor : NPE pour chaque OptionalLong null")
  void nullOptionalLongFieldsThrowNpe() {
    assertThatThrownBy(
            () ->
                new GoArgs(
                    null,
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalInt.empty(),
                    OptionalLong.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    false,
                    false,
                    List.of()))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("wtimeMs");

    assertThatThrownBy(
            () ->
                new GoArgs(
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalInt.empty(),
                    null,
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    false,
                    false,
                    List.of()))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("movetimeMs");
  }

  @Test
  @DisplayName("Compact constructor : NPE pour chaque OptionalInt null")
  void nullOptionalIntFieldsThrowNpe() {
    assertThatThrownBy(
            () ->
                new GoArgs(
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    null,
                    OptionalLong.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    false,
                    false,
                    List.of()))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("movestogo");

    assertThatThrownBy(
            () ->
                new GoArgs(
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalInt.empty(),
                    OptionalLong.empty(),
                    null,
                    OptionalInt.empty(),
                    false,
                    false,
                    List.of()))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("nodes");
  }

  @Test
  @DisplayName("Compact constructor : NPE si searchMoves null")
  void nullSearchMovesThrowsNpe() {
    assertThatThrownBy(
            () ->
                new GoArgs(
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    OptionalInt.empty(),
                    OptionalLong.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    false,
                    false,
                    null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("searchMoves");
  }
}
