package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Tests unitaires de {@link InfoFields} (cf. SPEC §3.2, §12 phase 1). */
class InfoFieldsTest {

  @Test
  @DisplayName("empty() : tous Optional vides, pv vide, string empty")
  void emptyDefault() {
    var f = InfoFields.empty();
    assertThat(f.depth()).isEmpty();
    assertThat(f.seldepth()).isEmpty();
    assertThat(f.nodes()).isEmpty();
    assertThat(f.nps()).isEmpty();
    assertThat(f.timeMs()).isEmpty();
    assertThat(f.scoreCp()).isEmpty();
    assertThat(f.scoreMate()).isEmpty();
    assertThat(f.pv()).isEmpty();
    assertThat(f.multipv()).isEmpty();
    assertThat(f.string()).isEmpty();
  }

  @Test
  @DisplayName("Construction valide avec tous les champs renseignés")
  void fullConstruction() {
    var f =
        new InfoFields(
            OptionalInt.of(15),
            OptionalInt.of(20),
            OptionalInt.of(50_000),
            OptionalInt.of(2_500),
            OptionalLong.of(20_000L),
            OptionalInt.of(35),
            OptionalInt.empty(),
            new int[] {100, 200, 300},
            OptionalInt.of(1),
            Optional.of("hashfull 250"));
    assertThat(f.depth()).hasValue(15);
    assertThat(f.scoreCp()).hasValue(35);
    assertThat(f.pv()).containsExactly(100, 200, 300);
    assertThat(f.string()).contains("hashfull 250");
  }

  @Test
  @DisplayName("pv : copie défensive (mutation externe sans effet)")
  void pvDefensiveCopy() {
    int[] pv = {1, 2, 3};
    var f =
        new InfoFields(
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalLong.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            pv,
            OptionalInt.empty(),
            Optional.empty());
    pv[0] = 999;
    assertThat(f.pv()).containsExactly(1, 2, 3);
  }

  @Test
  @DisplayName("Compact constructor : NPE pour chaque OptionalInt null")
  void nullOptionalIntFieldsThrowNpe() {
    assertThatThrownBy(
            () ->
                new InfoFields(
                    null,
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    OptionalLong.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    new int[0],
                    OptionalInt.empty(),
                    Optional.empty()))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("depth");

    assertThatThrownBy(
            () ->
                new InfoFields(
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    OptionalLong.empty(),
                    null,
                    OptionalInt.empty(),
                    new int[0],
                    OptionalInt.empty(),
                    Optional.empty()))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("scoreCp");
  }

  @Test
  @DisplayName("Compact constructor : NPE si OptionalLong timeMs null")
  void nullOptionalLongThrowsNpe() {
    assertThatThrownBy(
            () ->
                new InfoFields(
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    null,
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    new int[0],
                    OptionalInt.empty(),
                    Optional.empty()))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("timeMs");
  }

  @Test
  @DisplayName("Compact constructor : NPE si pv null")
  void nullPvThrowsNpe() {
    assertThatThrownBy(
            () ->
                new InfoFields(
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    OptionalLong.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    null,
                    OptionalInt.empty(),
                    Optional.empty()))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("pv");
  }

  @Test
  @DisplayName("Compact constructor : NPE si string Optional null")
  void nullStringThrowsNpe() {
    assertThatThrownBy(
            () ->
                new InfoFields(
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    OptionalLong.empty(),
                    OptionalInt.empty(),
                    OptionalInt.empty(),
                    new int[0],
                    OptionalInt.empty(),
                    null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("string");
  }

  @Test
  @DisplayName("pv tableau vide accepté")
  void emptyPvAccepted() {
    var f =
        new InfoFields(
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalLong.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            new int[0],
            OptionalInt.empty(),
            Optional.empty());
    assertThat(f.pv()).isEmpty();
  }
}
