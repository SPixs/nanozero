package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Tests unitaires des records {@link UciOption} (cf. SPEC §3.3, §12 phase 1). */
class UciOptionTest {

  // -------------------------------------------------------------------------------------------
  // Check
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Check : construction valide")
  void checkValidConstruction() {
    var c = new UciOption.Check("Ponder", true);
    assertThat(c.name()).isEqualTo("Ponder");
    assertThat(c.defaultValue()).isTrue();
  }

  @Test
  @DisplayName("Check avec name null → NPE")
  void checkNullNameThrowsNpe() {
    assertThatThrownBy(() -> new UciOption.Check(null, true))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("name");
  }

  @Test
  @DisplayName("Check avec name blank → IAE")
  void checkBlankNameThrowsIae() {
    assertThatThrownBy(() -> new UciOption.Check("", true))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("name must not be blank");
    assertThatThrownBy(() -> new UciOption.Check("   ", true))
        .isInstanceOf(IllegalArgumentException.class);
  }

  // -------------------------------------------------------------------------------------------
  // Spin
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Spin : construction valide aux bornes")
  void spinValidConstruction() {
    var s = new UciOption.Spin("Move Overhead", 30, 0, 5000);
    assertThat(s.name()).isEqualTo("Move Overhead");
    assertThat(s.defaultValue()).isEqualTo(30);
    assertThat(s.min()).isEqualTo(0);
    assertThat(s.max()).isEqualTo(5000);
  }

  @Test
  @DisplayName("Spin : default == min OK et default == max OK")
  void spinDefaultAtBoundsOk() {
    assertThat(new UciOption.Spin("X", 0, 0, 100).defaultValue()).isZero();
    assertThat(new UciOption.Spin("X", 100, 0, 100).defaultValue()).isEqualTo(100);
  }

  @Test
  @DisplayName("Spin : min == max légal (option à valeur unique)")
  void spinMinEqualMaxOk() {
    var s = new UciOption.Spin("X", 42, 42, 42);
    assertThat(s.defaultValue()).isEqualTo(42);
  }

  @Test
  @DisplayName("Spin avec name null → NPE")
  void spinNullNameThrowsNpe() {
    assertThatThrownBy(() -> new UciOption.Spin(null, 0, 0, 10))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("name");
  }

  @Test
  @DisplayName("Spin avec name blank → IAE")
  void spinBlankNameThrowsIae() {
    assertThatThrownBy(() -> new UciOption.Spin("", 0, 0, 10))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("name must not be blank");
  }

  @Test
  @DisplayName("Spin avec min > max → IAE")
  void spinMinGreaterMaxThrowsIae() {
    assertThatThrownBy(() -> new UciOption.Spin("X", 0, 100, 50))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("min must be <= max");
  }

  @Test
  @DisplayName("Spin avec defaultValue > max → IAE")
  void spinDefaultGreaterMaxThrowsIae() {
    assertThatThrownBy(() -> new UciOption.Spin("X", 100, 0, 50))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("defaultValue must be in [min, max]");
  }

  @Test
  @DisplayName("Spin avec defaultValue < min → IAE")
  void spinDefaultLessMinThrowsIae() {
    assertThatThrownBy(() -> new UciOption.Spin("X", -10, 0, 50))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("defaultValue must be in [min, max]");
  }

  // -------------------------------------------------------------------------------------------
  // String_
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("String_ : construction valide")
  void stringValidConstruction() {
    var s = new UciOption.String_("X", "default");
    assertThat(s.name()).isEqualTo("X");
    assertThat(s.defaultValue()).isEqualTo("default");
  }

  @Test
  @DisplayName("String_ : defaultValue empty string accepté")
  void stringEmptyDefaultAccepted() {
    var s = new UciOption.String_("X", "");
    assertThat(s.defaultValue()).isEmpty();
  }

  @Test
  @DisplayName("String_ avec name null → NPE")
  void stringNullNameThrowsNpe() {
    assertThatThrownBy(() -> new UciOption.String_(null, "default"))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("name");
  }

  @Test
  @DisplayName("String_ avec name blank → IAE")
  void stringBlankNameThrowsIae() {
    assertThatThrownBy(() -> new UciOption.String_("", "default"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("name must not be blank");
  }

  @Test
  @DisplayName("String_ avec defaultValue null → NPE")
  void stringNullDefaultThrowsNpe() {
    assertThatThrownBy(() -> new UciOption.String_("X", null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("defaultValue");
  }

  // -------------------------------------------------------------------------------------------
  // Sealed interface : exhaustivité du switch (smoke)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Sealed switch : exhaustivité au niveau name() pour les 3 sous-types")
  void sealedSwitchExhaustive() {
    UciOption[] options = {
      new UciOption.Check("c", true),
      new UciOption.Spin("s", 5, 0, 10),
      new UciOption.String_("st", "v")
    };
    for (var o : options) {
      String name =
          switch (o) {
            case UciOption.Check c -> c.name();
            case UciOption.Spin s -> s.name();
            case UciOption.String_ s -> s.name();
          };
      assertThat(name).isNotNull().isNotBlank();
    }
  }
}
