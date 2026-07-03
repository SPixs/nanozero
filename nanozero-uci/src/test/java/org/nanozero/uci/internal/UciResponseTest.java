package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.OptionalInt;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Tests unitaires des records {@link UciResponse} (cf. SPEC §3.2, §12 phase 1). */
class UciResponseTest {

  // -------------------------------------------------------------------------------------------
  // Records sans paramètres
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Construction triviale UciOk et ReadyOk")
  void emptyRecordsConstruction() {
    assertThat(new UciResponse.UciOk()).isNotNull();
    assertThat(new UciResponse.ReadyOk()).isNotNull();
    assertThat(new UciResponse.UciOk()).isEqualTo(new UciResponse.UciOk());
  }

  // -------------------------------------------------------------------------------------------
  // Id
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Id : construction valide")
  void idValidConstruction() {
    var id = new UciResponse.Id("NanoZero 1.0.0", "Mametz");
    assertThat(id.name()).isEqualTo("NanoZero 1.0.0");
    assertThat(id.author()).isEqualTo("Mametz");
  }

  @Test
  @DisplayName("Id avec name null → NPE")
  void idNullNameThrowsNpe() {
    assertThatThrownBy(() -> new UciResponse.Id(null, "Mametz"))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("name");
  }

  @Test
  @DisplayName("Id avec author null → NPE")
  void idNullAuthorThrowsNpe() {
    assertThatThrownBy(() -> new UciResponse.Id("NanoZero", null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("author");
  }

  @Test
  @DisplayName("Id avec name blank → IAE")
  void idBlankNameThrowsIae() {
    assertThatThrownBy(() -> new UciResponse.Id("", "Mametz"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("name must not be blank");
    assertThatThrownBy(() -> new UciResponse.Id("   ", "Mametz"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("Id avec author blank → IAE")
  void idBlankAuthorThrowsIae() {
    assertThatThrownBy(() -> new UciResponse.Id("NanoZero", ""))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("author must not be blank");
  }

  // -------------------------------------------------------------------------------------------
  // Option
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Option : construction valide")
  void optionValidConstruction() {
    var opt = new UciResponse.Option(new UciOption.Check("Ponder", true));
    assertThat(opt.option()).isInstanceOf(UciOption.Check.class);
  }

  @Test
  @DisplayName("Option avec option null → NPE")
  void optionNullThrowsNpe() {
    assertThatThrownBy(() -> new UciResponse.Option(null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("option");
  }

  // -------------------------------------------------------------------------------------------
  // Info
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Info : construction valide avec InfoFields.empty()")
  void infoValidConstruction() {
    var info = new UciResponse.Info(InfoFields.empty());
    // Note : InfoFields.equals utilise reference identity sur l'array pv (limitation Java
    // records) — donc deux instances de InfoFields.empty() ne sont PAS égales. On vérifie via
    // les champs pertinents directement.
    assertThat(info.fields()).isNotNull();
    assertThat(info.fields().pv()).isEmpty();
    assertThat(info.fields().depth()).isEmpty();
    assertThat(info.fields().nodes()).isEmpty();
  }

  @Test
  @DisplayName("Info avec fields null → NPE")
  void infoNullFieldsThrowsNpe() {
    assertThatThrownBy(() -> new UciResponse.Info(null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("fields");
  }

  // -------------------------------------------------------------------------------------------
  // BestMove
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("BestMove : construction valide sans ponderMove")
  void bestMoveValidWithoutPonder() {
    var bm = new UciResponse.BestMove(1234, OptionalInt.empty());
    assertThat(bm.move()).isEqualTo(1234);
    assertThat(bm.ponderMove()).isEmpty();
  }

  @Test
  @DisplayName("BestMove : construction valide avec ponderMove")
  void bestMoveValidWithPonder() {
    var bm = new UciResponse.BestMove(1234, OptionalInt.of(5678));
    assertThat(bm.move()).isEqualTo(1234);
    assertThat(bm.ponderMove()).hasValue(5678);
  }

  @Test
  @DisplayName("BestMove avec ponderMove null → NPE")
  void bestMoveNullPonderThrowsNpe() {
    assertThatThrownBy(() -> new UciResponse.BestMove(1234, null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("ponderMove");
  }

  // -------------------------------------------------------------------------------------------
  // Sealed switch exhaustiveness smoke
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Sealed switch : exhaustivité dispatch sur les 6 sous-types")
  void sealedSwitchExhaustive() {
    UciResponse[] all = {
      new UciResponse.Id("n", "a"),
      new UciResponse.UciOk(),
      new UciResponse.ReadyOk(),
      new UciResponse.Option(new UciOption.Check("X", true)),
      new UciResponse.Info(InfoFields.empty()),
      new UciResponse.InfoString("visits"),
      new UciResponse.BestMove(0, OptionalInt.empty())
    };
    for (var r : all) {
      String tag =
          switch (r) {
            case UciResponse.Id id -> "id";
            case UciResponse.UciOk u -> "uciok";
            case UciResponse.ReadyOk ok -> "readyok";
            case UciResponse.Option o -> "option";
            case UciResponse.Info i -> "info";
            case UciResponse.InfoString is -> "infostring";
            case UciResponse.BestMove bm -> "bestmove";
          };
      assertThat(tag).isNotNull();
    }
  }
}
