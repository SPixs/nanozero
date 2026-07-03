package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;

/**
 * Tests unitaires des records {@link UciCommand} (cf. SPEC §3.1, §12 phase 1). Vérifie les
 * invariants des compact constructors : non-nullité, copie défensive {@code int[]}, validation
 * blank pour {@code SetOption.name}.
 */
class UciCommandTest {

  // -------------------------------------------------------------------------------------------
  // Records sans paramètres : construction triviale
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Construction triviale des records sans paramètres")
  void emptyRecordsConstruction() {
    assertThat(new UciCommand.Uci()).isNotNull();
    assertThat(new UciCommand.IsReady()).isNotNull();
    assertThat(new UciCommand.UciNewGame()).isNotNull();
    assertThat(new UciCommand.Stop()).isNotNull();
    assertThat(new UciCommand.PonderHit()).isNotNull();
    assertThat(new UciCommand.Quit()).isNotNull();
  }

  @Test
  @DisplayName("Records sans paramètres : equals/hashCode reflètent identité de classe")
  void emptyRecordsEqualsHashCode() {
    assertThat(new UciCommand.Uci()).isEqualTo(new UciCommand.Uci());
    assertThat(new UciCommand.Uci().hashCode()).isEqualTo(new UciCommand.Uci().hashCode());
    // Deux types différents ne sont pas égaux malgré l'absence de champs.
    assertThat((Object) new UciCommand.Uci()).isNotEqualTo(new UciCommand.IsReady());
  }

  // -------------------------------------------------------------------------------------------
  // Position
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Position : construction valide")
  void positionValidConstruction() {
    var p = new UciCommand.Position(new GameState(), new int[] {1, 2, 3});
    assertThat(p.playedMoves()).containsExactly(1, 2, 3);
    assertThat(p.position()).isNotNull();
  }

  @Test
  @DisplayName("Position avec position null → NPE")
  void positionNullPositionThrowsNpe() {
    assertThatThrownBy(() -> new UciCommand.Position(null, new int[0]))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("position");
  }

  @Test
  @DisplayName("Position avec playedMoves null → NPE")
  void positionNullMovesThrowsNpe() {
    assertThatThrownBy(() -> new UciCommand.Position(new GameState(), null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("playedMoves");
  }

  @Test
  @DisplayName("Position : copie défensive de playedMoves (mutation externe sans effet)")
  void positionDefensiveCopyOnPlayedMoves() {
    int[] moves = {1, 2, 3};
    var p = new UciCommand.Position(new GameState(), moves);
    moves[0] = 999;
    assertThat(p.playedMoves()).containsExactly(1, 2, 3);
  }

  @Test
  @DisplayName("Position : tableau vide accepté")
  void positionEmptyMovesAccepted() {
    var p = new UciCommand.Position(new GameState(), new int[0]);
    assertThat(p.playedMoves()).isEmpty();
  }

  // -------------------------------------------------------------------------------------------
  // Go
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Go : construction valide avec GoArgs.empty()")
  void goValidConstruction() {
    var go = new UciCommand.Go(GoArgs.empty());
    assertThat(go.args()).isEqualTo(GoArgs.empty());
  }

  @Test
  @DisplayName("Go avec args null → NPE")
  void goNullArgsThrowsNpe() {
    assertThatThrownBy(() -> new UciCommand.Go(null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("args");
  }

  // -------------------------------------------------------------------------------------------
  // SetOption
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("SetOption : construction valide avec name + value")
  void setOptionValidConstruction() {
    var so = new UciCommand.SetOption("Ponder", "false");
    assertThat(so.name()).isEqualTo("Ponder");
    assertThat(so.value()).isEqualTo("false");
  }

  @Test
  @DisplayName("SetOption avec name null → NPE")
  void setOptionNullNameThrowsNpe() {
    assertThatThrownBy(() -> new UciCommand.SetOption(null, "value"))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("name");
  }

  @Test
  @DisplayName("SetOption avec name blank (\"\") → IAE")
  void setOptionEmptyNameThrowsIae() {
    assertThatThrownBy(() -> new UciCommand.SetOption("", "value"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("name must not be blank");
  }

  @Test
  @DisplayName("SetOption avec name uniquement whitespace → IAE")
  void setOptionWhitespaceNameThrowsIae() {
    assertThatThrownBy(() -> new UciCommand.SetOption("   ", "value"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("name must not be blank");
  }

  @Test
  @DisplayName("SetOption : value null acceptée (option type Button sans valeur)")
  void setOptionNullValueAccepted() {
    var so = new UciCommand.SetOption("Ponder", null);
    assertThat(so.value()).isNull();
  }

  // -------------------------------------------------------------------------------------------
  // Debug
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Debug : construction valide on/off")
  void debugValidConstruction() {
    assertThat(new UciCommand.Debug(true).enabled()).isTrue();
    assertThat(new UciCommand.Debug(false).enabled()).isFalse();
  }

  // -------------------------------------------------------------------------------------------
  // Unknown
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Unknown : construction valide avec ligne brute")
  void unknownValidConstruction() {
    var u = new UciCommand.Unknown("garbage 123");
    assertThat(u.rawLine()).isEqualTo("garbage 123");
  }

  @Test
  @DisplayName("Unknown : ligne vide acceptée")
  void unknownEmptyLineAccepted() {
    var u = new UciCommand.Unknown("");
    assertThat(u.rawLine()).isEmpty();
  }

  @Test
  @DisplayName("Unknown avec rawLine null → NPE")
  void unknownNullLineThrowsNpe() {
    assertThatThrownBy(() -> new UciCommand.Unknown(null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("rawLine");
  }
}
