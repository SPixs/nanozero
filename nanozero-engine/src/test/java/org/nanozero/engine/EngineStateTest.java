package org.nanozero.engine;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class EngineStateTest {

  @Test
  @DisplayName("Enum stable : 6 valeurs, noms figés (anti-régression sur renommage)")
  void enumValuesArePinned() {
    assertThat(EngineState.values()).hasSize(6);
    assertThat(EngineState.values())
        .containsExactly(
            EngineState.IDLE,
            EngineState.SEARCHING,
            EngineState.PONDERING,
            EngineState.STOPPING,
            EngineState.DONE,
            EngineState.CLOSED);
  }

  @Test
  @DisplayName("Noms textuels figés (UCI / logs dépendent de ces String)")
  void enumNamesArePinned() {
    assertThat(EngineState.IDLE.name()).isEqualTo("IDLE");
    assertThat(EngineState.SEARCHING.name()).isEqualTo("SEARCHING");
    assertThat(EngineState.PONDERING.name()).isEqualTo("PONDERING");
    assertThat(EngineState.STOPPING.name()).isEqualTo("STOPPING");
    assertThat(EngineState.DONE.name()).isEqualTo("DONE");
    assertThat(EngineState.CLOSED.name()).isEqualTo("CLOSED");
  }
}
