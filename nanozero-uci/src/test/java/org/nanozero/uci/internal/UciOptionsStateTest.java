package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Tests unitaires de {@link UciOptionsState} (cf. SPEC §6.2, ADR-003, §12 phase 1). */
class UciOptionsStateTest {

  // -------------------------------------------------------------------------------------------
  // Defaults
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Defaults : Ponder=true, Move Overhead=30")
  void defaults() {
    var s = new UciOptionsState();
    assertThat(s.ponder()).isTrue();
    assertThat(s.moveOverheadMs()).isEqualTo(30);
  }

  // -------------------------------------------------------------------------------------------
  // set Ponder
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("set Ponder false")
  void setPonderFalse() {
    var s = new UciOptionsState();
    s.set("Ponder", "false");
    assertThat(s.ponder()).isFalse();
  }

  @Test
  @DisplayName("set Ponder true (toggle from false)")
  void setPonderTrueToggle() {
    var s = new UciOptionsState();
    s.set("Ponder", "false");
    s.set("Ponder", "true");
    assertThat(s.ponder()).isTrue();
  }

  @Test
  @DisplayName("set Ponder TRUE case-insensitive (Boolean.parseBoolean)")
  void setPonderCaseInsensitive() {
    var s = new UciOptionsState();
    s.set("Ponder", "false");
    s.set("Ponder", "TRUE");
    assertThat(s.ponder()).isTrue();
  }

  @Test
  @DisplayName("set Ponder garbage → false (Boolean.parseBoolean retourne false pour non-true)")
  void setPonderGarbageBecomesFalse() {
    var s = new UciOptionsState();
    s.set("Ponder", "garbage");
    assertThat(s.ponder()).isFalse();
  }

  @Test
  @DisplayName("set Ponder avec value null → ignoré silencieusement, valeur courante préservée")
  void setPonderNullValueIgnored() {
    var s = new UciOptionsState();
    s.set("Ponder", null);
    assertThat(s.ponder()).isTrue();
  }

  @Test
  @DisplayName("set Ponder avec whitespace → trim avant parse")
  void setPonderTrimmed() {
    var s = new UciOptionsState();
    s.set("Ponder", "  false  ");
    assertThat(s.ponder()).isFalse();
  }

  // -------------------------------------------------------------------------------------------
  // set Move Overhead
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("set Move Overhead valeur valide")
  void setMoveOverheadValid() {
    var s = new UciOptionsState();
    s.set("Move Overhead", "100");
    assertThat(s.moveOverheadMs()).isEqualTo(100);
  }

  @Test
  @DisplayName("set Move Overhead négatif → clamp à 0")
  void setMoveOverheadNegativeClampedToZero() {
    var s = new UciOptionsState();
    s.set("Move Overhead", "-50");
    assertThat(s.moveOverheadMs()).isZero();
  }

  @Test
  @DisplayName("set Move Overhead > 5000 → clamp à 5000")
  void setMoveOverheadAboveMaxClampedTo5000() {
    var s = new UciOptionsState();
    s.set("Move Overhead", "10000");
    assertThat(s.moveOverheadMs()).isEqualTo(5000);
  }

  @Test
  @DisplayName("set Move Overhead garbage → unchanged (NumberFormatException ignored)")
  void setMoveOverheadGarbageIgnored() {
    var s = new UciOptionsState();
    s.set("Move Overhead", "garbage");
    assertThat(s.moveOverheadMs()).isEqualTo(30);
  }

  @Test
  @DisplayName("set Move Overhead avec value null → ignoré silencieusement")
  void setMoveOverheadNullValueIgnored() {
    var s = new UciOptionsState();
    s.set("Move Overhead", null);
    assertThat(s.moveOverheadMs()).isEqualTo(30);
  }

  @Test
  @DisplayName("set Move Overhead aux bornes [0, 5000] inclusif")
  void setMoveOverheadAtBounds() {
    var s = new UciOptionsState();
    s.set("Move Overhead", "0");
    assertThat(s.moveOverheadMs()).isZero();
    s.set("Move Overhead", "5000");
    assertThat(s.moveOverheadMs()).isEqualTo(5000);
  }

  @Test
  @DisplayName("set Move Overhead avec whitespace → trim avant parse")
  void setMoveOverheadTrimmed() {
    var s = new UciOptionsState();
    s.set("Move Overhead", "  100  ");
    assertThat(s.moveOverheadMs()).isEqualTo(100);
  }

  // -------------------------------------------------------------------------------------------
  // UCI tolérance : options inconnues, name null
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("set option inconnue → silently ignored, autres options unchanged")
  void setUnknownOptionSilentlyIgnored() {
    var s = new UciOptionsState();
    s.set("UnknownOption", "value");
    s.set("Ponder", null); // re-test que Ponder reste à true
    assertThat(s.ponder()).isTrue();
    assertThat(s.moveOverheadMs()).isEqualTo(30);
  }

  @Test
  @DisplayName("set name null → silently ignored (pas d'exception)")
  void setNullNameSilentlyIgnored() {
    var s = new UciOptionsState();
    s.set(null, "anything");
    assertThat(s.ponder()).isTrue();
    assertThat(s.moveOverheadMs()).isEqualTo(30);
  }

  // -------------------------------------------------------------------------------------------
  // declaredOptions
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("declaredOptions : Ponder + Move Overhead + Threads + BatchSize + NNCacheSize")
  void declaredOptionsList() {
    var options = UciOptionsState.declaredOptions();
    assertThat(options).hasSize(5);

    var ponder = (UciOption.Check) options.get(0);
    assertThat(ponder.name()).isEqualTo("Ponder");
    assertThat(ponder.defaultValue()).isTrue();

    var moveOverhead = (UciOption.Spin) options.get(1);
    assertThat(moveOverhead.name()).isEqualTo("Move Overhead");
    assertThat(moveOverhead.defaultValue()).isEqualTo(30);
    assertThat(moveOverhead.min()).isZero();
    assertThat(moveOverhead.max()).isEqualTo(5000);

    var threads = (UciOption.Spin) options.get(2);
    assertThat(threads.name()).isEqualTo("Threads");
    assertThat(threads.defaultValue()).isEqualTo(1);
    assertThat(threads.min()).isEqualTo(1);
    assertThat(threads.max()).isEqualTo(128);

    var batchSize = (UciOption.Spin) options.get(3);
    assertThat(batchSize.name()).isEqualTo("BatchSize");
    assertThat(batchSize.defaultValue()).isEqualTo(1);
    assertThat(batchSize.min()).isEqualTo(1);
    assertThat(batchSize.max()).isEqualTo(64);

    // (ADR-018) Cache d'évaluation NN optionnel.
    var nnCacheSize = (UciOption.Spin) options.get(4);
    assertThat(nnCacheSize.name()).isEqualTo("NNCacheSize");
    assertThat(nnCacheSize.defaultValue()).isZero();
    assertThat(nnCacheSize.min()).isZero();
    assertThat(nnCacheSize.max()).isEqualTo(1 << 24);
  }

  // -------------------------------------------------------------------------------------------
  // (ADR-018) Option NNCacheSize
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("ADR-018 NNCacheSize default : 0 (cache désactivé)")
  void nnCacheSizeDefaultIsZero() {
    assertThat(new UciOptionsState().nnCacheSize()).isZero();
  }

  @Test
  @DisplayName("ADR-018 set NNCacheSize valide")
  void setNnCacheSizeValid() {
    var s = new UciOptionsState();
    s.set("NNCacheSize", "200000");
    assertThat(s.nnCacheSize()).isEqualTo(200000);
  }

  @Test
  @DisplayName("ADR-018 set NNCacheSize négatif : clamp à 0")
  void setNnCacheSizeClampedToMin() {
    var s = new UciOptionsState();
    s.set("NNCacheSize", "-5");
    assertThat(s.nnCacheSize()).isZero();
  }

  @Test
  @DisplayName("ADR-018 set NNCacheSize au-dessus du max : clamp à 16M")
  void setNnCacheSizeClampedToMax() {
    var s = new UciOptionsState();
    s.set("NNCacheSize", String.valueOf(Integer.MAX_VALUE));
    assertThat(s.nnCacheSize()).isEqualTo(1 << 24);
  }

  @Test
  @DisplayName("ADR-018 set NNCacheSize garbage : ignoré (reste à 0)")
  void setNnCacheSizeGarbageIgnored() {
    var s = new UciOptionsState();
    s.set("NNCacheSize", "not-a-number");
    assertThat(s.nnCacheSize()).isZero();
  }

  // -------------------------------------------------------------------------------------------
  // (v1.3.0) Option BatchSize (cf. ADR-008-uci)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("v1.3.0 BatchSize default : 0 (sentinelle = non set, auto-config)")
  void batchSizeDefaultIsZeroSentinel() {
    var s = new UciOptionsState();
    assertThat(s.batchSizeOverride()).as("default sentinel 0 = auto-config").isZero();
  }

  @Test
  @DisplayName("v1.3.0 set BatchSize valide")
  void setBatchSizeValid() {
    var s = new UciOptionsState();
    s.set("BatchSize", "8");
    assertThat(s.batchSizeOverride()).isEqualTo(8);
  }

  @Test
  @DisplayName("v1.3.0 set BatchSize=1 explicite : override valide (Mode A debug)")
  void setBatchSize1Override() {
    var s = new UciOptionsState();
    s.set("BatchSize", "1");
    assertThat(s.batchSizeOverride()).as("BatchSize=1 set → override actif").isEqualTo(1);
  }

  @Test
  @DisplayName("v1.3.0 set BatchSize au-dessus de 64 : clamp à 64")
  void setBatchSizeClampedToMax() {
    var s = new UciOptionsState();
    s.set("BatchSize", "256");
    assertThat(s.batchSizeOverride()).isEqualTo(64);
  }

  @Test
  @DisplayName("v1.3.0 set BatchSize en dessous de 1 : clamp à 1")
  void setBatchSizeClampedToMin() {
    var s = new UciOptionsState();
    s.set("BatchSize", "-5");
    assertThat(s.batchSizeOverride()).isEqualTo(1);
  }

  @Test
  @DisplayName("v1.3.0 set BatchSize garbage : ignored (reste à default sentinelle)")
  void setBatchSizeGarbageIgnored() {
    var s = new UciOptionsState();
    s.set("BatchSize", "not-a-number");
    assertThat(s.batchSizeOverride()).as("parse fail → reste à 0 sentinelle").isZero();
  }

  @Test
  @DisplayName("declaredOptions : retourne une List immutable")
  void declaredOptionsImmutable() {
    var options = UciOptionsState.declaredOptions();
    org.assertj.core.api.Assertions.assertThatThrownBy(
            () -> options.add(new UciOption.Check("X", true)))
        .isInstanceOf(UnsupportedOperationException.class);
  }
}
