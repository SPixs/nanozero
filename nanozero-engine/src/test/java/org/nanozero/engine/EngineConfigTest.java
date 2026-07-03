package org.nanozero.engine;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class EngineConfigTest {

  @Test
  @DisplayName("defaults() : (2.5f, 0.0f, 1024)")
  void defaultsValuesArePinned() {
    EngineConfig cfg = EngineConfig.defaults();
    assertThat(cfg.cPuct()).isEqualTo(2.5f);
    assertThat(cfg.fpuValue()).isEqualTo(0.0f);
    assertThat(cfg.treeInitialCapacity()).isEqualTo(1024);
  }

  @Test
  @DisplayName("Construction valide : valeurs hors-défauts dans les bornes acceptées")
  void validNonDefaultValuesAccepted() {
    EngineConfig cfg = new EngineConfig(1.0f, -0.3f, 4096, 0.0f, 0.0f, 0L);
    assertThat(cfg.cPuct()).isEqualTo(1.0f);
    assertThat(cfg.fpuValue()).isEqualTo(-0.3f);
    assertThat(cfg.treeInitialCapacity()).isEqualTo(4096);
  }

  @Test
  @DisplayName("cPuct == 0 → IAE")
  void cPuctZeroRejected() {
    assertThatThrownBy(() -> new EngineConfig(0.0f, 0.0f, 1024, 0.0f, 0.0f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("cPuct")
        .hasMessageContaining("> 0");
  }

  @Test
  @DisplayName("cPuct négatif → IAE")
  void cPuctNegativeRejected() {
    assertThatThrownBy(() -> new EngineConfig(-1.0f, 0.0f, 1024, 0.0f, 0.0f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("cPuct");
  }

  @Test
  @DisplayName("fpuValue < -1 → IAE")
  void fpuValueBelowMinusOneRejected() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, -1.5f, 1024, 0.0f, 0.0f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("fpuValue")
        .hasMessageContaining("[-1, 1]");
  }

  @Test
  @DisplayName("fpuValue > +1 → IAE")
  void fpuValueAbovePlusOneRejected() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 1.5f, 1024, 0.0f, 0.0f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("fpuValue");
  }

  @Test
  @DisplayName("fpuValue aux bornes -1.0f et +1.0f acceptés (intervalle fermé)")
  void fpuValueBoundariesAccepted() {
    new EngineConfig(2.5f, -1.0f, 1024, 0.0f, 0.0f, 0L);
    new EngineConfig(2.5f, 1.0f, 1024, 0.0f, 0.0f, 0L);
  }

  @Test
  @DisplayName("treeInitialCapacity == 0 → IAE")
  void treeInitialCapacityZeroRejected() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 0, 0.0f, 0.0f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("treeInitialCapacity")
        .hasMessageContaining(">= 1");
  }

  @Test
  @DisplayName("treeInitialCapacity négatif → IAE")
  void treeInitialCapacityNegativeRejected() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, -10, 0.0f, 0.0f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("treeInitialCapacity");
  }

  // ---------------------------------------------------------------------------------------------
  // v1.1.0 — Dirichlet noise extensions (cf. ADR-012, SPEC §3.5)
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("defaults() : Dirichlet désactivé (alpha=0, epsilon=0, seed=0)")
  void defaults_disablesDirichlet() {
    EngineConfig c = EngineConfig.defaults();
    assertThat(c.dirichletAlpha()).isEqualTo(0.0f);
    assertThat(c.dirichletEpsilon()).isEqualTo(0.0f);
    assertThat(c.randomSeed()).isEqualTo(0L);
  }

  @Test
  @DisplayName("Construction valide avec Dirichlet activé (alpha=0.3, epsilon=0.25, seed=42)")
  void validDirichletConfig() {
    EngineConfig c = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);
    assertThat(c.dirichletAlpha()).isEqualTo(0.3f);
    assertThat(c.dirichletEpsilon()).isEqualTo(0.25f);
    assertThat(c.randomSeed()).isEqualTo(42L);
  }

  @Test
  @DisplayName("dirichletAlpha négatif → IAE")
  void invalidDirichletAlphaNegative() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, -0.1f, 0.25f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dirichletAlpha");
  }

  @Test
  @DisplayName("dirichletEpsilon négatif → IAE")
  void invalidDirichletEpsilonNegative() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, 0.3f, -0.1f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dirichletEpsilon");
  }

  @Test
  @DisplayName("dirichletEpsilon > 1 → IAE")
  void invalidDirichletEpsilonAboveOne() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 1.1f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dirichletEpsilon");
  }

  @Test
  @DisplayName("epsilon > 0 && alpha == 0 → IAE (combinaison interdite)")
  void invalidEpsilonPositiveAlphaZero() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.25f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dirichletAlpha");
  }

  @Test
  @DisplayName("Mode désactivé explicitement (alpha=0, epsilon=0) accepté")
  void disabledModeAlphaZeroEpsilonZero() {
    EngineConfig c = new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L);
    assertThat(c.dirichletAlpha()).isEqualTo(0.0f);
    assertThat(c.dirichletEpsilon()).isEqualTo(0.0f);
  }

  @Test
  @DisplayName("Bornes epsilon : 0 et 1 acceptés (intervalle fermé) si alpha > 0")
  void epsilonBoundariesAccepted() {
    new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.0f, 0L);
    new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 1.0f, 0L);
  }

  // ---------------------------------------------------------------------------------------------
  // v1.1.2 — Hardening Float.isFinite() + constructor 3-args délégant (cf. ADR-012 v1.1.2)
  // ---------------------------------------------------------------------------------------------

  @Test
  @DisplayName("v1.1.2 : cPuct = NaN → IAE (isFinite rejette)")
  void invalidCPuctNaN() {
    assertThatThrownBy(() -> new EngineConfig(Float.NaN, 0.0f, 1024, 0.0f, 0.0f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("cPuct")
        .hasMessageContaining("finite");
  }

  @Test
  @DisplayName("v1.1.2 : cPuct = +Infinity → IAE")
  void invalidCPuctPositiveInfinity() {
    assertThatThrownBy(() -> new EngineConfig(Float.POSITIVE_INFINITY, 0.0f, 1024, 0.0f, 0.0f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("cPuct");
  }

  @Test
  @DisplayName("v1.1.2 : cPuct = -Infinity → IAE")
  void invalidCPuctNegativeInfinity() {
    assertThatThrownBy(() -> new EngineConfig(Float.NEGATIVE_INFINITY, 0.0f, 1024, 0.0f, 0.0f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("cPuct");
  }

  @Test
  @DisplayName("v1.1.2 : fpuValue = NaN → IAE")
  void invalidFpuValueNaN() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, Float.NaN, 1024, 0.0f, 0.0f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("fpuValue")
        .hasMessageContaining("finite");
  }

  @Test
  @DisplayName("v1.1.2 : dirichletAlpha = NaN → IAE")
  void invalidDirichletAlphaNaN() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, Float.NaN, 0.0f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dirichletAlpha")
        .hasMessageContaining("finite");
  }

  @Test
  @DisplayName("v1.1.2 : dirichletAlpha = +Infinity → IAE")
  void invalidDirichletAlphaInfinity() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, Float.POSITIVE_INFINITY, 0.5f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dirichletAlpha");
  }

  @Test
  @DisplayName("v1.1.2 : dirichletEpsilon = NaN → IAE")
  void invalidDirichletEpsilonNaN() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, 0.3f, Float.NaN, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dirichletEpsilon")
        .hasMessageContaining("finite");
  }

  @Test
  @DisplayName("v1.1.2 : constructor 3-args produit Dirichlet désactivé (source-compat v1.0.0)")
  void threeArgsConstructorDisablesDirichlet() {
    EngineConfig config = new EngineConfig(2.5f, 0.0f, 1024);
    assertThat(config.cPuct()).isEqualTo(2.5f);
    assertThat(config.fpuValue()).isEqualTo(0.0f);
    assertThat(config.treeInitialCapacity()).isEqualTo(1024);
    assertThat(config.dirichletAlpha()).isEqualTo(0.0f);
    assertThat(config.dirichletEpsilon()).isEqualTo(0.0f);
    assertThat(config.randomSeed()).isZero();
  }

  @Test
  @DisplayName("v1.1.2 : constructor 3-args équivalent au 6-args avec Dirichlet 0")
  void threeArgsEquivalentToFullWithDisabled() {
    EngineConfig threeArgs = new EngineConfig(2.5f, 0.0f, 1024);
    EngineConfig fullArgs = new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L);
    assertThat(threeArgs).isEqualTo(fullArgs);
  }

  @Test
  @DisplayName("v1.1.2 : constructor 3-args applique les validations (NaN rejeté via délégation)")
  void threeArgsConstructorValidationsApplyViaCascade() {
    assertThatThrownBy(() -> new EngineConfig(Float.NaN, 0.0f, 1024))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("cPuct");
  }

  // -----------------------------------------------------------------------------------
  // v1.2.0 — searchThreads / batchSize / virtualLoss (cf. SPEC §15.3, ADR-013/014/015)
  // -----------------------------------------------------------------------------------

  @Test
  @DisplayName("v1.2.0 : defaults() inclut searchThreads=1, batchSize=1, virtualLoss=0.0f")
  void defaultsBatchedFieldsArePinned() {
    EngineConfig cfg = EngineConfig.defaults();
    assertThat(cfg.searchThreads()).isEqualTo(1);
    assertThat(cfg.batchSize()).isEqualTo(1);
    assertThat(cfg.virtualLoss()).isEqualTo(0.0f);
  }

  @Test
  @DisplayName("v1.2.0 : valeurs hors-défaut dans les bornes acceptées (4, 64, 3.0)")
  void validBatchedValuesAccepted() {
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 4, 64, 3.0f);
    assertThat(cfg.searchThreads()).isEqualTo(4);
    assertThat(cfg.batchSize()).isEqualTo(64);
    assertThat(cfg.virtualLoss()).isEqualTo(3.0f);
  }

  @Test
  @DisplayName("v1.2.0 : searchThreads=0 rejeté (< 1)")
  void searchThreadsBelowOneRejected() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 0, 1, 0.0f))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("searchThreads");
  }

  @Test
  @DisplayName("v1.2.0 : searchThreads=129 rejeté (> MAX_SEARCH_THREADS=128)")
  void searchThreadsAboveMaxRejected() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 129, 1, 0.0f))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("searchThreads");
  }

  @Test
  @DisplayName("v1.2.0 : searchThreads=128 accepté (borne haute MAX_SEARCH_THREADS)")
  void searchThreadsAtMaxAccepted() {
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 128, 1, 0.0f);
    assertThat(cfg.searchThreads()).isEqualTo(EngineConfig.MAX_SEARCH_THREADS);
  }

  @Test
  @DisplayName("v1.2.0 : batchSize=0 rejeté (< 1)")
  void batchSizeBelowOneRejected() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, 0, 0.0f))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("batchSize");
  }

  @Test
  @DisplayName("v1.2.0 : batchSize > Network.MAX_BATCH rejeté")
  void batchSizeAboveMaxRejected() {
    int over = org.nanozero.nn.Network.MAX_BATCH + 1;
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, over, 0.0f))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("batchSize");
  }

  @Test
  @DisplayName("v1.2.0 : batchSize = Network.MAX_BATCH accepté (borne haute)")
  void batchSizeAtNetworkMaxAccepted() {
    EngineConfig cfg =
        new EngineConfig(
            2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, org.nanozero.nn.Network.MAX_BATCH, 0.0f);
    assertThat(cfg.batchSize()).isEqualTo(org.nanozero.nn.Network.MAX_BATCH);
  }

  @Test
  @DisplayName("v1.2.0 : virtualLoss NaN rejeté (isFinite)")
  void virtualLossNaNRejected() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, 1, Float.NaN))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("virtualLoss");
  }

  @Test
  @DisplayName("v1.2.0 : virtualLoss +Infinity rejeté (isFinite)")
  void virtualLossPosInfRejected() {
    assertThatThrownBy(
            () -> new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, 1, Float.POSITIVE_INFINITY))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("virtualLoss");
  }

  @Test
  @DisplayName("v1.2.0 : virtualLoss négatif rejeté (< 0)")
  void virtualLossNegativeRejected() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, 1, -0.1f))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("virtualLoss");
  }

  @Test
  @DisplayName("v1.2.0 : virtualLoss > MAX_VIRTUAL_LOSS=10 rejeté")
  void virtualLossAboveMaxRejected() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, 1, 10.5f))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("virtualLoss");
  }

  @Test
  @DisplayName("v1.2.0 : virtualLoss = MAX_VIRTUAL_LOSS accepté (borne haute)")
  void virtualLossAtMaxAccepted() {
    EngineConfig cfg =
        new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, 1, EngineConfig.MAX_VIRTUAL_LOSS);
    assertThat(cfg.virtualLoss()).isEqualTo(EngineConfig.MAX_VIRTUAL_LOSS);
  }

  @Test
  @DisplayName("v1.2.0 : constructor 6-args produit mode batched désactivé (source-compat v1.1.0)")
  void sixArgsConstructorDisablesBatched() {
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);
    assertThat(cfg.cPuct()).isEqualTo(2.5f);
    assertThat(cfg.dirichletAlpha()).isEqualTo(0.3f);
    assertThat(cfg.dirichletEpsilon()).isEqualTo(0.25f);
    assertThat(cfg.randomSeed()).isEqualTo(42L);
    assertThat(cfg.searchThreads()).isEqualTo(1);
    assertThat(cfg.batchSize()).isEqualTo(1);
    assertThat(cfg.virtualLoss()).isEqualTo(0.0f);
  }

  @Test
  @DisplayName("v1.2.0 : constructor 6-args équivalent au 9-args avec batched désactivé")
  void sixArgsEquivalentToFullWithBatchedDisabled() {
    EngineConfig sixArgs = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);
    EngineConfig fullArgs = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L, 1, 1, 0.0f);
    assertThat(sixArgs).isEqualTo(fullArgs);
  }

  @Test
  @DisplayName("v1.2.0 : constructor 6-args applique validations batched (via délégation)")
  void sixArgsConstructorValidationsApplyViaCascade() {
    // Valid args batched-désactivé. La validation se déclenche sur les Dirichlet incohérents.
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.5f, 0L))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("dirichletAlpha");
  }

  @Test
  @DisplayName("v1.2.0 : constructor 3-args (v1.1.2) inclut batched désactivé (régression check)")
  void threeArgsConstructorAlsoDisablesBatched() {
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024);
    assertThat(cfg.searchThreads()).isEqualTo(1);
    assertThat(cfg.batchSize()).isEqualTo(1);
    assertThat(cfg.virtualLoss()).isEqualTo(0.0f);
  }

  // -----------------------------------------------------------------------------------
  // ADR-018 — nnCacheSize (cache d'évaluation NN optionnel)
  // -----------------------------------------------------------------------------------

  @Test
  @DisplayName("ADR-018 : defaults() inclut nnCacheSize=0 (cache désactivé)")
  void defaultsNnCacheSizeIsZero() {
    assertThat(EngineConfig.defaults().nnCacheSize()).isZero();
  }

  @Test
  @DisplayName("ADR-018 : nnCacheSize positif accepté (constructor 10-args)")
  void nnCacheSizePositiveAccepted() {
    EngineConfig cfg = new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, 1, 0.0f, 200_000);
    assertThat(cfg.nnCacheSize()).isEqualTo(200_000);
  }

  @Test
  @DisplayName("ADR-018 : nnCacheSize négatif rejeté (< 0)")
  void nnCacheSizeNegativeRejected() {
    assertThatThrownBy(() -> new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, 1, 1, 0.0f, -1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("nnCacheSize");
  }

  @Test
  @DisplayName("ADR-018 : constructor 9-args (compat) produit nnCacheSize=0")
  void nineArgsConstructorDisablesNnCache() {
    EngineConfig nineArgs = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L, 4, 8, 1.0f);
    assertThat(nineArgs.nnCacheSize()).isZero();
    // Équivalent au 10-args avec nnCacheSize=0.
    EngineConfig tenArgs = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L, 4, 8, 1.0f, 0);
    assertThat(nineArgs).isEqualTo(tenArgs);
  }
}
