package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.engine.EngineConfig;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/**
 * Tests v1.1.0 : hidden options Dirichlet via {@code setoption} (cf. SPEC §6.4, ADR-003 mise à jour
 * v1.1.0). Vérifie le storage dans {@link UciOptionsState}, l'injection dans {@link EngineConfig}
 * via {@link UciAdapterState#buildEngineConfigFromOptions}, et le pattern hidden (absence dans la
 * liste {@code declaredOptions()}).
 */
class UciDirichletOptionsTest {

  private static Network sharedNetwork;

  @BeforeAll
  static void loadNetwork() throws IOException {
    var url = UciDirichletOptionsTest.class.getResource("/npz/parity-model.npz");
    try {
      sharedNetwork = NetworkLoader.load(Paths.get(url.toURI()), LoadOptions.defaults());
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  @AfterAll
  static void clearNetwork() {
    sharedNetwork = null;
  }

  private static UciResponseWriter writerOver(ByteArrayOutputStream baos) {
    PrintStream ps = new PrintStream(baos, true, StandardCharsets.UTF_8);
    return new UciResponseWriter(ps);
  }

  // -------------------------------------------------------------------------------------------
  // UciOptionsState : 3 champs Dirichlet, defaults + set()
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("UciOptionsState defaults : alpha=0, epsilon=0, seed=\"0\" (Dirichlet désactivé)")
  void uciOptionsStateDefaultsDirichletDisabled() {
    UciOptionsState options = new UciOptionsState();
    assertThat(options.dirichletAlpha()).isZero();
    assertThat(options.dirichletEpsilon()).isZero();
    assertThat(options.dirichletSeed()).isEqualTo("0");
  }

  @Test
  @DisplayName("setoption DirichletAlpha valid : stocké tel quel")
  void setDirichletAlphaValid() {
    UciOptionsState options = new UciOptionsState();
    options.set("DirichletAlpha", "300");
    assertThat(options.dirichletAlpha()).isEqualTo(300);
  }

  @Test
  @DisplayName("setoption DirichletAlpha out-of-range : clampé [0, 10000]")
  void setDirichletAlphaOutOfRangeClamped() {
    UciOptionsState options = new UciOptionsState();
    options.set("DirichletAlpha", "99999");
    assertThat(options.dirichletAlpha()).isEqualTo(10000);
    options.set("DirichletAlpha", "-50");
    assertThat(options.dirichletAlpha()).isZero();
  }

  @Test
  @DisplayName("setoption DirichletAlpha invalid integer : ignored (UCI tolérance)")
  void setDirichletAlphaInvalidIgnored() {
    UciOptionsState options = new UciOptionsState();
    options.set("DirichletAlpha", "300");
    options.set("DirichletAlpha", "not-a-number");
    assertThat(options.dirichletAlpha()).isEqualTo(300); // value précédente conservée
  }

  @Test
  @DisplayName("setoption DirichletEpsilon valid : stocké")
  void setDirichletEpsilonValid() {
    UciOptionsState options = new UciOptionsState();
    options.set("DirichletEpsilon", "250");
    assertThat(options.dirichletEpsilon()).isEqualTo(250);
  }

  @Test
  @DisplayName("setoption DirichletEpsilon out-of-range : clampé [0, 1000]")
  void setDirichletEpsilonOutOfRangeClamped() {
    UciOptionsState options = new UciOptionsState();
    options.set("DirichletEpsilon", "5000");
    assertThat(options.dirichletEpsilon()).isEqualTo(1000);
    options.set("DirichletEpsilon", "-10");
    assertThat(options.dirichletEpsilon()).isZero();
  }

  @Test
  @DisplayName("setoption DirichletSeed valid long string : stocké tel quel")
  void setDirichletSeedValidLongString() {
    UciOptionsState options = new UciOptionsState();
    options.set("DirichletSeed", "123456789012345");
    assertThat(options.dirichletSeed()).isEqualTo("123456789012345");
  }

  @Test
  @DisplayName("setoption DirichletSeed invalid string : stocké, parsé en 0 plus tard")
  void setDirichletSeedInvalidStringStoredAsIs() {
    UciOptionsState options = new UciOptionsState();
    options.set("DirichletSeed", "not-a-number");
    // Stocké tel quel — la validation parse long est faite au build EngineConfig.
    assertThat(options.dirichletSeed()).isEqualTo("not-a-number");
  }

  @Test
  @DisplayName("setoption DirichletXxx value null : ignored")
  void setDirichletNullValueIgnored() {
    UciOptionsState options = new UciOptionsState();
    options.set("DirichletAlpha", "300");
    options.set("DirichletAlpha", null);
    assertThat(options.dirichletAlpha()).isEqualTo(300);
  }

  // -------------------------------------------------------------------------------------------
  // declaredOptions() : pattern hidden — DirichletXxx NON déclarées
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Pattern hidden : DirichletAlpha/Epsilon/Seed absentes de declaredOptions()")
  void hiddenOptionsNotInDeclaredList() {
    var declared = UciOptionsState.declaredOptions();
    var declaredNames = declared.stream().map(UciOption::name).toList();
    assertThat(declaredNames)
        .as("Pattern hidden : DirichletXxx NON déclarées (cf. SPEC §6.4)")
        .doesNotContain("DirichletAlpha")
        .doesNotContain("DirichletEpsilon")
        .doesNotContain("DirichletSeed");
  }

  // -------------------------------------------------------------------------------------------
  // UciAdapterState.buildEngineConfigFromOptions : injection Dirichlet
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("buildEngineConfigFromOptions defaults → comportement v1.0.0 (Dirichlet désactivé)")
  void buildEngineConfigFromOptionsDefaults() {
    UciOptionsState options = new UciOptionsState();
    EngineConfig config = UciAdapterState.buildEngineConfigFromOptions(options);
    assertThat(config.dirichletAlpha()).isEqualTo(0.0f);
    assertThat(config.dirichletEpsilon()).isEqualTo(0.0f);
    assertThat(config.randomSeed()).isZero();
    // Champs v1.0.0 inchangés (hérités de EngineConfig.defaults())
    EngineConfig d = EngineConfig.defaults();
    assertThat(config.cPuct()).isEqualTo(d.cPuct());
    assertThat(config.fpuValue()).isEqualTo(d.fpuValue());
    assertThat(config.treeInitialCapacity()).isEqualTo(d.treeInitialCapacity());
  }

  @Test
  @DisplayName("ADR-018 : buildEngineConfigFromOptions défaut → nnCacheSize=0 (cache désactivé)")
  void buildEngineConfigFromOptionsNnCacheDefaultZero() {
    UciOptionsState options = new UciOptionsState();
    EngineConfig config = UciAdapterState.buildEngineConfigFromOptions(options);
    assertThat(config.nnCacheSize()).isZero();
  }

  @Test
  @DisplayName("ADR-018 : buildEngineConfigFromOptions propage NNCacheSize → EngineConfig")
  void buildEngineConfigFromOptionsNnCacheWired() {
    UciOptionsState options = new UciOptionsState();
    options.set("NNCacheSize", "200000");
    EngineConfig config = UciAdapterState.buildEngineConfigFromOptions(options);
    assertThat(config.nnCacheSize()).isEqualTo(200_000);
  }

  @Test
  @DisplayName("buildEngineConfigFromOptions Dirichlet activé : alpha=0.3, epsilon=0.25, seed=42")
  void buildEngineConfigFromOptionsDirichletEnabled() {
    UciOptionsState options = new UciOptionsState();
    options.set("DirichletAlpha", "300");
    options.set("DirichletEpsilon", "250");
    options.set("DirichletSeed", "42");
    EngineConfig config = UciAdapterState.buildEngineConfigFromOptions(options);
    assertThat(config.dirichletAlpha()).isEqualTo(0.3f);
    assertThat(config.dirichletEpsilon()).isEqualTo(0.25f);
    assertThat(config.randomSeed()).isEqualTo(42L);
  }

  @Test
  @DisplayName("buildEngineConfigFromOptions seed invalid string → 0L (parse échec)")
  void buildEngineConfigFromOptionsSeedInvalidFallsBackToZero() {
    UciOptionsState options = new UciOptionsState();
    options.set("DirichletSeed", "not-a-number");
    EngineConfig config = UciAdapterState.buildEngineConfigFromOptions(options);
    assertThat(config.randomSeed()).isZero();
  }

  @Test
  @DisplayName("buildEngineConfigFromOptions seed très large (>Integer.MAX) accepté en long")
  void buildEngineConfigFromOptionsSeedLargeLongAccepted() {
    UciOptionsState options = new UciOptionsState();
    long bigSeed = 5_000_000_000L; // > Integer.MAX_VALUE
    options.set("DirichletSeed", Long.toString(bigSeed));
    EngineConfig config = UciAdapterState.buildEngineConfigFromOptions(options);
    assertThat(config.randomSeed()).isEqualTo(bigSeed);
  }

  // -------------------------------------------------------------------------------------------
  // (v1.3.0) Auto-config batchSize selon useCuda (cf. ADR-008-uci)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("v1.3.0 buildEngineConfig useCuda=false, threads=4 : batchSize=1 (Mode A)")
  void buildEngineConfigCpuMultiThreadBatchSize1() {
    UciOptionsState options = new UciOptionsState();
    options.set("Threads", "4");
    EngineConfig config = UciAdapterState.buildEngineConfigFromOptions(options, false);
    assertThat(config.searchThreads()).isEqualTo(4);
    assertThat(config.batchSize()).as("CPU multi-thread → Mode A batchSize=1").isEqualTo(1);
    assertThat(config.virtualLoss()).isEqualTo(1.0f);
  }

  @Test
  @DisplayName(
      "v1.3.0 buildEngineConfig useCuda=true, threads=4 : batchSize=4 (Mode B auto-config)")
  void buildEngineConfigCudaMultiThreadAutoBatchSize() {
    UciOptionsState options = new UciOptionsState();
    options.set("Threads", "4");
    EngineConfig config = UciAdapterState.buildEngineConfigFromOptions(options, true);
    assertThat(config.searchThreads()).isEqualTo(4);
    assertThat(config.batchSize())
        .as("CUDA multi-thread → auto-config batchSize=threads")
        .isEqualTo(4);
  }

  @Test
  @DisplayName("v1.3.0 buildEngineConfig useCuda=true, threads=1 : batchSize=1 (cas dégénéré)")
  void buildEngineConfigCudaSingleThreadBatchSize1() {
    UciOptionsState options = new UciOptionsState();
    EngineConfig config = UciAdapterState.buildEngineConfigFromOptions(options, true);
    assertThat(config.searchThreads()).isEqualTo(1);
    assertThat(config.batchSize()).as("CUDA mono-thread → pas de gain batching → 1").isEqualTo(1);
  }

  @Test
  @DisplayName("v1.3.0 buildEngineConfig BatchSize override : explicite > auto-config")
  void buildEngineConfigBatchSizeOverride() {
    UciOptionsState options = new UciOptionsState();
    options.set("Threads", "4");
    options.set("BatchSize", "16");
    // useCuda=true → auto-config aurait donné batchSize=4, mais override force 16.
    EngineConfig config = UciAdapterState.buildEngineConfigFromOptions(options, true);
    assertThat(config.batchSize()).as("Override BatchSize=16 > auto-config 4").isEqualTo(16);
  }

  @Test
  @DisplayName("v1.3.0 buildEngineConfig BatchSize=1 explicite force Mode A même avec CUDA")
  void buildEngineConfigBatchSize1ExplicitOverridesCudaAutoConfig() {
    UciOptionsState options = new UciOptionsState();
    options.set("Threads", "4");
    options.set("BatchSize", "1");
    // useCuda=true → auto-config aurait donné batchSize=4, mais override force 1 (Mode A debug).
    EngineConfig config = UciAdapterState.buildEngineConfigFromOptions(options, true);
    assertThat(config.batchSize()).as("Override BatchSize=1 force Mode A debug").isEqualTo(1);
  }

  // -------------------------------------------------------------------------------------------
  // UciAdapterState : constructor lazy + engine() lazy creation
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("UciAdapterState(Network, ...) : engine lazy-créé au premier engine() call")
  void lazyEngineCreationOnFirstAccess() {
    UciOptionsState options = new UciOptionsState();
    var baos = new ByteArrayOutputStream();
    try (var state = new UciAdapterState(sharedNetwork, writerOver(baos), options)) {
      var eng1 = state.engine();
      assertThat(eng1).isNotNull();
      // Second appel : même instance
      var eng2 = state.engine();
      assertThat(eng2).isSameAs(eng1);
    }
  }

  @Test
  @DisplayName(
      "Lazy creation lit les Dirichlet options courantes (set avant premier engine() call)")
  void lazyCreationReadsDirichletOptionsAtFirstUse() {
    UciOptionsState options = new UciOptionsState();
    options.set("DirichletAlpha", "300");
    options.set("DirichletEpsilon", "250");
    options.set("DirichletSeed", "42");
    var baos = new ByteArrayOutputStream();
    try (var state = new UciAdapterState(sharedNetwork, writerOver(baos), options)) {
      // engine() est lazy-créé maintenant, avec les options déjà set.
      // Pas d'API publique Engine pour exposer la config interne ; on vérifie indirectement
      // via le fait que buildEngineConfigFromOptions retourne la config attendue.
      EngineConfig config = UciAdapterState.buildEngineConfigFromOptions(options);
      assertThat(config.dirichletAlpha()).isEqualTo(0.3f);
      assertThat(config.dirichletEpsilon()).isEqualTo(0.25f);
      assertThat(config.randomSeed()).isEqualTo(42L);
      // L'engine est créé avec cette config.
      assertThat(state.engine()).isNotNull();
    }
  }

  @Test
  @DisplayName("UciAdapterState(Network=null, ...) : NPE")
  void lazyConstructorNullNetworkThrowsNpe() {
    assertThatThrownBy(
            () ->
                new UciAdapterState(
                    (Network) null, writerOver(new ByteArrayOutputStream()), new UciOptionsState()))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("network");
  }

  @Test
  @DisplayName("UciAdapterState close idempotent même si engine jamais créé (lazy unused)")
  void closeIdempotentWithLazyUnusedEngine() {
    UciOptionsState options = new UciOptionsState();
    var baos = new ByteArrayOutputStream();
    var state = new UciAdapterState(sharedNetwork, writerOver(baos), options);
    // Engine jamais accédé → reste null
    state.close(); // doit être no-op safe
    state.close(); // 2e fois aussi
  }

  // -------------------------------------------------------------------------------------------
  // Coverage : close() avec exception sur session.onUciStop()
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("close avec debugMode + session.onUciStop throws → log stderr (branche v1.0.0)")
  void closeWithDebugModeSessionThrowsLogsStderr() {
    UciOptionsState options = new UciOptionsState();
    var baos = new ByteArrayOutputStream();
    var state = new UciAdapterState(sharedNetwork, writerOver(baos), options);
    state.setDebugMode(true);
    var engine = state.engine();
    var session = new UciSession(engine, state.writer(), /* isPonder */ false, GoArgs.empty());
    state.setCurrentSession(session);
    // Fermer engine prématurément → engine.stop() lèvera IllegalStateException (état CLOSED).
    engine.close();

    var stderrCapture = new ByteArrayOutputStream();
    PrintStream originalErr = System.err;
    System.setErr(new PrintStream(stderrCapture, true, StandardCharsets.UTF_8));
    try {
      state.close(); // catch RuntimeException + log debug, ne doit pas crasher
    } finally {
      System.setErr(originalErr);
    }
    String stderr = stderrCapture.toString(StandardCharsets.UTF_8);
    assertThat(stderr).contains("error stopping session in close");
  }

  @Test
  @DisplayName("close sans debugMode + session.onUciStop throws → silent (branche v1.0.0 false)")
  void closeWithoutDebugModeSessionThrowsSilent() {
    UciOptionsState options = new UciOptionsState();
    var baos = new ByteArrayOutputStream();
    var state = new UciAdapterState(sharedNetwork, writerOver(baos), options);
    // PAS de setDebugMode(true)
    var engine = state.engine();
    var session = new UciSession(engine, state.writer(), /* isPonder */ false, GoArgs.empty());
    state.setCurrentSession(session);
    engine.close();

    var stderrCapture = new ByteArrayOutputStream();
    PrintStream originalErr = System.err;
    System.setErr(new PrintStream(stderrCapture, true, StandardCharsets.UTF_8));
    try {
      state.close(); // catch silent
    } finally {
      System.setErr(originalErr);
    }
    String stderr = stderrCapture.toString(StandardCharsets.UTF_8);
    assertThat(stderr).doesNotContain("error stopping session");
  }
}
