package org.nanozero.engine;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/**
 * Tests end-to-end de l'injection Dirichlet via {@link Engine} (cf. SPEC §3.1 amendement, §5.2
 * amendement, §5.3 amendement, ADR-012).
 *
 * <p>Périmètre phase 1.1.0-2 : validation que le comportement v1.0.0 est strictement préservé avec
 * {@code EngineConfig.defaults()}, et que le déterminisme/non-déterminisme attendus selon la seed
 * sont observables au niveau {@link SearchResult}.
 *
 * <p>Périmètre phase 1.1.0-3 : tests intégration plus larges (déterminisme strict consécutif,
 * mesure de diversification entropie Shannon, cas extrêmes PUCT injection).
 *
 * <p>Tag {@code slow} : classe lourde (~70 s pour 9 tests d'intégration NN end-to-end avec sims
 * 256-512). Skipped par défaut en CI rapide ; activable via {@code mvn verify -DexcludedGroups=}.
 */
@Tag("slow")
class EngineDirichletTest {

  private static Network sharedNetwork;

  @BeforeAll
  static void loadNetwork() throws IOException {
    var url = EngineDirichletTest.class.getResource("/npz/parity-model.npz");
    if (url == null) {
      throw new AssertionError("Fixture introuvable : /npz/parity-model.npz");
    }
    try {
      Path path = Paths.get(url.toURI());
      sharedNetwork = NetworkLoader.load(path, LoadOptions.defaults());
    } catch (Exception e) {
      throw new AssertionError("Impossible de charger parity-model.npz", e);
    }
  }

  @AfterAll
  static void clearNetwork() {
    sharedNetwork = null;
  }

  // -------------------------------------------------------------------------------------------
  // Équivalence v1.0.0 : defaults() preserve strictement le comportement v1.0.0
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("EngineConfig.defaults() : 2 engines produisent le même bestmove (equiv v1.0.0)")
  void engineWithDefaultsProducesSameBestmove() {
    EngineConfig config = EngineConfig.defaults();

    try (Engine e1 = new Engine(sharedNetwork, config);
        Engine e2 = new Engine(sharedNetwork, config)) {
      SearchResult r1 = e1.searchSync(new GameState(), SearchBudget.nodes(128));
      SearchResult r2 = e2.searchSync(new GameState(), SearchBudget.nodes(128));

      assertThat(r1.bestMove()).isEqualTo(r2.bestMove());
      assertThat(r1.simulationsCount()).isEqualTo(r2.simulationsCount());
      assertThat(r1.childVisits()).isEqualTo(r2.childVisits());
    }
  }

  // -------------------------------------------------------------------------------------------
  // Déterminisme avec seed fixée + Dirichlet activé
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Dirichlet activé + seed fixée : 2 engines produisent le même bestmove (bit-à-bit)")
  void engineWithDirichletEnabledDeterministicWithFixedSeed() {
    EngineConfig config = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);

    try (Engine e1 = new Engine(sharedNetwork, config);
        Engine e2 = new Engine(sharedNetwork, config)) {
      SearchResult r1 = e1.searchSync(new GameState(), SearchBudget.nodes(128));
      SearchResult r2 = e2.searchSync(new GameState(), SearchBudget.nodes(128));

      // Avec même seed et même config Dirichlet : résultats bit-à-bit identiques.
      assertThat(r1.bestMove()).isEqualTo(r2.bestMove());
      assertThat(r1.simulationsCount()).isEqualTo(r2.simulationsCount());
      assertThat(r1.childVisits()).isEqualTo(r2.childVisits());
    }
  }

  // -------------------------------------------------------------------------------------------
  // Effet visible du Dirichlet : différent de defaults() sur la distribution de visites
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Dirichlet activé vs defaults : distribution de visites diffère (effet observable)")
  void engineWithDirichletVisibleEffectVsDefaults() {
    // Avec parity-model.npz (priors uniformes), sans Dirichlet la distribution suit le
    // tie-breaking + l'exploration PUCT classique. Avec Dirichlet alpha=0.3 epsilon=0.25, le
    // noise peaky décale significativement la distribution de visites au root.
    EngineConfig defaultsCfg = EngineConfig.defaults();
    EngineConfig dirichletCfg = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);

    try (Engine eDefault = new Engine(sharedNetwork, defaultsCfg);
        Engine eDirichlet = new Engine(sharedNetwork, dirichletCfg)) {
      SearchResult rDefault = eDefault.searchSync(new GameState(), SearchBudget.nodes(256));
      SearchResult rDirichlet = eDirichlet.searchSync(new GameState(), SearchBudget.nodes(256));

      // Au moins une métrique observable doit différer (childVisits typiquement).
      // Possible cas pathologique : si bestMove identique ET childVisits identiques, le Dirichlet
      // n'a aucun effet observable (peu probable avec alpha=0.3 sur 20 coups + 256 sims).
      boolean visitDistributionDiffers =
          !java.util.Arrays.equals(rDefault.childVisits(), rDirichlet.childVisits());
      assertThat(visitDistributionDiffers)
          .as("childVisits distribution doit différer avec/sans Dirichlet noise")
          .isTrue();
    }
  }

  // -------------------------------------------------------------------------------------------
  // Axe A — Déterminisme strict 10 runs + non-déterminisme seeds + diversification entropie
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName(
      "Axe A : 3 runs consécutifs avec seed fixée + Dirichlet → résultats bit-à-bit identiques")
  void dirichletEnabled_deterministicOverConsecutiveRuns() {
    // 3 runs × 256 sims suffisent pour valider le déterminisme strict (un seul écart
    // bit-à-bit casserait le test). Trois runs valident transitively i==j pour tout i,j ∈ {0,1,2}.
    EngineConfig config = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);

    int firstBestMove = -1;
    int[] firstChildVisits = null;

    for (int i = 0; i < 3; i++) {
      try (Engine engine = new Engine(sharedNetwork, config)) {
        SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(256));
        if (i == 0) {
          firstBestMove = result.bestMove();
          firstChildVisits = result.childVisits().clone();
        } else {
          assertThat(result.bestMove()).as("run %d bestMove", i).isEqualTo(firstBestMove);
          assertThat(result.childVisits()).as("run %d childVisits", i).isEqualTo(firstChildVisits);
        }
      }
    }
  }

  @Test
  @DisplayName("Axe A : seeds différentes → métrique observable diverge (bestmove ou childVisits)")
  void dirichletEnabled_nonDeterministicAcrossDifferentSeeds() {
    EngineConfig cfg1 = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 1L);
    EngineConfig cfg2 = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 2L);

    SearchResult r1;
    SearchResult r2;
    try (Engine e1 = new Engine(sharedNetwork, cfg1);
        Engine e2 = new Engine(sharedNetwork, cfg2)) {
      r1 = e1.searchSync(new GameState(), SearchBudget.nodes(256));
      r2 = e2.searchSync(new GameState(), SearchBudget.nodes(256));
    }

    boolean bestMoveDiffers = r1.bestMove() != r2.bestMove();
    boolean visitsDiffer = !java.util.Arrays.equals(r1.childVisits(), r2.childVisits());
    assertThat(bestMoveDiffers || visitsDiffer)
        .as("Seeds différentes : au moins une métrique (bestmove ou childVisits) doit diverger")
        .isTrue();
  }

  @Test
  @DisplayName(
      "Axe A : entropie de Shannon des visites avec Dirichlet > entropie sans (diversification)")
  void dirichletIncreaseEntropy_diversificationVisible() {
    // Calibration (dépend du fixture parity-model.npz) :
    // - 256 sims : écart observé ~0.006 bit (variance noise écrase le signal).
    // - 512 sims (fixture scalaire ≤ v1.4.0) : écart > 0.1 bit (marge confortable).
    // - 512 sims (fixture WDL v1.5.0, régénéré en 061fc08) : écart tombé à ~0.015 bit
    //   → sous la marge 0.05, jamais détecté faute de run -Pslow post-WDL. Recalibré à
    //   1024 sims : écart mesuré 0.081 bit (2026-06-10), marge restaurée.
    EngineConfig configWithoutNoise = EngineConfig.defaults();
    EngineConfig configWithNoise = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);

    SearchResult rWithout;
    SearchResult rWith;
    try (Engine eWithout = new Engine(sharedNetwork, configWithoutNoise);
        Engine eWith = new Engine(sharedNetwork, configWithNoise)) {
      rWithout = eWithout.searchSync(new GameState(), SearchBudget.nodes(1024));
      rWith = eWith.searchSync(new GameState(), SearchBudget.nodes(1024));
    }

    double hWithout = shannonEntropy(rWithout.childVisits());
    double hWith = shannonEntropy(rWith.childVisits());

    assertThat(hWith)
        .as("H_with_dirichlet=%.4f doit être > H_without=%.4f + 0.05 (marge)", hWith, hWithout)
        .isGreaterThan(hWithout + 0.05);
  }

  /** Entropie de Shannon en bits : H(p) = -sum(p_i * log2(p_i)) où p_i = visits_i / total. */
  private static double shannonEntropy(int[] visits) {
    long total = 0;
    for (int v : visits) {
      total += v;
    }
    if (total == 0) {
      return 0.0;
    }
    double h = 0.0;
    double log2 = Math.log(2.0);
    for (int v : visits) {
      if (v == 0) {
        continue;
      }
      double p = (double) v / total;
      h -= p * (Math.log(p) / log2);
    }
    return h;
  }

  // -------------------------------------------------------------------------------------------
  // Axe D — Cas extrêmes PUCT injection (epsilon=0.5, epsilon=1.0, petit n)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Axe D : epsilon=0.5 (mix 50/50) → recherche complète + bestmove valide")
  void dirichletEpsilonHalf_mixesPriorsAndNoise5050() {
    EngineConfig config = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.5f, 42L);
    try (Engine engine = new Engine(sharedNetwork, config)) {
      SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(256));
      assertThat(result.simulationsCount()).isEqualTo(256);
      // bestMove != 0 (Move.NULL sentinelle) — la recherche a produit un coup réel
      assertThat(result.bestMove()).isNotZero();
    }
  }

  @Test
  @DisplayName(
      "Axe D : epsilon=1.0 (full noise, NN priors ignorés) → pas de crash, bestmove valide")
  void dirichletEpsilonOne_fullNoiseIgnoresNNPriors() {
    EngineConfig config = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 1.0f, 42L);
    try (Engine engine = new Engine(sharedNetwork, config)) {
      SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(256));
      assertThat(result.simulationsCount()).isEqualTo(256);
      assertThat(result.bestMove()).isNotZero();
    }
  }

  @Test
  @DisplayName("Axe D : position à 1 seul coup légal (Kg8 forcé) → Dirichlet noise n=1 trivial")
  void dirichletWithSingleLegalMove_noiseDegradesToTrivial() {
    // FEN : roi noir h8 attaqué par dame h6 ; seule case de fuite = g8 (Kh8-g8).
    // Coups Roi h8 : g7 (Dh6 diagonale), h7 (Dh6 colonne), g8 (libre). → 1 coup légal.
    GameState singleMovePos = new GameState("7k/8/7Q/8/8/8/8/7K b - - 0 1");
    EngineConfig config = new EngineConfig(2.5f, 0.0f, 1024, 0.3f, 0.25f, 42L);
    try (Engine engine = new Engine(sharedNetwork, config)) {
      SearchResult result = engine.searchSync(singleMovePos, SearchBudget.nodes(64));
      // Pas de crash sur n=1, retourne le coup unique légal.
      assertThat(result.simulationsCount()).isEqualTo(64);
      assertThat(result.bestMove()).isNotZero();
      // childMoves doit avoir taille 1 (1 seul coup légal)
      assertThat(result.childMoves()).hasSize(1);
    }
  }
}
