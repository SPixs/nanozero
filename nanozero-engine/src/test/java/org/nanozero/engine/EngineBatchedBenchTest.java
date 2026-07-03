package org.nanozero.engine;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/**
 * Bench informatif : throughput du mode batched vs mono-thread (cf. SPEC §15, phase 1.2.0-7).
 *
 * <p>Mesure wall-clock + sims/s pour {@code searchThreads ∈ {1, 2, 4}} sur startpos + parity-model.
 * Le but : valider empiriquement que le mode batched apporte du speedup (cible : ≥1.5× pour
 * justifier la release v1.2.0).
 *
 * <p>Tag {@code @Tag("perf")} : non-bloquant pour le profile CI rapide ; exécuté manuellement via
 * {@code mvn -pl nanozero-engine -P perf test -Dtest=EngineBatchedBenchTest}.
 */
@Tag("perf")
class EngineBatchedBenchTest {

  private static final int BUDGET = 2000;
  private static final int WARMUP = 200;

  private static Path fixturePath() {
    var url = EngineBatchedBenchTest.class.getResource("/npz/parity-model.npz");
    if (url == null) {
      throw new AssertionError("Fixture introuvable : /npz/parity-model.npz");
    }
    try {
      return Paths.get(url.toURI());
    } catch (Exception e) {
      throw new AssertionError("Impossible de résoudre /npz/parity-model.npz", e);
    }
  }

  private static Network loadParityNetwork() throws IOException {
    return NetworkLoader.load(fixturePath(), LoadOptions.defaults());
  }

  private static EngineConfig configFor(int searchThreads) {
    float vloss = searchThreads > 1 ? 3.0f : 0.0f;
    return new EngineConfig(2.5f, 0.0f, 1024, 0.0f, 0.0f, 0L, searchThreads, 1, vloss);
  }

  /** Bench unitaire : 1 warmup + 1 mesure, retourne sims/sec. */
  private static double bench(EngineConfig config) throws IOException {
    try (Engine engine = new Engine(loadParityNetwork(), config)) {
      // Warmup (charge JIT, alloc buffers).
      engine.searchSync(new GameState(), SearchBudget.nodes(WARMUP));

      // Mesure.
      long t0 = System.nanoTime();
      SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(BUDGET));
      long elapsedNanos = System.nanoTime() - t0;

      double simsPerSec = result.simulationsCount() * 1e9 / elapsedNanos;
      System.out.printf(
          "%n  threads=%d  budget=%d  sims=%d  elapsed=%.2fs  rate=%.0f sims/s%n",
          config.searchThreads(),
          BUDGET,
          result.simulationsCount(),
          elapsedNanos / 1e9,
          simsPerSec);
      return simsPerSec;
    }
  }

  @Test
  @DisplayName("Bench : N=1 vs N=2 vs N=4")
  void benchSpeedup() throws Exception {
    System.out.println("\n=== EngineBatchedBenchTest (informatif) ===");
    System.out.printf("Warmup=%d, Mesure=%d sims%n", WARMUP, BUDGET);

    double mono = bench(configFor(1));
    double batched2 = bench(configFor(2));
    double batched4 = bench(configFor(4));

    double speedup2 = batched2 / mono;
    double speedup4 = batched4 / mono;
    System.out.printf("%n  Speedup N=2 : %.2fx%n", speedup2);
    System.out.printf("  Speedup N=4 : %.2fx%n", speedup4);

    // Pas d'assertion stricte (bench, pas test fonctionnel). On valide juste que les recherches
    // tournent.
    assertThat(mono).isPositive();
    assertThat(batched2).isPositive();
    assertThat(batched4).isPositive();
  }
}
