package org.nanozero.uci.internal;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nanozero.engine.Engine;
import org.nanozero.engine.EngineConfig;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/**
 * Tests d'intégration de la boucle UCI complète via scripts {@code .uciscript} (cf. SPEC §8.2, §12
 * phase 7).
 *
 * <p>Pour chaque script trouvé dans {@code src/test/resources/uciscripts/}, on construit un {@link
 * UciAdapterState} frais (Engine + writer + options) et on exécute le script via {@link
 * UciScriptRunner#runScript}. Le test échoue si un pattern {@code < ...} ne matche pas ou timeout.
 */
class UciScriptIT {

  private static Network sharedNetwork;

  @BeforeAll
  static void loadNetwork() throws IOException {
    var url = UciScriptIT.class.getResource("/npz/parity-model.npz");
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

  /**
   * Provider de scripts pour {@link ParameterizedTest}. Lit le dossier {@code uciscripts/} via le
   * classpath.
   */
  static Stream<Path> scriptProvider() throws IOException, java.net.URISyntaxException {
    var url = UciScriptIT.class.getResource("/uciscripts");
    if (url == null) {
      return Stream.empty();
    }
    Path dir = Paths.get(url.toURI());
    List<Path> scripts = UciScriptRunner.listScripts(dir);
    return scripts.stream();
  }

  @ParameterizedTest(name = "{0}")
  @MethodSource("scriptProvider")
  void runUciScript(Path script) throws IOException {
    Engine engine = new Engine(sharedNetwork, EngineConfig.defaults());
    var baos = new ByteArrayOutputStream();
    PrintStream ps = new PrintStream(baos, /* autoFlush */ true, StandardCharsets.UTF_8);
    var writer = new UciResponseWriter(ps);
    try (var state = new UciAdapterState(engine, writer, new UciOptionsState())) {
      UciScriptRunner.runScript(script, state, baos);
    }
  }
}
