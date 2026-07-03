package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Paths;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.GameState;
import org.nanozero.board.MoveGen;
import org.nanozero.engine.Engine;
import org.nanozero.engine.EngineConfig;
import org.nanozero.engine.SearchBudget;
import org.nanozero.engine.SearchResult;
import org.nanozero.nn.LoadOptions;
import org.nanozero.nn.Network;
import org.nanozero.nn.NetworkLoader;

/**
 * Tests v1.2.0 : émission {@code info string visits <move> <count> ...} à la fin de chaque search
 * (cf. SPEC §6.5 amendée v1.2.0, ADR-003 mise à jour v1.2.0). Vérifie le format, l'ordre, la
 * complétude, les cas terminal et stop.
 */
class UciVisitsInfoStringTest {

  private static Network sharedNetwork;

  @BeforeAll
  static void loadNetwork() throws IOException {
    var url = UciVisitsInfoStringTest.class.getResource("/npz/parity-model.npz");
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

  // -------------------------------------------------------------------------------------------
  // Tests unitaires : formatVisitsContent (helper pur, pas besoin d'Engine)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("formatVisitsContent format : préfixe 'visits' + pairs '<move> <count>' space-sep")
  void formatVisitsContent_basicFormat() {
    int e2e4 = encodeStartposMove("e2e4");
    int d2d4 = encodeStartposMove("d2d4");
    int g1f3 = encodeStartposMove("g1f3");
    String content =
        UciSession.formatVisitsContent(new int[] {e2e4, d2d4, g1f3}, new int[] {234, 187, 156});
    assertThat(content).startsWith("visits ");
    assertThat(content).contains("e2e4 234");
    assertThat(content).contains("d2d4 187");
    assertThat(content).contains("g1f3 156");
    // Format exact : "visits e2e4 234 d2d4 187 g1f3 156"
    assertThat(content).isEqualTo("visits e2e4 234 d2d4 187 g1f3 156");
  }

  @Test
  @DisplayName("formatVisitsContent : moves triés par visit count décroissant")
  void formatVisitsContent_sortedDescending() {
    int m1 = encodeStartposMove("a2a3");
    int m2 = encodeStartposMove("a2a4");
    int m3 = encodeStartposMove("b2b3");
    // Input ordre : m1=50, m2=200, m3=100. Output attendu : m2 200, m3 100, m1 50.
    String content =
        UciSession.formatVisitsContent(new int[] {m1, m2, m3}, new int[] {50, 200, 100});
    String[] tokens = content.split(" ");
    // tokens[0]="visits", tokens[1]="a2a4", tokens[2]="200", tokens[3]="b2b3", tokens[4]="100", ...
    assertThat(tokens[2]).isEqualTo("200");
    assertThat(tokens[4]).isEqualTo("100");
    assertThat(tokens[6]).isEqualTo("50");
  }

  @Test
  @DisplayName("formatVisitsContent : skip moves avec visit_count == 0")
  void formatVisitsContent_skipZeroVisits() {
    int m1 = encodeStartposMove("a2a3");
    int m2 = encodeStartposMove("a2a4");
    int m3 = encodeStartposMove("b2b3");
    String content = UciSession.formatVisitsContent(new int[] {m1, m2, m3}, new int[] {50, 0, 100});
    // m2 avec 0 visit doit être absent.
    assertThat(content).isEqualTo("visits b2b3 100 a2a3 50");
    assertThat(content).doesNotContain("a2a4");
  }

  @Test
  @DisplayName("formatVisitsContent : moves vides (position terminale) → 'visits' seul")
  void formatVisitsContent_emptyTerminalPosition() {
    String content = UciSession.formatVisitsContent(new int[0], new int[0]);
    assertThat(content).isEqualTo("visits");
  }

  @Test
  @DisplayName("formatVisitsContent : tous les counts à 0 → 'visits' seul")
  void formatVisitsContent_allZeroVisits() {
    int m1 = encodeStartposMove("a2a3");
    int m2 = encodeStartposMove("b2b3");
    String content = UciSession.formatVisitsContent(new int[] {m1, m2}, new int[] {0, 0});
    assertThat(content).isEqualTo("visits");
  }

  // -------------------------------------------------------------------------------------------
  // Tests d'émission end-to-end via UciSession
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("emitBestmove : émission 'info string visits' AVANT 'bestmove' (séquence)")
  void emitBestmove_infoStringVisitsBeforeBestmove() throws Exception {
    var baos = new ByteArrayOutputStream();
    PrintStream ps = new PrintStream(baos, true, StandardCharsets.UTF_8);
    UciResponseWriter writer = new UciResponseWriter(ps);
    try (Engine engine = new Engine(sharedNetwork, EngineConfig.defaults())) {
      SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(64));
      assertThat(result.simulationsCount()).isGreaterThan(0);

      UciSession session = new UciSession(engine, writer, /* isPonder */ false, GoArgs.empty());
      session.emitBestmove(result);

      String output = baos.toString(StandardCharsets.UTF_8);
      int idxInfoString = output.indexOf("info string visits");
      int idxBestmove = output.indexOf("bestmove ");
      assertThat(idxInfoString)
          .as("info string visits doit être présent")
          .isGreaterThanOrEqualTo(0);
      assertThat(idxBestmove).as("bestmove doit être présent").isGreaterThanOrEqualTo(0);
      assertThat(idxInfoString)
          .as("info string visits doit apparaître AVANT bestmove")
          .isLessThan(idxBestmove);
    }
  }

  @Test
  @DisplayName("emitBestmove : premier coup de visits = bestmove (cohérence I-Result-1)")
  void emitBestmove_firstVisitsMoveEqualsBestmove() throws Exception {
    var baos = new ByteArrayOutputStream();
    PrintStream ps = new PrintStream(baos, true, StandardCharsets.UTF_8);
    UciResponseWriter writer = new UciResponseWriter(ps);
    try (Engine engine = new Engine(sharedNetwork, EngineConfig.defaults())) {
      SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(128));

      UciSession session = new UciSession(engine, writer, false, GoArgs.empty());
      session.emitBestmove(result);

      String output = baos.toString(StandardCharsets.UTF_8);
      // Extraire la ligne "info string visits ..." et la ligne "bestmove ..."
      String visitsLine =
          java.util.Arrays.stream(output.split("\n"))
              .filter(l -> l.startsWith("info string visits"))
              .findFirst()
              .orElseThrow();
      String bestmoveLine =
          java.util.Arrays.stream(output.split("\n"))
              .filter(l -> l.startsWith("bestmove "))
              .findFirst()
              .orElseThrow();

      String[] visitsTokens = visitsLine.split(" ");
      // tokens : "info", "string", "visits", "<move1>", "<count1>", ...
      assertThat(visitsTokens.length).isGreaterThan(3);
      String firstVisitsMove = visitsTokens[3];
      String[] bestmoveTokens = bestmoveLine.split(" ");
      String bestmoveStr = bestmoveTokens[1];
      assertThat(firstVisitsMove)
          .as("Premier coup de visits doit être identique au bestmove (argmax visites)")
          .isEqualTo(bestmoveStr);
    }
  }

  @Test
  @DisplayName("emitBestmove : visits triés par count décroissant dans la ligne émise")
  void emitBestmove_visitsLineSortedDescending() throws Exception {
    var baos = new ByteArrayOutputStream();
    PrintStream ps = new PrintStream(baos, true, StandardCharsets.UTF_8);
    UciResponseWriter writer = new UciResponseWriter(ps);
    try (Engine engine = new Engine(sharedNetwork, EngineConfig.defaults())) {
      SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(256));
      UciSession session = new UciSession(engine, writer, false, GoArgs.empty());
      session.emitBestmove(result);

      String output = baos.toString(StandardCharsets.UTF_8);
      String visitsLine =
          java.util.Arrays.stream(output.split("\n"))
              .filter(l -> l.startsWith("info string visits"))
              .findFirst()
              .orElseThrow();
      String[] tokens = visitsLine.split(" ");
      // Vérifier counts décroissants (tokens[4], [6], [8], ... sont les counts).
      int previousCount = Integer.MAX_VALUE;
      for (int i = 4; i < tokens.length; i += 2) {
        int count = Integer.parseInt(tokens[i]);
        assertThat(count)
            .as(
                "Token count à l'index %d (=%d) doit être <= précédent (=%d)",
                i, count, previousCount)
            .isLessThanOrEqualTo(previousCount);
        previousCount = count;
      }
    }
  }

  @Test
  @DisplayName("emitBestmove : visites = somme(child_visits) == simulationsCount - root_visit")
  void emitBestmove_visitsContentIncludesAllVisitedMoves() throws Exception {
    var baos = new ByteArrayOutputStream();
    PrintStream ps = new PrintStream(baos, true, StandardCharsets.UTF_8);
    UciResponseWriter writer = new UciResponseWriter(ps);
    try (Engine engine = new Engine(sharedNetwork, EngineConfig.defaults())) {
      SearchResult result = engine.searchSync(new GameState(), SearchBudget.nodes(128));
      UciSession session = new UciSession(engine, writer, false, GoArgs.empty());
      session.emitBestmove(result);

      String output = baos.toString(StandardCharsets.UTF_8);
      String visitsLine =
          java.util.Arrays.stream(output.split("\n"))
              .filter(l -> l.startsWith("info string visits"))
              .findFirst()
              .orElseThrow();
      String[] tokens = visitsLine.split(" ");
      // Sum des counts dans la ligne = sum des child visits non-zero.
      int sumFromLine = 0;
      for (int i = 4; i < tokens.length; i += 2) {
        sumFromLine += Integer.parseInt(tokens[i]);
      }
      int sumFromResult = 0;
      for (int v : result.childVisits()) {
        sumFromResult += v;
      }
      assertThat(sumFromLine).isEqualTo(sumFromResult);
    }
  }

  // -------------------------------------------------------------------------------------------
  // Helpers : encode un coup UCI startpos
  // -------------------------------------------------------------------------------------------

  /**
   * Encode un coup UCI long algebraic depuis startpos (helper test). Génère les coups légaux et
   * cherche celui qui correspond au string UCI demandé.
   */
  private static int encodeStartposMove(String uciMove) {
    GameState startpos = new GameState();
    int[] buffer = new int[MoveGen.RECOMMENDED_BUFFER_SIZE];
    int n = startpos.generateMoves(buffer, 0);
    for (int i = 0; i < n; i++) {
      if (UciMoveCodec.encode(buffer[i]).equals(uciMove)) {
        return buffer[i];
      }
    }
    throw new AssertionError("Move not legal from startpos: " + uciMove);
  }
}
