package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Tests unitaires de {@link UciCommandParser} (cf. SPEC §5.2, §14 grammaire, §12 phase 3).
 *
 * <p>Couvre :
 *
 * <ul>
 *   <li>Parsing valide pour chaque commande UCI subset (11 cas).
 *   <li>Tolérance erreurs : ligne null/empty, commande inconnue, casse, whitespace, malformations
 *       diverses → toutes mappées sur {@link UciCommand.Unknown}.
 *   <li>Cas particuliers position : startpos seul, startpos+moves, fen, fen+moves.
 *   <li>Cas particuliers go : tous params individuels et combinés, ordre quelconque, ponder.
 *   <li>Cas particuliers setoption : multi-word name, multi-word value, sans value.
 * </ul>
 */
class UciCommandParserTest {

  // -------------------------------------------------------------------------------------------
  // Commandes sans arguments
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("parse uci → Uci")
  void parseUci() {
    assertThat(UciCommandParser.parse("uci")).isInstanceOf(UciCommand.Uci.class);
  }

  @Test
  @DisplayName("parse isready → IsReady")
  void parseIsReady() {
    assertThat(UciCommandParser.parse("isready")).isInstanceOf(UciCommand.IsReady.class);
  }

  @Test
  @DisplayName("parse ucinewgame → UciNewGame")
  void parseUciNewGame() {
    assertThat(UciCommandParser.parse("ucinewgame")).isInstanceOf(UciCommand.UciNewGame.class);
  }

  @Test
  @DisplayName("parse stop → Stop")
  void parseStop() {
    assertThat(UciCommandParser.parse("stop")).isInstanceOf(UciCommand.Stop.class);
  }

  @Test
  @DisplayName("parse ponderhit → PonderHit")
  void parsePonderHit() {
    assertThat(UciCommandParser.parse("ponderhit")).isInstanceOf(UciCommand.PonderHit.class);
  }

  @Test
  @DisplayName("parse quit → Quit")
  void parseQuit() {
    assertThat(UciCommandParser.parse("quit")).isInstanceOf(UciCommand.Quit.class);
  }

  // -------------------------------------------------------------------------------------------
  // position
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("parse position startpos → Position avec moves vide")
  void parsePositionStartpos() {
    var cmd = (UciCommand.Position) UciCommandParser.parse("position startpos");
    assertThat(cmd.position()).isNotNull();
    assertThat(cmd.playedMoves()).isEmpty();
  }

  @Test
  @DisplayName("parse position startpos moves e2e4 e7e5 → Position avec 2 moves")
  void parsePositionStartposWithMoves() {
    var cmd = (UciCommand.Position) UciCommandParser.parse("position startpos moves e2e4 e7e5");
    assertThat(cmd.playedMoves()).hasSize(2);
  }

  @Test
  @DisplayName("parse position fen <FEN startpos equiv> → Position")
  void parsePositionFen() {
    var cmd =
        (UciCommand.Position)
            UciCommandParser.parse(
                "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
    assertThat(cmd.position()).isNotNull();
    assertThat(cmd.playedMoves()).isEmpty();
  }

  @Test
  @DisplayName("parse position fen <FEN> moves e2e4 → Position avec 1 move")
  void parsePositionFenWithMoves() {
    var cmd =
        (UciCommand.Position)
            UciCommandParser.parse(
                "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves e2e4");
    assertThat(cmd.playedMoves()).hasSize(1);
  }

  @Test
  @DisplayName("parse position avec extra whitespace → Position")
  void parsePositionExtraWhitespace() {
    var cmd = (UciCommand.Position) UciCommandParser.parse("position    startpos    moves    e2e4");
    assertThat(cmd.playedMoves()).hasSize(1);
  }

  // -------------------------------------------------------------------------------------------
  // go
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("parse go (sans args) → Go avec GoArgs default-empty")
  void parseGoEmptyArgs() {
    var cmd = (UciCommand.Go) UciCommandParser.parse("go");
    assertThat(cmd.args().wtimeMs()).isEmpty();
    assertThat(cmd.args().infinite()).isFalse();
    assertThat(cmd.args().ponder()).isFalse();
    assertThat(cmd.args().searchMoves()).isEmpty();
  }

  @Test
  @DisplayName("parse go infinite → Go.infinite=true")
  void parseGoInfinite() {
    var cmd = (UciCommand.Go) UciCommandParser.parse("go infinite");
    assertThat(cmd.args().infinite()).isTrue();
  }

  @Test
  @DisplayName("parse go movetime 5000 → movetimeMs=5000")
  void parseGoMovetime() {
    var cmd = (UciCommand.Go) UciCommandParser.parse("go movetime 5000");
    assertThat(cmd.args().movetimeMs()).hasValue(5000L);
  }

  @Test
  @DisplayName("parse go nodes 1000 → nodes=1000")
  void parseGoNodes() {
    var cmd = (UciCommand.Go) UciCommandParser.parse("go nodes 1000");
    assertThat(cmd.args().nodes()).hasValue(1000);
  }

  @Test
  @DisplayName("parse go depth 5 → depth=5")
  void parseGoDepth() {
    var cmd = (UciCommand.Go) UciCommandParser.parse("go depth 5");
    assertThat(cmd.args().depth()).hasValue(5);
  }

  @Test
  @DisplayName("parse go time control complet (wtime/btime/winc/binc/movestogo)")
  void parseGoTimeControl() {
    var cmd =
        (UciCommand.Go)
            UciCommandParser.parse("go wtime 10000 btime 10000 winc 100 binc 100 movestogo 30");
    assertThat(cmd.args().wtimeMs()).hasValue(10_000L);
    assertThat(cmd.args().btimeMs()).hasValue(10_000L);
    assertThat(cmd.args().wincMs()).hasValue(100L);
    assertThat(cmd.args().bincMs()).hasValue(100L);
    assertThat(cmd.args().movestogo()).hasValue(30);
  }

  @Test
  @DisplayName("parse go ponder + time → ponder=true + champs time présents")
  void parseGoPonderWithTime() {
    var cmd = (UciCommand.Go) UciCommandParser.parse("go ponder wtime 10000 btime 10000");
    assertThat(cmd.args().ponder()).isTrue();
    assertThat(cmd.args().wtimeMs()).hasValue(10_000L);
  }

  @Test
  @DisplayName("parse go params dans ordre quelconque (nodes avant infinite)")
  void parseGoArbitraryOrder() {
    var cmd = (UciCommand.Go) UciCommandParser.parse("go nodes 500 movestogo 20 wtime 30000");
    assertThat(cmd.args().nodes()).hasValue(500);
    assertThat(cmd.args().movestogo()).hasValue(20);
    assertThat(cmd.args().wtimeMs()).hasValue(30_000L);
  }

  @Test
  @DisplayName("parse go searchmoves e2e4 e2e3 → searchMoves ignoré silencieusement v1.0.0")
  void parseGoSearchmovesIgnored() {
    var cmd = (UciCommand.Go) UciCommandParser.parse("go searchmoves e2e4 e2e3 nodes 100");
    // searchmoves skip jusqu'au prochain go-keyword "nodes"
    assertThat(cmd.args().searchMoves()).isEmpty();
    assertThat(cmd.args().nodes()).hasValue(100);
  }

  // -------------------------------------------------------------------------------------------
  // setoption
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("parse setoption simple name + value")
  void parseSetOptionSimple() {
    var cmd = (UciCommand.SetOption) UciCommandParser.parse("setoption name Ponder value true");
    assertThat(cmd.name()).isEqualTo("Ponder");
    assertThat(cmd.value()).isEqualTo("true");
  }

  @Test
  @DisplayName("parse setoption multi-word name (Move Overhead)")
  void parseSetOptionMultiWordName() {
    var cmd =
        (UciCommand.SetOption) UciCommandParser.parse("setoption name Move Overhead value 100");
    assertThat(cmd.name()).isEqualTo("Move Overhead");
    assertThat(cmd.value()).isEqualTo("100");
  }

  @Test
  @DisplayName("parse setoption multi-word value (chemins, descriptions)")
  void parseSetOptionMultiWordValue() {
    var cmd =
        (UciCommand.SetOption)
            UciCommandParser.parse("setoption name SyzygyPath value /home/user/path with spaces");
    assertThat(cmd.name()).isEqualTo("SyzygyPath");
    assertThat(cmd.value()).isEqualTo("/home/user/path with spaces");
  }

  @Test
  @DisplayName("parse setoption sans value (option type Button)")
  void parseSetOptionWithoutValue() {
    var cmd = (UciCommand.SetOption) UciCommandParser.parse("setoption name Clear Hash");
    assertThat(cmd.name()).isEqualTo("Clear Hash");
    assertThat(cmd.value()).isNull();
  }

  // -------------------------------------------------------------------------------------------
  // debug
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("parse debug on → Debug(enabled=true)")
  void parseDebugOn() {
    var cmd = (UciCommand.Debug) UciCommandParser.parse("debug on");
    assertThat(cmd.enabled()).isTrue();
  }

  @Test
  @DisplayName("parse debug off → Debug(enabled=false)")
  void parseDebugOff() {
    var cmd = (UciCommand.Debug) UciCommandParser.parse("debug off");
    assertThat(cmd.enabled()).isFalse();
  }

  @Test
  @DisplayName("parse debug ON case-insensitive sur argument")
  void parseDebugCaseInsensitive() {
    var cmd1 = (UciCommand.Debug) UciCommandParser.parse("debug ON");
    var cmd2 = (UciCommand.Debug) UciCommandParser.parse("debug Off");
    assertThat(cmd1.enabled()).isTrue();
    assertThat(cmd2.enabled()).isFalse();
  }

  // -------------------------------------------------------------------------------------------
  // Tolérance : ligne null/empty/blank
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("parse null → Unknown(\"\")")
  void parseNullReturnsUnknownEmpty() {
    var cmd = (UciCommand.Unknown) UciCommandParser.parse(null);
    assertThat(cmd.rawLine()).isEmpty();
  }

  @Test
  @DisplayName("parse \"\" → Unknown(\"\")")
  void parseEmptyReturnsUnknown() {
    var cmd = (UciCommand.Unknown) UciCommandParser.parse("");
    assertThat(cmd.rawLine()).isEmpty();
  }

  @Test
  @DisplayName("parse \"   \" (whitespace only) → Unknown")
  void parseWhitespaceOnlyReturnsUnknown() {
    var cmd = (UciCommand.Unknown) UciCommandParser.parse("   ");
    assertThat(cmd.rawLine()).isEqualTo("   ");
  }

  // -------------------------------------------------------------------------------------------
  // Tolérance : commande inconnue, casse, whitespace
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("parse commande inconnue → Unknown avec rawLine préservée")
  void parseUnknownCommand() {
    var cmd = (UciCommand.Unknown) UciCommandParser.parse("foobar");
    assertThat(cmd.rawLine()).isEqualTo("foobar");
  }

  @Test
  @DisplayName("parse UCI / Uci (casse-insensitive sur nom commande) → Uci")
  void parseCommandCaseInsensitive() {
    assertThat(UciCommandParser.parse("UCI")).isInstanceOf(UciCommand.Uci.class);
    assertThat(UciCommandParser.parse("Uci")).isInstanceOf(UciCommand.Uci.class);
    assertThat(UciCommandParser.parse("Quit")).isInstanceOf(UciCommand.Quit.class);
  }

  @Test
  @DisplayName("parse \"   uci   \" (whitespace autour) → Uci")
  void parseWithExtraWhitespace() {
    assertThat(UciCommandParser.parse("   uci   ")).isInstanceOf(UciCommand.Uci.class);
  }

  @Test
  @DisplayName("parse \"position StartPos\" : sub-keyword strict → Unknown")
  void parseSubKeywordStrictCase() {
    // "startpos" doit être minuscule. "StartPos" → fall-through dans parsePosition →
    // IllegalArgumentException → Unknown via try/catch global.
    assertThat(UciCommandParser.parse("position StartPos")).isInstanceOf(UciCommand.Unknown.class);
  }

  // -------------------------------------------------------------------------------------------
  // Tolérance : commandes connues mais malformées → Unknown
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("parse position fen invalide → Unknown")
  void parsePositionInvalidFen() {
    assertThat(UciCommandParser.parse("position fen invalid"))
        .isInstanceOf(UciCommand.Unknown.class);
  }

  @Test
  @DisplayName("parse position avec sub-cmd inconnue → Unknown")
  void parsePositionInvalidSubcommand() {
    assertThat(UciCommandParser.parse("position invalid_subcmd"))
        .isInstanceOf(UciCommand.Unknown.class);
  }

  @Test
  @DisplayName("parse position startpos moves <coup illégal> → Unknown")
  void parsePositionIllegalMove() {
    assertThat(UciCommandParser.parse("position startpos moves invalidmove"))
        .isInstanceOf(UciCommand.Unknown.class);
    assertThat(UciCommandParser.parse("position startpos moves e7e5"))
        .as("e7e5 illégal sur startpos (mauvais tour)")
        .isInstanceOf(UciCommand.Unknown.class);
  }

  @Test
  @DisplayName("parse go wtime <non numérique> → Unknown")
  void parseGoInvalidNumber() {
    assertThat(UciCommandParser.parse("go wtime abc")).isInstanceOf(UciCommand.Unknown.class);
  }

  @Test
  @DisplayName("parse go movetime sans valeur → Unknown (argument manquant)")
  void parseGoMissingArgument() {
    assertThat(UciCommandParser.parse("go movetime")).isInstanceOf(UciCommand.Unknown.class);
  }

  @Test
  @DisplayName("parse setoption sans 'name' → Unknown")
  void parseSetOptionMissingName() {
    assertThat(UciCommandParser.parse("setoption Ponder value true"))
        .isInstanceOf(UciCommand.Unknown.class);
  }

  @Test
  @DisplayName("parse setoption tronqué → Unknown")
  void parseSetOptionTooShort() {
    assertThat(UciCommandParser.parse("setoption")).isInstanceOf(UciCommand.Unknown.class);
    assertThat(UciCommandParser.parse("setoption name")).isInstanceOf(UciCommand.Unknown.class);
  }

  @Test
  @DisplayName("parse debug sans argument → Unknown")
  void parseDebugMissingArg() {
    assertThat(UciCommandParser.parse("debug")).isInstanceOf(UciCommand.Unknown.class);
  }

  @Test
  @DisplayName("parse debug avec valeur invalide → Unknown")
  void parseDebugInvalidValue() {
    assertThat(UciCommandParser.parse("debug maybe")).isInstanceOf(UciCommand.Unknown.class);
  }

  // -------------------------------------------------------------------------------------------
  // Intégration : roundtrip position cohérent avec applyMove
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Intégration : position startpos moves e2e4 e7e5 g1f3 produit GameState cohérent")
  void integrationPositionRoundtrip() {
    var cmd =
        (UciCommand.Position) UciCommandParser.parse("position startpos moves e2e4 e7e5 g1f3");
    assertThat(cmd.playedMoves()).hasSize(3);
    // Vérifier que la position résultante a bien 3 coups joués (côté au trait noir).
    var pos = cmd.position().currentPosition();
    assertThat(pos).isNotNull();
    // Demi-coup count = 3 (post 1.e4 e5 2.Nf3 = halfmove 3, fullmove 2).
    assertThat(pos.fullmoveNumber()).isEqualTo(2);
  }
}
