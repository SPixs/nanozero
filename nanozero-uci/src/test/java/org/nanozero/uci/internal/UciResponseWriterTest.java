package org.nanozero.uci.internal;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.nanozero.board.Move;
import org.nanozero.board.MoveType;
import org.nanozero.board.Square;

/**
 * Tests unitaires de {@link UciResponseWriter} (cf. SPEC §3.2, §5.6, §7.2, §12 phase 4).
 *
 * <p>Couvre :
 *
 * <ul>
 *   <li>Format pur via {@link UciResponseWriter#format} (sans I/O) sur les 6 sous-types.
 *   <li>Émission via {@link UciResponseWriter#emit} avec capture {@link ByteArrayOutputStream}
 *       (vérifie LF strict, contenu, auto-flush).
 *   <li>Validation argument null (constructor + emit).
 * </ul>
 */
class UciResponseWriterTest {

  // -------------------------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------------------------

  /** Encode un Move int simple à partir de from/to algébriques (NORMAL move type). */
  private static int normalMove(String from, String to) {
    return Move.encode(Square.fromAlgebraic(from), Square.fromAlgebraic(to));
  }

  /** Encode une promotion sur from/to + promo bits 0..3. */
  private static int promoMove(String from, String to, int promoBits) {
    return Move.encode(
        Square.fromAlgebraic(from), Square.fromAlgebraic(to), MoveType.PROMOTION, promoBits);
  }

  // -------------------------------------------------------------------------------------------
  // format() : records sans paramètres
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("format UciOk → \"uciok\"")
  void formatUciOk() {
    assertThat(UciResponseWriter.format(new UciResponse.UciOk())).isEqualTo("uciok");
  }

  @Test
  @DisplayName("format ReadyOk → \"readyok\"")
  void formatReadyOk() {
    assertThat(UciResponseWriter.format(new UciResponse.ReadyOk())).isEqualTo("readyok");
  }

  // -------------------------------------------------------------------------------------------
  // format() : Id (2 lignes séparées par \n interne)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("format Id → \"id name X\\nid author Y\"")
  void formatId() {
    String s = UciResponseWriter.format(new UciResponse.Id("NanoZero 1.0.0", "Mametz"));
    assertThat(s).isEqualTo("id name NanoZero 1.0.0\nid author Mametz");
  }

  // -------------------------------------------------------------------------------------------
  // format() : Option (3 sous-types UciOption)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("format Option Check default true/false")
  void formatOptionCheck() {
    assertThat(
            UciResponseWriter.format(new UciResponse.Option(new UciOption.Check("Ponder", true))))
        .isEqualTo("option name Ponder type check default true");
    assertThat(
            UciResponseWriter.format(new UciResponse.Option(new UciOption.Check("Ponder", false))))
        .isEqualTo("option name Ponder type check default false");
  }

  @Test
  @DisplayName("format Option Spin avec range")
  void formatOptionSpin() {
    var opt = new UciResponse.Option(new UciOption.Spin("Move Overhead", 30, 0, 5000));
    assertThat(UciResponseWriter.format(opt))
        .isEqualTo("option name Move Overhead type spin default 30 min 0 max 5000");
  }

  @Test
  @DisplayName("format Option String_")
  void formatOptionString() {
    var opt = new UciResponse.Option(new UciOption.String_("SyzygyPath", "/path/to/syzygy"));
    assertThat(UciResponseWriter.format(opt))
        .isEqualTo("option name SyzygyPath type string default /path/to/syzygy");
  }

  @Test
  @DisplayName("format Option String_ avec defaultValue empty (espace traînant accepté)")
  void formatOptionStringEmptyDefault() {
    var opt = new UciResponse.Option(new UciOption.String_("Foo", ""));
    assertThat(UciResponseWriter.format(opt)).isEqualTo("option name Foo type string default ");
  }

  // -------------------------------------------------------------------------------------------
  // format() : BestMove
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("format BestMove sans ponder → \"bestmove e2e4\"")
  void formatBestMoveSimple() {
    int m = normalMove("e2", "e4");
    assertThat(UciResponseWriter.format(new UciResponse.BestMove(m, OptionalInt.empty())))
        .isEqualTo("bestmove e2e4");
  }

  @Test
  @DisplayName("format BestMove avec ponder → \"bestmove e2e4 ponder e7e5\"")
  void formatBestMoveWithPonder() {
    int m = normalMove("e2", "e4");
    int p = normalMove("e7", "e5");
    assertThat(UciResponseWriter.format(new UciResponse.BestMove(m, OptionalInt.of(p))))
        .isEqualTo("bestmove e2e4 ponder e7e5");
  }

  @Test
  @DisplayName("format BestMove promotion → \"bestmove e7e8q\"")
  void formatBestMovePromotion() {
    int m = promoMove("e7", "e8", 3); // queen
    assertThat(UciResponseWriter.format(new UciResponse.BestMove(m, OptionalInt.empty())))
        .isEqualTo("bestmove e7e8q");
  }

  // -------------------------------------------------------------------------------------------
  // format() : Info (champs vides, individuels, complet, ordre, score mate vs cp)
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("format Info empty fields → \"info\" tout court (légal UCI)")
  void formatInfoEmpty() {
    assertThat(UciResponseWriter.format(new UciResponse.Info(InfoFields.empty())))
        .isEqualTo("info");
  }

  @Test
  @DisplayName("format Info minimal (depth + nodes + nps)")
  void formatInfoMinimal() {
    var f =
        new InfoFields(
            OptionalInt.of(10),
            OptionalInt.empty(),
            OptionalInt.of(1234),
            OptionalInt.of(500),
            OptionalLong.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            new int[0],
            OptionalInt.empty(),
            Optional.empty());
    assertThat(UciResponseWriter.format(new UciResponse.Info(f)))
        .isEqualTo("info depth 10 nodes 1234 nps 500");
  }

  @Test
  @DisplayName(
      "format Info complet : ordre canonique depth seldepth multipv score nodes nps time pv")
  void formatInfoCompleteWithCp() {
    int e2e4 = normalMove("e2", "e4");
    int e7e5 = normalMove("e7", "e5");
    var f =
        new InfoFields(
            OptionalInt.of(10),
            OptionalInt.of(15),
            OptionalInt.of(50_000),
            OptionalInt.of(2_500),
            OptionalLong.of(20_000L),
            OptionalInt.of(25),
            OptionalInt.empty(),
            new int[] {e2e4, e7e5},
            OptionalInt.of(1),
            Optional.empty());
    assertThat(UciResponseWriter.format(new UciResponse.Info(f)))
        .isEqualTo(
            "info depth 10 seldepth 15 multipv 1 score cp 25 nodes 50000 nps 2500 time 20000 pv"
                + " e2e4 e7e5");
  }

  @Test
  @DisplayName("format Info score mate (sans cp) → \"score mate 5\"")
  void formatInfoScoreMate() {
    var f =
        new InfoFields(
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalLong.empty(),
            OptionalInt.empty(),
            OptionalInt.of(5),
            new int[0],
            OptionalInt.empty(),
            Optional.empty());
    assertThat(UciResponseWriter.format(new UciResponse.Info(f))).isEqualTo("info score mate 5");
  }

  @Test
  @DisplayName("format Info score mate prime sur cp si les deux présents (cas dégénéré)")
  void formatInfoScoreMatePrimesOverCp() {
    var f =
        new InfoFields(
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalLong.empty(),
            OptionalInt.of(150),
            OptionalInt.of(3),
            new int[0],
            OptionalInt.empty(),
            Optional.empty());
    String out = UciResponseWriter.format(new UciResponse.Info(f));
    assertThat(out).contains("score mate 3");
    assertThat(out).doesNotContain("score cp");
  }

  @Test
  @DisplayName("format Info scores négatifs (cp -150, mate -3)")
  void formatInfoNegativeScores() {
    var fCp =
        new InfoFields(
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalLong.empty(),
            OptionalInt.of(-150),
            OptionalInt.empty(),
            new int[0],
            OptionalInt.empty(),
            Optional.empty());
    assertThat(UciResponseWriter.format(new UciResponse.Info(fCp))).isEqualTo("info score cp -150");

    var fMate =
        new InfoFields(
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalLong.empty(),
            OptionalInt.empty(),
            OptionalInt.of(-3),
            new int[0],
            OptionalInt.empty(),
            Optional.empty());
    assertThat(UciResponseWriter.format(new UciResponse.Info(fMate)))
        .isEqualTo("info score mate -3");
  }

  @Test
  @DisplayName("format Info pv multiple moves")
  void formatInfoPvMultiple() {
    int e2e4 = normalMove("e2", "e4");
    int e7e5 = normalMove("e7", "e5");
    int g1f3 = normalMove("g1", "f3");
    var f =
        new InfoFields(
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalLong.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            new int[] {e2e4, e7e5, g1f3},
            OptionalInt.empty(),
            Optional.empty());
    assertThat(UciResponseWriter.format(new UciResponse.Info(f)))
        .isEqualTo("info pv e2e4 e7e5 g1f3");
  }

  @Test
  @DisplayName("format Info string générique en fin de ligne")
  void formatInfoString() {
    var f =
        new InfoFields(
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            OptionalLong.empty(),
            OptionalInt.empty(),
            OptionalInt.empty(),
            new int[0],
            OptionalInt.empty(),
            Optional.of("hashfull 250"));
    assertThat(UciResponseWriter.format(new UciResponse.Info(f)))
        .isEqualTo("info string hashfull 250");
  }

  // -------------------------------------------------------------------------------------------
  // emit() avec capture stdout
  // -------------------------------------------------------------------------------------------

  /** Crée un writer sur ByteArrayOutputStream + PrintStream auto-flush UTF_8. */
  private static UciResponseWriter writerOver(ByteArrayOutputStream baos) {
    PrintStream ps = new PrintStream(baos, /* autoFlush */ true, StandardCharsets.UTF_8);
    return new UciResponseWriter(ps);
  }

  @Test
  @DisplayName("emit UciOk → \"uciok\\n\" exact (LF strict)")
  void emitUciOkStrictLf() {
    var baos = new ByteArrayOutputStream();
    writerOver(baos).emit(new UciResponse.UciOk());
    assertThat(baos.toString(StandardCharsets.UTF_8)).isEqualTo("uciok\n");
  }

  @Test
  @DisplayName("emit Id → 2 lignes UCI conformes terminées chacune par \\n")
  void emitIdTwoLines() {
    var baos = new ByteArrayOutputStream();
    writerOver(baos).emit(new UciResponse.Id("NanoZero 1.0.0", "Mametz"));
    String s = baos.toString(StandardCharsets.UTF_8);
    // "id name X\nid author Y" + "\n" final = "id name X\nid author Y\n"
    assertThat(s).isEqualTo("id name NanoZero 1.0.0\nid author Mametz\n");
    assertThat(s.split("\n")).hasSize(2);
  }

  @Test
  @DisplayName("emit BestMove → \"bestmove e2e4\\n\"")
  void emitBestMove() {
    var baos = new ByteArrayOutputStream();
    int m = normalMove("e2", "e4");
    writerOver(baos).emit(new UciResponse.BestMove(m, OptionalInt.empty()));
    assertThat(baos.toString(StandardCharsets.UTF_8)).isEqualTo("bestmove e2e4\n");
  }

  @Test
  @DisplayName("emit auto-flush actif : sortie immédiate sans flush() explicite")
  void emitAutoFlushIsActive() {
    var baos = new ByteArrayOutputStream();
    var writer = writerOver(baos);
    // Écrire SANS appel flush() explicite — l'auto-flush du PrintStream sur \n doit suffire.
    writer.emit(new UciResponse.UciOk());
    // Lecture immédiate du contenu (sans close ni flush).
    assertThat(baos.toString(StandardCharsets.UTF_8)).isEqualTo("uciok\n");
  }

  @Test
  @DisplayName("emit séquence UCI typique : id × 2 lignes, options, uciok, readyok")
  void emitUciSessionSequence() {
    var baos = new ByteArrayOutputStream();
    var w = writerOver(baos);
    w.emit(new UciResponse.Id("NanoZero", "Mametz"));
    w.emit(new UciResponse.Option(new UciOption.Check("Ponder", true)));
    w.emit(new UciResponse.Option(new UciOption.Spin("Move Overhead", 30, 0, 5000)));
    w.emit(new UciResponse.UciOk());
    w.emit(new UciResponse.ReadyOk());
    String s = baos.toString(StandardCharsets.UTF_8);
    String[] lines = s.split("\n");
    assertThat(lines)
        .containsExactly(
            "id name NanoZero",
            "id author Mametz",
            "option name Ponder type check default true",
            "option name Move Overhead type spin default 30 min 0 max 5000",
            "uciok",
            "readyok");
    // Vérifie la terminaison \n finale.
    assertThat(s).endsWith("\n");
  }

  // -------------------------------------------------------------------------------------------
  // Validation arguments
  // -------------------------------------------------------------------------------------------

  @Test
  @DisplayName("Constructor avec null PrintStream → NPE")
  void constructorNullStreamThrowsNpe() {
    assertThatThrownBy(() -> new UciResponseWriter(null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("out");
  }

  @Test
  @DisplayName("emit avec null response → NPE")
  void emitNullResponseThrowsNpe() {
    var baos = new ByteArrayOutputStream();
    var w = writerOver(baos);
    assertThatThrownBy(() -> w.emit(null))
        .isInstanceOf(NullPointerException.class)
        .hasMessageContaining("response");
  }
}
