package org.nanozero.sprt.tournament;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Duration;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class TimeControlTest {

  @Test
  @DisplayName("parse '10+0.1' = Fischer 10s + 0.1s")
  void parseFischer() {
    TimeControl tc = TimeControl.parse("10+0.1");
    assertThat(tc.baseTime()).isEqualTo(Duration.ofSeconds(10));
    assertThat(tc.increment()).isEqualTo(Duration.ofMillis(100));
    assertThat(tc.movesPerControl()).isEqualTo(0);
    assertThat(tc.unlimited()).isFalse();
  }

  @Test
  @DisplayName("parse '60' = sudden death 60s")
  void parseSuddenDeath() {
    TimeControl tc = TimeControl.parse("60");
    assertThat(tc.baseTime()).isEqualTo(Duration.ofSeconds(60));
    assertThat(tc.increment()).isEqualTo(Duration.ZERO);
    assertThat(tc.movesPerControl()).isEqualTo(0);
  }

  @Test
  @DisplayName("parse '5+0.05' = standard TCEC short")
  void parseTcecShort() {
    TimeControl tc = TimeControl.parse("5+0.05");
    assertThat(tc.baseTime()).isEqualTo(Duration.ofSeconds(5));
    assertThat(tc.increment()).isEqualTo(Duration.ofMillis(50));
  }

  @Test
  @DisplayName("parse '40/60' = 40 moves per 60s")
  void parseClassical() {
    TimeControl tc = TimeControl.parse("40/60");
    assertThat(tc.baseTime()).isEqualTo(Duration.ofSeconds(60));
    assertThat(tc.movesPerControl()).isEqualTo(40);
    assertThat(tc.increment()).isEqualTo(Duration.ZERO);
  }

  @Test
  @DisplayName("parse '40/60+0.1' = classical + Fischer")
  void parseClassicalFischer() {
    TimeControl tc = TimeControl.parse("40/60+0.1");
    assertThat(tc.baseTime()).isEqualTo(Duration.ofSeconds(60));
    assertThat(tc.movesPerControl()).isEqualTo(40);
    assertThat(tc.increment()).isEqualTo(Duration.ofMillis(100));
  }

  @Test
  @DisplayName("parse 'inf' = unlimited")
  void parseInfinite() {
    TimeControl tc = TimeControl.parse("inf");
    assertThat(tc.unlimited()).isTrue();
    assertThat(tc.baseTime()).isEqualTo(Duration.ZERO);
  }

  @Test
  @DisplayName("parse 'INFINITE' (case-insensitive) = unlimited")
  void parseInfiniteAllCaps() {
    TimeControl tc = TimeControl.parse("INFINITE");
    assertThat(tc.unlimited()).isTrue();
  }

  @Test
  @DisplayName("parse with whitespace tolerated")
  void parseWhitespace() {
    TimeControl tc = TimeControl.parse("  10+0.1  ");
    assertThat(tc.baseTime()).isEqualTo(Duration.ofSeconds(10));
    assertThat(tc.increment()).isEqualTo(Duration.ofMillis(100));
  }

  @Test
  @DisplayName("parse invalid throws")
  void parseInvalid() {
    assertThatThrownBy(() -> TimeControl.parse("abc")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> TimeControl.parse("10++0.1"))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> TimeControl.parse("")).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> TimeControl.parse("+0.1"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("parse null throws")
  void parseNull() {
    assertThatThrownBy(() -> TimeControl.parse(null)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("factory sudden(d)")
  void factorySudden() {
    TimeControl tc = TimeControl.sudden(Duration.ofSeconds(120));
    assertThat(tc.baseTime()).isEqualTo(Duration.ofSeconds(120));
    assertThat(tc.increment()).isEqualTo(Duration.ZERO);
    assertThat(tc.unlimited()).isFalse();
  }

  @Test
  @DisplayName("factory fischer(d, inc)")
  void factoryFischer() {
    TimeControl tc = TimeControl.fischer(Duration.ofSeconds(10), Duration.ofMillis(100));
    assertThat(tc.baseTime()).isEqualTo(Duration.ofSeconds(10));
    assertThat(tc.increment()).isEqualTo(Duration.ofMillis(100));
  }

  @Test
  @DisplayName("factory unlimited()")
  void factoryUnlimited() {
    TimeControl tc = TimeControl.infinite();
    assertThat(tc.unlimited()).isTrue();
  }

  @Test
  @DisplayName("factory sudden rejects negative")
  void factoryRejectsNegative() {
    assertThatThrownBy(() -> TimeControl.sudden(Duration.ofSeconds(-1)))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> TimeControl.fischer(Duration.ofSeconds(10), Duration.ofMillis(-1)))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  @DisplayName("toSpec round-trip")
  void toSpecRoundTrip() {
    assertThat(TimeControl.parse("10+0.1").toSpec()).isEqualTo("10+0.1");
    assertThat(TimeControl.parse("60").toSpec()).isEqualTo("60");
    assertThat(TimeControl.parse("40/60").toSpec()).isEqualTo("40/60");
    assertThat(TimeControl.parse("40/60+0.1").toSpec()).isEqualTo("40/60+0.1");
    assertThat(TimeControl.parse("inf").toSpec()).isEqualTo("inf");
  }
}
