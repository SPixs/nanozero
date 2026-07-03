package org.nanozero.sprt.tournament;

import java.time.Duration;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Time control immutable pour une game SPRT.
 *
 * <p>Formats supportés (cf. SPEC-sprt §5.1) :
 *
 * <ul>
 *   <li><strong>{@code "tc+inc"}</strong> : sudden death + Fischer increment. Ex: {@code "10+0.1"}
 *       = 10 secondes base, 0.1 sec / coup.
 *   <li><strong>{@code "tc"}</strong> : sudden death sans increment. Ex: {@code "60"} = 60 sec pour
 *       la game entière.
 *   <li><strong>{@code "x/tc"}</strong> : x coups en tc secondes. Ex: {@code "40/60"}.
 *   <li><strong>{@code "x/tc+inc"}</strong> : combination. Ex: {@code "40/60+0.1"}.
 *   <li><strong>{@code "inf"}</strong> ou {@code "infinite"} : illimité (debug, non-SPRT).
 * </ul>
 *
 * <p>Unités : secondes (peut être décimal, ex {@code 0.1} = 100 ms).
 *
 * <p>Immutable. Thread-safe.
 *
 * @param baseTime temps de base par engine au début de la game (≥ 0)
 * @param increment Fischer increment ajouté après chaque coup (≥ 0)
 * @param movesPerControl 0 = sudden death, &gt; 0 = nombre de coups pour le contrôle de temps
 *     (style classical : N moves in baseTime, puis reset)
 * @param unlimited {@code true} si pas de limite (mode {@code "inf"})
 */
public record TimeControl(
    Duration baseTime, Duration increment, int movesPerControl, boolean unlimited) {

  private static final Pattern PATTERN =
      Pattern.compile(
          "^(?:(?<moves>\\d+)/)?(?<base>\\d+(?:\\.\\d+)?)(?:\\+(?<inc>\\d+(?:\\.\\d+)?))?$");

  /**
   * Construit un time control sudden death sans increment.
   *
   * @param base temps total de la game pour chaque engine
   * @return time control {@code "{base}"} équivalent
   */
  public static TimeControl sudden(Duration base) {
    requireNonNegative(base, "base");
    return new TimeControl(base, Duration.ZERO, 0, false);
  }

  /**
   * Construit un time control Fischer (sudden death + increment).
   *
   * @param base temps de base
   * @param inc increment par coup
   * @return time control {@code "{base}+{inc}"} équivalent
   */
  public static TimeControl fischer(Duration base, Duration inc) {
    requireNonNegative(base, "base");
    requireNonNegative(inc, "increment");
    return new TimeControl(base, inc, 0, false);
  }

  /** Time control «illimité» pour debug — engine peut prendre tout le temps qu'il veut. */
  public static TimeControl infinite() {
    return new TimeControl(Duration.ZERO, Duration.ZERO, 0, true);
  }

  /**
   * Parse une spec sous l'un des formats listés dans la Javadoc classe.
   *
   * @param spec chaîne à parser, non null
   * @return {@link TimeControl} correspondant
   * @throws IllegalArgumentException si format invalide
   */
  public static TimeControl parse(String spec) {
    if (spec == null) {
      throw new IllegalArgumentException("spec must not be null");
    }
    String trimmed = spec.trim().toLowerCase();
    if (trimmed.equals("inf") || trimmed.equals("infinite")) {
      return infinite();
    }
    Matcher m = PATTERN.matcher(trimmed);
    if (!m.matches()) {
      throw new IllegalArgumentException(
          "invalid time control format: '"
              + spec
              + "' (expected 'tc', 'tc+inc', 'x/tc', 'x/tc+inc', or 'inf')");
    }
    String movesStr = m.group("moves");
    String baseStr = m.group("base");
    String incStr = m.group("inc");

    int moves = movesStr != null ? Integer.parseInt(movesStr) : 0;
    Duration base = toDuration(Double.parseDouble(baseStr));
    Duration inc = incStr != null ? toDuration(Double.parseDouble(incStr)) : Duration.ZERO;
    if (moves < 0) {
      throw new IllegalArgumentException("moves per control must be ≥ 0, got " + moves);
    }
    return new TimeControl(base, inc, moves, false);
  }

  /** Sérialise au format canonique. */
  public String toSpec() {
    if (unlimited) {
      return "inf";
    }
    StringBuilder sb = new StringBuilder();
    if (movesPerControl > 0) {
      sb.append(movesPerControl).append('/');
    }
    sb.append(formatSeconds(baseTime));
    if (!increment.isZero()) {
      sb.append('+').append(formatSeconds(increment));
    }
    return sb.toString();
  }

  private static Duration toDuration(double seconds) {
    long millis = Math.round(seconds * 1000.0);
    return Duration.ofMillis(millis);
  }

  private static String formatSeconds(Duration d) {
    double s = d.toMillis() / 1000.0;
    if (s == Math.floor(s)) {
      return String.valueOf((long) s);
    }
    return String.valueOf(s);
  }

  private static void requireNonNegative(Duration d, String name) {
    if (d == null) {
      throw new IllegalArgumentException(name + " must not be null");
    }
    if (d.isNegative()) {
      throw new IllegalArgumentException(name + " must be non-negative, got " + d);
    }
  }
}
