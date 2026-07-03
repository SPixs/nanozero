package org.nanozero.uci.internal;

import java.util.Arrays;
import java.util.Objects;
import java.util.Optional;
import java.util.OptionalInt;
import java.util.OptionalLong;

/**
 * Champs agrégés pour une ligne UCI {@code info ...} (cf. SPEC §3.2, §5.6, §12 phase 1).
 *
 * <p>Tout ou partie des champs peut être renseigné. La ligne {@code info} émise par {@code
 * UciResponseWriter} (phase 4) ne mentionne que les champs présents (Optional non vide ou {@code
 * pv} non vide).
 *
 * <p><strong>Invariants normatifs</strong> (cf. SPEC §3.2 I-Rsp-2) : {@link InfoFields} peut avoir
 * tous ses champs Optional vides ; dans ce cas la ligne {@code info} émise n'émet que les champs
 * présents. {@code multipv} est toujours 1 en v1.0.0 (pas de MultiPV, cf. SPEC §5.6).
 *
 * <p><strong>Equality limitation</strong> : par défaut Java records, {@code equals} / {@code
 * hashCode} utilisent reference identity sur l'array {@code pv}. Acceptable pour ce data carrier
 * (les info lines sont des snapshots transitoires, pas des clés de map).
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-uci} module.
 */
public record InfoFields(
    OptionalInt depth,
    OptionalInt seldepth,
    OptionalInt nodes,
    OptionalInt nps,
    OptionalLong timeMs,
    OptionalInt scoreCp,
    OptionalInt scoreMate,
    int[] pv,
    OptionalInt multipv,
    Optional<String> string) {

  /**
   * Compact constructor : valide non-nullité (chaque {@code Optional*} est obligatoire ; {@code
   * Optional.empty()} acceptable, {@code null} non) et copie défensive de {@code pv}.
   *
   * @throws NullPointerException si l'un des {@code Optional*} ou {@code pv} est null
   */
  public InfoFields {
    Objects.requireNonNull(depth, "depth must not be null");
    Objects.requireNonNull(seldepth, "seldepth must not be null");
    Objects.requireNonNull(nodes, "nodes must not be null");
    Objects.requireNonNull(nps, "nps must not be null");
    Objects.requireNonNull(timeMs, "timeMs must not be null");
    Objects.requireNonNull(scoreCp, "scoreCp must not be null");
    Objects.requireNonNull(scoreMate, "scoreMate must not be null");
    Objects.requireNonNull(pv, "pv must not be null (use new int[0] if empty)");
    Objects.requireNonNull(multipv, "multipv must not be null");
    Objects.requireNonNull(string, "string must not be null");
    pv = Arrays.copyOf(pv, pv.length);
  }

  /**
   * Factory pour des {@code InfoFields} entièrement vides : tous les {@code Optional*} en {@code
   * empty()}, {@code pv} vide. Point de départ typique en phase 5 ({@code InfoReporter}) avant
   * remplissage incrémental.
   */
  public static InfoFields empty() {
    return new InfoFields(
        OptionalInt.empty(),
        OptionalInt.empty(),
        OptionalInt.empty(),
        OptionalInt.empty(),
        OptionalLong.empty(),
        OptionalInt.empty(),
        OptionalInt.empty(),
        new int[0],
        OptionalInt.empty(),
        Optional.empty());
  }
}
