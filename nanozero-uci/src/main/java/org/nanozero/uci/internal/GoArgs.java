package org.nanozero.uci.internal;

import java.util.List;
import java.util.Objects;
import java.util.OptionalInt;
import java.util.OptionalLong;

/**
 * Arguments parsés de la commande UCI {@code go} (cf. SPEC §3.1, §12 phase 1).
 *
 * <p>Tous les champs {@code Optional*} peuvent être vides. Le caller (typiquement {@code
 * TimeManagementPolicy} en phase 5) interprète selon le côté au trait et les options présentes.
 * {@code searchMoves} est généralement vide en pratique : peu de GUIs UCI émettent ce paramètre.
 *
 * <p><strong>Invariants</strong> :
 *
 * <ul>
 *   <li>Tous les {@code Optional*} sont non-null (validé au compact constructor).
 *   <li>{@code searchMoves} est non-null ; le compact constructor en fait une copie immutable via
 *       {@link List#copyOf} pour empêcher toute mutation post-construction.
 * </ul>
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-uci} module.
 */
public record GoArgs(
    OptionalLong wtimeMs,
    OptionalLong btimeMs,
    OptionalLong wincMs,
    OptionalLong bincMs,
    OptionalInt movestogo,
    OptionalLong movetimeMs,
    OptionalInt nodes,
    OptionalInt depth,
    boolean infinite,
    boolean ponder,
    List<int[]> searchMoves) {

  /**
   * Compact constructor : valide non-nullité et copie défensive {@code searchMoves}.
   *
   * @throws NullPointerException si l'un des {@code Optional*} ou {@code searchMoves} est {@code
   *     null}
   */
  public GoArgs {
    Objects.requireNonNull(wtimeMs, "wtimeMs must not be null");
    Objects.requireNonNull(btimeMs, "btimeMs must not be null");
    Objects.requireNonNull(wincMs, "wincMs must not be null");
    Objects.requireNonNull(bincMs, "bincMs must not be null");
    Objects.requireNonNull(movestogo, "movestogo must not be null");
    Objects.requireNonNull(movetimeMs, "movetimeMs must not be null");
    Objects.requireNonNull(nodes, "nodes must not be null");
    Objects.requireNonNull(depth, "depth must not be null");
    Objects.requireNonNull(searchMoves, "searchMoves must not be null (use List.of() if empty)");
    searchMoves = List.copyOf(searchMoves);
  }

  /**
   * Factory pour des arguments {@code go} entièrement vides : tous les {@code Optional*} en {@link
   * OptionalInt#empty()} / {@link OptionalLong#empty()}, {@code infinite} et {@code ponder} à
   * {@code false}, {@code searchMoves} vide. Utilisé en phase 3 par le parser comme point de départ
   * avant remplissage incrémental, et en tests phase 1 pour vérifier les defaults.
   */
  public static GoArgs empty() {
    return new GoArgs(
        OptionalLong.empty(),
        OptionalLong.empty(),
        OptionalLong.empty(),
        OptionalLong.empty(),
        OptionalInt.empty(),
        OptionalLong.empty(),
        OptionalInt.empty(),
        OptionalInt.empty(),
        false,
        false,
        List.of());
  }
}
