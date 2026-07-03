package org.nanozero.uci.internal;

import java.util.Arrays;
import java.util.Objects;
import org.nanozero.board.GameState;

/**
 * Commande UCI parsée depuis stdin (cf. SPEC §3.1, §12 phase 1). Sealed interface avec 11
 * implémentations record.
 *
 * <p>Chaque ligne UCI reçue sur stdin est parsée par le {@code UciCommandParser} (phase 3) en une
 * instance d'un type dérivé. Les lignes invalides (commande inconnue, syntaxe erronée) sont
 * représentées par {@link Unknown} plutôt que de lever une exception : le protocole UCI exige de
 * tolérer les commandes inconnues silencieusement (cf. SPEC §1.2).
 *
 * <p><strong>Invariants normatifs</strong> (cf. SPEC §3.1) :
 *
 * <ul>
 *   <li><strong>I-Cmd-1</strong> : {@code Position.position} correspond à l'état FINAL après
 *       application des {@code playedMoves}. Le caller (parser) applique les coups au moment du
 *       parsing et fournit la position finale.
 *   <li><strong>I-Cmd-2</strong> : {@code Position.playedMoves} est conservé pour information /
 *       debug, mais la position est déjà à jour.
 *   <li><strong>I-Cmd-3</strong> : {@link GoArgs} peut avoir des champs {@code Optional*} vides ;
 *       le caller (TimeManagementPolicy) interprète selon le côté au trait et les options
 *       présentes.
 * </ul>
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-uci} module.
 */
public sealed interface UciCommand
    permits UciCommand.Uci,
        UciCommand.IsReady,
        UciCommand.UciNewGame,
        UciCommand.Position,
        UciCommand.Go,
        UciCommand.Stop,
        UciCommand.PonderHit,
        UciCommand.Quit,
        UciCommand.SetOption,
        UciCommand.Debug,
        UciCommand.Unknown {

  /** Commande UCI {@code uci} : poignée de main initiale. */
  record Uci() implements UciCommand {}

  /** Commande UCI {@code isready} : ping de synchronisation. */
  record IsReady() implements UciCommand {}

  /** Commande UCI {@code ucinewgame} : signal de nouvelle partie (no-op v1.0.0, cf. ADR-004). */
  record UciNewGame() implements UciCommand {}

  /**
   * Commande UCI {@code position [startpos|fen ...] [moves ...]}.
   *
   * <p>{@code position} est l'état FINAL après application des {@code playedMoves} (cf. I-Cmd-1).
   * {@code playedMoves} est conservé pour information / debug.
   *
   * <p><strong>Equality limitation</strong> : par défaut Java {@code records}, {@code equals} /
   * {@code hashCode} utilisent {@code Object.equals(Object)} sur les champs, donc reference
   * identity sur l'array {@code playedMoves}. Acceptable pour ce data carrier — l'identité
   * sémantique d'une {@code Position} est portée par le hash Zobrist de sa {@code GameState}. Une
   * comparaison structurelle nécessiterait une override manuelle (différée si besoin).
   *
   * @param position état d'échecs final, non null
   * @param playedMoves coups joués depuis la position de départ ; copié défensivement, non null
   */
  record Position(GameState position, int[] playedMoves) implements UciCommand {

    /**
     * Compact constructor : valide non-nullité et copie défensive de {@code playedMoves}.
     *
     * @throws NullPointerException si {@code position} ou {@code playedMoves} est null
     */
    public Position {
      Objects.requireNonNull(position, "position must not be null");
      Objects.requireNonNull(playedMoves, "playedMoves must not be null");
      playedMoves = Arrays.copyOf(playedMoves, playedMoves.length);
    }
  }

  /**
   * Commande UCI {@code go [params...]}.
   *
   * @param args arguments parsés (tous Optional, cf. {@link GoArgs}), non null
   */
  record Go(GoArgs args) implements UciCommand {

    /**
     * @throws NullPointerException si {@code args} est null
     */
    public Go {
      Objects.requireNonNull(args, "args must not be null");
    }
  }

  /** Commande UCI {@code stop} : arrêt de la recherche en cours. */
  record Stop() implements UciCommand {}

  /** Commande UCI {@code ponderhit} : conversion ponder → search (cf. SPEC-engine §4.2). */
  record PonderHit() implements UciCommand {}

  /** Commande UCI {@code quit} : terminaison du processus. */
  record Quit() implements UciCommand {}

  /**
   * Commande UCI {@code setoption name <name> [value <value>]}.
   *
   * @param name nom de l'option (non null, non blank)
   * @param value valeur (peut être {@code null} pour les options de type Button sans valeur, et
   *     pour {@link UciOptionsState#set} qui tolère le {@code null})
   */
  record SetOption(String name, String value) implements UciCommand {

    /**
     * @throws NullPointerException si {@code name} est null
     * @throws IllegalArgumentException si {@code name} est blank
     */
    public SetOption {
      Objects.requireNonNull(name, "name must not be null");
      if (name.isBlank()) {
        throw new IllegalArgumentException("name must not be blank");
      }
    }
  }

  /**
   * Commande UCI {@code debug on|off}. Active ou désactive les logs de debug stderr (cf. ADR-007).
   */
  record Debug(boolean enabled) implements UciCommand {}

  /**
   * Commande UCI inconnue ou syntaxiquement invalide. Le parser (phase 3) émet ce record plutôt que
   * de lever une exception : le protocole UCI exige de tolérer les commandes inconnues
   * silencieusement (cf. SPEC §1.2 et AGENTS.md uci convention « UCI tolérance »).
   *
   * @param rawLine ligne brute reçue, non null (peut être vide)
   */
  record Unknown(String rawLine) implements UciCommand {

    /**
     * @throws NullPointerException si {@code rawLine} est null
     */
    public Unknown {
      Objects.requireNonNull(rawLine, "rawLine must not be null");
    }
  }
}
