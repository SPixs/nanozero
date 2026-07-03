package org.nanozero.uci.internal;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.OptionalInt;
import java.util.OptionalLong;
import java.util.Set;
import org.nanozero.board.GameState;

/**
 * Parser des lignes UCI reçues sur stdin (cf. SPEC §5.2, §14 grammaire, §12 phase 3).
 *
 * <p>API minimaliste : une méthode statique {@link #parse} qui transforme une ligne brute en {@link
 * UciCommand} typé. Parsing pur : pas d'appel à {@code Engine}, pas de side-effects, pas
 * d'allocation autre que les records et l'array de tokens.
 *
 * <p><strong>Tolérance UCI</strong> : conformément au protocole, une ligne malformée ou une
 * commande inconnue retourne {@link UciCommand.Unknown}, pas une exception. Le caller (handler
 * phase 7) peut logger en debug et continuer la boucle UCI. Toute exception lancée par les
 * sous-parsers (parseLong sur token non numérique, FEN invalide, coup illégal, etc.) est attrapée
 * par le {@code try/catch} global et convertie en {@code Unknown(rawLine)}.
 *
 * <p><strong>Conventions de casse</strong> :
 *
 * <ul>
 *   <li>Casse-insensitive sur le nom de commande principal : {@code uci}, {@code Uci}, {@code UCI}
 *       sont tous valides.
 *   <li>Strict (case-sensitive) sur les sub-keywords : {@code startpos}, {@code fen}, {@code
 *       moves}, {@code wtime}, etc. doivent être minuscules.
 *   <li>Le caractère de promotion UCI tolère la casse (cf. {@link UciMoveCodec#decode}).
 * </ul>
 *
 * <p><strong>searchmoves v1.0.0</strong> : ignoré silencieusement (cf. {@code GoArgs.empty()}). Les
 * tokens entre {@code searchmoves} et le prochain mot-clé {@code go} sont skippés sans conversion.
 * Cas d'usage rare en pratique. Reconsidération v1.1.0+ si cas concret.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-uci} module.
 */
public final class UciCommandParser {

  /** Mots-clés UCI qui suivent {@code go} ; utilisé pour borner la liste {@code searchmoves}. */
  private static final Set<String> GO_KEYWORDS =
      Set.of(
          "wtime",
          "btime",
          "winc",
          "binc",
          "movestogo",
          "movetime",
          "nodes",
          "depth",
          "infinite",
          "ponder",
          "searchmoves");

  private UciCommandParser() {
    throw new AssertionError("Non-instantiable");
  }

  /**
   * Parse une ligne UCI brute (sans le LF terminal) en {@link UciCommand} typé.
   *
   * <p>Sémantique tolérante : ligne {@code null}, vide, blank, commande inconnue, ou parsing
   * détaillé qui échoue → {@link UciCommand.Unknown} avec le {@code rawLine} (vide pour {@code
   * null}). Aucune exception propagée au caller.
   *
   * @param line ligne UCI reçue, peut être {@code null}
   * @return {@link UciCommand} jamais null
   */
  public static UciCommand parse(String line) {
    if (line == null) {
      return new UciCommand.Unknown("");
    }
    String trimmed = line.trim();
    if (trimmed.isEmpty()) {
      return new UciCommand.Unknown(line);
    }
    String[] tokens = trimmed.split("\\s+");
    if (tokens.length == 0) {
      return new UciCommand.Unknown(line);
    }
    String cmd = tokens[0].toLowerCase(Locale.ROOT);
    try {
      return switch (cmd) {
        case "uci" -> new UciCommand.Uci();
        case "isready" -> new UciCommand.IsReady();
        case "ucinewgame" -> new UciCommand.UciNewGame();
        case "stop" -> new UciCommand.Stop();
        case "ponderhit" -> new UciCommand.PonderHit();
        case "quit" -> new UciCommand.Quit();
        case "position" -> parsePosition(tokens);
        case "go" -> parseGo(tokens);
        case "setoption" -> parseSetOption(tokens);
        case "debug" -> parseDebug(tokens);
        default -> new UciCommand.Unknown(line);
      };
    } catch (RuntimeException e) {
      // Filet de sécurité ultime : toute erreur de parsing détaillé (FEN invalide, coup
      // illégal, parseLong échoue, IndexOutOfBounds sur tokens manquants...) → Unknown.
      return new UciCommand.Unknown(line);
    }
  }

  // -------------------------------------------------------------------------------------------
  // position [startpos | fen <6 tokens>] [moves m1 m2 ...]
  // -------------------------------------------------------------------------------------------

  private static UciCommand.Position parsePosition(String[] tokens) {
    if (tokens.length < 2) {
      throw new IllegalArgumentException("position: missing startpos|fen");
    }
    int idx = 1;
    GameState state;
    if ("startpos".equals(tokens[idx])) {
      state = new GameState();
      idx++;
    } else if ("fen".equals(tokens[idx])) {
      // FEN = 6 champs séparés par espace (placement, side, castling, ep, halfmove, fullmove).
      if (idx + 6 >= tokens.length) {
        throw new IllegalArgumentException("position fen: expected 6 fields");
      }
      String fen = String.join(" ", Arrays.copyOfRange(tokens, idx + 1, idx + 7));
      state = new GameState(fen);
      idx += 7;
    } else {
      throw new IllegalArgumentException("position: expected startpos or fen, got " + tokens[idx]);
    }

    int[] playedMoves;
    if (idx < tokens.length && "moves".equals(tokens[idx])) {
      idx++;
      int n = tokens.length - idx;
      playedMoves = new int[n];
      for (int i = 0; i < n; i++) {
        int move = UciMoveCodec.decode(tokens[idx + i], state);
        playedMoves[i] = move;
        state.applyMove(move);
      }
    } else {
      playedMoves = new int[0];
    }

    return new UciCommand.Position(state, playedMoves);
  }

  // -------------------------------------------------------------------------------------------
  // go [wtime|btime|winc|binc|movestogo|movetime|nodes|depth|infinite|ponder|searchmoves ...]
  // -------------------------------------------------------------------------------------------

  private static UciCommand.Go parseGo(String[] tokens) {
    OptionalLong wtime = OptionalLong.empty();
    OptionalLong btime = OptionalLong.empty();
    OptionalLong winc = OptionalLong.empty();
    OptionalLong binc = OptionalLong.empty();
    OptionalInt movestogo = OptionalInt.empty();
    OptionalLong movetime = OptionalLong.empty();
    OptionalInt nodes = OptionalInt.empty();
    OptionalInt depth = OptionalInt.empty();
    boolean infinite = false;
    boolean ponder = false;
    List<int[]> searchMoves = new ArrayList<>();

    int idx = 1;
    while (idx < tokens.length) {
      String tok = tokens[idx];
      switch (tok) {
        case "wtime" -> {
          wtime = OptionalLong.of(Long.parseLong(tokens[idx + 1]));
          idx += 2;
        }
        case "btime" -> {
          btime = OptionalLong.of(Long.parseLong(tokens[idx + 1]));
          idx += 2;
        }
        case "winc" -> {
          winc = OptionalLong.of(Long.parseLong(tokens[idx + 1]));
          idx += 2;
        }
        case "binc" -> {
          binc = OptionalLong.of(Long.parseLong(tokens[idx + 1]));
          idx += 2;
        }
        case "movestogo" -> {
          movestogo = OptionalInt.of(Integer.parseInt(tokens[idx + 1]));
          idx += 2;
        }
        case "movetime" -> {
          movetime = OptionalLong.of(Long.parseLong(tokens[idx + 1]));
          idx += 2;
        }
        case "nodes" -> {
          nodes = OptionalInt.of(Integer.parseInt(tokens[idx + 1]));
          idx += 2;
        }
        case "depth" -> {
          depth = OptionalInt.of(Integer.parseInt(tokens[idx + 1]));
          idx += 2;
        }
        case "infinite" -> {
          infinite = true;
          idx++;
        }
        case "ponder" -> {
          ponder = true;
          idx++;
        }
        case "searchmoves" -> {
          // v1.0.0 : ignoré silencieusement (cf. AGENTS.md uci). Skip jusqu'au prochain
          // keyword go ou fin de ligne.
          idx++;
          while (idx < tokens.length && !GO_KEYWORDS.contains(tokens[idx])) {
            idx++;
          }
        }
        default -> idx++; // token inconnu, skip silently (tolérance UCI)
      }
    }

    GoArgs args =
        new GoArgs(
            wtime,
            btime,
            winc,
            binc,
            movestogo,
            movetime,
            nodes,
            depth,
            infinite,
            ponder,
            searchMoves);
    return new UciCommand.Go(args);
  }

  // -------------------------------------------------------------------------------------------
  // setoption name <X> [value <Y>]
  // -------------------------------------------------------------------------------------------

  private static UciCommand.SetOption parseSetOption(String[] tokens) {
    if (tokens.length < 3 || !"name".equals(tokens[1])) {
      throw new IllegalArgumentException("setoption: expected 'name <X>'");
    }
    // Localiser le mot-clé littéral "value" qui sépare le nom de la valeur (les deux peuvent
    // contenir des espaces).
    int valueIdx = -1;
    for (int i = 2; i < tokens.length; i++) {
      if ("value".equals(tokens[i])) {
        valueIdx = i;
        break;
      }
    }
    String name;
    String value;
    if (valueIdx == -1) {
      name = String.join(" ", Arrays.copyOfRange(tokens, 2, tokens.length));
      value = null;
    } else {
      if (valueIdx == 2) {
        throw new IllegalArgumentException("setoption: empty name before 'value'");
      }
      name = String.join(" ", Arrays.copyOfRange(tokens, 2, valueIdx));
      value =
          (valueIdx + 1 < tokens.length)
              ? String.join(" ", Arrays.copyOfRange(tokens, valueIdx + 1, tokens.length))
              : "";
    }
    return new UciCommand.SetOption(name, value);
  }

  // -------------------------------------------------------------------------------------------
  // debug [on | off]
  // -------------------------------------------------------------------------------------------

  private static UciCommand.Debug parseDebug(String[] tokens) {
    if (tokens.length < 2) {
      throw new IllegalArgumentException("debug: missing on|off");
    }
    String arg = tokens[1].toLowerCase(Locale.ROOT);
    return switch (arg) {
      case "on" -> new UciCommand.Debug(true);
      case "off" -> new UciCommand.Debug(false);
      default -> throw new IllegalArgumentException("debug: expected on|off, got " + tokens[1]);
    };
  }
}
