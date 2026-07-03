package org.nanozero.uci.internal;

import java.util.Objects;
import java.util.OptionalInt;

/**
 * Réponse UCI à émettre sur stdout (cf. SPEC §3.2, §12 phase 1). Sealed interface avec 6
 * implémentations record.
 *
 * <p>Chaque réponse est convertible en une ligne UCI valide par le {@code UciResponseWriter} (phase
 * 4).
 *
 * <p><strong>Invariants normatifs</strong> (cf. SPEC §3.2) :
 *
 * <ul>
 *   <li><strong>I-Rsp-1</strong> : tout {@code UciResponse} est convertible en une ligne UCI
 *       valide. Le {@code UciResponseWriter} garantit le format conforme.
 *   <li><strong>I-Rsp-2</strong> : {@link Info} peut avoir tous ses champs {@link InfoFields}
 *       Optional vides ; dans ce cas la ligne {@code info} émise n'émet que les champs présents.
 *   <li><strong>I-Rsp-3</strong> : {@link BestMove#ponderMove()} est rempli uniquement si l'engine
 *       fournit un coup à pondérer (extrait de la PV). Sinon {@link OptionalInt#empty()}.
 * </ul>
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-uci} module.
 */
public sealed interface UciResponse
    permits UciResponse.Id,
        UciResponse.UciOk,
        UciResponse.Option,
        UciResponse.ReadyOk,
        UciResponse.Info,
        UciResponse.InfoString,
        UciResponse.BestMove {

  /**
   * Réponse UCI {@code id name <name>} et {@code id author <author>}, émises en réponse à {@code
   * uci}.
   *
   * @param name nom du moteur, non null, non blank
   * @param author auteur, non null, non blank
   */
  record Id(String name, String author) implements UciResponse {

    /**
     * @throws NullPointerException si {@code name} ou {@code author} est null
     * @throws IllegalArgumentException si {@code name} ou {@code author} est blank
     */
    public Id {
      Objects.requireNonNull(name, "name must not be null");
      Objects.requireNonNull(author, "author must not be null");
      if (name.isBlank()) {
        throw new IllegalArgumentException("name must not be blank");
      }
      if (author.isBlank()) {
        throw new IllegalArgumentException("author must not be blank");
      }
    }
  }

  /** Réponse UCI {@code uciok}, terminant la séquence de poignée de main initiale. */
  record UciOk() implements UciResponse {}

  /** Réponse UCI {@code readyok}, en réponse à {@code isready}. */
  record ReadyOk() implements UciResponse {}

  /**
   * Réponse UCI {@code option name <name> type <type> ...} déclarant une option supportée. Émise
   * pour chaque option de {@link UciOptionsState#declaredOptions()} après {@code uci}.
   *
   * @param option déclaration d'option, non null
   */
  record Option(UciOption option) implements UciResponse {

    /**
     * @throws NullPointerException si {@code option} est null
     */
    public Option {
      Objects.requireNonNull(option, "option must not be null");
    }
  }

  /**
   * Réponse UCI {@code info ...}, émission périodique pendant une recherche (cf. SPEC §5.6).
   *
   * @param fields champs renseignés, non null
   */
  record Info(InfoFields fields) implements UciResponse {

    /**
     * @throws NullPointerException si {@code fields} est null
     */
    public Info {
      Objects.requireNonNull(fields, "fields must not be null");
    }
  }

  /**
   * (v1.2.0) Réponse UCI {@code info string <text>}, émission non-standard tolérée par tous les
   * GUIs UCI standards (cf. SPEC §6.5 amendée v1.2.0, ADR-003 mise à jour v1.2.0 « hidden behaviors
   * pattern »).
   *
   * <p>Usage v1.2.0 : émission de la distribution des visites MCTS au root pour clients training
   * via le contenu {@code "visits <move1> <count1> <move2> <count2> ..."}. Les GUIs ignorent
   * silencieusement les {@code info string} qu'ils ne comprennent pas.
   *
   * @param text contenu du info string, après le préfixe {@code "info string "}. Non null, peut
   *     être vide (cas position terminale = ligne {@code "info string visits"} seule).
   */
  record InfoString(String text) implements UciResponse {

    /**
     * @throws NullPointerException si {@code text} est null
     */
    public InfoString {
      Objects.requireNonNull(text, "text must not be null");
    }
  }

  /**
   * Réponse UCI {@code bestmove <move> [ponder <ponderMove>]}, signal de fin de recherche.
   *
   * <p>L'encodage du Move 16-bit n'est pas validé ici (à la charge du {@code UciResponseWriter} en
   * phase 4). Ce record est un data carrier pur.
   *
   * @param move coup choisi, encodage Move 16-bit
   * @param ponderMove coup à pondérer extrait de la PV ({@code OptionalInt} non null, peut être
   *     {@link OptionalInt#empty()})
   */
  record BestMove(int move, OptionalInt ponderMove) implements UciResponse {

    /**
     * @throws NullPointerException si {@code ponderMove} est null
     */
    public BestMove {
      Objects.requireNonNull(ponderMove, "ponderMove must not be null");
    }
  }
}
