package org.nanozero.uci.internal;

import java.util.Objects;

/**
 * Déclaration d'une option UCI (cf. SPEC §3.3, §6.1, ADR-003, §12 phase 1). Sealed interface avec 3
 * implémentations record correspondant aux types UCI standards utilisés en v1.0.0 :
 *
 * <ul>
 *   <li>{@link Check} — booléen on/off (ex. {@code Ponder}).
 *   <li>{@link Spin} — entier dans une plage {@code [min, max]} avec valeur par défaut (ex. {@code
 *       Move Overhead}).
 *   <li>{@link String_} — chaîne de caractères avec valeur par défaut.
 * </ul>
 *
 * <p>Les types UCI {@code combo} et {@code button} ne sont pas supportés en v1.0.0 (cf. ADR-003).
 * Le suffix underscore sur {@link String_} évite la collision avec {@link java.lang.String}.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-uci} module.
 */
public sealed interface UciOption permits UciOption.Check, UciOption.Spin, UciOption.String_ {

  /** Nom de l'option UCI tel qu'émis dans la déclaration et reconnu dans {@code setoption}. */
  String name();

  /**
   * Option UCI de type {@code check} (booléen on/off).
   *
   * @param name nom de l'option, non null, non blank
   * @param defaultValue valeur initiale au boot
   */
  record Check(String name, boolean defaultValue) implements UciOption {

    /**
     * @throws NullPointerException si {@code name} est null
     * @throws IllegalArgumentException si {@code name} est blank
     */
    public Check {
      Objects.requireNonNull(name, "name must not be null");
      if (name.isBlank()) {
        throw new IllegalArgumentException("name must not be blank");
      }
    }
  }

  /**
   * Option UCI de type {@code spin} (entier borné). Le compact constructor valide {@code min ≤ max}
   * et {@code defaultValue ∈ [min, max]} pour garantir la cohérence dès la construction (un GUI UCI
   * mal configuré ne peut pas produire une option dégénérée silencieuse).
   *
   * @param name nom de l'option, non null, non blank
   * @param defaultValue valeur initiale au boot, doit être dans {@code [min, max]}
   * @param min borne inférieure (inclusive)
   * @param max borne supérieure (inclusive), doit être {@code ≥ min}
   */
  record Spin(String name, int defaultValue, int min, int max) implements UciOption {

    /**
     * @throws NullPointerException si {@code name} est null
     * @throws IllegalArgumentException si {@code name} est blank, si {@code min > max}, ou si
     *     {@code defaultValue} hors de {@code [min, max]}
     */
    public Spin {
      Objects.requireNonNull(name, "name must not be null");
      if (name.isBlank()) {
        throw new IllegalArgumentException("name must not be blank");
      }
      if (min > max) {
        throw new IllegalArgumentException("min must be <= max, got min=" + min + ", max=" + max);
      }
      if (defaultValue < min || defaultValue > max) {
        throw new IllegalArgumentException(
            "defaultValue must be in [min, max], got defaultValue="
                + defaultValue
                + ", min="
                + min
                + ", max="
                + max);
      }
    }
  }

  /**
   * Option UCI de type {@code string} (chaîne libre).
   *
   * <p>Le suffixe underscore évite la collision avec {@link java.lang.String}. {@code defaultValue}
   * peut être la chaîne vide {@code ""} (légitime UCI : le GUI peut afficher un champ vide à
   * initialisation), mais pas {@code null}.
   *
   * @param name nom de l'option, non null, non blank
   * @param defaultValue valeur initiale au boot, non null (la chaîne vide {@code ""} est acceptée)
   */
  record String_(String name, String defaultValue) implements UciOption {

    /**
     * @throws NullPointerException si {@code name} ou {@code defaultValue} est null
     * @throws IllegalArgumentException si {@code name} est blank
     */
    public String_ {
      Objects.requireNonNull(name, "name must not be null");
      if (name.isBlank()) {
        throw new IllegalArgumentException("name must not be blank");
      }
      Objects.requireNonNull(defaultValue, "defaultValue must not be null");
    }
  }
}
