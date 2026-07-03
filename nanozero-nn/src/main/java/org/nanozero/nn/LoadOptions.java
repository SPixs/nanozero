package org.nanozero.nn;

/**
 * Options de chargement transmises à {@link NetworkLoader#load(java.nio.file.Path, LoadOptions)}
 * (cf. SPEC §4.2.2, §6.6 étape 8).
 *
 * <p>Le contrôle d'intégrité par recalcul SHA-256 sur les poids concaténés (cf. §6.5) est activé
 * <strong>par défaut</strong>. Coût indicatif : ~100 ms pour 1.4 M paramètres au boot, négligeable
 * pour un usage typique. {@link #skipHashCheck()} désactive cette vérification ; à n'utiliser que
 * pour les tests massifs chargeant de nombreux modèles ou pour mesurer une baseline de boot.
 *
 * <p>L'ordre des autres validations §6.6 (header {@code _meta_*}, présence + shape + dtype des 42
 * tenseurs, sanity values) n'est pas affecté par {@link LoadOptions} et reste obligatoire.
 *
 * @param verifyHash si {@code true}, recalcul SHA-256 sur les 42 tenseurs de poids triés
 *     alphabétiquement (excluant les {@code _meta_*}) et comparaison avec {@code _meta_model_hash}.
 *     Tout mismatch lève {@link IllegalArgumentException}.
 */
public record LoadOptions(boolean verifyHash) {

  /** Options par défaut : hash check activé (cf. §6.6 étape 8). */
  public static LoadOptions defaults() {
    return new LoadOptions(true);
  }

  /**
   * Options désactivant le hash check. Réduit le boot d'environ 100 ms ; à n'utiliser qu'en
   * connaissance de cause (tests massifs, baselines de perf).
   */
  public static LoadOptions skipHashCheck() {
    return new LoadOptions(false);
  }
}
