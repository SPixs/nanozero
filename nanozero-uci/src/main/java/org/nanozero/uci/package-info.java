/**
 * Module {@code nanozero-uci} : adaptateur protocole UCI au-dessus de {@code nanozero-engine}.
 *
 * <p>Seule la classe {@link org.nanozero.uci.UciMain} est exposée comme API publique du module
 * (point d'entrée exécutable, {@code main(String[])}). Toutes les autres classes du module sont
 * dans le sub-package {@code org.nanozero.uci.internal} et marquées {@code @apiNote Internal — do
 * not depend on this from outside the nanozero-uci module}.
 *
 * <p>Cf. {@code docs/SPEC-uci.md} §11 pour les conventions de visibilité et §1 pour le périmètre
 * fonctionnel (subset UCI v1 : {@code uci}, {@code isready}, {@code ucinewgame}, {@code position},
 * {@code go}, {@code stop}, {@code ponderhit}, {@code quit}, {@code setoption}, {@code debug}).
 *
 * @since 1.0.0
 */
package org.nanozero.uci;
