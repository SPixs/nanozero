/**
 * Implémentation interne du moteur de recherche MCTS (cf. SPEC §11.2).
 *
 * <p>Classes utilisées uniquement à l'intérieur du module {@code nanozero-engine}. Toute classe
 * dans ce package porte une note Javadoc {@code @apiNote Internal — do not depend on this from
 * outside the nanozero-engine module}. Techniquement publiques en l'absence de {@code
 * module-info.java}, mais doivent être traitées comme privées par les callers externes.
 *
 * <p>Cf. convention identique pour les sub-packages {@code internal/} et {@code kernels/} de {@code
 * nanozero-nn}.
 */
package org.nanozero.engine.internal;
