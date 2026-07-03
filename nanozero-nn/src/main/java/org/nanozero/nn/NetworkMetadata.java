package org.nanozero.nn;

/**
 * Métadonnées d'un réseau chargé (cf. SPEC §4.2.7). Hors hot path : utilisé pour logging et
 * vérification de version au boot. Renseigné par {@code NetworkLoader.load} en phase 8 depuis le
 * header {@code _meta_*} du fichier {@code .npz} ; en phase 7, des valeurs factices sont passées
 * par les constructeurs de test.
 *
 * @param architectureVersion ex. {@code "resnet8x96-v1"} (figé, cf. ADR-009)
 * @param modelHash hash SHA-256 hex des poids concaténés
 * @param trainingStep nombre d'étapes d'entraînement
 * @param exportDate ISO-8601 UTC, ex. {@code "2026-05-15T14:30:00Z"}
 * @param inputPlaneFormat ex. {@code "alphazero-119"} (figé)
 */
public record NetworkMetadata(
    String architectureVersion,
    String modelHash,
    long trainingStep,
    String exportDate,
    String inputPlaneFormat) {}
