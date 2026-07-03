package org.nanozero.nn;

import org.nanozero.board.BitboardPlaneEncoder;
import org.nanozero.board.GameState;

/**
 * Interface NN AlphaZero polymorphe (Phase v1.1.0+ ONNX backend support).
 *
 * <p>Existait avant comme {@code final class Network}. Refactor en interface pour permettre 2
 * implémentations interchangeables :
 *
 * <ul>
 *   <li>{@link NetworkVectorApi} : current Vector API SIMD CPU (legacy v1.0.0, BN-folded {@code
 *       .npz} format).
 *   <li>{@link NetworkOnnx} : ONNX Runtime Java avec CPU EP ou CUDA EP (Phase 12 PoC, ~17× speedup
 *       CPU, format {@code .onnx} BN-folded).
 * </ul>
 *
 * <p>{@link NetworkLoader#load(java.nio.file.Path)} dispatche sur l'extension du fichier ({@code
 * .npz} → VectorApi ; {@code .onnx} → Onnx).
 *
 * <p>Le FQN {@code org.nanozero.nn.Network} est préservé : downstream callers (nanozero-engine,
 * nanozero-uci) continuent d'importer/utiliser {@code Network} sans modification.
 */
public interface Network {

  /** Capacité maximale de batch (cf. SPEC §3.2). */
  int MAX_BATCH = 64;

  /** Capacité minimale de batch. */
  int MIN_BATCH = 1;

  /**
   * Retourne la capacité maximale de batch de CETTE instance (amendement v1.6.0, ADR-013-nn).
   *
   * <p>Par défaut {@link #MAX_BATCH} (contrat historique : SIMD Vector API, ONNX CPU EP). Les
   * implémentations à chemin batché réel (ONNX CUDA EP par paliers) PEUVENT exposer une capacité
   * supérieure. Les callers génériques DOIVENT dimensionner {@code planes} et {@link NNOutput} sur
   * cette valeur, pas sur la constante.
   *
   * @return capacité maximale de batch acceptée par {@link #forward}
   */
  default int maxBatch() {
    return MAX_BATCH;
  }

  /**
   * Exécute le forward pass sur un batch de positions.
   *
   * @param planes buffer de planes, longueur exacte {@code maxBatch() × 119 × 64}
   * @param batchSize positions effectives à inférer ({@code 1..maxBatch()})
   * @param output buffer pré-alloué rempli en place (capacité {@code >= maxBatch()})
   */
  void forward(float[] planes, int batchSize, NNOutput output);

  /**
   * Commodité pour inférer une seule position. Allouante par appel.
   *
   * @param state position à évaluer
   * @return {@link NNSingleResult} avec logits + value
   */
  NNSingleResult forwardSingle(GameState state);

  /** Retourne les métadonnées du modèle chargé. */
  NetworkMetadata metadata();

  /** Retourne l'encoder Vector API. */
  BitboardPlaneEncoder planeEncoder();
}
