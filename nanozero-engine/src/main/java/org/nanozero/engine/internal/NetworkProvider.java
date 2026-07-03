package org.nanozero.engine.internal;

import org.nanozero.nn.NNOutput;

/**
 * Contrat minimaliste pour l'évaluation NN consommée par {@link LeafEvaluator}. Permet de mocker
 * {@code Network} dans les tests sans avoir à charger un modèle {@code .npz}.
 *
 * <p>La signature est volontairement réduite par rapport à {@code org.nanozero.nn.Network#forward}
 * (batch fixe à 1, conforme ADR-011) : {@code LeafEvaluator} opère toujours sur une seule position,
 * donc l'interface fige cette dimension.
 *
 * <p>Une implémentation par défaut wrappant {@code Network} sera fournie en phase 7 ({@code
 * NetworkProviderImpl}). En phase 4, les tests injectent un mock direct.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
@FunctionalInterface
interface NetworkProvider {

  /**
   * Forward d'une seule position (batch=1 conforme ADR-011). L'implémentation remplit les buffers
   * internes de {@code output} ({@code logits[0..POLICY_INDICES)} et {@code values[0]}).
   *
   * @param planes buffer de {@code 119 × 64 = 7616} floats au format NCHW d'une seule position
   * @param output conteneur pré-alloué dont les buffers internes seront remplis
   */
  void forward(float[] planes, NNOutput output);
}
