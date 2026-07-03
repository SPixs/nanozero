package org.nanozero.nn;

/**
 * Résultat d'une inférence sur une position unique, retourné par {@link
 * Network#forwardSingle(org.nanozero.board.GameState)} (cf. SPEC §4.2.4).
 *
 * <p>Le tableau {@code logits} est une copie indépendante (allocation par appel), sécurisée pour
 * usage par le caller même si {@code forwardSingle} est appelé à nouveau sur le même thread.
 *
 * @param logits {@code float[POLICY_INDICES = 4672]}, format §3.5.1 ({@code logitIndex = fromSquare
 *     * 73 + planeIndex})
 * @param value scalaire dans {@code [-1, 1]} (post-tanh)
 */
public record NNSingleResult(float[] logits, float value) {

  /**
   * Vérifie que {@code logits} a la longueur attendue {@link MoveEncoding#POLICY_INDICES}.
   *
   * @throws IllegalArgumentException si {@code logits.length != POLICY_INDICES}
   */
  public NNSingleResult {
    if (logits.length != MoveEncoding.POLICY_INDICES) {
      throw new IllegalArgumentException(
          "logits length must be " + MoveEncoding.POLICY_INDICES + ", got " + logits.length);
    }
  }
}
