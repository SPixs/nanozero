package org.nanozero.engine.internal;

import java.util.Objects;

/**
 * Propagation alternée d'une valeur depuis une feuille jusqu'à la racine (cf. SPEC §5.4).
 *
 * <p>À chaque nœud du chemin, {@code totalVisits += 1} et {@code totalValueSum += value}. La {@code
 * value} est <strong>inversée à chaque niveau</strong> parce que les échecs sont un jeu zero-sum :
 * ce qui est bon pour le joueur au trait à un nœud est mauvais pour le parent (qui jouait depuis
 * l'autre côté).
 *
 * <p>Algorithme strict §5.4 :
 *
 * <pre>
 *     current ← leaf
 *     value ← leafValue
 *     TANT QUE current != null :
 *         current.totalVisits += 1
 *         current.totalValueSum += value
 *         value ← -value
 *         current ← current.parent
 * </pre>
 *
 * <p>L'inversion est faite <strong>après</strong> l'addition au nœud courant (pas avant) — c'est le
 * piège classique. La feuille reçoit donc {@code leafValue} sans inversion, son parent reçoit
 * {@code -leafValue}, le grand-parent {@code +leafValue}, etc.
 *
 * <p><strong>Validation de la value</strong> : {@code leafValue} n'est pas validé ; une valeur
 * pathologique (NaN, ±Infinity, hors {@code [-1, +1]}) sera propagée telle quelle. C'est la
 * responsabilité du caller (typiquement {@code LeafEvaluator}, dont {@code Network} garantit {@code
 * value ∈ [-1, +1]}). Pas de protection défensive.
 *
 * <p><strong>Invariant maintenu</strong> : {@code totalVisits} reste cohérent avec I-Node-4. Pour
 * une simulation MCTS donnée, exactement un chemin de la racine à la feuille est mis à jour ; les
 * sous-arbres non parcourus restent intacts.
 *
 * <p><strong>Concurrence</strong> (mise à jour v1.2.0/v1.3.0) : le backup multi-thread est
 * <strong>implémenté</strong> via les méthodes atomic de {@link Node} ({@code
 * totalValueSum.atomicAdd} puis {@code totalVisits.incrementAndGet} — ordre <strong>W puis
 * N</strong>, cf. ADR-014 amendé). Lock-free, sans synchronisation explicite. En mono-thread, la
 * sémantique v1.1.2 bit-pour-bit est préservée. La Javadoc « aucune synchronisation, mono-thread
 * v1.0.0 » d'origine est obsolète.
 *
 * <p>Classe utilitaire ({@code final}, constructeur privé).
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
final class Backup {

  private Backup() {
    throw new AssertionError("Utility class — do not instantiate");
  }

  /**
   * Propage {@code leafValue} depuis {@code leaf} jusqu'à la racine en remontant la chaîne {@code
   * parent}. Mise à jour {@code totalVisits} et {@code totalValueSum} à chaque nœud, avec inversion
   * alternée du signe.
   *
   * <p>Allocation : aucune. Boucle simple sans varargs ni autoboxing.
   *
   * @param leaf nœud de départ (feuille évaluée), non null
   * @param leafValue value du point de vue du joueur au trait à {@code leaf}
   * @throws NullPointerException si {@code leaf} est null
   */
  static void backup(Node leaf, float leafValue) {
    Objects.requireNonNull(leaf, "leaf must not be null");
    Node current = leaf;
    float value = leafValue;
    while (current != null) {
      // (v1.2.0, ADR-014, v1.3.0 fix P1.2) Ordre W PUIS N — corrige le drift ADR-014 vs code.
      // L'ADR original §82 prétendait que "ordre writer N puis W garantit que lecteur ne voit
      // jamais N stale + W frais". Faux : si writer fait N puis W, le lecteur (PUCT, buildSnapshot)
      // qui lit N puis W peut voir (N_old, W_new) → Q = W/N peut excéder [-1,1] sous contention.
      // En inversant (W puis N), le lecteur voit soit (N_old, W_old) soit (N_old, W_new_partiel
      // → bounded car W = somme bornée des deltas ∈ [-1, 1]). Le pire cas devient
      // "W stale d'1 tick avec N frais" → Q sous-estimé d'au plus 1/(N+1), borné dans [-1, 1].
      // AtomicFloat.atomicAdd = CAS-loop sur les bits IEEE-754, lock-free.
      // AtomicInteger.incrementAndGet = LOCK XADD intrinsic, lock-free.
      current.totalValueSum.atomicAdd(value);
      current.totalVisits.incrementAndGet();
      value = -value;
      current = current.parent;
    }
  }
}
