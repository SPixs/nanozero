package org.nanozero.engine;

/**
 * Résultat immutable d'une recherche, retourné par {@code Engine.stop()} ou snapshot via {@code
 * Engine.currentBest()} (cf. SPEC §3.3).
 *
 * <p>Invariants normatifs §3.3 :
 *
 * <ul>
 *   <li><strong>I-Result-1</strong> : {@code bestMove ∈ childMoves}, à l'index correspondant à
 *       {@code argmax childVisits}.
 *   <li><strong>I-Result-2</strong> : {@code principalVariation[0] == bestMove}. La PV est extraite
 *       par descente récursive « max visits » depuis chaque nœud.
 *   <li><strong>I-Result-3</strong> : {@code simulationsCount ≥ 0}. Vaut 0 si l'arrêt arrive avant
 *       la première simulation complète.
 *   <li><strong>I-Result-4</strong> : {@code value ∈ [-1.0f, +1.0f]}. Vaut {@code Float.NaN} si
 *       {@code simulationsCount == 0} (pas de Q calculé).
 * </ul>
 *
 * <p>Phase 1 : seul le squelette du record + sa validation arguments + {@link #valueAsCentipawns()}
 * sont fournis. La logique de remplissage (descente PV depuis {@code SearchTree}, agrégation des
 * visites enfants) est implémentée à partir de la phase 5.
 *
 * @param bestMove coup retenu (encodage {@code Move} 16-bit). Cohérent avec {@code
 *     childMoves[argmax(childVisits)]} dès lors que {@code simulationsCount > 0}.
 * @param principalVariation séquence de coups du meilleur chemin (longueur typique ~30 plies, peut
 *     être vide si {@code simulationsCount == 0}). Non null. {@code principalVariation[0] ==
 *     bestMove} quand non vide.
 * @param value Q estimée pour {@code bestMove} depuis la racine, dans {@code [-1, +1]}. Vaut {@code
 *     Float.NaN} ssi {@code simulationsCount == 0}.
 * @param simulationsCount nombre total de simulations terminées (≥ 0).
 * @param elapsedNanos durée écoulée depuis le début de la recherche, en nanosecondes (≥ 0).
 * @param childVisits distribution {@code N(root, a)} pour chaque coup légal racine. Non null. Même
 *     longueur que {@code childMoves}.
 * @param childMoves coups légaux racine, ordre identique à {@code childVisits}. Non null.
 * @param terminated {@code true} si la recherche a atteint son budget (arrêt naturel), {@code
 *     false} si stop précoce (UCI {@code stop}, {@code AtomicBoolean} flippé, etc.).
 */
public record SearchResult(
    int bestMove,
    int[] principalVariation,
    float value,
    int simulationsCount,
    long elapsedNanos,
    int[] childVisits,
    int[] childMoves,
    boolean terminated) {

  /**
   * Validation des arguments au compact constructor.
   *
   * @throws NullPointerException si {@code principalVariation}, {@code childVisits} ou {@code
   *     childMoves} sont null
   * @throws IllegalArgumentException si une autre contrainte est violée (longueurs incohérentes,
   *     {@code value} hors {@code [-1, +1]} et non-NaN, {@code simulationsCount < 0}, {@code
   *     elapsedNanos < 0})
   */
  public SearchResult {
    if (principalVariation == null) {
      throw new NullPointerException("principalVariation must not be null");
    }
    if (childVisits == null) {
      throw new NullPointerException("childVisits must not be null");
    }
    if (childMoves == null) {
      throw new NullPointerException("childMoves must not be null");
    }
    if (childVisits.length != childMoves.length) {
      throw new IllegalArgumentException(
          "childVisits.length ("
              + childVisits.length
              + ") must equal childMoves.length ("
              + childMoves.length
              + ")");
    }
    if (!Float.isNaN(value) && (value < -1.0f || value > 1.0f)) {
      throw new IllegalArgumentException("value must be in [-1, 1] or NaN, got " + value);
    }
    if (simulationsCount < 0) {
      throw new IllegalArgumentException("simulationsCount must be >= 0, got " + simulationsCount);
    }
    if (elapsedNanos < 0) {
      throw new IllegalArgumentException("elapsedNanos must be >= 0, got " + elapsedNanos);
    }
  }

  /**
   * Q convertie en centipawns approximatifs pour affichage UCI ({@code Math.round(value * 100)}).
   * Retourne 0 si {@code value} est {@link Float#NaN} (cas {@code simulationsCount == 0}), conforme
   * au contrat de {@link Math#round(float)}.
   *
   * @return entier dans {@code [-100, +100]} en pratique
   */
  public int valueAsCentipawns() {
    return Math.round(value * 100);
  }
}
