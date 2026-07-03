package org.nanozero.engine.internal;

import org.nanozero.engine.EngineConfig;

/**
 * Sélection PUCT (Predictor + Upper Confidence bound applied to Trees) variante AlphaZero (cf. SPEC
 * §5.2, ADR-001).
 *
 * <p>À un nœud {@code s} expandé non-terminal, le child à explorer est celui qui maximise :
 *
 * <pre>
 *     score(s, a_i) = Q(s, a_i) + c_puct × P(s, a_i) × √N(s) / (1 + N(s, a_i))
 * </pre>
 *
 * <p>avec :
 *
 * <ul>
 *   <li>{@code Q(s, a_i)} : valeur moyenne pour le child {@code a_i} ; vaut {@code
 *       child.totalValueSum / child.totalVisits} si visité, sinon {@link EngineConfig#fpuValue}
 *       (First Play Urgency, cf. §6.2).
 *   <li>{@code P(s, a_i)} : prior policy issu du NN, lu dans {@code node.childPriors[i]}.
 *   <li>{@code N(s)} : visites du parent, lues dans {@code node.totalVisits}.
 *   <li>{@code N(s, a_i)} : visites du child, ou 0 si non instancié / non visité.
 * </ul>
 *
 * <p><strong>Tie-breaking déterministe</strong> (cf. ADR-010) : si plusieurs scores sont
 * strictement égaux (rare avec floats), le <strong>premier index</strong> dans l'ordre {@code
 * MoveGen} l'emporte. Implémenté via la comparaison stricte {@code score > bestScore} (jamais
 * {@code >=}).
 *
 * <p>Classe utilitaire ({@code final}, constructeur privé). Toutes les méthodes sont {@code
 * static}.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
final class PUCTSelector {

  private PUCTSelector() {
    throw new AssertionError("Utility class — do not instantiate");
  }

  /**
   * Sélectionne l'index du child maximisant le score PUCT.
   *
   * <p><strong>Préconditions</strong> (responsabilité du caller {@code SearcherCore}) :
   *
   * <ul>
   *   <li>{@code node.expanded == true}
   *   <li>{@code node.terminal == false}
   *   <li>{@code node.childMoves.length >= 1}
   * </ul>
   *
   * <p>Une violation cause {@link NullPointerException} ou {@link ArrayIndexOutOfBoundsException} ;
   * aucune validation explicite (hot path).
   *
   * <p>Cas {@code node.totalVisits.get() == 0} (nœud fraîchement expandé jamais re-sélectionné) :
   * {@code √N = 0} donc {@code U = 0} pour tous les enfants ; le score se réduit alors à {@code Q =
   * fpuValue} (constant), donc l'argmax retourne {@code 0} (premier index par tie-breaking).
   * Comportement strict §5.2.
   *
   * @param node parent dont on sélectionne le meilleur child
   * @param config hyperparamètres ({@link EngineConfig#cPuct}, {@link EngineConfig#fpuValue})
   * @return index dans {@code [0, node.childMoves.length)} du child à explorer
   */
  static int argmax(Node node, EngineConfig config) {
    // (v1.2.0) Lectures atomiques via .get() — overhead négligeable (1 volatile load).
    float sqrtN = (float) Math.sqrt(node.totalVisits.get());
    float cPuct = config.cPuct();
    float fpu = config.fpuValue();
    int n = node.childMoves.length;

    // v1.1.0 — Détection root via dirichletNoise != null (cf. SPEC §5.2 amendement). Seul le root
    // peut avoir ce champ non-null (responsabilité de SearcherCore.runOneSimulation). Si non-null,
    // mélanger les priors NN avec le noise selon la formule AlphaZero :
    //   P_effective[i] = (1 - epsilon) * P_nn[i] + epsilon * noise[i]
    // Pour les nodes non-root (99 % des cas), la branche est skip rapidement (test de référence).
    float[] noise = node.dirichletNoise;
    float epsilon = (noise != null) ? config.dirichletEpsilon() : 0f;
    float oneMinusEpsilon = 1f - epsilon;

    // (v1.2.0) Virtual loss (KataGo style, cf. ADR-013 §15.3) : pénalise les chemins déjà pris
    // par d'autres threads concurrents pour diversifier l'exploration. Fast-path : si
    // childInFlight==null OU vloss==0, la branche est court-circuitée et le code path est
    // strictement v1.1.2 bit-pour-bit (régression check critique).
    //
    // Formule (POV child) : W_eff = W + vloss*inFlight, N_eff = N + vloss*inFlight (on simule
    // des "victoires côté child" = défaites côté parent → POV parent voit le chemin moins
    // attractif).
    java.util.concurrent.atomic.AtomicIntegerArray inFlight = node.childInFlight;
    float vloss = (inFlight != null) ? config.virtualLoss() : 0f;
    boolean vlossActive = vloss > 0f;

    float bestScore = Float.NEGATIVE_INFINITY;
    int bestIdx = 0;
    for (int i = 0; i < n; i++) {
      Node child = node.children[i];
      float qSa;
      int nSa;
      // (v1.2.0) Snapshot N puis W : si lecteur concurrent voit N stale (ancien), le Q est
      // sous-évalué mais jamais aberrant. Lecture N en premier garantit que pour N > 0,
      // W est >= "N-1 ticks de retard" — pas de NaN ni divergence.
      int childVisits = (child == null) ? 0 : child.totalVisits.get();
      // (v1.2.0) Application virtual loss : pondère N et W par vloss * inFlight[i].
      // En mode mono-thread (inFlight==null) ou vloss=0, vlossN reste 0 → identité v1.1.2.
      float vlossN = vlossActive ? vloss * inFlight.get(i) : 0f;
      float effN = childVisits + vlossN;
      if (effN <= 0f) {
        qSa = fpu;
        nSa = 0;
      } else {
        // Convention zero-sum (cf. SPEC §5.4 amendé) : child.totalValueSum est stocké du POV
        // du child (cohérent avec l'alternance de signe à chaque niveau de Backup). Pour que
        // PUCT calcule le score du POV du parent (= côté à jouer à `node`), on négie : un
        // child avec Q élevé pour LUI est un mauvais coup pour le parent.
        // Virtual loss : W_eff = W + vlossN → Q_child_eff plus élevé → -Q_child_eff plus
        // négatif POV parent → chemin moins attractif pour les autres threads.
        float wRaw = (child == null) ? 0f : child.totalValueSum.get();
        qSa = -(wRaw + vlossN) / effN;
        nSa = childVisits;
      }
      float prior =
          (noise != null)
              ? oneMinusEpsilon * node.childPriors[i] + epsilon * noise[i]
              : node.childPriors[i];
      // U term : on garde 1 + N (pas N_eff) pour préserver la convention v1.1.2 stricte sur le
      // bonus exploration (U ne dépend que de visites réelles), seul Q est pondéré par vloss.
      float u = cPuct * prior * sqrtN / (1f + nSa);
      float score = qSa + u;
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }
    return bestIdx;
  }
}
