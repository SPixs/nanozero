package org.nanozero.engine.internal;

import java.util.Objects;
import org.nanozero.board.GameState;
import org.nanozero.nn.BitboardPlaneEncoderVector;
import org.nanozero.nn.MoveEncoding;
import org.nanozero.nn.NNOutput;
import org.nanozero.nn.Network;

/**
 * Évaluation d'une feuille MCTS via le NN (cf. SPEC §5.3, §12 phase 4). Enchaîne :
 *
 * <ol>
 *   <li>Encoder la position en plans 119 via {@link BitboardPlaneEncoderVector#INSTANCE}.
 *   <li>Forward NN (batch=1) via le {@link NetworkProvider} injecté.
 *   <li>Décoder la policy via {@link MoveEncoding#decodePolicy} (softmax masqué sur les coups
 *       légaux).
 *   <li>Retourner la value scalaire post-tanh.
 * </ol>
 *
 * <p><strong>Cache d'évaluation optionnel</strong> (ADR-018) : si un {@link NNEvalCache} est
 * injecté (non null), {@link #evaluate} consulte le cache AVANT l'étape 1 (HIT ⇒ refill {@code
 * priorsDest} + retour de la value, en sautant {@code toPlanes/forward/decode}) et y stocke le
 * résultat APRÈS un MISS. Avec {@code cache == null} ({@code nnCacheSize=0}, défaut) le
 * comportement est strictement préservé bit-pour-bit — aucun calcul de clé, aucun overhead.
 *
 * <p><strong>Zéro allocation</strong> en hot path : {@code planeBuffer}, {@code nnOutput} et {@code
 * logitsBuffer} sont pré-alloués au constructeur. Le caller fournit son propre {@code priorsDest}
 * (réutilisé entre nœuds, typiquement remplit {@code Node.childPriors}).
 *
 * <p><strong>Non thread-safe</strong> : conforme au modèle de threading mono-thread search v1.0.0
 * (ADR-002). Une instance par {@code SearcherCore} suffit.
 *
 * <p><strong>Positions terminales</strong> : {@code LeafEvaluator} ne gère pas les positions
 * terminales. Le caller ({@code SearcherCore} en phase 6+) DOIT détecter le terminal AVANT
 * d'appeler {@link #evaluate}. Un appel avec {@code numLegalMoves < 1} lève {@link
 * IllegalArgumentException}.
 *
 * @apiNote Internal — do not depend on this from outside the {@code nanozero-engine} module.
 */
final class LeafEvaluator {

  private final NetworkProvider networkProvider;

  /**
   * (ADR-018) Cache d'évaluation NN OPTIONNEL, partagé entre tous les {@code SearcherCore}. {@code
   * null} ⇒ cache désactivé ({@code nnCacheSize=0}) : comportement strictement préservé
   * bit-pour-bit (aucun calcul de clé, aucun lookup/store — tout est gardé derrière le null-check).
   */
  private final NNEvalCache cache;

  /**
   * Buffer plans NCHW pré-alloué une fois au constructeur, dimensionné à {@code MAX_BATCH × 119 ×
   * 64 = 487 424 floats} (~1.86 MiB) tel qu'attendu par le contrat strict de {@link
   * Network#forward} (cf. SPEC-nn §4.2). Bien que {@code LeafEvaluator} n'écrit que les 7 616
   * premiers floats (batch=1), {@code Network.forward} valide la taille totale du buffer.
   */
  private final float[] planeBuffer = new float[Network.MAX_BATCH * GameState.NN_PLANES * 64];

  /** Conteneur de sortie NN pré-alloué (taille {@code MAX_BATCH × POLICY_INDICES} en interne). */
  private final NNOutput nnOutput = new NNOutput();

  /**
   * Buffer des 4 672 logits de la position courante, copiés depuis {@code nnOutput} par {@code
   * copyLogitsTo(0, …)} pour rester zéro alloc (vs {@code logitsOf(0)} qui allouerait une nouvelle
   * copie à chaque évaluation).
   */
  private final float[] logitsBuffer = new float[MoveEncoding.POLICY_INDICES];

  /**
   * Construit un {@code LeafEvaluator} avec son {@link NetworkProvider}. Toutes les allocations
   * sont faites ici ; les appels ultérieurs à {@link #evaluate} sont zéro-alloc.
   *
   * @param networkProvider provider NN (mockable) ; non null
   * @throws NullPointerException si {@code networkProvider} est null
   */
  LeafEvaluator(NetworkProvider networkProvider) {
    this(networkProvider, null);
  }

  /**
   * (ADR-018) Construit un {@code LeafEvaluator} avec un {@link NNEvalCache} optionnel. Si {@code
   * cache != null}, {@link #evaluate} consulte le cache AVANT le forward et y stocke le résultat
   * APRÈS un MISS. Si {@code cache == null}, comportement v1.x strictement préservé bit-pour-bit.
   *
   * @param networkProvider provider NN (mockable) ; non null
   * @param cache cache d'évaluation NN partagé, ou {@code null} pour désactiver
   * @throws NullPointerException si {@code networkProvider} est null
   */
  LeafEvaluator(NetworkProvider networkProvider, NNEvalCache cache) {
    this.networkProvider =
        Objects.requireNonNull(networkProvider, "networkProvider must not be null");
    this.cache = cache;
  }

  /**
   * Évalue une position non terminale et remplit les priors normalisés sur les coups légaux.
   *
   * @param state position à évaluer (non terminale ; cohérence à la charge du caller)
   * @param legalMoves coups légaux depuis {@code state}, encodés au format {@code Move} 16-bit
   * @param numLegalMoves nombre de coups légaux dans {@code legalMoves} (≥ 1)
   * @param priorsDest buffer destination ; les indices {@code [0, numLegalMoves)} seront écrits
   * @return value du NN post-tanh, dans {@code [-1, +1]}
   * @throws NullPointerException si {@code state}, {@code legalMoves} ou {@code priorsDest} sont
   *     null
   * @throws IllegalArgumentException si {@code numLegalMoves < 1}, {@code numLegalMoves >
   *     legalMoves.length}, ou {@code priorsDest.length < numLegalMoves}
   */
  float evaluate(GameState state, int[] legalMoves, int numLegalMoves, float[] priorsDest) {
    if (state == null) {
      throw new NullPointerException("state must not be null");
    }
    if (legalMoves == null) {
      throw new NullPointerException("legalMoves must not be null");
    }
    if (priorsDest == null) {
      throw new NullPointerException("priorsDest must not be null");
    }
    if (numLegalMoves < 1) {
      throw new IllegalArgumentException(
          "numLegalMoves must be >= 1 (terminal must be detected by caller); got " + numLegalMoves);
    }
    if (numLegalMoves > legalMoves.length) {
      throw new IllegalArgumentException(
          "numLegalMoves ("
              + numLegalMoves
              + ") exceeds legalMoves buffer length ("
              + legalMoves.length
              + ")");
    }
    if (priorsDest.length < numLegalMoves) {
      throw new IllegalArgumentException(
          "priorsDest length ("
              + priorsDest.length
              + ") must be >= numLegalMoves ("
              + numLegalMoves
              + ")");
    }

    // 0. (ADR-018) Lookup cache d'évaluation NN, si activé. Tout est gardé derrière le null-check :
    //    cache désactivé ⇒ aucun calcul de clé, aucun overhead, comportement v1.x bit-pour-bit.
    long cacheKey = 0L;
    if (cache != null) {
      cacheKey = NNEvalCache.key(state);
      float cached = cache.lookup(cacheKey, numLegalMoves, priorsDest);
      if (!Float.isNaN(cached)) {
        // HIT : priorsDest[0..numLegalMoves) déjà rempli par lookup (arraycopy, zéro alloc).
        // On saute toPlanes + forward + decodePolicy.
        return cached;
      }
    }

    // 1. Encoder la position en plans 119 NCHW (cf. SPEC-nn §5.4).
    state.toPlanes(planeBuffer, 0, BitboardPlaneEncoderVector.INSTANCE);

    // 2. Forward NN (batch=1).
    networkProvider.forward(planeBuffer, nnOutput);

    // 3. Copier les logits du sample 0 dans le buffer pré-alloué (zéro alloc), puis décoder
    //    la policy via softmax masqué sur les coups légaux (cf. SPEC-nn §5.5).
    nnOutput.copyLogitsTo(0, logitsBuffer);
    MoveEncoding.decodePolicy(
        logitsBuffer, legalMoves, numLegalMoves, state.currentPosition().sideToMove(), priorsDest);

    // 4. Value scalaire (déjà passée par tanh / WDL V=P(W)−P(L) côté NN).
    float value = nnOutput.valueOf(0);

    // 5. (ADR-018) Store dans le cache, si activé (chemin MISS uniquement).
    if (cache != null) {
      cache.store(cacheKey, value, priorsDest, numLegalMoves);
    }
    return value;
  }

  // -------------------------------------------------------------------------------------------
  // Accesseurs package-private réservés aux tests (vérification de la pré-allocation)
  // -------------------------------------------------------------------------------------------

  /** Référence du buffer plans pré-alloué — réservé aux tests pour vérifier la réutilisation. */
  float[] planeBufferForTest() {
    return planeBuffer;
  }
}
