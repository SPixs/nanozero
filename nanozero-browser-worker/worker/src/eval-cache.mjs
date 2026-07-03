// eval-cache.mjs — cache COMPACT d'évaluations NN, par Web Worker (CPU/WASM uniquement).
//
// Miroir JS du `NNEvalCache` Java (ADR-018), adapté au worker self-play navigateur. La leçon clé
// (revue mémoire) : NE JAMAIS cacher le résultat NN COMPLET `{policy:Float32Array(4672), value, wdl}`
// (~19 Ko/entrée) — à 65536 entrées × K Web Workers = ~1,3 Go/worker → OOM (mobile surtout). On cache
// une entrée COMPACTE :
//
//     { value:number, wdl:[pw,pd,pl], priors:Float32Array(numLegalMoves) }
//
// où `priors` = les priors par coup légal DÉJÀ MASQUÉS/normalisés (softmax sur les coups légaux —
// ce que produit `expandNode`), PAS la policy dense [4672]. ~35 floats ≈ 200 o/entrée → à 8192
// entrées ≈ 1,6 Mo/worker. Sur un HIT le MCTS renvoie (value/wdl, priors) et SAUTE à la fois
// `net.evaluate` ET le masquage dense→légal.
//
// CLÉ Lc0-tronquée (calculée par `GameState.cacheKey()`) : `${lo},${hi},${rule50},${rep}` — zobrist
// (pièces+trait+roque+ep) + compteur 50-coups + indicateur de répétition. Ignore VOLONTAIREMENT
// l'historique 8-ply cosmétique encodé par les plans (deux positions de même clé tronquée mais
// d'historique 8-ply différent partagent une entrée — approximation assumée, identique au cache Java) ;
// rep + rule50 RESTENT dans la clé car ils changent la valeur de la position.
//
// Mono-thread (Web Worker) → AUCUN verrou. Borné (cap par défaut 8192) avec éviction FIFO.

const DEFAULT_CAPACITY = 8192;

export class EvalCache {
  /** @param {number} [capacity=8192]  nombre max d'entrées (FIFO au-delà). */
  constructor(capacity = DEFAULT_CAPACITY) {
    this.capacity = capacity;
    // Map JS = ordre d'insertion garanti → la 1ʳᵉ clé énumérée est la plus ancienne (FIFO).
    this.map = new Map();
    this.hits = 0;
    this.lookups = 0;
  }

  /** Nombre d'entrées actuellement stockées. */
  get size() {
    return this.map.size;
  }

  /** hits / lookups (0 si aucun lookup) — pour l'A/B débit. */
  get hitRate() {
    return this.lookups > 0 ? this.hits / this.lookups : 0;
  }

  /**
   * Recherche une entrée. Compte TOUJOURS un lookup. Compte un hit UNIQUEMENT si l'entrée existe ET
   * que `priors.length === expectedLen` (garde anti-collision : une clé tronquée partagée par deux
   * positions au nombre de coups légaux différent = MISS → ré-évaluation propre + écrasement par set).
   *
   * @param {string} key  clé tronquée (GameState.cacheKey()).
   * @param {number} expectedLen  nombre de coups légaux de la position courante.
   * @returns {{value:number, wdl:number[], priors:Float32Array}|undefined} entrée valide, ou undefined.
   */
  get(key, expectedLen) {
    this.lookups++;
    const e = this.map.get(key);
    if (e !== undefined && e.priors.length === expectedLen) {
      this.hits++;
      return e;
    }
    return undefined;
  }

  /**
   * Insère/écrase une entrée. Éviction FIFO (supprime la plus ancienne) quand le cap est atteint et
   * que la clé est nouvelle. Ré-insérer une clé existante ne déplace PAS son rang FIFO (overwrite).
   *
   * @param {string} key
   * @param {{value:number, wdl:number[], priors:Float32Array}} entry
   */
  set(key, entry) {
    if (!this.map.has(key) && this.map.size >= this.capacity) {
      const oldest = this.map.keys().next().value;
      this.map.delete(oldest);
    }
    this.map.set(key, entry);
  }
}
