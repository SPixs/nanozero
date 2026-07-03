// nn.mjs — adaptateur d'inférence (interface commune node ORT / browser ORT Web).
// Le modèle gen-027 : input 'board' (n,119,8,8) -> 'policy_logits' (n,4672) + 'value' (n,3 WDL).
// V = P(W) − P(L) (ADR-012). Interface engine inchangée (un scalaire).

export function softmax3(a, b, c) {
  const m = Math.max(a, b, c);
  const ea = Math.exp(a - m), eb = Math.exp(b - m), ec = Math.exp(c - m);
  const s = ea + eb + ec;
  return [ea / s, eb / s, ec / s];
}

export function wdlToValue(w, d, l) {
  const [pw, , pl] = softmax3(w, d, l);
  return pw - pl;
}

export class NeuralNet {
  /**
   * @param {object} session  InferenceSession ORT (node ou web)
   * @param {{Tensor:Function, inputName?:string, policyName?:string, valueName?:string}} opts
   */
  constructor(session, opts = {}) {
    this.session = session;
    this.Tensor = opts.Tensor;
    this.inputName = opts.inputName || session.inputNames[0];
    this.policyName = opts.policyName || 'policy_logits';
    this.valueName = opts.valueName || 'value';
  }

  // 1 position : planes Float32(7616) -> {policy:Float32Array(4672), value:number}
  async evaluate(planes) {
    return (await this.evaluateBatch(planes, 1))[0];
  }

  // batch : planesBatch Float32(n*7616) -> [{policy, value}] (n entrées)
  async evaluateBatch(planesBatch, n) {
    const input = new this.Tensor('float32', planesBatch, [n, 119, 8, 8]);
    const feeds = {}; feeds[this.inputName] = input;
    const _t0 = performance.now();
    const out = await this.session.run(feeds);
    // Temps d'inférence RÉEL (session.run) — DISTINCT du « mur » mesuré autour de l'await dans
    // BatchedEvaluator : sur un thread JS unique ce mur absorbe le MCTS et NE reflète PAS le GPU
    // (cf. diag 2026-06-20 : run réel ~30-40 ms vs mur ~629 ms quand le thread est saturé).
    this.lastRunMs = performance.now() - _t0;
    const pol = out[this.policyName].data; // n*4672 float32
    const val = out[this.valueName].data;  // n*3   float32
    const res = new Array(n);
    for (let i = 0; i < n; i++) {
      const [pw, pd, pl] = softmax3(val[i * 3], val[i * 3 + 1], val[i * 3 + 2]); // probas W/D/L
      res[i] = {
        // COPIE détachée (`.slice` = memcpy, pas une vue `.subarray`) : `pol` est le buffer de
        // sortie ORT ; une vue resterait valide tant que la consommation est synchrone avant le
        // run suivant, mais devient corrompue si ORT réutilise/préalloue le buffer (IO-binding).
        policy: pol.slice(i * 4672, (i + 1) * 4672),
        value: pw - pl, // scalaire V = P(W)−P(L) (inchangé) — utilisé par le MCTS
        wdl: [pw, pd, pl], // probas (POV trait) — pour la barre d'éval UI (Epic C)
      };
    }
    // Libère les buffers des tenseurs E/S APRÈS la copie (.slice ci-dessus a déjà détaché les
    // données). ORT-Web/WebGPU ne recycle pas tout seul ses ressources entre les `session.run`
    // → sans ça la mémoire monte par run jusqu'à l'OOM onglet (incident W3090, fuite ∝ K runs).
    // `?.()` : no-op sur ORT-node (pas de dispose) et si la sortie est déjà un tenseur CPU.
    input.dispose?.();
    for (const name in out) out[name].dispose?.();
    return res;
  }
}
