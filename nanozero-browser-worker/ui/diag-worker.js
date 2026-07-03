// diag-worker.js — Web Worker du diagnostic #2, chargé en MÊME ORIGINE (comme worker.bundle.js).
// Un blob: worker échoue sur importScripts(CDN) sous certains COEP/CSP ; un worker same-origin
// (le schéma de la prod, worker-entry.mjs) le charge sans souci. Reçoit {modelBytes, CDN}.
let ort;
self.onmessage = async (e) => {
  const { modelBytes, CDN } = e.data;
  const post = (m) => self.postMessage(m);
  try {
    /* global importScripts */
    importScripts(CDN + 'ort.webgpu.min.js'); // EXACTEMENT comme worker-entry.mjs:12
    ort = self.ort;
    ort.env.wasm.wasmPaths = CDN;

    const TENSOR_SIZE = 119 * 8 * 8; // 7616
    const N = 267;                   // la taille de batch observée en prod
    const planes = new Float32Array(N * TENSOR_SIZE);

    // config PROD à l'identique (browser-api.mjs:36) — on teste le contexte réel
    const session = await ort.InferenceSession.create(new Uint8Array(modelBytes), {
      executionProviders: ['webgpu', 'wasm'], graphOptimizationLevel: 'all',
    });
    const inName = session.inputNames[0];
    const outs = session.outputNames;
    const polName = outs.includes('policy_logits') ? 'policy_logits' : outs[0];
    const valName = outs.includes('value') ? 'value' : outs[outs.length - 1];
    post({ type: 'log', msg: 'session créée DANS LE WORKER. in=' + inName + ' out=' + JSON.stringify(outs) });

    const feeds = () => { const f = {}; f[inName] = new ort.Tensor('float32', planes, [N, 119, 8, 8]); return f; };
    const readAndSoftmax = (o) => { // reproduit nn.mjs:40-53 (read-back + softmax JS), pour rester fidèle
      const v = o[valName].data; o[polName].data;
      let s = 0; for (let j = 0; j < N; j++) { const a = v[j * 3], b = v[j * 3 + 1], c = v[j * 3 + 2], m = Math.max(a, b, c); s += Math.exp(a - m) / (Math.exp(a - m) + Math.exp(b - m) + Math.exp(c - m)); } return s;
    };
    const busy = (ms) => { const end = performance.now() + ms; let x = 0; while (performance.now() < end) { x += Math.sqrt(end - x); } return x; }; // charge JS SYNCHRONE (≈ MCTS chess.js)

    // warmup (compile la forme 267)
    for (let i = 0; i < 3; i++) readAndSoftmax(await session.run(feeds()));

    // --- Exp A : session.run PUR @267, DANS LE WORKER, sans charge (run pur) -----------------
    const pure = [];
    for (let i = 0; i < 8; i++) {
      const t0 = performance.now(); const o = await session.run(feeds()); const t1 = performance.now();
      const s = readAndSoftmax(o); const t2 = performance.now();
      pure.push({ run: t1 - t0, total: t2 - t0, sink: s });
    }
    post({ type: 'expA', pure });

    // --- Exp B : analogue EXACT de gpuMs (mur autour de l'await) + charge JS L injectée -------
    // Démarrer le run, occuper le thread JS L ms (synchrone), PUIS await → la résolution est
    // retardée par la charge, comme l'event-loop du worker live noyé sous le MCTS.
    const loads = [0, 150, 300, 600, 900];
    const expB = [];
    for (const L of loads) {
      let wall = 0; const reps = 5;
      for (let i = 0; i < reps; i++) {
        const t0 = performance.now();        // ← t0 (comme batched-evaluator.mjs:64)
        const p = session.run(feeds());      //   GPU démarre (async)
        busy(L);                             //   charge JS synchrone PENDANT la fenêtre de l'await
        const o = await p;                   //   continuation retardée par la charge
        readAndSoftmax(o);
        wall += performance.now() - t0;      // ← gpuMs += now()-t0 (batched-evaluator.mjs:66)
        post({ type: 'progress', msg: 'L=' + L + ' ms · rep ' + (i + 1) + '/' + reps });
      }
      expB.push({ L, wallAvg: wall / reps });
    }
    post({ type: 'expB', expB, done: true });
    try { await session.release(); } catch { /* ok */ }
  } catch (err) { post({ type: 'error', msg: String((err && err.stack) || err) }); }
};
