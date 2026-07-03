// Web Worker : charge ORT Web + le modèle gen-027, benchmark l'inférence.
// Tourne hors du thread principal -> la page ne freeze jamais.
// bundle WebGPU explicite (le ort.min.js par défaut n'enregistre pas le backend webgpu)
importScripts('https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.min.js');
// chemin des .wasm (dont le jsep requis par WebGPU) -> CDN
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

const log = (t, c) => postMessage({ t: String(t), c: c || '' });

async function bench(ep) {
  log(`\n━━━ Provider: ${ep.toUpperCase()} ━━━`, 'b');
  let s;
  try {
    s = await ort.InferenceSession.create('gen-027-promoted.onnx',
      { executionProviders: [ep], graphOptimizationLevel: 'all' });
  } catch (e) { log(`  indisponible: ${e.message || e}`, 'r'); return; }
  const inName = s.inputNames[0];
  for (const bs of [1, 16]) {
    const d = new Float32Array(bs * 119 * 8 * 8);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() < 0.12 ? 1 : 0;
    const feed = {}; feed[inName] = new ort.Tensor('float32', d, [bs, 119, 8, 8]);
    try {
      for (let i = 0; i < 5; i++) await s.run(feed);          // warmup
      const N = bs === 1 ? 200 : 60;
      const t0 = performance.now();
      for (let i = 0; i < N; i++) await s.run(feed);
      const dt = (performance.now() - t0) / 1000, pps = N * bs / dt;
      log(`  batch=${String(bs).padStart(2)} : ${pps.toFixed(0).padStart(6)} pos/s  (${(1000 * dt / N).toFixed(2)} ms/batch)`);
      if (bs === 1) { const inf = 800 * 80; log(`         → 1 partie self-play (${inf} inf.) ≈ ${(inf / pps / 60).toFixed(1)} min`, 'g'); }
    } catch (e) { log(`  batch=${bs}: échec (${e.message || e})`, 'r'); }
  }
  try { await s.release(); } catch (e) {}
}

(async () => {
  try {
    const nav = self.navigator || {};
    log(`Worker démarré.`);
    log(`crossOriginIsolated (threads): ${self.crossOriginIsolated} | WebGPU: ${!!nav.gpu} | cœurs: ${nav.hardwareConcurrency || '?'}`);
    ort.env.wasm.simd = true;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? (nav.hardwareConcurrency || 4) : 1;
    log(`ORT WASM threads utilisés : ${ort.env.wasm.numThreads}`);
    await bench('wasm');
    if (nav.gpu) await bench('webgpu'); else log(`\nWebGPU indisponible dans ce worker.`, 'r');
    log(`\n━━━ TERMINÉ ━━━`, 'b');
    log(`(réf worker natif x86 ≈ 200-600 pos/s/cœur)`);
  } catch (e) { log('ERREUR worker: ' + (e.message || e), 'r'); }
})();
