// browser-api.mjs — entrée navigateur : bench du débit self-play (runBench) + contribution
// RÉELLE au jobserver (runContribution, A4-D2). Bundlé via esbuild (expose NZ.*).

import { runRollingPool } from './self-play.mjs';
import { runConcurrentContributionLoop } from './run-loop.mjs';
import { NeuralNet } from './nn.mjs';
import { EvalCache } from './eval-cache.mjs';
import { TENSOR_SIZE } from './encoding/plane-encoding.mjs';

// Cap du cache d'éval NN par worker (CPU/WASM). Entrée COMPACTE ~200 o → 8192 ≈ 1,6 Mo/worker
// (vs ~1,3 Go si on cachait la policy dense [4672] × 65536 — cf. eval-cache.mjs). FIFO au-delà.
const EVAL_CACHE_CAPACITY = 8192;

/**
 * A4-D2 — Entrypoint de CONTRIBUTION : télécharge le champion, crée la session, lance le pool.
 *
 * Glue entre A.3 (client claim/submit + getChampionModel) et A4-D1 (boucle N-parties batchée) :
 *   getChampionModel() → ort.InferenceSession.create(bytes) → NeuralNet → runConcurrentContributionLoop.
 * `ort` et `client` sont INJECTÉS (en navigateur : onnxruntime-web + JobserverClient ; en test :
 * onnxruntime-node + client mémoire). La session est créée DEPUIS les bytes déjà vérifiés sha256
 * par le client — jamais une URL brute.
 *
 * @param {object} ort  runtime ONNX (InferenceSession, Tensor) — web (WebGPU) ou node.
 * @param {object} opts
 * @param {{getChampionModel:Function, claim:Function, submitGame:Function}} opts.client
 * @param {number} [opts.poolSize=16]  parties concurrentes (= batch cible).
 * @param {object} [opts.sessionOptions]  options ORT (défaut WebGPU + opt 'all').
 * @param {object} [opts.gameOpts]  options playGame (sims, tempMoves, maxPlies...).
 * @param {(p:{contributed:number, modelVersion:number, plies:number})=>void} [opts.onProgress]  émis par submit.
 * @param {()=>boolean} [opts.shouldStop]  arrêt propre (bouton Stop, Epic C).
 * @param {(m:string)=>void} [opts.log]
 * @param {boolean} [opts.warmup=true]
 * @returns {Promise<object>} stats agrégées de la boucle + modelVersion.
 */
export async function runContribution(ort, opts = {}) {
  const {
    client,
    poolSize = 16,
    // webgpu d'abord, REPLI wasm (CPU) : `navigator.gpu` peut exister sans adaptateur réel →
    // l'init webgpu échoue ; avec wasm en repli, ORT bascule en CPU au lieu de « no available backend ».
    sessionOptions = { executionProviders: ['webgpu', 'wasm'], graphOptimizationLevel: 'all' },
    gameOpts = { sims: 800, tempMoves: 30, maxPlies: 512 },
    onProgress = () => {},
    onModel = () => {}, // appelé dès que le modèle champion est téléchargé : onModel(version)
    onPly, // hook live per-ply (Epic C) : onPly(slotIndex, {ply, fen, wdl, gameId}) — agrégé par l'appelant (Web Worker)
    onSearchProgress, // hook « réflexion » live : onSearchProgress(slotIndex, done, total) pendant la recherche du coup
    progressSlots = 0, // nb de slots affichés → onSearchProgress câblé seulement pour eux (pas d'overhead sur les invisibles)
    onEvaluator, // hook : reçoit le BatchedEvaluator (stats live gpuMs/avgBatch) — diag latence
    onEvalCache, // hook : reçoit l'EvalCache (ou null) → lecture live hits/lookups pour l'A/B débit
    enableEvalCache = false, // cache d'éval NN — ON en CPU/WASM, OFF en WebGPU (gate posé par worker-entry)
    shouldStop = () => false,
    log = () => {},
    warmup = true,
  } = opts;

  if (!ort || !ort.InferenceSession || !ort.Tensor) throw new Error('runContribution : ort requis (InferenceSession + Tensor)');
  if (!client || typeof client.getChampionModel !== 'function') throw new Error('runContribution : client requis (getChampionModel/claim/submitGame)');

  log('téléchargement du modèle champion…');
  const model = await client.getChampionModel(); // {version, sha256, bytes} — sha déjà vérifié
  onModel(model.version); // remonte la version à l'UI AVANT la création de session / le 1er submit
  log(`modèle v${model.version} (${model.bytes.length} o) → création session…`);
  const session = await ort.InferenceSession.create(model.bytes, sessionOptions);
  const net = new NeuralNet(session, { Tensor: ort.Tensor });
  if (warmup) await net.evaluateBatch(new Float32Array(poolSize * TENSOR_SIZE), poolSize);
  log(`session v${model.version} prête → pool de ${poolSize} parties concurrentes`);

  // Cache d'éval NN COMPACT, PARTAGÉ par tous les slots de CE worker (CPU/WASM only). En WebGPU le
  // goulot est la bulle JS de feeding (pas l'inférence) et le coût mémoire ×K-workers n'est pas
  // justifié → désactivé (null) ⇒ chemin MCTS strictement inchangé.
  const evalCache = enableEvalCache ? new EvalCache(EVAL_CACHE_CAPACITY) : null;
  if (onEvalCache) onEvalCache(evalCache); // expose hits/lookups à l'UI (peut être null en WebGPU)
  if (evalCache) log(`cache d'éval NN activé (cap ${EVAL_CACHE_CAPACITY}, CPU/WASM)`);

  // Progression COARSE (par submit). Le flux per-ply fin pour les mini-échiquiers = Epic C (C.2).
  // On wrappe submitGame sans toucher la signature de runConcurrentContributionLoop.
  let contributed = 0;
  const wrappedClient = {
    claim: () => client.claim(),
    submitGame: async (jobId, samples) => {
      const r = await client.submitGame(jobId, samples);
      contributed += 1;
      onProgress({ contributed, modelVersion: model.version, plies: samples.length });
      return r;
    },
  };

  const stats = await runConcurrentContributionLoop({
    client: wrappedClient,
    net,
    poolSize,
    maxBatch: poolSize,
    // evalCache injecté dans gameOpts → propagé tel quel par les slots jusqu'à playGame → search/searchBatched.
    gameOpts: { ...gameOpts, evalCache },
    onPly,
    onSearchProgress,
    progressSlots,
    onEvaluator,
    shouldStop,
  });
  stats.modelVersion = model.version;
  return stats;
}

// Burst pour le test multi-worker : 1 session + self-play pool roulant pendant durationMs.
export async function runBurst(ort, modelUrl, poolSize, durationMs) {
  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['webgpu'], graphOptimizationLevel: 'all',
  });
  const net = new NeuralNet(session, { Tensor: ort.Tensor });
  await net.evaluateBatch(new Float32Array(poolSize * TENSOR_SIZE), poolSize); // warmup
  const t0 = performance.now();
  const r = await runRollingPool(poolSize, net, {
    sims: 100, tempMoves: 30, maxPlies: 400, maxBatch: poolSize, deadline: t0 + durationMs,
  });
  const dt = (performance.now() - t0) / 1000;
  return { evals: r.stats.positions, dt: +dt.toFixed(2), pps: Math.round(r.stats.positions / dt) };
}

// Un burst de mesure (~burstMs) pour une config {pool,wave} sur un `net` donné. Renvoie pos/s + avgBatch.
// Le warmup de la shape de batch est EXCLU de la mesure (1er run paie la compilation/cache des buffers).
async function measureConfig(net, pool, wave, burstMs) {
  // ⚠️ BUG FIX (CALIBRATION-DESIGN §« Deux actions ») : le batch réel est PLAFONNÉ à 256 par
  // BatchedEvaluator (MAX_GPU_BATCH) → on évalue toujours `warmN = min(pool×wave, 256)` lignes. Le
  // warmup DOIT allouer EXACTEMENT `warmN` planes : allouer `pool×wave` (> 256 dès wave≥8 à pool≥32)
  // tout en déclarant la shape [256,…] levait « Tensor size mismatch » et plantait la calib GPU.
  const warmN = Math.max(1, Math.min(pool * wave, 256));
  const warm = new Float32Array(warmN * TENSOR_SIZE);
  await net.evaluateBatch(warm, warmN);
  const t0 = performance.now();
  const r = await runRollingPool(pool, net, {
    sims: 100, tempMoves: 30, maxPlies: 400, maxBatch: pool * wave, waveSize: wave,
    deadline: t0 + burstMs,
  });
  const dt = (performance.now() - t0) / 1000;
  return { pps: dt > 0 ? Math.round(r.stats.positions / dt) : 0, avgBatch: +(r.avgBatch || 0).toFixed(1) };
}

/**
 * CALIBRATION VALIDÉE (CALIBRATION-DESIGN « Design FINAL validé ») — Bench de CALIBRATION mono-process :
 * mesure le débit (évaluations NN/s = pos/s) d'1+ config(s) {pool,wave} dans CETTE session. NE soumet
 * RIEN (self-play local pour timing seul) — télécharge juste le modèle champion via getChampionModel
 * (mis en cache → le vrai démarrage le réutilise).
 *
 * UNE SEULE SESSION, backend webgpu+wasm (IDENTIQUE à runContribution). Le numThreads CPU/WASM est
 * FIXE (posé une fois avant create) : le design validé NE sweepe PLUS les threads (CPU = `nt=1`, le
 * levier est K — cf. sweep 2D §« Algo FINAL »). GPU : numThreads null (WASM only, hors-sujet).
 *
 * Le débit AGRÉGÉ multi-worker (le vrai levier — K Web Workers pour le CPU, K=2 pour le GPU) est
 * orchestré par le THREAD PRINCIPAL (app.mjs) qui spawn K de ces workers en parallèle et SOMME les
 * pos/s. Ce sweep-ci ne mesure donc QU'UN process.
 *
 * ⚠️ Session créée avec `['webgpu','wasm']` — PAS `['webgpu']` seul, sinon une machine sans adaptateur
 * réel n'aurait AUCUN repli CPU. La mesure reflète le backend RÉEL.
 *
 * @param {object} ort  runtime ONNX (InferenceSession + Tensor).
 * @param {object} opts
 * @param {{getChampionModel:Function}} opts.client  client (getChampionModel uniquement).
 * @param {Array<{pool:number, wave:number}>} opts.configs  configs pool/wave à mesurer.
 * @param {number|null} [opts.numThreads]  numThreads CPU/WASM fixe (null = GPU/WASM-only laissé à ORT).
 * @param {(t:number)=>void} [opts.setThreads]  pose ort.env.wasm.numThreads (AVANT create, défensif).
 * @param {number} [opts.burstMs=1500]  durée de chaque burst.
 * @param {(p:{i:number,total:number,pool:number,wave:number,pps:number,numThreads?:number})=>void} [opts.onProgress]
 * @param {(m:string)=>void} [opts.log]
 * @returns {Promise<{results, modelVersion}>}
 */
export async function runBenchSweep(ort, opts = {}) {
  const {
    client,
    configs = [],
    numThreads = null, // CPU/WASM fixe ; null = GPU (WASM only laissé à ORT)
    setThreads = () => {}, // injecté par worker-entry : (t) => { ort.env.wasm.numThreads = t; }
    burstMs = 1500,
    sessionOptions = { executionProviders: ['webgpu', 'wasm'], graphOptimizationLevel: 'all' },
    onProgress = () => {},
    log = () => {},
  } = opts;

  if (!ort || !ort.InferenceSession || !ort.Tensor) throw new Error('runBenchSweep : ort requis (InferenceSession + Tensor)');
  if (!client || typeof client.getChampionModel !== 'function') throw new Error('runBenchSweep : client requis (getChampionModel)');

  log('téléchargement du modèle (bench)…');
  const model = await client.getChampionModel(); // {version, sha256, bytes} — mis en cache, réutilisé au vrai start

  // numThreads CPU/WASM posé AVANT la création de session (le seul moment où ORT-Web le lit).
  if (Number.isFinite(numThreads) && numThreads > 0) setThreads(numThreads);
  log(`modèle v${model.version} → création session (bench${numThreads ? `, nt=${numThreads}` : ''})…`);
  const session = await ort.InferenceSession.create(model.bytes, sessionOptions);
  const net = new NeuralNet(session, { Tensor: ort.Tensor });

  const results = [];
  const total = configs.length;
  for (let i = 0; i < total; i++) {
    const { pool, wave } = configs[i];
    const m = await measureConfig(net, pool, wave, burstMs);
    results.push({ pool, wave, numThreads: numThreads || undefined, pps: m.pps, avgBatch: m.avgBatch });
    log(`bench ${i + 1}/${total} pool=${pool} wave=${wave} → ${m.pps} pos/s (avgBatch ${m.avgBatch})`);
    onProgress({ i: i + 1, total, pool, wave, numThreads: numThreads || undefined, pps: m.pps });
  }
  return { results, modelVersion: model.version };
}

export async function runBench(ort, modelUrl, opts = {}) {
  const log = opts.log || (() => {});
  log('session create...');
  const session = await ort.InferenceSession.create(modelUrl, {
    executionProviders: ['webgpu'], graphOptimizationLevel: 'all',
  });
  log('session OK, outputs=' + JSON.stringify(session.outputNames));
  const net = new NeuralNet(session, { Tensor: ort.Tensor });
  log('1er evaluate (lecture .data)...');
  const probe = await net.evaluate(new Float32Array(TENSOR_SIZE));
  log('1er evaluate OK: value=' + (probe ? probe.value : '?') + ' polLen=' + (probe && probe.policy ? probe.policy.length : '?'));

  const result = { secure: self.isSecureContext, sweep: {} };
  try {
    const a = await self.navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    result.adapter = a && a.info ? { vendor: a.info.vendor, architecture: a.info.architecture } : 'na';
  } catch (e) { result.adapter = 'err'; }
  log('GPU: ' + JSON.stringify(result.adapter));

  for (const N of (opts.pools || [16, 32, 64, 128, 256])) {
    log('pool ' + N + ' warmup...');
    const warm = new Float32Array(N * TENSOR_SIZE);
    for (let i = 0; i < 3; i++) await net.evaluateBatch(warm, N);
    log('pool ' + N + ' warmup OK, run 5s...');

    // 5 s de self-play réel (pool roulant maintient N parties vivantes)
    const t0 = performance.now();
    const r = await runRollingPool(N, net, {
      sims: 100, tempMoves: 30, maxPlies: 400, maxBatch: N, deadline: t0 + 5000,
    });
    const dt = (performance.now() - t0) / 1000;
    result.sweep[N] = {
      pps: Math.round(r.stats.positions / dt),
      avgBatch: +r.avgBatch.toFixed(1),
      evals: r.stats.positions,
    };
    log(`pool=${N}: ${result.sweep[N].pps} pos/s (avgBatch ${result.sweep[N].avgBatch})`);
  }
  return result;
}
