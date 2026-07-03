// worker-entry.mjs — Web Worker de CONTRIBUTION (Epic C). Bundlé via esbuild (IIFE, classic worker) :
//   esbuild ui/worker-entry.mjs --bundle --format=iife --outfile=ui/worker.bundle.js
// Tourne hors du thread principal → la page ne freeze jamais (NFR4). ORT Web (WebGPU) chargé via
// importScripts (CDN). Reçoit {type:'start'|'stop'} ; poste progress / boards (coalescés) / log / done.

import { JobserverClient } from '../worker/src/jobserver-client.mjs';
import { runContribution, runBenchSweep } from '../worker/src/browser-api.mjs';
import { createCoalescer } from './coalesce.mjs';

// ORT auto-hébergé (ex-CDN jsdelivr : SPOF tiers + tension privacy — servi par nanozero.org).
const ORT_BASE = '/vendor/ort-1.20.1-r2/';
/* global importScripts */
importScripts(ORT_BASE + 'ort.webgpu.min.js'); // → global self.ort (classic worker)
const ort = self.ort;
ort.env.wasm.wasmPaths = ORT_BASE;

const post = (msg) => self.postMessage(msg);
let stop = false;
let activeClient = null; // client de la session en cours → permet d'appliquer un pseudo en direct (STORY-015)
let liveSims = 100; // réglage sims LIVE (modifiable en cours de session via {type:'config'})
let liveBoards = 9; // nb de boards AFFICHÉS : seul ce préfixe de slots calcule l'éval W/D/L (1 inférence/coup) + push board

// Per-ply → coalescé ~3 fps (latest par gameId) avant postMessage : l'UI ne vole pas de CPU au self-play.
const boards = createCoalescer({
  emit: (batch) => post({ type: 'boards', batch }),
  intervalMs: 300,
});

self.onmessage = (e) => {
  const m = e.data || {};
  if (m.type === 'start') {
    stop = false;
    session(m).catch((err) => post({ type: 'error', message: String((err && err.message) || err) }));
  } else if (m.type === 'bench') {
    // STORY-014 — MODE BENCH : mesure le débit (pos/s) à plusieurs configs puis poste les résultats.
    // Chemin SÉPARÉ de la contribution : self-play LOCAL pour timing seul, AUCUN submit au jobserver.
    bench(m).catch((err) => post({ type: 'bench-error', message: String((err && err.message) || err) }));
  } else if (m.type === 'stop') {
    stop = true;
  } else if (m.type === 'set-pseudo') {
    // STORY-015 — pseudo choisi en cours de session : appliqué SANS redémarrer le worker.
    if (activeClient) activeClient.setPseudo(m.pseudo);
  } else if (m.type === 'config') {
    if (m.sims) liveSims = m.sims; // réglage sims appliqué EN DIRECT (prochain coup)
    if (m.boards != null) liveBoards = m.boards; // nb de boards affichés en direct → gate éval/push
  }
};

// CALIBRATION VALIDÉE — Bench de calibration (mode séparé). Crée un client UNIQUEMENT pour
// getChampionModel (pas de claim/submit), puis mesure le débit d'1+ config(s) via runBenchSweep.
// Session créée avec le MÊME backend que la contribution (webgpu+wasm) → la mesure reflète la perf RÉELLE.
//
// CE worker mesure UN SEUL processus (1 session WASM/WebGPU). Le débit AGRÉGÉ multi-worker (le levier
// K du design) est orchestré par le THREAD PRINCIPAL (app.mjs) : il spawn K de ces workers en
// parallèle et SOMME leurs pos/s. Chaque worker reçoit donc des `configs` (≥ 1) et un `numThreads`
// fixe (CPU/WASM ; absent/null en GPU). `setThreads` pose `ort.env.wasm.numThreads` AVANT la création
// de session (le seul moment où ORT-Web lit la valeur).
async function bench({ baseUrl, configs, numThreads, burstMs }) {
  const client = new JobserverClient({ baseUrl, workerId: 'browser-calib', pseudo: null });
  // numThreads CPU/WASM posé une fois avant la session ; null/absent (GPU) → on laisse ORT (WASM only).
  const setThreads = (t) => { if (Number.isFinite(t) && t > 0) ort.env.wasm.numThreads = t; };
  if (Number.isFinite(numThreads) && numThreads > 0) setThreads(numThreads);
  const { results, modelVersion } = await runBenchSweep(self.ort, {
    client,
    configs,
    numThreads: Number.isFinite(numThreads) && numThreads > 0 ? numThreads : null,
    setThreads, // (déjà appliqué ci-dessus ; runBenchSweep le re-pose défensivement avant create)
    burstMs: Number.isFinite(burstMs) && burstMs > 0 ? burstMs : undefined,
    onProgress: (p) => post({ type: 'bench-progress', ...p }), // {i, total, numThreads?, pool, wave, pps}
    log: (message) => post({ type: 'log', message }),
  });
  post({ type: 'bench-done', results, modelVersion });
}

// NB : `boards:` est renommé `displayBoards` ici — sinon le paramètre masquerait le coalesceur
// `boards` (scope module) → `boards.flush()`/`boards.push()` planteraient (« .flush is not a function »).
async function session({ baseUrl, workerId, poolSize = 8, sims = 100, boards: displayBoards = 9, waveSize = 8, pseudo = null, contributorToken = null, numThreads = null }) {
  liveSims = sims; // valeur initiale ; modifiable en direct via {type:'config'}
  liveBoards = displayBoards; // valeur initiale ; modifiable en direct via {type:'config', boards}
  // calibration incr.1 — `numThreads` (CPU/WASM) DOIT être posé AVANT InferenceSession.create
  // (runContribution crée la session). app.mjs ne l'envoie QU'EN CPU (omis en GPU → WASM only laissé
  // à ORT). Rétro-compat : absent → aucune écriture (ancien comportement). Le seul levier ~×4-8 (P1).
  if (Number.isFinite(numThreads) && numThreads > 0) {
    ort.env.wasm.numThreads = numThreads;
    post({ type: 'log', message: `numThreads=${numThreads} (WASM CPU)` });
  }
  post({ type: 'log', message: `worker démarré (pool ${poolSize}, sims/coup dictés par le serveur, leaf-batch K=${waveSize})` });
  // STORY-007 — `pseudo` (label crédits « adresse ouverte ») opt-in : null → aucun header X-Pseudo.
  // STORY-015 — `contributorToken` (identité stable de décompte) → header X-Contributor au submit.
  const client = new JobserverClient({ baseUrl, workerId: workerId || 'browser-volunteer', pseudo, contributorToken });
  activeClient = client; // exposé au handler 'set-pseudo' (application live du pseudo)
  // Débit LIVE : on compte chaque coup joué (onPly) et on poste le taux toutes les 2 s — dès le
  // 1ᵉʳ coup, sans attendre qu'une partie (longue sur CPU) se termine.
  let played = 0;
  const thinkThrottle = new Map(); // slot → dernier post 'thinking' (throttle ~5/s par slot affiché)
  let tFirstPly = 0; // horodatage du 1ᵉʳ coup → le débit cumulé final exclut le download + warmup
  let evaluator = null; // capté via onEvaluator → lecture de la latence d'inférence en direct
  let evalCache = null; // capté via onEvalCache → hits/lookups (CPU/WASM only ; null en WebGPU)
  // GATE cache d'éval NN : ON en CPU/WASM (numThreads posé, même nt=1), OFF en WebGPU (numThreads
  // null/absent). MÊME signal que progressSlots : en GPU le goulot est la bulle JS (pas l'inférence)
  // et le coût mémoire ×K-workers n'est pas justifié. CPU = lent par inférence → le cache débloque.
  const enableEvalCache = Number.isFinite(numThreads) && numThreads > 0;
  // Débit INSTANTANÉ sur fenêtre glissante (dernier intervalle), PAS un cumul depuis le start :
  // l'ancien cumul mettait le download modèle + warmup (played=0) au dénominateur → pos/s sous-estimé
  // ET incohérent avec ms/appel & batch (eux comptés par flush, pas sur le temps mur total). La
  // fenêtre reflète aussi les changements de réglage live au lieu de traîner sur toute la session.
  let prevT = performance.now(), prevPlayed = 0, prevGpuMs = 0, prevRunMs = 0, prevFlushes = 0, prevPositions = 0;
  const perfTimer = setInterval(() => {
    const now = performance.now();
    const dt = (now - prevT) / 1000;
    const s = evaluator && evaluator.stats;
    let pps = 0;
    let msPerCall = null, realMsPerCall = null, avgBatch = null, gpuPct = null;
    if (s) {
      const dF = s.flushes - prevFlushes, dG = s.gpuMs - prevGpuMs, dR = s.runMs - prevRunMs, dP = s.positions - prevPositions;
      // « pos/s » = ÉVALUATIONS NN/s : s.positions s'incrémente à CHAQUE flush (≈continu) → lisse et
      // de l'ordre du millier/s. L'ancien calcul comptait `played` (coups JOUÉS), qui ne bouge qu'à la
      // FIN d'un coup (après ~sims simulations) → 1-6/s en rafale, jamais convergent (cf. retour W3090).
      pps = dt > 0 ? Math.round(dP / dt) : 0;
      // ⚠️ msPerCall = MUR autour de l'await : sur un thread JS unique il ABSORBE le MCTS → ce n'est
      // PAS le temps GPU (cf. mémoire browser-worker-gpu-metric-jsbound). On expose donc les DEUX :
      msPerCall = dF > 0 ? +(dG / dF).toFixed(1) : null;      // « fenêtre » (inférence + JS bloquant)
      realMsPerCall = dF > 0 ? +(dR / dF).toFixed(1) : null;  // session.run RÉEL = le vrai GPU (~30-40 ms)
      avgBatch = dF > 0 ? +(dP / dF).toFixed(1) : null;
      // util GPU HONNÊTE = temps d'inférence RÉEL / temps mur (et non mur/mur, qui sature à 100 %).
      gpuPct = dt > 0 ? Math.min(100, Math.round(dR / (dt * 1000) * 100)) : null;
      prevFlushes = s.flushes; prevGpuMs = s.gpuMs; prevRunMs = s.runMs; prevPositions = s.positions;
    }
    prevPlayed = played; prevT = now;
    // cacheHitRate : hits/lookups cumulés du cache d'éval NN (null si désactivé/WebGPU) — pour l'A/B débit.
    const cacheHitRate = evalCache && evalCache.lookups > 0 ? +evalCache.hitRate.toFixed(3) : null;
    post({ type: 'perf', pps, msPerCall, realMsPerCall, avgBatch, gpuPct, cacheHitRate });
  }, 2000);
  try {
    const stats = await runContribution(ort, {
      client,
      poolSize,
      // sims = getter live (réglable sans restart) ; waveSize figé (l'évaluateur idle-gap/maxBatch
      // est créé une fois selon waveSize → paramètre de session, changé par un restart).
      gameOpts: { sims: () => liveSims, waveSize, tempMoves: 30, maxPlies: 512 },
      onProgress: (p) => post({ type: 'progress', ...p }),
      onModel: (version) => post({ type: 'model', version }),
      onEvaluator: (ev) => { evaluator = ev; }, // pour lire gpuMs/avgBatch dans le perfTimer
      enableEvalCache, // gate CPU/WASM (cf. ci-dessus) → cache d'éval NN compact partagé par le worker
      onEvalCache: (c) => { evalCache = c; }, // pour lire hits/lookups dans le perfTimer

      // WDL = agrégé par la recherche (gratuit, dans info.wdl) ; on ne PUSH le board que pour les
      // slots affichés (< liveBoards) — coupe le postMessage des parties non visibles, pas une éval.
      onPly: (slot, info) => {
        if (!tFirstPly) tFirstPly = performance.now(); // 1ᵉʳ coup : démarre le chrono du débit (post-warmup)
        played += 1; // débit = TOUS les coups joués (même slots non affichés), pour un pos/s juste
        if (slot < liveBoards) {
          boards.push(slot, { slot, gameId: info.gameId, ply: info.ply, fen: info.fen, lastMove: info.lastMove, wdl: info.wdl });
          thinkThrottle.delete(slot); // coup joué → reset l'indicateur « réflexion » du slot
        }
      },
      // « Réflexion » live : signale la progression de la recherche du coup courant (done/total sims)
      // POUR LES SLOTS AFFICHÉS → le board montre qu'il travaille pendant la 1ʳᵉ recherche (longue sur
      // CPU), au lieu de paraître gelé. THROTTLÉ ~5/s. Data-neutre (n'affecte ni la recherche ni la cible).
      onSearchProgress: (slot, done, total) => {
        if (slot >= liveBoards) return;
        const now = performance.now();
        if (done < total && now - (thinkThrottle.get(slot) || 0) < 200) return;
        thinkThrottle.set(slot, now);
        post({ type: 'thinking', slot, done, total });
      },
      // « Réflexion » UNIQUEMENT en CPU/WASM (numThreads posé). MESURÉ : sur GPU le worker est
      // feeding-bound (bulle JS) → le callback de progression coûte ~30 % de débit (inférence-mur
      // 35→50 ms) et les coups y sont rapides (barre inutile). En CPU les coups sont lents (barre
      // NÉCESSAIRE) et l'inférence WASM domine → coût relatif faible.
      progressSlots: (Number.isFinite(numThreads) && numThreads > 0) ? liveBoards : 0,
      shouldStop: () => stop,
      log: (message) => post({ type: 'log', message }),
    });
    boards.flush();
    // débit moyen de la session = ÉVALUATIONS NN / temps DEPUIS LE 1ᵉʳ COUP (exclut download + warmup).
    const elapsed = tFirstPly ? (performance.now() - tFirstPly) / 1000 : 0;
    const totalPos = evaluator ? evaluator.stats.positions : 0;
    post({ type: 'perf', pps: elapsed > 0 ? Math.round(totalPos / elapsed) : 0 });
    post({ type: 'done', stats });
  } finally {
    clearInterval(perfTimer);
    activeClient = null; // fin de session → plus de cible pour 'set-pseudo'
  }
}
