// apps/bench/probe.mjs — spike latence moteur in-browser (story F4).
// Bundlé en dist/probe.bundle.js (ESM). ORT-Web vient du global `window.ort` (chargé par CDN dans le
// HTML) ; le cœur (mcts/nn/évaluateur/game-state) est bundlé depuis core/ (chessops inclus).
//
// Mesure, sur la position de départ, la latence du PREMIER coup moteur (`searchBatched`) à différents
// `sims`, en WebGPU et en WASM-CPU, pour fixer les valeurs de STRENGTH_LEVELS (J1).
// Réutilise le wiring PROUVÉ du worker : InferenceSession.create → NeuralNet → BatchedEvaluator.

import { GameState } from '../../core/game-state.mjs';
import { NeuralNet } from '../../core/nn.mjs';
import { BatchedEvaluator } from '../../core/batched-evaluator.mjs';
import { searchBatched } from '../../core/mcts.mjs';
import { TENSOR_SIZE } from '../../core/encoding/plane-encoding.mjs';

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
export const SIMS_GRID = [50, 100, 200, 400, 800];
const WAVE = 8;

// --- Harnais pur (exporté, réutilisable) ------------------------------------------------------

export async function buildNet(ort, modelBytes, backend) {
  // backend 'wasm' = CPU forcé ; sinon WebGPU avec repli WASM (même config que le worker).
  const executionProviders = backend === 'wasm' ? ['wasm'] : ['webgpu', 'wasm'];
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders,
    graphOptimizationLevel: 'all',
  });
  const net = new NeuralNet(session, { Tensor: ort.Tensor });
  await net.evaluateBatch(new Float32Array(WAVE * TENSOR_SIZE), WAVE); // warmup (compile shaders / threads)
  return { net, sessionMs: Math.round(performance.now() - t0) };
}

// 1 recherche complète (1 coup) sur la startpos, arbre neuf. Retourne la durée en ms.
async function oneMove(net, sims) {
  const ev = new BatchedEvaluator(net, WAVE, { idleGap: true });
  ev.setLive(1);
  const gs = new GameState(START);
  const t0 = performance.now();
  await searchBatched(gs, ev, sims, { addNoise: false, waveSize: WAVE });
  ev.workerDone();
  return performance.now() - t0;
}

export async function runBackend(ort, modelBytes, backend, simsList = SIMS_GRID) {
  const { net, sessionMs } = await buildNet(ort, modelBytes, backend);
  const rows = [];
  for (const sims of simsList) {
    const cold = await oneMove(net, sims); // 1er coup à ce sims (arbre froid)
    const warm = await oneMove(net, sims); // coup suivant (cache chaud)
    rows.push({ sims, coldMs: Math.round(cold), warmMs: Math.round(warm) });
  }
  return { backend, sessionMs, rows };
}

// Modèle champion via le jobserver (même origine si servi sur nanozero.org).
export async function fetchChampionModel(apiBase = '/api/v1') {
  const cur = await (await fetch(`${apiBase}/models/current`)).json();
  const buf = await (await fetch(`${apiBase}/models/${cur.version}/download`)).arrayBuffer();
  return { version: cur.version, bytes: new Uint8Array(buf) };
}

// --- Pilote DOM (page latency-probe.html) -----------------------------------------------------

function el(id) { return document.getElementById(id); }
function log(msg) { const o = el('out'); if (o) o.textContent += msg + '\n'; }

async function loadModelBytes() {
  const file = el('modelFile').files[0];
  if (file) { log(`modèle local : ${file.name} (${file.size} o)`); return { version: 'local', bytes: new Uint8Array(await file.arrayBuffer()) }; }
  const base = el('apiBase').value.trim() || '/api/v1';
  log(`fetch modèle via ${base}…`);
  return fetchChampionModel(base);
}

async function run() {
  const ort = window.ort;
  if (!ort) { log('ERREUR : ORT-Web non chargé (script CDN).'); return; }
  el('runBtn').disabled = true;
  try {
    const m = await loadModelBytes();
    log(`modèle v${m.version} (${m.bytes.length} o)`);
    const backends = [];
    if (el('bgpu').checked) backends.push('webgpu');
    if (el('bwasm').checked) backends.push('wasm');
    const results = [];
    for (const b of backends) {
      log(`\n=== ${b} ===`);
      const r = await runBackend(ort, m.bytes, b);
      log(`session+warmup : ${r.sessionMs} ms`);
      for (const row of r.rows) log(`sims=${String(row.sims).padStart(4)} | 1er coup ${String(row.coldMs).padStart(6)} ms | suivant ${String(row.warmMs).padStart(6)} ms`);
      results.push(r);
    }
    log('\n--- JSON (à coller dans spike-latence-F4.md) ---');
    log(JSON.stringify({ ua: navigator.userAgent, deviceMemory: navigator.deviceMemory, results }, null, 2));
  } catch (e) {
    log('ERREUR : ' + (e && e.message ? e.message : e));
  } finally {
    el('runBtn').disabled = false;
  }
}

if (typeof document !== 'undefined') {
  const wire = () => { const b = el('runBtn'); if (b) b.addEventListener('click', run); };
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', wire); else wire();
}
