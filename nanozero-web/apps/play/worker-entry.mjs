// apps/play/worker-entry.mjs — Web Worker d'inférence de la surface « Jouer » (J2).
// Bundlé (esbuild, IIFE classic worker). ORT-Web via CDN (importScripts) ; cœur depuis core/.
// Messages :
//   {type:'start', apiBase, backend, thresholds?} → charge le champion, warmup → {type:'model-ready', version}
//   {type:'move',  fen, sims, temperature, addNoise} → searchBatched+selectMove → {type:'move-result', move, complexity, wdl}
//   progression de la recherche → {type:'progress', done, total}

import { GameState } from '../../core/game-state.mjs';
import { NeuralNet } from '../../core/nn.mjs';
import { BatchedEvaluator } from '../../core/batched-evaluator.mjs';
import { searchBatched, selectMove } from '../../core/mcts.mjs';
import { computeComplexity } from '../../core/complexity.mjs';
import { TENSOR_SIZE } from '../../core/encoding/plane-encoding.mjs';
import { packMove } from '../../core/encoding/move-encoding.mjs';

// ORT auto-hébergé (ex-CDN jsdelivr : SPOF tiers + tension privacy). Build choisi selon le backend
// passé dans l'URL du worker (?backend=wasm|webgpu) : le build wasm-only est plus léger pour mobile.
const ORT_BASE = '/vendor/ort-1.20.1-r2/';
const ORT_BUILD = new URLSearchParams(self.location.search).get('backend') === 'wasm'
  ? 'ort.wasm.min.js' : 'ort.webgpu.min.js';
/* global importScripts */
importScripts(ORT_BASE + ORT_BUILD);
const ort = self.ort;
ort.env.wasm.wasmPaths = ORT_BASE;

const post = (m) => self.postMessage(m);
const WAVE = 8;
let net = null;
let thresholds = { calm: 0.10, tense: 0.25 };

self.onmessage = async (e) => {
  const m = e.data || {};
  try {
    if (m.type === 'start') {
      if (m.thresholds) thresholds = m.thresholds;
      const apiBase = m.apiBase || '/api/v1';
      const cur = await (await fetch(`${apiBase}/models/current`)).json();
      const buf = await (await fetch(`${apiBase}/models/${cur.version}/download`)).arrayBuffer();
      // backend device-aware (décision F4) : mobile → wasm (WebGPU plus lent sur le 8×96) ; desktop → webgpu.
      const eps = m.backend === 'wasm' ? ['wasm'] : ['webgpu', 'wasm'];
      const session = await ort.InferenceSession.create(new Uint8Array(buf), {
        executionProviders: eps, graphOptimizationLevel: 'all',
      });
      net = new NeuralNet(session, { Tensor: ort.Tensor });
      await net.evaluateBatch(new Float32Array(WAVE * TENSOR_SIZE), WAVE); // pré-chauffage (compile shaders / threads)
      const wasmThreads = (ort.env && ort.env.wasm && ort.env.wasm.numThreads) || 0;
      post({ type: 'model-ready', version: cur.version, backend: m.backend, wasmThreads });
    } else if (m.type === 'move') {
      if (!net) { post({ type: 'error', message: 'modèle non chargé' }); return; }
      const gs = new GameState(m.fen);
      const ev = new BatchedEvaluator(net, WAVE, { idleGap: true });
      ev.setLive(1);
      const root = await searchBatched(gs, ev, m.sims, {
        addNoise: !!m.addNoise, waveSize: WAVE,
        onProgress: (p) => post({ type: 'progress', done: p.done, total: p.total }),
      });
      ev.workerDone();
      // selectMove renvoie un coup PACKÉ (int, cf. mcts:84) → le matcher au coup légal enrichi
      // (même convention packMove 1-arg que l'expansion) pour récupérer l'algébrique {from,to,promotion}.
      const packed = selectMove(root, m.temperature || 0);
      const legal = gs.legalMoves().find((x) => packMove(x) === packed);
      if (!legal) { post({ type: 'error', message: 'coup moteur introuvable' }); return; }
      const cjs = legal.cjs;
      post({
        type: 'move-result',
        move: { from: cjs.from, to: cjs.to, promotion: cjs.promotion || undefined },
        complexity: computeComplexity(root, thresholds),
        wdl: root.wdl,
      });
    }
  } catch (err) {
    post({ type: 'error', message: String((err && err.message) || err) });
  }
};
