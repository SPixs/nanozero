// apps/sts/sts-worker.mjs — Web Worker STS : pour chaque position, joue le MEILLEUR coup de Nano
// à un budget `sims` donné (déterministe, temp 0) et renvoie son UCI. Le scoring (rank dans c9 →
// points c8) est fait côté thread principal. Bundlé (esbuild IIFE). ORT-Web via CDN ; cœur depuis core/.

import { GameState } from '../../core/game-state.mjs';
import { NeuralNet } from '../../core/nn.mjs';
import { BatchedEvaluator } from '../../core/batched-evaluator.mjs';
import { searchBatched, selectMove } from '../../core/mcts.mjs';
import { TENSOR_SIZE } from '../../core/encoding/plane-encoding.mjs';
import { packMove } from '../../core/encoding/move-encoding.mjs';

const ORT_BASE = '/vendor/ort-1.20.1-r2/';
/* global importScripts */
importScripts(ORT_BASE + 'ort.webgpu.min.js');
const ort = self.ort;
ort.env.wasm.wasmPaths = ORT_BASE;

const post = (m) => self.postMessage(m);
const WAVE = 8;
let net = null;
let stop = false;

self.onmessage = async (e) => {
  const m = e.data || {};
  try {
    if (m.type === 'start') {
      const apiBase = m.apiBase || '/api/v1';
      const cur = await (await fetch(`${apiBase}/models/current`)).json();
      const buf = await (await fetch(`${apiBase}/models/${cur.version}/download`)).arrayBuffer();
      const eps = m.backend === 'wasm' ? ['wasm'] : ['webgpu', 'wasm'];
      const session = await ort.InferenceSession.create(new Uint8Array(buf), {
        executionProviders: eps, graphOptimizationLevel: 'all',
      });
      net = new NeuralNet(session, { Tensor: ort.Tensor });
      await net.evaluateBatch(new Float32Array(WAVE * TENSOR_SIZE), WAVE);
      post({ type: 'model-ready', version: cur.version });
    } else if (m.type === 'run') {
      stop = false;
      const { sims, fens } = m;
      for (let i = 0; i < fens.length; i++) {
        if (stop) break;
        let uci = null;
        try {
          const gs = new GameState(fens[i]);
          const ev = new BatchedEvaluator(net, WAVE, { idleGap: true });
          ev.setLive(1);
          const root = await searchBatched(gs, ev, sims, { addNoise: false, waveSize: WAVE });
          ev.workerDone();
          const packed = selectMove(root, 0); // meilleur coup déterministe (STS)
          const legal = gs.legalMoves().find((x) => packMove(x) === packed);
          uci = legal ? legal.uci : null;
        } catch { uci = null; }
        post({ type: 'pos', idx: i, uci });
      }
      post({ type: 'done' });
    } else if (m.type === 'stop') {
      stop = true;
    }
  } catch (err) {
    post({ type: 'error', message: String((err && err.message) || err) });
  }
};
