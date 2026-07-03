// apps/analyse/worker-entry.mjs — Web Worker d'analyse (S-07). Bundlé esbuild (IIFE classic worker).
// ORT-Web via CDN (importScripts) ; cœur depuis core/. Recherche DÉTERMINISTE (addNoise:false).
// Messages :
//   {type:'start', apiBase, backend} → charge le champion → {type:'model-ready', version, backend}
//   {type:'analyze', fen, sims}      → searchBatched → {type:'result', fen, wdl}  (wdl POV trait)
//   progression → {type:'progress', done, total}
// Cache interne Map<fen, wdl> : re-naviguer vers une position déjà analysée = réponse immédiate.

import { GameState } from '../../core/game-state.mjs';
import { NeuralNet } from '../../core/nn.mjs';
import { BatchedEvaluator } from '../../core/batched-evaluator.mjs';
import { searchBatched } from '../../core/mcts.mjs';
import { topVariations } from '../../core/mcts-pv.mjs';
import { computeCriticality } from '../../core/complexity.mjs';
import { TENSOR_SIZE } from '../../core/encoding/plane-encoding.mjs';

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
let aborted = false;
const bgCache = new Map();                       // `${startFen}|${moves}@${sims}` → {wdl, topPV, criticality}
let reuseRoot = null, reuseKey = null, reuseGs = null; // avant-plan : arbre approfondi en continu (tree-reuse)
const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

// Recopie-fill (à la Lc0) : si l'historique NN est incomplet (position ISOLÉE sans coups), on recopie le plus
// ancien état dans les timestamps vides (rep1/rep2=false) → entrée cohérente, vs des plans vides = aberrants.
function repeatFillState(st) {
  if (!st.history || st.history.length >= 8) return st;
  const o = st.history[st.history.length - 1], pad = [];
  for (let i = st.history.length; i < 8; i++) pad.push({ white: o.white, black: o.black, rep1: false, rep2: false });
  return { ...st, history: [...st.history, ...pad] };
}

// Construit le GameState AVEC l'historique réel : depuis startFen, on REJOUE les coups → plans NN 8-plis corrects
// (sinon éval hors-distribution). Fallbacks : rejeu impossible → FEN nue ; position isolée sans coups et ≠ startpos
// (FEN collée) → recopie-fill à la Lc0. En jeu (startpos + coups) : historique réel, partiel accepté en début (comme à l'entraînement).
function buildGs(m) {
  const startFen = m.startFen || m.fen;
  const gs = new GameState(startFen);
  if (m.moves && m.moves.length) {
    for (const u of m.moves) {
      const mv = gs.legalMoves().find((x) => x.uci === u);
      if (!mv) return new GameState(m.fen); // rejeu cassé → repli sûr sur la FEN
      gs.applyMove(mv);
    }
  } else if (startFen.split(' ')[0] !== START_FEN.split(' ')[0]) {
    const orig = gs.toEncoderState.bind(gs);
    gs.toEncoderState = () => repeatFillState(orig()); // FEN isolée non-startpos → recopie-fill
  }
  return gs;
}

async function runSearch(gs, sims, root) {
  const ev = new BatchedEvaluator(net, WAVE, { idleGap: true });
  ev.setLive(1);
  const r = await searchBatched(gs, ev, sims, { addNoise: false, waveSize: WAVE, shouldAbort: () => aborted, root });
  ev.workerDone();
  return r;
}

self.onmessage = async (e) => {
  const m = e.data || {};
  if (m.type === 'cancel') { aborted = true; return; }
  if (m.type === 'reset') { aborted = true; bgCache.clear(); return; } // l'avant-plan repart seul au changement de FEN
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
      await net.evaluateBatch(new Float32Array(WAVE * TENSOR_SIZE), WAVE); // pré-chauffage
      post({ type: 'model-ready', version: cur.version, backend: m.backend });
    } else if (m.type === 'analyze') {
      if (!net) { post({ type: 'error', message: 'modèle non chargé' }); return; }
      const sims = m.sims || 256;
      aborted = false;
      const pathKey = `${m.startFen || m.fen}|${(m.moves || []).join(' ')}`; // identité = position + historique
      if (m.reuse) {
        // AVANT-PLAN : approfondissement continu (tree-reuse). Repart à zéro si le CHEMIN change (une même FEN
        // atteinte par transposition a un historique différent → éval différente).
        if (pathKey !== reuseKey) { reuseKey = pathKey; reuseRoot = null; reuseGs = buildGs(m); }
        reuseRoot = await runSearch(reuseGs, sims, reuseRoot);
        const wdl = aborted ? null : reuseRoot.wdl;
        const topPV = aborted ? null : topVariations(reuseRoot, 3, reuseGs);
        // criticité SIM-STABLE (v3e) : converge/se précise avec l'approfondissement, ne dérive plus → pas de freeze.
        const criticality = aborted ? null : computeCriticality(reuseRoot);
        post({ type: 'result', idx: m.idx, gen: m.gen, fen: m.fen, wdl, topPV, criticality, n: reuseRoot.N });
      } else {
        // ARRIÈRE-PLAN : une position du graphe, profondeur fixe (cache par (CHEMIN,sims) → transpositions OK).
        const key = `${pathKey}@${sims}`;
        if (bgCache.has(key)) { const c = bgCache.get(key); post({ type: 'result', idx: m.idx, gen: m.gen, fen: m.fen, wdl: c.wdl, topPV: c.topPV, criticality: c.criticality, n: sims, cached: true }); return; }
        const gs = buildGs(m);
        const root = await runSearch(gs, sims, null);
        const wdl = aborted ? null : root.wdl;
        const topPV = aborted ? null : topVariations(root, 3, gs);
        const criticality = aborted ? null : computeCriticality(root);
        if (!aborted && wdl) bgCache.set(key, { wdl, topPV, criticality });
        post({ type: 'result', idx: m.idx, gen: m.gen, fen: m.fen, wdl, topPV, criticality, n: root.N });
      }
    }
  } catch (err) {
    post({ type: 'error', message: String((err && err.message) || err) });
  }
};
