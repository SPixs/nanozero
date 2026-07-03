// gate_tree_feats.mjs — features ARBRE (mêmes que strat_tree2) sur les positions du gate curated.
// Recherche 800 sims snapshotée à 128 (découverte tardive) via reprise de root, addNoise=false.
// FEN nue assumée (pas d'historique connu) — même condition que l'utilisateur qui colle une position.
// Sortie JSON → /tmp/gate_tree_feats.json. Run : node gate_tree_feats.mjs (depuis nanozero-web/)
import { createRequire } from 'module';
const require = createRequire(new URL('../../nanozero-web/package.json', import.meta.url));
const ort = require('onnxruntime-node');
const fs = require('fs');
import { GameState } from '../../nanozero-web/core/game-state.mjs';
import { NeuralNet } from '../../nanozero-web/core/nn.mjs';
import { BatchedEvaluator } from '../../nanozero-web/core/batched-evaluator.mjs';
import { searchBatched } from '../../nanozero-web/core/mcts.mjs';
import { TENSOR_SIZE } from '../../nanozero-web/core/encoding/plane-encoding.mjs';

const MODEL = process.env.NZ_ONNX || `${process.env.HOME}/nanozero-night/models/gen-031-promoted.onnx`;
const GATE = new URL('../data/lichess_sample/curated_gate.json', import.meta.url).pathname;
const WAVE = 8, L2 = Math.LN2;
const sc = (w) => w[0] + 0.5 * w[1], sw = (w) => [w[2], w[1], w[0]];

function treeFeats(root) {
  const all = root.children || [];
  const ch = all.filter((c) => c.node && c.node.N > 0);
  if (ch.length < 2) return null;
  let pv = ch[0]; for (const c of ch) if (c.node.N > pv.node.N) pv = c;
  const refV = -pv.node.Q, refE = sc(sw(pv.node.wdl));
  let psum = 0; for (const c of ch) psum += (c.P || 0);
  let betaMcts = 0, betaWdl = 0, mean = 0;
  for (const c of ch) {
    const p = (c.P || 0) / (psum || 1), vm = -c.node.Q;
    betaMcts += p * Math.max(0, refV - vm);
    betaWdl += p * Math.max(0, refE - sc(sw(c.node.wdl)));
    mean += p * vm;
  }
  let betaVar = 0; for (const c of ch) { const p = (c.P || 0) / (psum || 1), vm = -c.node.Q; betaVar += p * (vm - mean) ** 2; }
  let pAll = 0; for (const c of all) pAll += (c.P || 0);
  let entPol = 0; for (const c of all) { const p = (c.P || 0) / (pAll || 1); if (p > 0) entPol += -p * Math.log(p) / L2; }
  return { refv: refV, q_root: root.N > 0 ? root.Q : 0, beta_mcts_800: betaMcts, beta_wdl: betaWdl,
           beta_var: betaVar, ent_policy: entPol, decisiveness: root.wdl ? 1 - root.wdl[1] : 0 };
}

const session = await ort.InferenceSession.create(MODEL, {
  executionProviders: ['cpu'], intraOpNumThreads: 2, graphOptimizationLevel: 'all',
});
const net = new NeuralNet(session, { Tensor: ort.Tensor });
await net.evaluateBatch(new Float32Array(WAVE * TENSOR_SIZE), WAVE);

const out = [];
for (const r of JSON.parse(fs.readFileSync(GATE, 'utf8'))) {
  let rec = { name: r.name, fen: r.fen, cat: r.cat, sf_cp: r.sf_cp, ok: false };
  try {
    const gs = new GameState(r.fen);
    const ev = new BatchedEvaluator(net, WAVE, { idleGap: false }); ev.setLive(1);
    let root = await searchBatched(gs, ev, 128, { addNoise: false, waveSize: WAVE });      // snapshot 128
    const q128 = root.N > 0 ? root.Q : 0;
    root = await searchBatched(gs, ev, 800 - 128, { addNoise: false, waveSize: WAVE, root }); // reprise → 800
    ev.workerDone();
    const f = treeFeats(root);
    if (f) rec = { ...rec, ok: true, ...f, q_root_128: q128, q_root_800: root.Q };
  } catch (e) { rec.err = String(e).split('\n')[0]; }
  out.push(rec);
  console.error(`${r.name}: refv=${rec.refv != null ? rec.refv.toFixed(3) : '—'}`);
}
fs.writeFileSync('/tmp/gate_tree_feats.json', JSON.stringify(out, null, 1));
console.error('→ /tmp/gate_tree_feats.json');
process.exit(0);
