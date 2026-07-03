// naturalness.mjs — calcule le signal de « naturalité » (confiance de la policy) par position, pour
// distinguer forcé-ÉVIDENT (recapture) de forcé-DUR. pmax = plus gros prior policy ; p_pv = prior du
// coup le plus visité (32 sims). Un coup évident (recapture) → pmax/p_pv élevés → PAS critique.
import { createRequire } from 'module';
import { fileURLToPath } from 'url';
import { readFileSync, writeFileSync } from 'fs';
const require = createRequire(fileURLToPath(new URL('../../nanozero-web/package.json', import.meta.url)));
const ort = require('onnxruntime-node');
const CORE = new URL('../../nanozero-web/core/', import.meta.url);
const { GameState } = await import(new URL('game-state.mjs', CORE));
const { NeuralNet } = await import(new URL('nn.mjs', CORE));
const { BatchedEvaluator } = await import(new URL('batched-evaluator.mjs', CORE));
const { searchBatched } = await import(new URL('mcts.mjs', CORE));
const { packMove } = await import(new URL('encoding/move-encoding.mjs', CORE));
const { TENSOR_SIZE } = await import(new URL('encoding/plane-encoding.mjs', CORE));
const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const IN = arg('in', '../data/lichess_sample/betav2_hist.csv');
const OUT = arg('out', '/tmp/nat.csv');
const [SI, SN] = arg('shard', '0/1').split('/').map(Number);
const MODEL = arg('model', 'models/gen-031-promoted.onnx');
const SIMS = 32, WAVE = 8;
const session = await ort.InferenceSession.create(MODEL, { executionProviders: ['cpu'], graphOptimizationLevel: 'all', intraOpNumThreads: 1, interOpNumThreads: 1 });
const net = new NeuralNet(session, { Tensor: ort.Tensor });
await net.evaluateBatch(new Float32Array(WAVE * TENSOR_SIZE), WAVE);

const lines = readFileSync(IN, 'utf8').split('\n').filter((l) => l.length);
const h = lines[0].split(','); const idx = {}; h.forEach((k, i) => (idx[k] = i));
let rows = lines.slice(1).filter((_, i) => i % SN === SI);
const buf = ['game_id,ply,n_legal,pmax,p_pv,pv_uci'];
const t0 = Date.now();
for (let n = 0; n < rows.length; n++) {
  const c = rows[n].split(','); const gid = c[idx.game_id], ply = c[idx.ply];
  let out = `${gid},${ply},,,,`;
  try {
    const gs = new GameState(START); const line = (c[idx.line] || '').trim();
    let bad = false;
    if (line) for (const u of line.split(/\s+/)) { const m = gs.legalMoves().find((x) => x.uci === u); if (!m) { bad = true; break; } gs.applyMove(m); }
    if (!bad && !gs.isGameOver() && gs.legalMoves().length >= 2) {
      const legal = gs.legalMoves(); const uci = new Map(); for (const m of legal) uci.set(packMove(m), m.uci);
      const ev = new BatchedEvaluator(net, WAVE, { idleGap: false }); ev.setLive(1);
      const root = await searchBatched(gs, ev, SIMS, { addNoise: false, waveSize: WAVE }); ev.workerDone();
      let pmax = 0, pv = root.children[0];
      for (const ch of root.children) { if ((ch.P || 0) > pmax) pmax = ch.P || 0; if (ch.node.N > pv.node.N) pv = ch; }
      out = `${gid},${ply},${legal.length},${pmax.toFixed(4)},${(pv.P || 0).toFixed(4)},${uci.get(pv.move)}`;
    }
  } catch (e) { /* garde la ligne vide */ }
  buf.push(out);
  if ((n + 1) % 200 === 0) { writeFileSync(OUT, buf.join('\n') + '\n'); process.stderr.write(`${SI}/${SN}: ${n + 1}/${rows.length} (${((n + 1) / ((Date.now() - t0) / 1000)).toFixed(1)}/s)\n`); }
}
writeFileSync(OUT, buf.join('\n') + '\n');
process.stderr.write(`${SI}/${SN}: DONE ${rows.length}\n`);
process.exit(0);
