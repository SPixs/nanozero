// mcts_features.mjs — EXP β-v3 : extrait des features de complexité Stockfish-free depuis le VRAI
// core/mcts.mjs (l'algo qui shippe), sur les 15k positions Lichess labellisées Stockfish.
//
// Fidélité : GameState(startpos) + rejeu de `line` = historique NN 8-plies (leçon dure : jamais FEN nu).
// Une recherche 800-sims snapshotée (32/128/512/800) via reprise de root → β_mcts(sweep) + trajectoire
// Q racine (β_instab) + PV. Passe 1-ply → V_head par coup (β_v1_js + β_swing). Passe multipv-Nano (flag
// --multipv) → β_nanompv (trap_mass Stockfish-free). Sortie CSV jointe par (game_id, ply).
//
// Usage : node mcts_features.mjs --in <csv> --out <csv> [--shard i/n] [--multipv] [--limit N] [--model P]

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
const { encodePlanes, TENSOR_SIZE } = await import(new URL('encoding/plane-encoding.mjs', CORE));

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

// ---- args ----
const argv = process.argv.slice(2);
const arg = (k, d) => { const i = argv.indexOf('--' + k); return i >= 0 ? argv[i + 1] : d; };
const has = (k) => argv.includes('--' + k);
const IN = arg('in', '../data/lichess_sample/betav2_hist.csv');
const OUT = arg('out', '/tmp/mcts_features.csv');
const MODEL = arg('model', 'models/gen-031-promoted.onnx');
const [SHARD_I, SHARD_N] = (arg('shard', '0/1')).split('/').map(Number);
const LIMIT = parseInt(arg('limit', '0'), 10);
const MULTIPV = has('multipv');
const SNAPS = [32, 128, 512, 800];      // budgets snapshot
const K_MPV = 8, SIMS_MPV = 64;         // multipv-Nano : top-k coups, sims/coup
const WAVE = 8;

// ---- réseau ----
const session = await ort.InferenceSession.create(MODEL, {
  executionProviders: ['cpu'], graphOptimizationLevel: 'all',
  intraOpNumThreads: Math.max(1, parseInt(process.env.NANOZERO_THREADS || '1', 10)), interOpNumThreads: 1,
});
const net = new NeuralNet(session, { Tensor: ort.Tensor });
await net.evaluateBatch(new Float32Array(WAVE * TENSOR_SIZE), WAVE); // warmup

// ---- helpers ----
const log2 = (x) => Math.log(x) / Math.LN2;
const swapWDL = (w) => [w[2], w[1], w[0]];               // remontée d'un niveau (échange W↔L)
const scoreFromWDL = (w) => w[0] + 0.5 * w[1];           // espérance de score (POV du triplet)

// β depuis un arbre : lit root.children {P, node:{N,Q,wdl}} ; ref = PV (coup le plus visité).
function treeBetas(root, uciOf) {
  const ch = (root.children || []).filter((c) => c.node && c.node.N > 0);
  if (ch.length < 2) return null;
  let pv = ch[0]; for (const c of ch) if (c.node.N > pv.node.N) pv = c;
  const refV = -pv.node.Q;                                // valeur PV, POV trait
  const refE = scoreFromWDL(swapWDL(pv.node.wdl));        // espérance PV, POV trait
  let psum = 0, betaMcts = 0, betaWdl = 0, mean = 0, Ntot = 0;
  for (const c of ch) { psum += (c.P || 0); Ntot += c.node.N; }
  for (const c of ch) {
    const p = (c.P || 0) / (psum || 1);
    const vm = -c.node.Q;                                 // valeur coup, POV trait
    const em = scoreFromWDL(swapWDL(c.node.wdl));
    betaMcts += p * Math.max(0, refV - vm);
    betaWdl += p * Math.max(0, refE - em);
    mean += p * vm;
  }
  let betaVar = 0, klNum = 0, entVis = 0;
  for (const c of ch) {
    const p = (c.P || 0) / (psum || 1);
    const vm = -c.node.Q;
    betaVar += p * (vm - mean) * (vm - mean);
    const pi = c.node.N / Ntot;
    if (pi > 0) { entVis += -pi * log2(pi); if (p > 0) klNum += pi * log2(pi / p); }
  }
  let entPol = 0; for (const c of root.children) { const p = c.P || 0; if (p > 0) entPol += -p * log2(p); }
  return {
    betaMcts, betaWdl, betaVar, betaKl: klNum, entVis, entPol,
    qroot: root.N > 0 ? root.Q : 0,                       // valeur racine, POV trait
    refv: refV,                                           // valeur du PV (= éval runtime), POV trait
    decis: root.wdl ? 1 - root.wdl[1] : 0,
    pv: uciOf(pv.move), nVisited: ch.length,
  };
}

function parseHeader(line) { const h = line.trim().split(',').map((k) => k.trim()); const idx = {}; h.forEach((k, i) => (idx[k] = i)); return idx; } // trim : les CSV python finissent en \r\n → sans trim la DERNIÈRE clé devient 'line\r' et idx.line=undefined (bug 12k startpos, 2026-07-02)

async function processRow(cols, idx) {
  const fen = cols[idx.fen], line = (cols[idx.line] || '').trim();
  const out = { game_id: cols[idx.game_id], ply: cols[idx.ply], ok: 0 };
  const gs = new GameState(START);
  if (line) for (const uci of line.split(/\s+/)) {
    const m = gs.legalMoves().find((x) => x.uci === uci);
    if (!m) return { ...out, err: 'illegal_line' };
    gs.applyMove(m);
  }
  out.fen_ok = gs.fen().split(' ')[0] === fen.split(' ')[0] ? 1 : 0;
  if (gs.isGameOver()) return { ...out, err: 'terminal' };
  const legal = gs.legalMoves();
  if (legal.length < 2) return { ...out, err: 'few_moves' };
  const uciByPacked = new Map(); for (const m of legal) uciByPacked.set(packMove(m), m.uci);
  const uciOf = (packed) => uciByPacked.get(packed);

  // ---- recherche principale, snapshots par reprise de root (addNoise=false) ----
  const ev = new BatchedEvaluator(net, WAVE, { idleGap: false }); ev.setLive(1);
  let root = null; const snaps = {};
  let prev = 0;
  for (const target of SNAPS) {
    root = await searchBatched(gs, ev, target - prev, { addNoise: false, waveSize: WAVE, root });
    prev = target;
    snaps[target] = treeBetas(root, uciOf);
  }
  ev.workerDone();
  const s800 = snaps[800]; if (!s800) return { ...out, err: 'no_tree' };

  // β_instab : variance de la trajectoire Q racine + flip de PV + nb de PV distincts
  const qs = SNAPS.map((s) => snaps[s] && snaps[s].qroot).filter((x) => x != null);
  const qm = qs.reduce((a, b) => a + b, 0) / qs.length;
  const qvar = qs.reduce((a, b) => a + (b - qm) * (b - qm), 0) / qs.length;
  const pvs = SNAPS.map((s) => snaps[s] && snaps[s].pv).filter(Boolean);
  const pvFlip = pvs[0] !== pvs[pvs.length - 1] ? 1 : 0;
  const nPv = new Set(pvs).size;

  // ---- passe 1-ply : V_head par coup (β_v1_js + β_swing) ----
  const rootEval = await net.evaluate(encodePlanes(gs.toEncoderState()));
  const vHeadRoot = rootEval.value;                       // POV trait
  const oneP = new Map();
  for (const m of legal) {
    gs.applyMove(m);
    let vm;
    if (gs.isGameOver()) vm = gs.isCheckmate() ? 1 : 0;   // trait a maté / pat
    else { const e = await net.evaluate(encodePlanes(gs.toEncoderState())); vm = -e.value; }
    oneP.set(m.uci, vm);
    gs.undoMove();
  }
  // priors depuis l'arbre (mêmes que le MCTS) ; β_v1_js = regret 1-ply pondéré policy vs maxq
  const priors = new Map(); for (const c of root.children) priors.set(uciOf(c.move), c.P || 0);
  let maxV = -Infinity; for (const v of oneP.values()) if (v > maxV) maxV = v;
  let b1 = 0, ps1 = 0, swingChild = 0, psS = 0;
  for (const c of root.children) {
    if (!c.node || c.node.N === 0) continue;
    const uci = uciOf(c.move); const p = c.P || 0; const v1 = oneP.get(uci);
    if (v1 == null) continue;
    b1 += p * Math.max(0, maxV - v1); ps1 += p;
    swingChild += p * Math.abs(v1 - (-c.node.Q)); psS += p;    // |1-ply − recherché| par coup
  }
  const betaV1js = ps1 > 0 ? b1 / ps1 : 0;
  const betaSwingChild = psS > 0 ? swingChild / psS : 0;
  const betaSwingRoot = Math.abs(vHeadRoot - s800.qroot);

  // ---- passe multipv-Nano (optionnelle) : β_nanompv ----
  let betaNanompv = '';
  if (MULTIPV) {
    const kids = [...root.children].filter((c) => c.P > 0).sort((a, b) => b.P - a.P).slice(0, K_MPV);
    const ev2 = new BatchedEvaluator(net, WAVE, { idleGap: false }); ev2.setLive(1);
    let sBest = -Infinity; const sc = [];
    for (const c of kids) {
      const m = legal.find((x) => x.uci === uciOf(c.move)); if (!m) continue;
      gs.applyMove(m);
      let s;
      if (gs.isGameOver()) s = gs.isCheckmate() ? 1 : 0;
      else { const sub = await searchBatched(gs, ev2, SIMS_MPV, { addNoise: false, waveSize: WAVE }); s = -sub.Q; }
      gs.undoMove();
      sc.push({ p: c.P, s }); if (s > sBest) sBest = s;
    }
    ev2.workerDone();
    let num = 0, den = 0; for (const { p, s } of sc) { num += p * Math.max(0, sBest - s); den += p; }
    betaNanompv = den > 0 ? num / den : 0;
  }

  return {
    ...out, ok: 1, n_legal: legal.length, n_visited: s800.nVisited,
    beta_mcts_32: snaps[32] ? snaps[32].betaMcts : '', beta_mcts_128: snaps[128] ? snaps[128].betaMcts : '',
    beta_mcts_512: snaps[512] ? snaps[512].betaMcts : '', beta_mcts_800: s800.betaMcts,
    beta_v1_js: betaV1js, beta_swing_root: betaSwingRoot, beta_swing_child: betaSwingChild,
    q_var: qvar, pv_flip: pvFlip, n_pv: nPv,
    beta_kl: s800.betaKl, beta_wdl: s800.betaWdl, decisiveness: s800.decis,
    beta_var: s800.betaVar, ent_visits: s800.entVis, ent_policy: s800.entPol,
    beta_nanompv: betaNanompv,
    // éval racine par budget (POV trait) : amplitude runtime-cohérente + « découverte tardive » (128 vs 800)
    q_root_32: snaps[32] ? snaps[32].qroot : '', q_root_128: snaps[128] ? snaps[128].qroot : '',
    q_root_512: snaps[512] ? snaps[512].qroot : '', q_root_800: s800.qroot,
    refv_800: s800.refv,
  };
}

// ---- main ----
const lines = readFileSync(IN, 'utf8').split('\n').filter((l) => l.length);
const idx = parseHeader(lines[0]);
let rows = lines.slice(1);
rows = rows.filter((_, i) => i % SHARD_N === SHARD_I);
if (LIMIT) rows = rows.slice(0, LIMIT);

const FIELDS = ['game_id', 'ply', 'ok', 'err', 'fen_ok', 'n_legal', 'n_visited',
  'beta_mcts_32', 'beta_mcts_128', 'beta_mcts_512', 'beta_mcts_800', 'beta_v1_js',
  'beta_swing_root', 'beta_swing_child', 'q_var', 'pv_flip', 'n_pv', 'beta_kl', 'beta_wdl',
  'decisiveness', 'beta_var', 'ent_visits', 'ent_policy', 'beta_nanompv',
  'q_root_32', 'q_root_128', 'q_root_512', 'q_root_800', 'refv_800'];
const buf = [FIELDS.join(',')];
const fmt = (v) => (v == null ? '' : typeof v === 'number' ? (Number.isFinite(v) ? v.toFixed(6) : '') : v);

const t0 = Date.now();
for (let i = 0; i < rows.length; i++) {
  const cols = rows[i].replace(/\r$/, '').split(',');
  let r; try { r = await processRow(cols, idx); } catch (e) { r = { game_id: cols[idx.game_id], ply: cols[idx.ply], ok: 0, err: 'exc:' + (e.message || e) }; }
  buf.push(FIELDS.map((f) => fmt(r[f])).join(','));
  if ((i + 1) % 50 === 0) {
    writeFileSync(OUT, buf.join('\n') + '\n');
    process.stderr.write(`shard ${SHARD_I}/${SHARD_N}: ${i + 1}/${rows.length}  (${((i + 1) / ((Date.now() - t0) / 1000)).toFixed(2)} pos/s)\n`);
  }
}
writeFileSync(OUT, buf.join('\n') + '\n');
process.stderr.write(`shard ${SHARD_I}/${SHARD_N}: DONE ${rows.length} pos in ${((Date.now() - t0) / 60000).toFixed(1)} min\n`);
process.exit(0); // onnxruntime-node retient l'event loop → sortie forcée (sinon process hang au teardown)
