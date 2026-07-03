// Cross-validation JAVA↔JS : compare plans (7616) + indices policy contre le dump
// de référence Java (/tmp/reference.jsonl produit par tools/CrossValHarness.java).
// LE gate anti-poison : prouve la parité bit-à-bit des encodages.

import { readFileSync } from 'fs';
import { GameState } from '../src/game-state.mjs';
import { encodePlanes, TENSOR_SIZE } from '../src/encoding/plane-encoding.mjs';
import { encode } from '../src/encoding/move-encoding.mjs';

// ⚠️ DOIVENT correspondre exactement à tools/CrossValHarness.java
const FENS = [
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
  'rnbq1rk1/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w - - 6 5',
  'n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1',
  '8/8/8/3k4/8/3K4/8/8 w - - 5 30',
  'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2',
  // --- FENs durs Story 0 (silent-poison) — DOIVENT correspondre à CrossValHarness.java ---
  'rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3',
  '4k3/8/8/8/5P2/8/8/4K3 b - f3 0 1',
  'r3k2r/8/8/8/4r3/8/8/R3K2R w KQkq - 0 1',
  'n1n1k3/1P6/8/8/8/8/8/4K3 w - - 0 1',
];
const UCIS = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f3g1', 'c6b8', 'g1f3', 'b8c6', 'f3g1', 'c6b8', 'g1f3', 'b8c6'];

const ref = readFileSync('/tmp/reference.jsonl', 'utf8').trim().split('\n').map((l) => JSON.parse(l));
let di = 0, fail = 0;
const uciOf = (m) => m.cjs.from + m.cjs.to + (m.cjs.promotion || '');

function jsDump(gs) {
  const planes = encodePlanes(gs.toEncoderState());
  const side = gs.sideToMove();
  const moves = gs.legalMoves().map((m) => ({ uci: uciOf(m), idx: encode(m, side) }));
  return { planes, moves, side };
}

function compare(label, java, js) {
  const sideOk = java.side === js.side;
  let pdiff = -1;
  for (let i = 0; i < TENSOR_SIZE; i++) {
    if (Math.abs(js.planes[i] - java.planes[i]) > 1e-5) { pdiff = i; break; }
  }
  const jMap = new Map(java.moves.map((m) => [m.uci, m.idx]));
  const sMap = new Map(js.moves.map((m) => [m.uci, m.idx]));
  const onlyJava = [...jMap.keys()].filter((u) => !sMap.has(u));
  const onlyJs = [...sMap.keys()].filter((u) => !jMap.has(u));
  let idxBad = null;
  for (const [u, idx] of jMap) if (sMap.has(u) && sMap.get(u) !== idx) { idxBad = { u, java: idx, js: sMap.get(u) }; break; }

  const ok = sideOk && pdiff === -1 && !onlyJava.length && !onlyJs.length && !idxBad;
  if (ok) {
    console.log(`  ✓ ${label} : side + plans(7616) + ${jMap.size} coups (uci&idx) == Java`);
  } else {
    fail++;
    console.error(`  ✗ ${label} : side=${sideOk} planesDiff@${pdiff} onlyJava=${JSON.stringify(onlyJava)} onlyJs=${JSON.stringify(onlyJs)} idxBad=${JSON.stringify(idxBad)}`);
    if (pdiff >= 0) console.error(`      plane[${pdiff}] (p=${Math.floor(pdiff / 64)} sq=${pdiff % 64}) : java=${java.planes[pdiff]} js=${js.planes[pdiff]}`);
  }
}

console.log('=== Mode 1 : positions isolées (t0 ; t1-7 = 0) ===');
for (let i = 0; i < FENS.length; i++) { compare(ref[di].label, ref[di], jsDump(new GameState(FENS[i]))); di++; }

console.log('=== Mode 2 : replay startpos (historique 8-timestamps + répétition) ===');
const gs = new GameState();
compare(ref[di].label, ref[di], jsDump(gs)); di++;
for (const u of UCIS) {
  gs.applyMove({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u[4] });
  compare(ref[di].label, ref[di], jsDump(gs)); di++;
}

if (fail === 0) console.log('\n✅✅ CROSS-VALIDATION JAVA↔JS : PARITÉ TOTALE (plans + policy)');
else { console.error(`\n❌ ${fail} position(s) divergente(s)`); process.exit(1); }
