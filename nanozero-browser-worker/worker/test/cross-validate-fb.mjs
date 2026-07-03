// cross-validate-fb.mjs — Story 3 GATE PRINCIPAL.
// Identique à cross-validate.mjs MAIS substitue FastBoard à GameState :
//   - encodePlanes(fb.toEncoderState()) → les 7616 plans (Story 3) ;
//   - encode({from,to,type,promo}, side) sur les coups de fb.legalMoves() → indices policy ;
//   - comparaison BIT-À-BIT contre le dump Java de référence (/tmp/reference.jsonl).
//
// L'index policy ne dépend du `type` que pour la SOUS-PROMOTION (move-encoding.mjs:82) ;
// roque/ep/promo-dame sont classés par géométrie. On dérive donc {type,promo} minimalement
// depuis le format interne {from,to,promotion} de FastBoard (Story 4 le formalisera) :
//   promotion défini → type=PROMOTION, promo=PROMO[role] ; sinon type=NORMAL, promo=0.
//
// node test/cross-validate-fb.mjs   (régénérer /tmp/reference.jsonl via tools/run-fastboard-gates.sh --java)

import { readFileSync } from 'fs';
import { FastBoard } from '../src/fast-board.mjs';
import { encodePlanes, TENSOR_SIZE } from '../src/encoding/plane-encoding.mjs';
import { encode } from '../src/encoding/move-encoding.mjs';

// ⚠️ DOIVENT correspondre exactement à tools/CrossValHarness.java + cross-validate.mjs.
const FENS = [
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
  'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
  'rnbq1rk1/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w - - 6 5',
  'n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1',
  '8/8/8/3k4/8/3K4/8/8 w - - 5 30',
  'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2',
  'rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3',
  '4k3/8/8/8/5P2/8/8/4K3 b - f3 0 1',
  'r3k2r/8/8/8/4r3/8/8/R3K2R w KQkq - 0 1',
  'n1n1k3/1P6/8/8/8/8/8/4K3 w - - 0 1',
];
const UCIS = ['e2e4', 'e7e5', 'g1f3', 'b8c6', 'f3g1', 'c6b8', 'g1f3', 'b8c6', 'f3g1', 'c6b8', 'g1f3', 'b8c6'];

// --- helpers FastBoard ---
const sqUci = (s) => (Number(s[1]) - 1) * 8 + (s.charCodeAt(0) - 97); // 'e2' → sq (LSB=a1)
const algOf = (sq) => String.fromCharCode(97 + (sq & 7)) + ((sq >> 3) + 1); // sq → 'e2'
const PROMO_ROLE = { queen: 'q', rook: 'r', bishop: 'b', knight: 'n' };
const uciOfFb = (m) => algOf(m.from) + algOf(m.to) + (m.promotion ? PROMO_ROLE[m.promotion] : '');

// Story 4 : legalMoves() émet DÉJÀ {from,to,type,promo}. On encode DIRECTEMENT ces champs —
// la classification FastBoard (CASTLING/EN_PASSANT/PROMOTION/NORMAL + promo 0/1/2/3) pilote
// l'index policy. Tout écart de type/promo désaligne l'index → cross-validate le détecte.
// (Garde-fou : le move DOIT exposer type/promo en Story 4, sinon dérivation legacy + erreur.)
function toEncodeMove(m) {
  if (m.type !== undefined) return { from: m.from, to: m.to, type: m.type, promo: m.promo };
  throw new Error(`Story 4 régressée : legalMoves() n'émet pas {type,promo} sur ${uciOfFb(m)}`);
}

// Résout un UCI vers le move interne de fb.legalMoves() (pour le replay Mode 2 sans applyMove).
function resolveUci(fb, uci) {
  const from = sqUci(uci.slice(0, 2));
  const to = sqUci(uci.slice(2, 4));
  const promo = uci.length > 4 ? { q: 'queen', r: 'rook', b: 'bishop', n: 'knight' }[uci[4]] : undefined;
  for (const m of fb.legalMoves()) {
    if (m.from === from && m.to === to && m.promotion === promo) return m;
  }
  return null;
}

function fbDump(fb) {
  const planes = encodePlanes(fb.toEncoderState());
  const side = fb.sideToMove();
  const moves = fb.legalMoves().map((m) => ({ uci: uciOfFb(m), idx: encode(toEncodeMove(m), side) }));
  return { planes, moves, side };
}

const ref = readFileSync('/tmp/reference.jsonl', 'utf8').trim().split('\n').map((l) => JSON.parse(l));
let di = 0, fail = 0, okPlanes = 0, okMoves = 0;

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
    okPlanes += TENSOR_SIZE;
    okMoves += jMap.size;
    console.log(`  ✓ ${label} : side + plans(7616) + ${jMap.size} coups (uci&idx) == Java`);
  } else {
    fail++;
    console.error(`  ✗ ${label} : side=${sideOk} planesDiff@${pdiff} onlyJava=${JSON.stringify(onlyJava)} onlyJs=${JSON.stringify(onlyJs)} idxBad=${JSON.stringify(idxBad)}`);
    if (pdiff >= 0) console.error(`      plane[${pdiff}] (p=${Math.floor(pdiff / 64)} sq=${pdiff % 64}) : java=${java.planes[pdiff]} js=${js.planes[pdiff]}`);
  }
}

console.log('=== Mode 1 : positions isolées (t0 ; t1-7 = 0) — FastBoard substitué ===');
for (let i = 0; i < FENS.length; i++) { compare(ref[di].label, ref[di], fbDump(new FastBoard(FENS[i]))); di++; }

console.log('=== Mode 2 : replay startpos (historique 8-timestamps + répétition) — FastBoard ===');
const fb = new FastBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
compare(ref[di].label, ref[di], fbDump(fb)); di++;
for (const u of UCIS) {
  const m = resolveUci(fb, u);
  if (!m) { console.error(`  ✗ UCI ${u} introuvable dans legalMoves()`); process.exit(1); }
  fb._make(m);
  compare(ref[di].label, ref[di], fbDump(fb)); di++;
}

if (fail === 0) {
  console.log(`\n✅✅ CROSS-VALIDATION JAVA↔FastBoard : PARITÉ TOTALE`);
  console.log(`   ${di} positions · ${okPlanes} valeurs de plans bit-à-bit · ${okMoves} coups (uci+index) == Java`);
} else {
  console.error(`\n❌ ${fail} position(s) divergente(s)`);
  process.exit(1);
}
