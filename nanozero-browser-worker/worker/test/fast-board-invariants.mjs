// fast-board-invariants.mjs — Filet make/unmake + silent-poison (Story 0), pur-JS sur GameState.
//
// Trois gates :
//   A. INVARIANT make/unmake : pour chaque FEN, chaque coup légal -> applyMove(m) puis undoMove()
//      DOIT restaurer toEncoderState() ET legalMoves() à l'identique. Attrape roque-unmake
//      incomplet (tour), corruption ep, snapshot non dépilé.
//   B. EP-CONVENTION-JAVA : la clé de répétition contient l'ep "toujours-après-double-poussée"
//      même quand aucun pion ne peut capturer (sinon plans rép12/13 faux). Le snapshot ne se
//      compte PAS lui-même à la 1re occurrence.
//   C. 3-FOLD : sur une séquence aller-retour, rep1 devient True à la 2e occurrence (pas la 1re),
//      rep2 à la 3e. Le snapshot courant ne se compte jamais lui-même.
//
// Baseline = chess.js (oracle). Aucun dump Java requis.

import { GameState } from '../src/game-state.mjs';
import { ALL_INVARIANT_FENS, HARD_FENS, THREEFOLD_SEQUENCES } from './fast-board-fens.mjs';

let fail = 0;
const err = (msg) => { fail++; console.error(`  ✗ ${msg}`); };

// --- Sérialisation canonique d'un encoderState (BigInt -> string) pour comparaison exacte. ---
function serEncoder(st) {
  const histStr = st.history
    .map((h) => `${h.white.join(',')}|${h.black.join(',')}|${h.rep1 ? 1 : 0}${h.rep2 ? 1 : 0}`)
    .join(';');
  const c = st.castling;
  return [
    st.sideToMove, st.fullmoveNumber, st.halfmoveClock,
    `${c.wk ? 1 : 0}${c.wq ? 1 : 0}${c.bk ? 1 : 0}${c.bq ? 1 : 0}`,
    histStr,
  ].join('#');
}

// Coups triés/canoniques (indépendant de l'ordre d'énumération).
function serMoves(moves) {
  return moves
    .map((m) => `${m.from}-${m.to}-${m.type}-${m.promo}`)
    .sort()
    .join(',');
}

function snap(gs) {
  return { enc: serEncoder(gs.toEncoderState()), mv: serMoves(gs.legalMoves()) };
}

// =====================================================================================
// GATE A — INVARIANT make/unmake (chaque coup légal de chaque FEN)
// =====================================================================================
console.log('=== GATE A : invariant make/unmake (toEncoderState + legalMoves) ===');
let aChecked = 0, aDiv = 0;
for (const { label, fen } of ALL_INVARIANT_FENS) {
  const gs = new GameState(fen);
  const before = snap(gs);
  const moves = gs.legalMoves();
  let localDiv = 0;
  for (const m of moves) {
    gs.applyMove(m);
    gs.undoMove();
    const after = snap(gs);
    aChecked++;
    if (after.enc !== before.enc) {
      localDiv++; aDiv++;
      err(`[A:${label}] coup ${m.from}->${m.to} : ENCODER divergent après apply+undo`);
      if (localDiv === 1) console.error(`      before.enc=${before.enc.slice(0, 90)}...\n      after.enc =${after.enc.slice(0, 90)}...`);
    }
    if (after.mv !== before.mv) {
      localDiv++; aDiv++;
      err(`[A:${label}] coup ${m.from}->${m.to} : LEGAL MOVES divergent après apply+undo`);
    }
    if (localDiv > 3) { console.error(`      ... (${label} : >3 divergences, on coupe)`); break; }
  }
  if (localDiv === 0) console.log(`  ✓ ${label} : ${moves.length} coups × (apply+undo) → 0 divergence`);
}
console.log(`  → ${aChecked} coups vérifiés, ${aDiv} divergence(s).`);

// Variante profondeur-2 sur un sous-ensemble : apply(m1) apply(m2) undo undo (roque + ep imbriqués).
console.log('--- GATE A2 : make/unmake imbriqué profondeur 2 (Kiwipete + roque-sous-échec) ---');
let a2Checked = 0, a2Div = 0;
for (const fen of [
  'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
  'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', // roque des deux côtés dispo
]) {
  const gs = new GameState(fen);
  const before = snap(gs);
  const m1s = gs.legalMoves();
  for (const m1 of m1s) {
    gs.applyMove(m1);
    const mid = snap(gs);
    const m2s = gs.legalMoves();
    for (const m2 of m2s) {
      gs.applyMove(m2);
      gs.undoMove();
      a2Checked++;
      if (snap(gs).enc !== mid.enc || snap(gs).mv !== mid.mv) {
        a2Div++; err(`[A2] ${m1.from}->${m1.to} puis ${m2.from}->${m2.to} : état mid corrompu`);
      }
    }
    gs.undoMove();
  }
  if (snap(gs).enc !== before.enc || snap(gs).mv !== before.mv) {
    a2Div++; err(`[A2] ${fen} : état initial NON restauré après remontée complète`);
  }
}
console.log(`  → ${a2Checked} paires (apply×2/undo×2) vérifiées, ${a2Div} divergence(s).`);

// =====================================================================================
// GATE B — EP convention-Java (clé de répétition contient l'ep même si non capturable)
// ⚠️ Story 5 : GameState est désormais un adaptateur sur FastBoard. La clé de répétition
// n'est plus une string FEN (`snapshots[].key`) mais le Zobrist convention-Java {lo,hi}
// porté par `_fb._epFile`. On reconstruit l'ep algébrique depuis ce file (+ le trait) et on
// prouve la MÊME propriété : l'ep est conservé dans la clé même quand il n'est PAS capturable.
// =====================================================================================
console.log('\n=== GATE B : ep convention-Java dans la clé de répétition (via _fb._epFile) ===');
let bDiv = 0;
// ep file (0-7) -> algébrique : le rang est dicté par le trait (ep noir→rang3, ep blanc→rang6).
const epAlgFromFile = (gs) => {
  const file = gs._fb._epFile;
  if (file < 0) return '-';
  const rank = gs.sideToMove() === 0 /* WHITE au trait */ ? 6 : 3; // ep des Noirs (rang6) / des Blancs (rang3)
  return String.fromCharCode(97 + file) + rank;
};
for (const h of HARD_FENS) {
  if (!h.epInKey) continue;
  const gs = new GameState(h.fen);
  const keyEp = epAlgFromFile(gs);
  const fenEp = h.fen.trim().split(/\s+/)[3];
  const epCapturable = gs.legalMoves().some((m) => m.type === 3 /* EN_PASSANT */);
  const ok = keyEp === h.epInKey;
  if (ok) {
    console.log(`  ✓ ${h.label} : clé ep="${keyEp}" (FEN ep=${fenEp}, chess.js ep-jouable=${epCapturable})`);
  } else {
    bDiv++; err(`[B:${h.label}] clé ep="${keyEp}", ATTENDU "${h.epInKey}" (FEN ep=${fenEp})`);
  }
}
// Preuve clé du piège : sur ep-isolated, AUCUN coup ep n'est émis MAIS la clé garde l'ep.
{
  const iso = HARD_FENS.find((h) => h.label === 'ep-java-divergent-isolated');
  const gs = new GameState(iso.fen);
  const epMoves = gs.legalMoves().filter((m) => m.type === 3);
  const keyEp = epAlgFromFile(gs);
  if (epMoves.length === 0 && keyEp === 'f3') {
    console.log(`  ✓ DIVERGENCE PROUVÉE : ep f3 non-capturable (0 coup ep) MAIS clé contient f3 (convention Java)`);
  } else {
    bDiv++; err(`[B:isolated] epMoves=${epMoves.length} keyEp=${keyEp} (attendu 0 coup ep + clé f3)`);
  }
}

// =====================================================================================
// GATE C — 3-fold : rep1 à la 2e occurrence, rep2 à la 3e ; le snapshot ne se compte pas lui-même
// =====================================================================================
console.log('\n=== GATE C : détection 3-fold (rep1@2e, rep2@3e occurrence) ===');
let cDiv = 0;
for (const seq of THREEFOLD_SEQUENCES) {
  const gs = new GameState(seq.startFen || undefined);
  // occurrence #1 (ply 0) : startpos vue 1 fois -> rep1 doit être FALSE (snapshot ne se compte pas)
  const occ = []; // capture le rep1/rep2 de la position courante (t=0) à chaque retour startpos
  const recordCurrent = () => {
    const h0 = gs.toEncoderState().history[0]; // t=0 = position courante
    return { rep1: h0.rep1, rep2: h0.rep2 };
  };
  occ.push({ ply: 0, ...recordCurrent() }); // 1re occurrence startpos
  for (let i = 0; i < seq.ucis.length; i++) {
    const u = seq.ucis[i];
    gs.applyMove({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u[4] });
    // après 4 demi-coups (i=3) -> 2e occurrence startpos ; après 8 (i=7) -> 3e occurrence
    if (i === 3) occ.push({ ply: 4, ...recordCurrent() });
    if (i === 7) occ.push({ ply: 8, ...recordCurrent() });
  }
  // Attendu : occ0 (1re) rep1=false ; occ1 (2e) rep1=true rep2=false ; occ2 (3e) rep1=true rep2=true
  const [o0, o1, o2] = occ;
  const checks = [
    ['1re occurrence (ply0) rep1=false', o0.rep1 === false],
    ['1re occurrence (ply0) rep2=false', o0.rep2 === false],
    ['2e occurrence (ply4) rep1=true', o1.rep1 === true],
    ['2e occurrence (ply4) rep2=false', o1.rep2 === false],
    ['3e occurrence (ply8) rep1=true', o2.rep1 === true],
    ['3e occurrence (ply8) rep2=true', o2.rep2 === true],
  ];
  for (const [desc, pass] of checks) {
    if (pass) console.log(`  ✓ ${seq.label} : ${desc}`);
    else { cDiv++; err(`[C:${seq.label}] ${desc} — FAUX (o0=${JSON.stringify(o0)} o1=${JSON.stringify(o1)} o2=${JSON.stringify(o2)})`); }
  }
}

// =====================================================================================
console.log('');
if (fail === 0) {
  console.log('✅✅ INVARIANTS : make/unmake (0 div) + ep-convention-Java + 3-fold TOUS VERTS');
} else {
  console.error(`❌ INVARIANTS : ${fail} divergence(s) (A=${aDiv}+${a2Div}, B=${bDiv}, C=${cDiv})`);
  process.exit(1);
}
