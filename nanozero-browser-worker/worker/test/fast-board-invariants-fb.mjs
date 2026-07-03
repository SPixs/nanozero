// fast-board-invariants-fb.mjs — Story 1 gate : invariant _make/_unmake sur FastBoard.
//
// Pour chaque FEN de fast-board-fens.mjs, chaque coup légal → _make puis _unmake → le `Board`
// chessops doit être BIT-IDENTIQUE à avant (occupied, white, black, pawn..king) PLUS l'état
// scalaire (turn, epSquare, castlingRights, halfmoves, fullmoves). Attrape : roque-unmake
// incomplet (tour), corruption ep (case to∓8), promotion (rôle mover), droits roque après
// prise de tour, snapshot/undo non dépilé.
//
// Inclut GATE A2 : profondeur-2 imbriquée (apply×2/undo×2) sur Kiwipete + roque-des-deux-côtés.

import { FastBoard } from '../src/fast-board.mjs';
import { ALL_INVARIANT_FENS } from './fast-board-fens.mjs';

let fail = 0;
const err = (msg) => { fail++; console.error(`  ✗ ${msg}`); };

const ROLES = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king'];

// Sérialisation canonique de l'état complet (Board bitboards lo/hi + scalaires).
function ser(fb) {
  const pos = fb.position;
  const b = pos.board;
  const ss = (s) => `${s.lo}.${s.hi}`;
  const parts = [
    ss(b.occupied), ss(b.white), ss(b.black), ss(b.promoted),
    ...ROLES.map((r) => ss(b[r])),
    `turn:${pos.turn}`,
    `ep:${pos.epSquare === undefined ? '-' : pos.epSquare}`,
    `cr:${ss(pos.castles.castlingRights)}`,
    `rk:${pos.castles.rook.white.a},${pos.castles.rook.white.h},${pos.castles.rook.black.a},${pos.castles.rook.black.h}`,
    `hm:${pos.halfmoves}`,
    `fm:${pos.fullmoves}`,
  ];
  return parts.join('|');
}

const mvStr = (m) => `${m.from}->${m.to}${m.promotion ? '=' + m.promotion[0] : ''}`;

// =====================================================================================
// GATE A — invariant _make/_unmake (chaque coup légal de chaque FEN)
// =====================================================================================
console.log('=== GATE A : invariant _make/_unmake (Board bit-identique + scalaires) ===');
let aChecked = 0, aDiv = 0;
for (const { label, fen } of ALL_INVARIANT_FENS) {
  const fb = new FastBoard(fen);
  const before = ser(fb);
  const moves = fb.legalMoves();
  let localDiv = 0;
  for (const m of moves) {
    fb._make(m);
    fb._unmake();
    const after = ser(fb);
    aChecked++;
    if (after !== before) {
      localDiv++; aDiv++;
      err(`[A:${label}] coup ${mvStr(m)} : ÉTAT divergent après _make+_unmake`);
      if (localDiv === 1) {
        // diff ciblé : montre le 1er champ divergent
        const bp = before.split('|'), ap = after.split('|');
        for (let i = 0; i < bp.length; i++) {
          if (bp[i] !== ap[i]) { console.error(`      champ[${i}] before=${bp[i]} after=${ap[i]}`); break; }
        }
      }
    }
    if (localDiv > 3) { console.error(`      ... (${label} : >3 divergences, on coupe)`); break; }
  }
  if (localDiv === 0) console.log(`  ✓ ${label} : ${moves.length} coups × (_make+_unmake) → 0 divergence`);
}
console.log(`  → ${aChecked} coups vérifiés, ${aDiv} divergence(s).`);

// =====================================================================================
// GATE A2 — make/unmake imbriqué profondeur 2 (roque + ep imbriqués)
// =====================================================================================
console.log('\n--- GATE A2 : make/unmake imbriqué profondeur 2 (Kiwipete + roque-des-deux-côtés) ---');
let a2Checked = 0, a2Div = 0;
for (const fen of [
  'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
  'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', // roque des deux côtés dispo
]) {
  const fb = new FastBoard(fen);
  const before = ser(fb);
  for (const m1 of fb.legalMoves()) {
    fb._make(m1);
    const mid = ser(fb);
    for (const m2 of fb.legalMoves()) {
      fb._make(m2);
      fb._unmake();
      a2Checked++;
      if (ser(fb) !== mid) {
        a2Div++; err(`[A2] ${mvStr(m1)} puis ${mvStr(m2)} : état mid corrompu`);
      }
    }
    fb._unmake();
  }
  if (ser(fb) !== before) {
    a2Div++; err(`[A2] ${fen} : état initial NON restauré après remontée complète`);
  }
}
console.log(`  → ${a2Checked} paires (make×2/unmake×2) vérifiées, ${a2Div} divergence(s).`);

// =====================================================================================
console.log('');
if (fail === 0) {
  console.log('✅✅ INVARIANT FastBoard : _make/_unmake bit-identique (A + A2) — 0 divergence');
} else {
  console.error(`❌ INVARIANT FastBoard : ${fail} divergence(s) (A=${aDiv}, A2=${a2Div})`);
  process.exit(1);
}
