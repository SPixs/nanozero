// plane-encoding.mjs
// Port fidèle de l'encodage des 119 plans d'entrée NN (SPEC-board §7).
//   119 = 8 timestamps × 14 plans + 7 plans constants
//   timestamp (14) : 0-5 = P1 (PNBRQK), 6-11 = P2 (PNBRQK), 12 = rép≥1, 13 = rép≥2
//   constants : 112 color, 113 fullmove, 114-117 castling (P1KS,P1QS,P2KS,P2QS), 118 halfmove
//   P1 = côté au trait COURANT (constant sur les 8 timestamps).
//   Flip vertical (reverseBytes) sur chaque bitboard si side-to-move == BLACK.
//   Layout : planes[p*64 + sq], sq = rank*8+file (LSB=a1=0).
// ⚠️ Toute divergence vs Java désaligne l'entrée NN → données empoisonnées.

import { WHITE, BLACK } from './move-encoding.mjs';

export const NUM_PLANES = 119;
export const PLANE_SIZE = 64;
export const TENSOR_SIZE = NUM_PLANES * PLANE_SIZE; // 7616

// Long.reverseBytes : inverse l'ordre des 8 octets (= flip vertical de l'échiquier).
export function reverseBytes(bb) {
  let r = 0n;
  for (let i = 0; i < 8; i++) r |= ((bb >> BigInt(i * 8)) & 0xffn) << BigInt((7 - i) * 8);
  return r;
}

function bitboardToPlane(bb, planes, off) {
  // Optim : le buffer est déjà zéro-init ; on ne pose QUE les bits set (plans pièces creux).
  // BigInt -> 2× uint32, itération des bits set en arithmétique 32-bit (rapide, pas de BigInt
  // par cellule). Reste bit-identique au scalaire Java (mêmes positions de bits).
  let lo = Number(bb & 0xffffffffn) >>> 0;
  let hi = Number((bb >> 32n) & 0xffffffffn) >>> 0;
  while (lo !== 0) { const b = lo & -lo; planes[off + (31 - Math.clz32(b))] = 1; lo ^= b; }
  while (hi !== 0) { const b = hi & -hi; planes[off + 32 + (31 - Math.clz32(b))] = 1; hi ^= b; }
}

function fillPlane(planes, off, val) {
  for (let sq = 0; sq < 64; sq++) planes[off + sq] = val;
}

/**
 * Encode l'état (position + historique) en tenseur 119×8×8 (Float32Array, NCHW).
 * @param {{
 *   sideToMove:number,
 *   castling:{wk:boolean,wq:boolean,bk:boolean,bq:boolean},  // droits absolus W/B KS/QS
 *   fullmoveNumber:number, halfmoveClock:number,
 *   history: Array<{white:bigint[6], black:bigint[6], rep1:boolean, rep2:boolean}>  // t=0 courant .. t=7
 * }} state
 * @returns {Float32Array} longueur 7616
 */
export function encodePlanes(state) {
  const { sideToMove, castling, fullmoveNumber, halfmoveClock, history } = state;
  const planes = new Float32Array(TENSOR_SIZE);
  const p1White = sideToMove === WHITE;
  const flip = sideToMove === BLACK;
  const orient = (bb) => (flip ? reverseBytes(bb) : bb);

  for (let t = 0; t < 8; t++) {
    if (t >= history.length) continue; // historique insuffisant -> 14 plans à 0.0
    const snap = history[t];
    const base = t * 14 * PLANE_SIZE;
    const p1 = p1White ? snap.white : snap.black;
    const p2 = p1White ? snap.black : snap.white;
    for (let pt = 0; pt < 6; pt++) bitboardToPlane(orient(p1[pt]), planes, base + pt * PLANE_SIZE);
    for (let pt = 0; pt < 6; pt++) bitboardToPlane(orient(p2[pt]), planes, base + (6 + pt) * PLANE_SIZE);
    if (snap.rep1) fillPlane(planes, base + 12 * PLANE_SIZE, 1.0);
    if (snap.rep2) fillPlane(planes, base + 13 * PLANE_SIZE, 1.0);
  }

  // Plans constants 112-118
  if (p1White) fillPlane(planes, 112 * PLANE_SIZE, 1.0);             // 112 color (P1==WHITE)
  fillPlane(planes, 113 * PLANE_SIZE, Math.min(fullmoveNumber, 99) / 99); // 113 fullmove
  const p1ks = p1White ? castling.wk : castling.bk;
  const p1qs = p1White ? castling.wq : castling.bq;
  const p2ks = p1White ? castling.bk : castling.wk;
  const p2qs = p1White ? castling.bq : castling.wq;
  if (p1ks) fillPlane(planes, 114 * PLANE_SIZE, 1.0);  // 114 P1 kingside
  if (p1qs) fillPlane(planes, 115 * PLANE_SIZE, 1.0);  // 115 P1 queenside
  if (p2ks) fillPlane(planes, 116 * PLANE_SIZE, 1.0);  // 116 P2 kingside
  if (p2qs) fillPlane(planes, 117 * PLANE_SIZE, 1.0);  // 117 P2 queenside
  fillPlane(planes, 118 * PLANE_SIZE, halfmoveClock / 100); // 118 no-progress
  return planes;
}
