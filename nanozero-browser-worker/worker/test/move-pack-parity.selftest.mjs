// move-pack-parity.selftest — HARDENING : prouve que packMove/unpack est LOSSLESS, sur TOUS les
// coups de toutes les positions de test (+ descente 2 plies → promotions ×4, roque, EP, normal).
//   (1) encode : encodePacked(packMove(m), side) === encode(m, side)   (index policy bit-à-bit)
//   (2) replay : appliquer m  ==  appliquer unpackForApply(packMove(m))  (FEN résultante identique)
//   (3) round-trip : unpackMove(packMove(m)) préserve from/to/(promo si promotion)
import { GameState } from '../src/game-state.mjs';
import {
  encode, packMove, unpackMove, encodePacked, unpackForApply,
  WHITE, PROMOTION, CASTLING, EN_PASSANT,
} from '../src/encoding/move-encoding.mjs';
import { ALL_INVARIANT_FENS } from './fast-board-fens.mjs';

let fail = 0, checked = 0, promos = 0, castles = 0, eps = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } };

function verifyMove(gs, m) {
  checked++;
  const side = gs.sideToMove();
  const p = packMove(m);
  // (1) encode lossless
  check(encodePacked(p, side) === encode(m, side), `encode mismatch ${m.uci} @ ${gs.fen()}`);
  // (3) round-trip from/to (+promo si promotion)
  const u = unpackMove(p);
  check(u.from === m.from && u.to === m.to, `round-trip from/to ${m.uci}`);
  if (m.promotion != null) { check(u.type === PROMOTION && u.promo === m.promo, `round-trip promo ${m.uci}`); promos++; }
  if (m.type === CASTLING) castles++;
  if (m.type === EN_PASSANT) eps++;
  // (2) replay lossless : objet original vs reconstruit depuis l'int -> même position (FEN complète)
  const a = new GameState(gs.fen()); a.applyMove(m);
  const b = new GameState(gs.fen()); b.applyMove(unpackForApply(p));
  check(a.fen() === b.fen(), `replay FEN mismatch ${m.uci}:\n      orig=${a.fen()}\n      int =${b.fen()}`);
}

function walk(gs, depth) {
  const moves = gs.legalMoves();
  for (const m of moves) verifyMove(gs, m);
  if (depth > 0) {
    for (let i = 0; i < moves.length; i++) { // descente complète d'1 ply (couvre tous les sous-arbres)
      const g = new GameState(gs.fen()); g.applyMove(moves[i]);
      if (!g.isGameOver()) walk(g, depth - 1);
    }
  }
}

for (const tc of ALL_INVARIANT_FENS) {
  const gs = new GameState(tc.fen);
  if (!gs.isGameOver()) walk(gs, 1);
}
console.log(`coups vérifiés: ${checked}  (promotions: ${promos}, roques: ${castles}, en-passant: ${eps})`);
check(promos > 0, 'au moins une promotion couverte (sinon le test ne prouve pas le cas promo)');
console.log(fail === 0 ? `\n✅ pack/unpack LOSSLESS (encode + replay + round-trip) sur ${checked} coups` : `\n❌ ${fail} échec(s)`);
process.exit(fail === 0 ? 0 : 1);
