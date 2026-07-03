// legal-moves-equality-fb.mjs — Story 4 GATE 2 : égalité DIRECTE
//   FastBoard.legalMoves()  ==  GameState.legalMoves()
// sur toutes les FENs du filet, en comparant l'ENSEMBLE des coups au format NanoZero
// {from,to,type,promo} ET l'index policy encode({from,to,type,promo}, side).
//
// Couvre les 3 pièges Story 4 :
//   - CASTLING (détection file-diff===2) — Kiwipete / roque-des-deux-côtés ;
//   - EN_PASSANT (case ep-convention-Java, PAS pos.epSquare) — ep réel d6 jouable ;
//   - PROMOTION × 4 enfants distincts N/B/R/Q (promo 0/1/2/3) — sous-promo n3k3/1P6/...
//
// GameState (chess.js) est l'ORACLE : ses {from,to,type,promo} viennent de m.flags
// (k/q→CASTLING, e→EN_PASSANT, promotion→PROMOTION). FastBoard doit produire le MÊME
// ensemble. La clé de comparaison = `from-to-type-promo` (canonique, indépendant de
// l'ordre d'énumération). L'index `encode()` doit lui aussi coïncider coup à coup.
//
// node test/legal-moves-equality-fb.mjs

import { FastBoard } from '../src/fast-board.mjs';
import { GameState } from '../src/game-state.mjs';
import { encode, NORMAL, CASTLING, EN_PASSANT, PROMOTION } from '../src/encoding/move-encoding.mjs';
import { ALL_INVARIANT_FENS } from './fast-board-fens.mjs';

let fail = 0;
const err = (msg) => { fail++; console.error(`  ✗ ${msg}`); };

const TYPE_NAME = { [NORMAL]: 'NORMAL', [CASTLING]: 'CASTLING', [EN_PASSANT]: 'EN_PASSANT', [PROMOTION]: 'PROMOTION' };
const algOf = (sq) => String.fromCharCode(97 + (sq & 7)) + ((sq >> 3) + 1);
const keyOf = (m) => `${m.from}-${m.to}-${m.type}-${m.promo}`;
const human = (m) => `${algOf(m.from)}${algOf(m.to)} ${TYPE_NAME[m.type]}/p${m.promo}`;

// Construit Map(key → {move, idx}) à partir d'une liste de coups {from,to,type,promo} + side.
function moveMap(moves, side) {
  const map = new Map();
  for (const m of moves) {
    const key = keyOf(m);
    const idx = encode({ from: m.from, to: m.to, type: m.type, promo: m.promo }, side);
    map.set(key, { move: m, idx });
  }
  return map;
}

console.log('=== Story 4 GATE 2 : legalMoves() FastBoard == GameState ({from,to,type,promo}+index) ===');
let totalMoves = 0;
for (const { label, fen } of ALL_INVARIANT_FENS) {
  const fb = new FastBoard(fen);
  const gs = new GameState(fen);
  const side = fb.sideToMove();
  if (side !== gs.sideToMove()) { err(`${label} : sideToMove FastBoard≠GameState`); continue; }

  const fbMap = moveMap(fb.legalMoves(), side);
  const gsMap = moveMap(gs.legalMoves(), side);

  const onlyFb = [...fbMap.keys()].filter((k) => !gsMap.has(k));
  const onlyGs = [...gsMap.keys()].filter((k) => !fbMap.has(k));
  let idxBad = null;
  for (const [k, v] of gsMap) {
    if (fbMap.has(k) && fbMap.get(k).idx !== v.idx) {
      idxBad = { mv: human(v.move), gs: v.idx, fb: fbMap.get(k).idx };
      break;
    }
  }

  if (onlyFb.length === 0 && onlyGs.length === 0 && !idxBad) {
    totalMoves += gsMap.size;
    console.log(`  ✓ ${label} : ${gsMap.size} coups identiques (type/promo + index policy)`);
  } else {
    const fmt = (keys, src) => keys.map((k) => human(src.get(k).move)).join(', ');
    err(`${label} : onlyFastBoard=[${fmt(onlyFb, fbMap)}] onlyGameState=[${fmt(onlyGs, gsMap)}] idxBad=${JSON.stringify(idxBad)}`);
  }
}

// ===========================================================================
// FOCUS sous-promotion : les 4 enfants N/B/R/Q DOIVENT exister et avoir 4 index distincts.
// (3 sous-promo plans 64-72 + 1 promo-dame queen-style → 4 index différents par case d'arrivée.)
// ===========================================================================
console.log('\n=== FOCUS sous-promotion : n3k3/1P6/... → 4 coups promo N/B/R/Q distincts ===');
{
  const fen = 'n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1'; // b7→b8 (push) + b7xa8 (capture) — Cf. cross-validate FEN #4
  const fb = new FastBoard(fen);
  const side = fb.sideToMove();
  // groupe les coups de promotion par (from,to) → on attend 4 promo (0,1,2,3) par cible.
  const byTarget = new Map();
  for (const m of fb.legalMoves()) {
    if (m.type !== PROMOTION) continue;
    const tk = `${m.from}->${m.to}`;
    if (!byTarget.has(tk)) byTarget.set(tk, []);
    byTarget.get(tk).push(m);
  }
  if (byTarget.size === 0) err('aucun coup PROMOTION détecté sur la FEN sous-promo');
  for (const [tk, group] of byTarget) {
    const promos = group.map((m) => m.promo).sort((a, b) => a - b);
    const idxs = group.map((m) => encode(m, side));
    const distinct = new Set(idxs);
    const promosOk = promos.length === 4 && promos.every((p, i) => p === i); // [0,1,2,3]
    const idxOk = distinct.size === 4;
    if (promosOk && idxOk) {
      console.log(`  ✓ ${algOf(group[0].from)}${algOf(group[0].to)} : 4 enfants promo [N,B,R,Q] → 4 index distincts {${idxs.join(',')}}`);
    } else {
      err(`${tk} : promos=[${promos}] (attendu 0,1,2,3) index=[${idxs}] distincts=${distinct.size}`);
    }
  }
  // Croise avec GameState : même ensemble de promotions.
  const gs = new GameState(fen);
  const fbPromo = new Set(fb.legalMoves().filter((m) => m.type === PROMOTION).map(keyOf));
  const gsPromo = new Set(gs.legalMoves().filter((m) => m.type === PROMOTION).map(keyOf));
  const sameSet = fbPromo.size === gsPromo.size && [...fbPromo].every((k) => gsPromo.has(k));
  if (sameSet) console.log(`  ✓ ensemble PROMOTION FastBoard == GameState (${fbPromo.size} coups)`);
  else err(`ensemble PROMOTION divergent : FastBoard=${fbPromo.size} GameState=${gsPromo.size}`);
}

// ===========================================================================
// FOCUS en passant RÉEL : ep d6 jouable → exactement 1 coup EN_PASSANT (e5xd6).
// ===========================================================================
console.log('\n=== FOCUS en passant réel : ...3pP3...w KQkq d6 → 1 coup EN_PASSANT (e5d6) ===');
{
  const fen = 'rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3';
  const fb = new FastBoard(fen);
  const gs = new GameState(fen);
  const fbEp = fb.legalMoves().filter((m) => m.type === EN_PASSANT);
  const gsEp = gs.legalMoves().filter((m) => m.type === EN_PASSANT);
  const fbSet = new Set(fbEp.map(keyOf));
  const gsSet = new Set(gsEp.map(keyOf));
  const sameSet = fbSet.size === gsSet.size && [...fbSet].every((k) => gsSet.has(k));
  if (fbEp.length === 1 && sameSet) {
    console.log(`  ✓ ${fbEp.map(human).join(', ')} classé EN_PASSANT (== GameState, via _epFile non pos.epSquare)`);
  } else {
    err(`ep réel : FastBoard=[${fbEp.map(human)}] GameState=[${gsEp.map(human)}]`);
  }
}

// ===========================================================================
// FOCUS roque (file-diff===2) : Kiwipete → O-O et O-O-O blancs classés CASTLING.
// ===========================================================================
console.log('\n=== FOCUS roque (file-diff===2) : Kiwipete → 2 coups CASTLING (e1g1, e1c1) ===');
{
  const fen = 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1';
  const fb = new FastBoard(fen);
  const gs = new GameState(fen);
  const fbCa = fb.legalMoves().filter((m) => m.type === CASTLING);
  const gsCa = gs.legalMoves().filter((m) => m.type === CASTLING);
  const fbSet = new Set(fbCa.map(keyOf));
  const gsSet = new Set(gsCa.map(keyOf));
  const sameSet = fbSet.size === gsSet.size && [...fbSet].every((k) => gsSet.has(k));
  // Vérifie aussi que le `to` est bien la case du ROI (g1=6 / c1=2), file-diff vs e1(4)===2.
  const filesOk = fbCa.every((m) => Math.abs((m.to & 7) - (m.from & 7)) === 2);
  if (fbCa.length === 2 && sameSet && filesOk) {
    console.log(`  ✓ ${fbCa.map(human).join(', ')} classés CASTLING (to=case roi, file-diff=2, == GameState)`);
  } else {
    err(`roque : FastBoard=[${fbCa.map(human)}] GameState=[${gsCa.map(human)}] filesOk=${filesOk}`);
  }
}

if (fail === 0) {
  console.log(`\n✅✅ legalMoves() FastBoard == GameState : ENSEMBLES {from,to,type,promo} + index IDENTIQUES`);
  console.log(`   ${ALL_INVARIANT_FENS.length} FENs · ${totalMoves} coups · sous-promo×4 · ep réel · roque file-diff`);
} else {
  console.error(`\n❌ ${fail} divergence(s) legalMoves() FastBoard↔GameState`);
  process.exit(1);
}
