// fast-board-perft-fb.mjs — Oracle perft sur FastBoard (legalMoves + _make + _unmake).
// Story 1 gate bloquant :
//   - startpos d=4 = 197 281   (chessprogramming.org perft startpos)
//   - pos2  d=3 = 97 862       (chessprogramming.org Position 2 "Kiwipete")
// La descente joue via _make(move) puis remonte via _unmake(). Si _unmake ne restaure pas
// EXACTEMENT l'état (roque tour, ep-case, promo, droits roque, epSquare, halfmoves), les
// comptes des frères/parents divergent → détection immédiate du bug make/unmake.

import { FastBoard } from '../src/fast-board.mjs';
import { PERFT_CASES } from './fast-board-fens.mjs';

function perft(fb, depth) {
  if (depth === 0) return 1;
  const moves = fb.legalMoves();
  if (depth === 1) return moves.length;
  let nodes = 0;
  for (const m of moves) {
    fb._make(m);
    nodes += perft(fb, depth - 1);
    fb._unmake();
  }
  return nodes;
}

let fail = 0;
console.log('=== PERFT (FastBoard : legalMoves + _make + _unmake) ===');
for (const c of PERFT_CASES) {
  console.log(`\n[${c.label}] ${c.fen}`);
  for (const dStr of Object.keys(c.expected)) {
    const d = Number(dStr);
    const exp = c.expected[d];
    const fb = new FastBoard(c.fen);
    const t0 = Date.now();
    const got = perft(fb, d);
    const ms = Date.now() - t0;
    const gate = d === c.gateDepth ? ' [GATE]' : '';
    if (got === exp) {
      console.log(`  ✓ d=${d} : ${got} nœuds (attendu ${exp}) — ${ms} ms${gate}`);
    } else {
      fail++;
      console.error(`  ✗ d=${d} : ${got} nœuds, ATTENDU ${exp} (Δ=${got - exp}) — ${ms} ms${gate}`);
    }
  }
}

if (fail === 0) {
  console.log('\n✅✅ PERFT FastBoard : tous les comptes EXACTS (make/unmake cohérents)');
} else {
  console.error(`\n❌ PERFT FastBoard : ${fail} compte(s) divergent(s) — bug _make/_unmake`);
  process.exit(1);
}
