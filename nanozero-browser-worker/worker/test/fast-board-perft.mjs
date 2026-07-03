// fast-board-perft.mjs — Oracle perft sur GameState (legalMoves + applyMove + undoMove).
// Gate anti-poison : si le triplet movegen/apply/undo est incohérent, le compte diverge.
//   - startpos d=4 = 197 281   (chessprogramming.org perft startpos)
//   - pos2  d=3 = 97 862       (chessprogramming.org Position 2 "Kiwipete")
// Pur-JS, self-contained (constantes d'échecs universelles). Aucun dump Java requis.
//
// Le perft descend via applyMove(move) puis remonte via undoMove(). Si undoMove ne
// restaure pas EXACTEMENT l'état (ep, roque, tour de roque, snapshots), les comptes
// des frères/parents divergent -> détection immédiate.

import { GameState } from '../src/game-state.mjs';
import { PERFT_CASES } from './fast-board-fens.mjs';

// Récursion : compte les nœuds-feuilles à profondeur `depth`.
// Joue chaque coup via son objet .cjs natif (legalMoves() le fournit) pour éviter le
// re-parse SAN ; applyMove/undoMove restent le chemin testé.
function perft(gs, depth) {
  if (depth === 0) return 1;
  const moves = gs.legalMoves();
  if (depth === 1) return moves.length; // micro-optim : pas besoin de descendre d'un cran
  let nodes = 0;
  for (const m of moves) {
    gs.applyMove(m);
    nodes += perft(gs, depth - 1);
    gs.undoMove();
  }
  return nodes;
}

let fail = 0;
console.log('=== PERFT (GameState : legalMoves + applyMove + undoMove) ===');
for (const c of PERFT_CASES) {
  console.log(`\n[${c.label}] ${c.fen}`);
  for (const dStr of Object.keys(c.expected)) {
    const d = Number(dStr);
    const exp = c.expected[d];
    const gs = new GameState(c.fen);
    const t0 = Date.now();
    const got = perft(gs, d);
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
  console.log('\n✅✅ PERFT : tous les comptes EXACTS (movegen/apply/undo cohérents)');
} else {
  console.error(`\n❌ PERFT : ${fail} compte(s) divergent(s) — le triplet baseline est FAUX`);
  process.exit(1);
}
