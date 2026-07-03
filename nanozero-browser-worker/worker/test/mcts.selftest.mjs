// Test du MCTS PUCT avec le vrai gen-027 : ouverture sensée, mat-en-1, mini self-play.

import { loadNet } from '../src/nn-node.mjs';
import { GameState } from '../src/game-state.mjs';
import { search, selectMove, visitPolicy } from '../src/mcts.mjs';
import { unpackMove } from '../src/encoding/move-encoding.mjs';

// Les coups d'arbre sont désormais des INT compacts (packMove). Décodage en UCI pour les asserts.
const A = (s) => 'abcdefgh'[s & 7] + ((s >> 3) + 1);
const uciOf = (p) => { const u = unpackMove(p); return A(u.from) + A(u.to); };

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };
const net = await loadNet('./gen-027-promoted.onnx');

// 1. Startpos, 400 sims sans bruit -> coup le plus visité sensé, Q≈0, gs restauré
{
  const gs = new GameState();
  const root = await search(gs, net, 400, { addNoise: false });
  const side = gs.sideToMove();
  const ranked = root.children.map((ch) => ({ uci: uciOf(ch.move), N: ch.node.N })).sort((a, b) => b.N - a.N);
  console.log(`  startpos 400 sims: ${ranked.slice(0, 4).map((r) => `${r.uci}=${r.N}`).join(' ')} | rootQ=${root.Q.toFixed(3)}`);
  check(['d2d4', 'g1f3', 'e2e4', 'c2c4'].includes(ranked[0].uci), `coup le + visité "${ranked[0].uci}" = ouverture sensée`);
  check(Math.abs(root.Q) < 0.3, `rootQ ≈ 0 (${root.Q.toFixed(3)})`);
  let s = 0; for (const v of visitPolicy(root, side)) s += v;
  check(Math.abs(s - 1) < 1e-5, 'visitPolicy somme à 1');
  check(gs.legalMoves().length === 20, 'gs restauré après search (descente/remontée propre)');
}

// 2. Mat en 1 (back-rank) : MCTS doit choisir Ra8#, rootQ gagnant
{
  const gs = new GameState('6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1');
  const root = await search(gs, net, 300, { addNoise: false });
  const best = selectMove(root, 0);
  console.log(`  mat-en-1: coup choisi = ${uciOf(best)} | rootQ=${root.Q.toFixed(3)}`);
  check(uciOf(best) === 'a1a8', `trouve le mat (a joué ${uciOf(best)})`);
  check(root.Q > 0.6, `rootQ gagnant (${root.Q.toFixed(3)})`);
}

// 3. Mini self-play : 6 plies, progression + cohérence de l'historique
{
  const gs = new GameState();
  let plies = 0;
  for (; plies < 6 && !gs.isGameOver(); plies++) {
    const root = await search(gs, net, 100, { addNoise: true });
    const mv = selectMove(root, plies < 30 ? 1.0 : 0.0);
    gs.applyMove(mv);
  }
  console.log(`  self-play 6 plies: fen=${gs.fen()}`);
  check(plies === 6, '6 plies self-play joués');
  // Story 5 : la pile de snapshots vit dans le FastBoard sous-jacent (gs._fb), même propriété.
  check(gs._fb.snapshots.length === 7, 'historique cohérent (1 + 6 snapshots)');
}

if (fail === 0) console.log('\n✅ mcts self-test : MCTS PUCT fonctionnel');
else { console.error(`\n❌ ${fail} échec(s)`); process.exit(1); }
