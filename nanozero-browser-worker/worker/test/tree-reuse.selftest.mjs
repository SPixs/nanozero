// tree-reuse.selftest.mjs — vérifie le TREE REUSE (parité self-play Java) :
//  A. search() séquentiel : reuse ajoute numSims sims NEUVES par-dessus les visites conservées
//     (root.N final == priorN + numSims → eff_sims croît, comme SearcherCore.runSearch + tryReroot).
//  B. searchBatched() (chemin worker réel) : reuse + invariant C4 (Σenfants.N == root.N−1) tenu.
//  C. Partie complète waveSize>1 (reuse à chaque coup) : finit sans throw, samples valides.

import { loadNet } from '../src/nn-node.mjs';
import { GameState } from '../src/game-state.mjs';
import { search, searchBatched, selectMove } from '../src/mcts.mjs';
import { runParallelGames } from '../src/self-play.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };
const net = await loadNet('./gen-027-promoted.onnx');
const SIMS = 100;

// A. search() séquentiel : reuse = sims NEUVES par-dessus le sous-arbre conservé.
{
  const gs = new GameState();
  const root1 = await search(gs, net, SIMS, { addNoise: false });
  check(root1.N === SIMS, `fresh: root.N == ${SIMS} (got ${root1.N})`);
  const mv = selectMove(root1, 0);
  const child = root1.children.find((c) => c.move === mv);
  const priorN = child.node.N;
  check(priorN > 0, `enfant joué déjà visité (priorN=${priorN})`);
  gs.applyMove(mv);
  const root2 = await search(gs, net, SIMS, { addNoise: false, root: child.node });
  check(root2 === child.node, 'reuse: root2 EST le sous-arbre conservé (même objet)');
  check(root2.N === priorN + SIMS, `reuse: root.N == priorN+${SIMS} == ${priorN + SIMS} (got ${root2.N})`);
  check(root2.N > SIMS, `eff_sims a CRÛ au-delà de ${SIMS} (got ${root2.N})`);
  check(root2.children.length === gs.legalMoves().length, 'reuse: enfants == coups légaux de la nouvelle position');
}

// B. searchBatched() : reuse + invariant anti-vloss C4 (throw si Σenfants.N != root.N−1).
{
  const gs = new GameState();
  const root1 = await searchBatched(gs, net, SIMS, { addNoise: false, waveSize: 8 });
  const mv = selectMove(root1, 0);
  const child = root1.children.find((c) => c.move === mv);
  const priorN = child.node.N;
  gs.applyMove(mv);
  let ok = true, root2 = null;
  try { root2 = await searchBatched(gs, net, SIMS, { addNoise: false, waveSize: 8, root: child.node }); }
  catch (e) { ok = false; console.error('  C4 throw: ' + e.message); }
  check(ok, 'searchBatched reuse: invariant C4 tenu (pas de throw vloss)');
  check(root2 && root2.N === priorN + SIMS, `batched reuse: root.N == priorN+${SIMS} (got ${root2 && root2.N})`);
  let sumN = 0; if (root2) for (const ch of root2.children) sumN += ch.node.N;
  check(root2 && Math.abs(sumN - (root2.N - 1)) < 1e-9, `Σenfants.N == root.N−1 (${sumN} vs ${root2 && root2.N - 1})`);
}

// C. Partie complète via le chemin worker réel (waveSize>1 → reuse chaque coup, C4 interne garde l'intégrité).
{
  const { results } = await runParallelGames(1, net, { waveSize: 8, sims: 60, maxPlies: 24, tempMoves: 8 });
  const r = results[0];
  check(!r.error, `partie reuse complète sans erreur${r.error ? ' — ' + r.error : ''}`);
  check(r.samples.length > 0 && r.samples.length === r.plies, `samples produits (${r.samples.length} == ${r.plies} plies)`);
  const valid = r.samples.every((s) => { let sum = 0; for (const v of s.policy) sum += v; return Math.abs(sum - 1) < 1e-4; });
  check(valid, 'toutes les policies des samples somment à 1');
}

console.log(fail === 0 ? '\n✅ tree-reuse self-test : reuse OK (eff_sims↑, C4 tenu, partie OK)' : `\n❌ ${fail} échec(s)`);
process.exit(fail === 0 ? 0 : 1);
