// mcts-pv.mjs — extraction des variantes principales (MultiPV) depuis un arbre MCTS (surface Analyse).
// Top-K enfants de la racine par nombre de visites ; pour chacun, la ligne principale (PV) = descente
// en suivant l'enfant le plus visité à chaque niveau. Valeur v du POV du joueur AU TRAIT à la racine
// (l'app oriente ensuite vers les Blancs + convertit en cp). N'altère pas durablement `gs` (applyMove
// puis undoMove en remontant).

import { packMove } from './encoding/move-encoding.mjs';

// Coup compact MCTS (int packMove) → coup légal enrichi (porte .san) de la position courante.
function legalFor(gs, moveInt) {
  const lm = gs.legalMoves();
  for (let i = 0; i < lm.length; i++) if (packMove(lm[i]) === moveInt) return lm[i];
  return null;
}

/**
 * @param {object} root  racine MCTS (children = [{move:int, P, node:{N,Q,children}}])
 * @param {number} k     nombre de variantes (ex. 3)
 * @param {object} gs    GameState positionné sur la racine (restauré en sortie)
 * @returns {Array<{v:number, wdl:number[]|null, n:number, pv:string[]}>}
 *   v = valeur POV trait racine ; wdl = [W,D,L] de l'enfant (POV du trait APRÈS le coup) ; pv = SAN
 */
export function topVariations(root, k, gs, { maxDepth = 10, minN = 2 } = {}) {
  if (!root || !root.children || !root.children.length) return [];
  const sorted = root.children.slice().sort((a, b) => b.node.N - a.node.N).slice(0, k);
  const out = [];
  for (const ch of sorted) {
    const v = -ch.node.Q; // valeur de jouer ch, POV du joueur au trait à la racine
    const pv = [];
    let w = ch, applied = 0, from = null, to = null;
    while (w && pv.length < maxDepth) {
      const lm = legalFor(gs, w.move);
      if (!lm) break;
      if (!pv.length && lm.cjs) { from = lm.cjs.from; to = lm.cjs.to; } // 1er coup → flèche
      pv.push(lm.san);
      gs.applyMove(w.move); applied++;
      const kids = w.node && w.node.children;
      if (!kids || !kids.length) break;
      let best = null;
      for (const c of kids) if (c.node.N >= minN && (!best || c.node.N > best.node.N)) best = c;
      if (!best) break;
      w = best;
    }
    for (let i = 0; i < applied; i++) gs.undoMove();
    out.push({ v, wdl: ch.node.wdl, n: ch.node.N, pv, from, to });
  }
  return out;
}
