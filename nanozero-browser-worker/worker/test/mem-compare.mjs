// Mesure le gain mémoire du refactor "coup compact int" dans les children du MCTS.
// Compare, à structure d'arbre IDENTIQUE, le coût d'un child {move,P,node} selon que `move` soit :
//   - l'OBJET FastBoard de legalMoves() (porte 3 getters paresseux san/cjs/uci = closures retenues)
//   - l'INT packMove (8 o).
// Heap mesuré avec --expose-gc (gc() avant chaque snapshot). Lancer : node --expose-gc test/mem-compare.mjs
import { GameState } from '../src/game-state.mjs';
import { packMove } from '../src/encoding/move-encoding.mjs';

const gc = global.gc || (() => {});
const KIWIPETE = 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'; // ~48 coups légaux
const gs = new GameState(KIWIPETE);
const EXPANSIONS = 4000; // ~4000 nœuds expansés -> ~190k children (ordre d'un gros arbre reuse)

function build(pack) {
  gc(); gc();
  const before = process.memoryUsage().heapUsed;
  const trees = [];
  for (let i = 0; i < EXPANSIONS; i++) {
    const moves = gs.legalMoves();
    // mime exactement expandNode : children = moves.map(m => ({move, P, node})).
    const children = moves.map((m) => ({ move: pack ? packMove(m) : m, P: 0.02, node: { N: 0, W: 0, Q: 0, children: null, P: 0 } }));
    trees.push(children);
  }
  gc(); gc();
  const after = process.memoryUsage().heapUsed;
  let count = 0; for (const c of trees) count += c.length;
  return { bytes: after - before, count, _keep: trees };
}

const obj = build(false);
const int = build(true);
const perObj = obj.bytes / obj.count;
const perInt = int.bytes / int.count;
const M = (b) => (b / 1048576).toFixed(1);

console.log(`children construits: ${obj.count} (obj) / ${int.count} (int)`);
console.log(`OBJET  (move FastBoard + 3 getters) : ${M(obj.bytes)} Mo  (~${Math.round(perObj)} o/child)`);
console.log(`INT    (packMove)                   : ${M(int.bytes)} Mo  (~${Math.round(perInt)} o/child)`);
console.log(`gain : ${(perObj / perInt).toFixed(1)}× ; économie ~${Math.round(perObj - perInt)} o/coup`);
// Projection sur le scénario prod qui déclenchait l'OOM Max : ~91k coups/arbre reuse × pool 32 parties.
const proj = ((perObj - perInt) * 91000 * 32) / 1073741824;
console.log(`→ projection arbre reuse ~91k coups × pool=32 : économie ~${proj.toFixed(2)} Go de heap renderer`);
