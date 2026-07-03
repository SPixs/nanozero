// search-golden.selftest — PARITÉ : verrou de non-régression sur search() déterministe.
// Signature = root.N + hash de la distribution de visites (via encode -> index policy) sur 3 positions
// fixes (startpos / kiwipete / promo), 200 sims, addNoise=false. Capturée AVANT le refactor "coup
// compact int" (children: objet move -> packMove). Réussir ce test prouve que la recherche est restée
// BIT-À-BIT identique malgré le changement de représentation des coups dans l'arbre.
import { loadNet } from '../src/nn-node.mjs';
import { GameState } from '../src/game-state.mjs';
import { search, visitPolicy } from '../src/mcts.mjs';

// Oracle figé (capture AVANT wiring int, re-vérifié IDENTIQUE APRÈS). NE PAS éditer sans recapturer.
const GOLDEN = '200:2159360954|200:2768439701|200:4207953274|';
const POS = [
  'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',           // startpos
  'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1', // kiwipete
  'n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1',                                    // promo (pion b7)
];
const SIMS = 200;

const net = await loadNet('./gen-027-promoted.onnx');
let sig = '';
for (const fen of POS) {
  const gs = new GameState(fen);
  const side = gs.sideToMove();
  const root = await search(gs, net, SIMS, { addNoise: false });
  const pi = visitPolicy(root, side);
  let h = 2166136261 >>> 0;
  for (let i = 0; i < pi.length; i++) {
    if (pi[i] !== 0) { h ^= i; h = Math.imul(h, 16777619) >>> 0; h ^= Math.round(pi[i] * 1e6); h = Math.imul(h, 16777619) >>> 0; }
  }
  sig += `${root.N}:${h >>> 0}|`;
}

if (sig === GOLDEN) {
  console.log(`  ✓ signature search() identique au golden\n\n✅ search-golden self-test : parité bit-à-bit (refactor coup compact int sans effet sur la recherche)`);
  process.exit(0);
} else {
  console.error(`  ✗ DIVERGENCE search()\n      golden = ${GOLDEN}\n      obtenu = ${sig}\n\n❌ search-golden self-test : la recherche a CHANGÉ`);
  process.exit(1);
}
