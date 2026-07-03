// Test end-to-end de l'adaptateur NN : plans JS -> vrai modèle gen-027 -> value + policy.
// Si l'encodage était faux, la value/policy seraient absurdes -> re-valide tout le pipeline.

import { loadNet } from '../src/nn-node.mjs';
import { GameState } from '../src/game-state.mjs';
import { encodePlanes } from '../src/encoding/plane-encoding.mjs';
import { encode } from '../src/encoding/move-encoding.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };

const net = await loadNet('./gen-027-promoted.onnx');

function priors(gs, policy) {
  const side = gs.sideToMove();
  const ms = gs.legalMoves().map((m) => ({ san: m.san, logit: policy[encode(m, side)] }));
  const mx = Math.max(...ms.map((x) => x.logit));
  let s = 0; ms.forEach((x) => { x.e = Math.exp(x.logit - mx); s += x.e; });
  ms.forEach((x) => { x.p = x.e / s; });
  ms.sort((a, b) => b.p - a.p);
  return ms;
}

// 1. Startpos : value ≈ 0, top move = ouverture sensée
{
  const gs = new GameState();
  const { policy, value } = await net.evaluate(encodePlanes(gs.toEncoderState()));
  const top = priors(gs, policy);
  console.log(`  startpos: value=${value.toFixed(4)} | top5: ${top.slice(0, 5).map((x) => `${x.san}=${(x.p * 100).toFixed(1)}%`).join(' ')}`);
  check(Math.abs(value) < 0.4, `value startpos ≈ 0 (${value.toFixed(3)})`);
  check(['e4', 'd4', 'Nf3', 'c4', 'g3', 'Nc3', 'b3'].includes(top[0].san), `top move "${top[0].san}" = ouverture sensée`);
  const sum = top.reduce((a, x) => a + x.p, 0);
  check(Math.abs(sum - 1) < 1e-5, `priors somment à 1 (${sum.toFixed(6)})`);
}

// 2. Position où les Blancs gagnent une dame -> value nettement positive (trait Blancs)
{
  const gs = new GameState('4k3/8/8/8/8/8/8/3QK3 w - - 0 1'); // Blancs ont une dame de plus
  const { value } = await net.evaluate(encodePlanes(gs.toEncoderState()));
  console.log(`  K+Q vs K (trait Blancs): value=${value.toFixed(4)}`);
  check(value > 0.5, `value très positive quand Blancs ont +dame (${value.toFixed(3)})`);
}

// 3. Même matériel mais trait aux Noirs perdants -> value négative (POV side-to-move)
{
  const gs = new GameState('3qk3/8/8/8/8/8/8/4K3 b - - 0 1'); // Noirs ont +dame, trait Noirs
  const { value } = await net.evaluate(encodePlanes(gs.toEncoderState()));
  console.log(`  trait Noirs avec +dame: value=${value.toFixed(4)}`);
  check(value > 0.5, `value positive pour le camp au trait qui a +dame (${value.toFixed(3)})`);
}

if (fail === 0) console.log('\n✅ nn self-test : pipeline plans JS -> gen-027 COHÉRENT');
else { console.error(`\n❌ ${fail} échec(s)`); process.exit(1); }
