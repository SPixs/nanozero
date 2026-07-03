// Test P3 : N parties concurrentes batchées. Vérifie que le batching opère (avgBatch ~ N),
// que les samples sont valides (policy normalisée, value ∈ {-1,0,1}).

import { loadNet } from '../src/nn-node.mjs';
import { runParallelGames } from '../src/self-play.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };
const net = await loadNet('./gen-027-promoted.onnx');

const N = 8;
const t0 = Date.now();
const { results, stats, avgBatch } = await runParallelGames(N, net, { sims: 30, maxPlies: 10, maxBatch: 64 });
const dt = (Date.now() - t0) / 1000;
const totalSamples = results.reduce((a, r) => a + r.samples.length, 0);

console.log(`  ${N} parties || : ${stats.flushes} batches, ${stats.positions} évals, avgBatch=${avgBatch.toFixed(1)}, maxBatch=${stats.maxBatch}`);
console.log(`  ${totalSamples} samples en ${dt.toFixed(1)}s = ${(stats.positions / dt).toFixed(0)} évals/s (batché)`);

check(avgBatch > N * 0.5, `batching effectif : avgBatch ${avgBatch.toFixed(1)} (N=${N})`);
check(stats.maxBatch === N || stats.maxBatch >= N - 1, `batch atteint ~N (max ${stats.maxBatch})`);
check(totalSamples === N * 10, `${N}×10 = ${N * 10} samples produits`);
check(results.every((r) => r.samples.every((s) => [-1, 0, 1].includes(s.value))), 'value ∈ {-1,0,1}');
check(results.every((r) => r.samples.every((s) => {
  let sum = 0; for (const v of s.policy) sum += v;
  return Math.abs(sum - 1) < 1e-4;
})), 'policy targets normalisées (somme=1)');
check(results.every((r) => r.samples.every((s) => s.planes.length === 119 * 64)), 'planes 7616 par sample');

// comparaison débit : le même nombre d'évals en batch=1 prendrait ~N× plus de flushes
console.log(`  → batché en ${stats.flushes} passes NN au lieu de ${stats.positions} (×${(stats.positions / stats.flushes).toFixed(1)} moins d'appels)`);

if (fail === 0) console.log('\n✅ self-play self-test : batching N parties concurrentes OK');
else { console.error(`\n❌ ${fail} échec(s)`); process.exit(1); }
