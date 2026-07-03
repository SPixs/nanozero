// Stress : beaucoup de parties FULL via runRollingPool avec reuse (waveSize>1). Chasse les
// erreurs/terminaisons anormales (throw C4, reset prématuré) que le slot-catch masquerait.
import { loadNet } from '../src/nn-node.mjs';
import { runRollingPool } from '../src/self-play.mjs';

const net = await loadNet('./gen-027-promoted.onnx');
const { results } = await runRollingPool(6, net, {
  waveSize: 8, sims: 64, maxPlies: 160, tempMoves: 15, targetGames: 24,
});
let errors = 0, natural = 0, maxplies = 0, short = 0; const plies = [];
for (const r of results) {
  if (r.error) { errors++; console.log('  ✗ ERREUR partie:', String(r.error).split('\n')[0]); }
  else { if (r.over) natural++; else maxplies++; plies.push(r.plies); if (r.plies < 10) short++; }
}
plies.sort((a, b) => a - b);
console.log(`\ngames=${results.length} | errors=${errors} | fin-naturelle=${natural} | maxplies=${maxplies} | courtes(<10)=${short}`);
console.log('plies (triés):', plies.join(','));
console.log(errors === 0 ? '\n✅ aucune erreur reuse sur 24 parties full' : `\n❌ ${errors} parties en erreur (bug reuse ?)`);
process.exit(errors === 0 ? 0 : 1);
