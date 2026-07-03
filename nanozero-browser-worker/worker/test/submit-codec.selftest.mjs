// submit-codec.selftest.mjs — round-trip + cas limites du codec binaire (Story A.1).
import { encode, decode, PLANES_LEN, POLICY_LEN } from '../src/submit-codec.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };

// sample déterministe (planes 0/1 + un plan constant fractionnaire ; policy creuse)
function sample(seed) {
  const planes = new Float32Array(PLANES_LEN);
  for (let i = 0; i < PLANES_LEN; i++) planes[i] = ((i + seed) % 7 === 0) ? 1 : 0;
  planes[113 * 64] = 0.42; // plan constant fractionnaire (fullmove)
  const policy = new Float32Array(POLICY_LEN);
  [[3, 0.5], [17, 0.25], [800, 0.2], [4671, 0.04], [(seed * 13) % POLICY_LEN, 0.01]]
    .forEach(([i, v]) => { policy[i] = v; });
  return { planes, policy, outcome: seed % 2 === 0 ? 1.0 : -1.0 };
}

console.log('A) round-trip encode→decode');
{
  const samples = [sample(0), sample(1), sample(2)];
  const back = decode(encode(samples));
  check(back.length === 3, 'nb positions conservé');
  let ok = true;
  for (let k = 0; k < 3; k++) {
    if (back[k].outcome !== samples[k].outcome) ok = false;
    for (let i = 0; i < PLANES_LEN; i++) if (back[k].planes[i] !== samples[k].planes[i]) { ok = false; break; }
    for (let i = 0; i < POLICY_LEN; i++) if (back[k].policy[i] !== samples[k].policy[i]) { ok = false; break; }
  }
  check(ok, 'planes/policy/outcome identiques après round-trip');
}

console.log('B) policy creuse (taille compacte)');
{
  const s = sample(5);
  const nnzExpected = s.policy.reduce((a, v) => a + (v !== 0 ? 1 : 0), 0);
  const buf = encode([s]);
  const dv = new DataView(buf);
  const nnz = dv.getUint16(4 + 4 + PLANES_LEN * 4, true);
  check(nnz === nnzExpected, `nnz encodé = ${nnz} (attendu ${nnzExpected})`);
  const expectedSize = 4 + 4 + PLANES_LEN * 4 + 2 + nnzExpected * 6;
  check(buf.byteLength === expectedSize, `taille = ${buf.byteLength} o (attendu ${expectedSize})`);
}

console.log('C) planes en vue (byteOffset != 0) → slice exact');
{
  const big = new Float32Array(PLANES_LEN + 10);
  for (let i = 0; i < PLANES_LEN; i++) big[i + 5] = (i % 3 === 0) ? 1 : 0;
  const view = big.subarray(5, 5 + PLANES_LEN);
  const s = { planes: view, policy: new Float32Array(POLICY_LEN), outcome: 0 };
  s.policy[10] = 1.0;
  const back = decode(encode([s]));
  let ok = true;
  for (let i = 0; i < PLANES_LEN; i++) if (back[0].planes[i] !== view[i]) { ok = false; break; }
  check(ok, 'slice exact d’une vue (byteOffset) correct');
}

console.log('D) encode valide ses entrées (lève à la source)');
{
  const throws = (fn) => { try { fn(); return false; } catch { return true; } };
  const good = { planes: new Float32Array(PLANES_LEN), policy: new Float32Array(POLICY_LEN), outcome: 0 };
  check(throws(() => encode([{ ...good, planes: new Float32Array(PLANES_LEN - 1) }])), 'planes trop courte → throw');
  check(throws(() => encode([{ ...good, policy: new Float32Array(POLICY_LEN + 1) }])), 'policy mauvaise taille → throw');
  check(throws(() => encode([{ ...good, outcome: NaN }])), 'outcome non-fini → throw');
  check(!throws(() => encode([good])), 'sample valide → OK');
}

console.log(fail === 0 ? '\nOK — submit-codec selftest vert' : `\nFAIL — ${fail} check(s)`);
process.exit(fail ? 1 : 0);
