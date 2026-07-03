// cross-codec-parity.mjs — écrit une fixture pour la parité bi-langage (Story A.1).
// Produit : parity-fixture.bin (encode binaire JS) + parity-fixture.json (représentation
// "chemin JSON" = input_planes_b64 / policy_target_b64 / outcome, comme PositionPayload).
// cross-codec-parity.py vérifie que decode(bin) côté serveur == ces bytes denses.
import { writeFileSync } from 'node:fs';
import { encode, PLANES_LEN, POLICY_LEN } from '../worker/src/submit-codec.mjs';

function sample(seed) {
  const planes = new Float32Array(PLANES_LEN);
  for (let i = 0; i < PLANES_LEN; i++) planes[i] = ((i + seed) % 5 === 0) ? 1 : 0;
  planes[113 * 64] = 0.123; // plan constant fractionnaire
  const policy = new Float32Array(POLICY_LEN);
  [[3, 0.5], [17, 0.25], [4671, 0.2], [(seed * 7) % POLICY_LEN, 0.05]]
    .forEach(([i, v]) => { policy[i] = v; });
  return { planes, policy, outcome: seed % 2 === 0 ? 1.0 : -1.0 };
}

const samples = [sample(0), sample(1)];
const bin = Buffer.from(encode(samples));

// représentation "chemin JSON" : bytes denses f32 LE en base64 (comme PositionPayload Java)
const jsonPath = samples.map((s) => ({
  input_planes_b64: Buffer.from(s.planes.buffer, s.planes.byteOffset, PLANES_LEN * 4).toString('base64'),
  policy_target_b64: Buffer.from(s.policy.buffer, s.policy.byteOffset, POLICY_LEN * 4).toString('base64'),
  outcome: s.outcome,
}));

writeFileSync(new URL('./parity-fixture.bin', import.meta.url), bin);
writeFileSync(new URL('./parity-fixture.json', import.meta.url), JSON.stringify(jsonPath));
console.log('fixture écrite : parity-fixture.bin (' + bin.length + ' o) + parity-fixture.json (' + jsonPath.length + ' positions)');
