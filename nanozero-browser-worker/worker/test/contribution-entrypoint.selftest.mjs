// contribution-entrypoint.selftest.mjs — A4-D2 : runContribution bout-en-bout (node ORT + modèle réel).
// Valide la GLUE : getChampionModel(bytes) → session ORT DEPUIS bytes → NeuralNet → pool concurrent
// → submit. Seul le WebGPU diffère en navigateur (ici EP CPU node). Lancer depuis worker/.
import * as ort from 'onnxruntime-node';
import { readFileSync } from 'node:fs';
import { runContribution } from '../src/browser-api.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };

const modelBytes = new Uint8Array(readFileSync(new URL('../gen-027-promoted.onnx', import.meta.url)));

console.log('A) garde-fous : ort / client absents → erreur claire');
{
  let threwOrt = false, threwClient = false;
  try { await runContribution(null, { client: {} }); } catch { threwOrt = true; }
  try { await runContribution(ort, {}); } catch { threwClient = true; }
  check(threwOrt, 'A1 ort absent → throw');
  check(threwClient, 'A2 client absent → throw');
}

console.log('B) contribution bout-en-bout : modèle (bytes) → session → pool → submit');
{
  let claimed = 0;
  const submitted = [];
  const progress = [];
  let gotBytes = null;
  const client = {
    // getChampionModel rend les BYTES réels (comme le vrai client après vérif sha256)
    getChampionModel: async () => { gotBytes = modelBytes; return { version: 27, sha256: 'a'.repeat(64), bytes: modelBytes }; },
    claim: async () => { claimed += 1; return { job_id: `j${claimed}`, model_version: 27, num_sims: 6 }; },
    submitGame: async (jobId, samples) => { submitted.push({ jobId, n: samples.length }); return { positions_stored: samples.length }; },
  };

  const stats = await runContribution(ort, {
    client,
    poolSize: 2,
    sessionOptions: { graphOptimizationLevel: 'all' }, // node CPU (pas de webgpu)
    gameOpts: { sims: 6, tempMoves: 0, maxPlies: 8 },   // parties courtes → test rapide
    onProgress: (p) => progress.push(p),
    shouldStop: () => submitted.length >= 2,
  });

  check(gotBytes === modelBytes, 'B1 session créée DEPUIS les bytes du champion (pas une URL)');
  check(submitted.length >= 2, `B2 parties soumises via le vrai net+pool (${submitted.length})`);
  check(submitted.every((s) => s.n > 0), 'B3 chaque partie a des samples (planes/policy/value réels)');
  check(progress.length >= 2 && progress.every((p) => p.modelVersion === 27 && p.contributed > 0),
    'B4 onProgress émis par submit (contributed, modelVersion)');
  check(stats.modelVersion === 27 && stats.maxBatch >= 1 && !stats.aborted, 'B5 stats cohérentes (modelVersion, batch, pas d\'abort)');
}

console.log(fail === 0 ? '\nOK — contribution-entrypoint selftest vert' : `\nFAIL — ${fail} check(s)`);
process.exit(fail ? 1 : 0);
