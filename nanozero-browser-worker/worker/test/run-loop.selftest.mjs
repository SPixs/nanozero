// run-loop.selftest.mjs — Story A.4 : submitGame (round-trip gzip) + boucle (mocks).
import { JobserverClient } from '../src/jobserver-client.mjs';
import { decode, PLANES_LEN, POLICY_LEN } from '../src/submit-codec.mjs';
import { runContributionLoop } from '../src/run-loop.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };

async function gunzip(bytes) {
  const stream = new Response(bytes).body.pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function sample(seed) {
  const planes = new Float32Array(PLANES_LEN);
  for (let i = 0; i < PLANES_LEN; i++) planes[i] = ((i + seed) % 97 === 0) ? 1 : 0;
  const policy = new Float32Array(POLICY_LEN);
  [[3, 0.5], [17, 0.3], [4671, 0.2]].forEach(([i, v]) => { policy[i] = v; });
  return { planes, policy, value: seed % 2 === 0 ? 1.0 : -1.0, side: 0, ply: seed };
}

console.log('A) submitGame — round-trip gzip + headers');
{
  let captured = null;
  const fakeFetch = async (url, init) => {
    captured = { url, init };
    return { ok: true, status: 200, json: async () => ({ status: 'ok', positions_stored: 2 }) };
  };
  const c = new JobserverClient({ baseUrl: 'http://h', workerId: 'w', fetch: fakeFetch });
  const samples = [sample(0), sample(1)];
  const r = await c.submitGame('job-42', samples);

  check(r.positions_stored === 2, 'A1 réponse serveur retournée');
  check(captured.url === 'http://h/jobs/job-42/submit' && captured.init.method === 'POST', 'A2 POST sur le bon endpoint');
  check(captured.init.headers['Content-Type'] === 'application/x-nanozero-submit-v1', 'A3 Content-Type binaire');
  check(captured.init.headers['Content-Encoding'] === 'gzip', 'A4 Content-Encoding gzip');

  // décompresse + décode → doit reconstituer les samples (outcome = value)
  const raw = await gunzip(captured.init.body);
  const decoded = decode(raw.buffer);
  check(decoded.length === 2, 'A5 2 positions décodées');
  let ok = true;
  for (let k = 0; k < 2; k++) {
    if (decoded[k].outcome !== samples[k].value) ok = false;
    for (let i = 0; i < POLICY_LEN; i++) if (decoded[k].policy[i] !== samples[k].policy[i]) { ok = false; break; }
    for (let i = 0; i < PLANES_LEN; i++) if (decoded[k].planes[i] !== samples[k].planes[i]) { ok = false; break; }
  }
  check(ok, 'A6 planes/policy/outcome identiques (value→outcome) après round-trip');
}

console.log('B) boucle happy — claim → play → submit');
{
  let claimed = 0, played = 0, submitted = 0;
  const client = {
    claim: async () => { claimed += 1; return { job_id: `j${claimed}`, model_version: 30 }; },
    submitGame: async () => { submitted += 1; return {}; },
  };
  const playGameBatch = async () => { played += 1; return [sample(0)]; };
  const stats = await runContributionLoop({ client, playGameBatch, backoffMs: 1, shouldStop: () => submitted >= 2 });
  check(claimed >= 2 && played === submitted && submitted === 2, 'B1 2 cycles claim→play→submit');
  check(stats.submitted === 2 && stats.errors === 0, 'B2 stats cohérentes');
}

console.log('C) claim 204 → backoff, pas de play/submit');
{
  let idle = 0, played = 0;
  const client = {
    claim: async () => { idle += 1; return null; },
    submitGame: async () => { throw new Error('ne doit pas soumettre'); },
  };
  const playGameBatch = async () => { played += 1; return []; };
  const stats = await runContributionLoop({ client, playGameBatch, backoffMs: 1, shouldStop: () => idle >= 3 });
  check(stats.idle >= 1 && played === 0, 'C1 204 → backoff, aucun play/submit');
}

console.log('D) circuit-breaker — N échecs consécutifs → stop');
{
  const client = { claim: async () => ({ job_id: 'j', model_version: 30 }), submitGame: async () => ({}) };
  const playGameBatch = async () => { throw new Error('net cassé'); };
  const stats = await runContributionLoop({ client, playGameBatch, backoffMs: 1, maxConsecutiveFailures: 3 });
  check(stats.errors === 3 && typeof stats.aborted === 'string', 'D1 circuit-breaker stoppe après 3 échecs (pas de spin-loop)');
}

console.log('E) batch vide → skip (pas de submit, pas de breaker)');
{
  let submitted = 0, played = 0;
  const client = {
    claim: async () => ({ job_id: 'j', model_version: 30 }),
    submitGame: async () => { submitted += 1; return {}; },
  };
  const playGameBatch = async () => { played += 1; return []; }; // toujours vide
  const stats = await runContributionLoop({
    client, playGameBatch, backoffMs: 1, maxConsecutiveFailures: 3, shouldStop: () => played >= 5,
  });
  check(submitted === 0 && stats.skipped >= 1, 'E1 batch vide → skip, aucun submit');
  check(!stats.aborted, 'E2 batch vide ne trip PAS le circuit-breaker');
}

console.log('F) échec submit (HTTP non-ok) → compte vers le breaker');
{
  const fakeFetch = async (url) => ({ ok: false, status: 400, json: async () => ({}) });
  const realClient = new JobserverClient({ baseUrl: 'http://h', workerId: 'w', fetch: fakeFetch });
  const client = {
    claim: async () => ({ job_id: 'j', model_version: 30 }),
    submitGame: realClient.submitGame.bind(realClient), // vrai submitGame → POST 400 → throw
  };
  const playGameBatch = async () => [sample(0)];
  const stats = await runContributionLoop({ client, playGameBatch, backoffMs: 1, maxConsecutiveFailures: 3 });
  check(stats.errors === 3 && typeof stats.aborted === 'string', 'F1 submit 400 → breaker après 3 échecs');
}

console.log(fail === 0 ? '\nOK — run-loop selftest vert' : `\nFAIL — ${fail} check(s)`);
process.exit(fail ? 1 : 0);
