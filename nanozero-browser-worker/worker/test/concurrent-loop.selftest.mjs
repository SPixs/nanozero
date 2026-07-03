// concurrent-loop.selftest.mjs — A4-D1 : boucle N-parties concurrentes (évaluateur partagé).
// Vérifie : fusion du batch (maxBatch>1, avgBatch>1), anti-stall avec slots idle (multi-parties
// sans hang), file vide→backoff, partie vide→skip, breaker partagé, shouldStop drain.
import { runConcurrentContributionLoop } from '../src/run-loop.mjs';
import { TENSOR_SIZE } from '../src/encoding/plane-encoding.mjs';
import { POLICY_LEN } from '../src/submit-codec.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };

// garde anti-hang : un stall (quorum jamais atteint) ferait tourner Promise.all à l'infini →
// on borne dans le temps et on échoue proprement au lieu de bloquer le selftest.
function withTimeout(p, ms, label) {
  let t;
  const timeout = new Promise((_, rej) => { t = setTimeout(() => rej(new Error('TIMEOUT ' + label)), ms); t.unref?.(); });
  return Promise.race([p.finally(() => clearTimeout(t)), timeout]);
}

// net factice : evaluateBatch(planes, n) → n résultats neutres (le MCTS réel n'est pas utilisé).
const fakeNet = {
  evaluateBatch: async (_planesBatch, n) =>
    Array.from({ length: n }, () => ({ policy: new Float32Array(POLICY_LEN), value: 0 })),
};

// playGame factice : "joue" en passant `steps` recherches par l'évaluateur PARTAGÉ, puis rend des samples.
const makePlayGame = (steps = 6) => async (evaluator) => {
  for (let i = 0; i < steps; i++) await evaluator.evaluate(new Float32Array(TENSOR_SIZE));
  return { samples: [{ planes: new Float32Array(TENSOR_SIZE), policy: new Float32Array(POLICY_LEN), value: 0 }] };
};

console.log('A) happy concurrent — fusion du batch + tous soumis + anti-stall (multi-parties)');
{
  let claimed = 0, submitted = 0;
  const client = {
    claim: async () => { claimed += 1; return { job_id: `j${claimed}`, model_version: 30 }; },
    submitGame: async () => { submitted += 1; return {}; },
  };
  const stats = await withTimeout(
    runConcurrentContributionLoop({
      client, net: fakeNet, poolSize: 4, playGame: makePlayGame(6),
      shouldStop: () => submitted >= 12, // ~3 parties/slot → transitions actif↔idle exercées
    }),
    5000, 'A',
  ).catch((e) => ({ error: String(e.message || e) }));

  check(!stats.error, 'A0 boucle terminée sans stall (pas de TIMEOUT)');
  check(stats.submitted >= 12 && stats.claimed >= 12, 'A1 toutes les parties claimées→soumises');
  check(stats.slots === 4 && !stats.aborted, 'A2 4 slots, pas d\'abort');
  check(stats.maxBatch >= 2, `A3 batch FUSIONNÉ (maxBatch=${stats.maxBatch} ≥ 2, vs séquentiel=1)`);
  check(stats.avgBatch > 1, `A4 avgBatch=${stats.avgBatch.toFixed(2)} > 1 (inférence partagée)`);
}

console.log('B) file vide → backoff (claim null), aucune partie');
{
  let idle = 0, played = 0;
  const client = {
    claim: async () => { idle += 1; return null; },
    submitGame: async () => { throw new Error('ne doit pas soumettre'); },
  };
  const playGame = async () => { played += 1; return { samples: [] }; };
  const stats = await withTimeout(
    runConcurrentContributionLoop({
      client, net: fakeNet, poolSize: 3, playGame, backoffMs: 1, shouldStop: () => idle >= 6,
    }),
    5000, 'B',
  ).catch((e) => ({ error: String(e.message || e) }));
  check(!stats.error && played === 0 && stats.idle >= 3 && stats.submitted === 0, 'B1 204→backoff, 0 play/submit');
}

console.log('C) partie vide → skip (pas de submit, pas de breaker)');
{
  let submitted = 0, played = 0;
  const client = {
    claim: async () => ({ job_id: 'j', model_version: 30 }),
    submitGame: async () => { submitted += 1; return {}; },
  };
  const playGame = async () => { played += 1; return { samples: [] }; };
  const stats = await withTimeout(
    runConcurrentContributionLoop({
      client, net: fakeNet, poolSize: 2, playGame, backoffMs: 1,
      maxConsecutiveFailures: 3, shouldStop: () => played >= 6,
    }),
    5000, 'C',
  ).catch((e) => ({ error: String(e.message || e) }));
  check(!stats.error && submitted === 0 && stats.skipped >= 1, 'C1 partie vide → skip, aucun submit');
  check(!stats.aborted, 'C2 partie vide ne trip PAS le breaker');
}

console.log('D) circuit-breaker PARTAGÉ — N échecs consécutifs → tous stoppent');
{
  const client = { claim: async () => ({ job_id: 'j', model_version: 30 }), submitGame: async () => ({}) };
  const playGame = async () => { throw new Error('net cassé'); };
  const stats = await withTimeout(
    runConcurrentContributionLoop({
      client, net: fakeNet, poolSize: 4, playGame, backoffMs: 1, maxConsecutiveFailures: 3,
    }),
    5000, 'D',
  ).catch((e) => ({ error: String(e.message || e) }));
  check(!stats.error && typeof stats.aborted === 'string' && stats.errors >= 3, 'D1 breaker partagé stoppe (pas de spin-loop)');
}

console.log('E) shouldStop pendant une partie → drain propre (la partie en cours est soumise)');
{
  let submitted = 0;
  let stop = false;
  const client = {
    claim: async () => ({ job_id: 'j', model_version: 30 }),
    submitGame: async () => { submitted += 1; stop = true; return {}; }, // arrête après le 1er submit
  };
  const stats = await withTimeout(
    runConcurrentContributionLoop({
      client, net: fakeNet, poolSize: 2, playGame: makePlayGame(3), shouldStop: () => stop,
    }),
    5000, 'E',
  ).catch((e) => ({ error: String(e.message || e) }));
  check(!stats.error && stats.submitted >= 1, 'E1 drain propre : la/les partie(s) en cours soumises puis arrêt');
}

console.log('F) parties vides en rafale → BACKOFF, pas de spin-loop CPU (anti-régression vecteur 5)');
{
  // Toutes les parties sont vides (cas deadline/avortée). Avec backoff : ~poolSize×(deadline/backoff)
  // appels. SANS backoff (bug spin) : des millions. On borne par une deadline wall-clock et on
  // vérifie que le nb d'appels reste PETIT (différentiel de plusieurs ordres de grandeur).
  let plays = 0;
  const client = {
    claim: async () => ({ job_id: 'j', model_version: 30 }),
    submitGame: async () => { throw new Error('ne doit pas soumettre (parties vides)'); },
  };
  const playGame = async () => { plays += 1; return { samples: [] }; };
  const t0 = performance.now();
  const stats = await withTimeout(
    runConcurrentContributionLoop({
      client, net: fakeNet, poolSize: 3, playGame, backoffMs: 40,
      shouldStop: () => performance.now() - t0 > 120, // ~3 cycles/slot si backoff respecté
    }),
    5000, 'F',
  ).catch((e) => ({ error: String(e.message || e) }));
  check(!stats.error, 'F0 boucle terminée (pas de hang)');
  check(plays < 200, `F1 backoff respecté sur parties vides : ${plays} appels (un spin en ferait des millions)`);
  check(stats.skipped >= 1 && stats.submitted === 0, 'F2 parties vides → skip, aucun submit');
}

console.log('G) num_sims DICTÉ PAR LE SERVEUR — job.num_sims fait autorité sur gameOpts.sims');
{
  // G.1 : le job porte num_sims=256 → playGame doit voir sims=256, PAS le défaut client 800.
  const seen = [];
  let submitted = 0;
  const client = {
    claim: async () => ({ job_id: 'j', model_version: 30, num_sims: 256 }),
    submitGame: async () => { submitted += 1; return {}; },
  };
  const playGame = async (_evaluator, opts) => {
    seen.push(opts.sims);
    return { samples: [{ planes: new Float32Array(TENSOR_SIZE), policy: new Float32Array(POLICY_LEN), value: 0 }] };
  };
  const stats = await withTimeout(
    runConcurrentContributionLoop({
      client, net: fakeNet, poolSize: 2, playGame,
      gameOpts: { sims: 800 }, // défaut client — doit être ÉCRASÉ par le serveur
      shouldStop: () => submitted >= 4,
    }),
    5000, 'G1',
  ).catch((e) => ({ error: String(e.message || e) }));
  check(!stats.error && seen.length >= 1 && seen.every((s) => s === 256),
    `G1 sims serveur fait autorité : parties à sims=256 (vu=${[...new Set(seen)].join(',')}), défaut client 800 ignoré`);
}
{
  // G.2 : repli — le job ne précise PAS num_sims → on garde gameOpts.sims (défaut client).
  const seen = [];
  let submitted = 0;
  const client = {
    claim: async () => ({ job_id: 'j', model_version: 30 }), // sans num_sims
    submitGame: async () => { submitted += 1; return {}; },
  };
  const playGame = async (_evaluator, opts) => {
    seen.push(opts.sims);
    return { samples: [{ planes: new Float32Array(TENSOR_SIZE), policy: new Float32Array(POLICY_LEN), value: 0 }] };
  };
  const stats = await withTimeout(
    runConcurrentContributionLoop({
      client, net: fakeNet, poolSize: 2, playGame,
      gameOpts: { sims: 512 },
      shouldStop: () => submitted >= 4,
    }),
    5000, 'G2',
  ).catch((e) => ({ error: String(e.message || e) }));
  check(!stats.error && seen.length >= 1 && seen.every((s) => s === 512),
    `G2 repli sur gameOpts.sims quand le job n'a pas num_sims (vu=${[...new Set(seen)].join(',')})`);
}

console.log(fail === 0 ? '\nOK — concurrent-loop selftest vert' : `\nFAIL — ${fail} check(s)`);
process.exit(fail ? 1 : 0);
