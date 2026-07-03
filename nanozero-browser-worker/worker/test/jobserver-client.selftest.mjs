// jobserver-client.selftest.mjs — Story A.3, avec un fake fetch (zéro réseau réel).
import { JobserverClient } from '../src/jobserver-client.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };

// Réponse minimale façon fetch Response.
function resp({ status = 200, json, bytes }) {
  return {
    status,
    ok: status >= 200 && status < 300,
    json: async () => json,
    arrayBuffer: async () =>
      bytes ? bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) : new ArrayBuffer(0),
  };
}

async function sha256Hex(bytes) {
  const d = await crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

// fake fetch : route par pathname, expose le compteur d'appels par chemin.
function makeFetch(handler) {
  const calls = {};
  const fn = async (url, init) => {
    const path = new URL(url, 'http://host').pathname;
    calls[path] = (calls[path] || 0) + 1;
    return handler(path, init);
  };
  fn.calls = calls;
  return fn;
}

console.log('A) claim');
{
  let f = makeFetch(() => resp({ status: 204 }));
  let c = new JobserverClient({ baseUrl: 'http://h', workerId: 'w', fetch: f });
  check((await c.claim()) === null, 'A1 claim 204 → null');

  f = makeFetch(() => resp({ status: 200, json: { job_id: 'j1', model_version: 30, num_sims: 1600 } }));
  c = new JobserverClient({ baseUrl: 'http://h/', workerId: 'w', fetch: f });
  const job = await c.claim();
  check(job && job.job_id === 'j1' && job.model_version === 30, 'A2 claim 200 → objet parsé');
}

console.log('B-E) getChampionModel (download + sha + cache)');
{
  const modelBytes = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
  const sha = await sha256Hex(modelBytes);
  let downloads = 0;
  let curV = 30;
  const f = makeFetch((path) => {
    if (path === '/models/current') return resp({ status: 200, json: { version: curV, name: 'gen', sha256_onnx: sha } });
    if (path === `/models/${curV}/download`) { downloads += 1; return resp({ status: 200, bytes: modelBytes }); }
    return resp({ status: 404 });
  });
  const c = new JobserverClient({ baseUrl: 'http://h', workerId: 'w', fetch: f });

  const m1 = await c.getChampionModel();
  check(m1.version === 30 && m1.sha256 === sha, 'B1 modèle v30, sha vérifié');
  check([...m1.bytes].join(',') === [...modelBytes].join(','), 'B2 bytes corrects');
  check(downloads === 1, 'B3 un seul download');

  await c.getChampionModel();
  check(downloads === 1, 'C cache hit (toujours 1 download)');

  curV = 31; // promotion → nouvelle version → re-download
  const m3 = await c.getChampionModel();
  check(m3.version === 31, 'D1 champion changé → v31');
  check(downloads === 2, 'D2 version change → re-download');
}

console.log('E) sha mismatch');
{
  const modelBytes = new Uint8Array([9, 9, 9]);
  const wrongSha = 'de'.repeat(32); // 64 hex faux
  let cachedAfterThrow = 0;
  const f = makeFetch((path) => {
    if (path === '/models/current') return resp({ status: 200, json: { version: 5, name: 'g', sha256_onnx: wrongSha } });
    if (path === '/models/5/download') { cachedAfterThrow += 1; return resp({ status: 200, bytes: modelBytes }); }
    return resp({ status: 404 });
  });
  const c = new JobserverClient({ baseUrl: 'http://h', workerId: 'w', fetch: f });

  let threw = false;
  try { await c.getChampionModel(); } catch { threw = true; }
  check(threw, 'E1 sha mismatch → throw');

  let threw2 = false;
  try { await c.getChampionModel(); } catch { threw2 = true; }
  check(threw2 && cachedAfterThrow === 2, 'E2 pas de cache pourri (re-tente + re-throw)');
}

console.log('F) validation & normalisation des réponses');
{
  const modelBytes = new Uint8Array([1, 2, 3]);
  const sha = await sha256Hex(modelBytes); // minuscule

  // F1 : sha en MAJUSCULES côté serveur → normalisé → accepté
  let f = makeFetch((path) => {
    if (path === '/models/current') return resp({ status: 200, json: { version: 7, sha256_onnx: sha.toUpperCase() } });
    if (path === '/models/7/download') return resp({ status: 200, bytes: modelBytes });
    return resp({ status: 404 });
  });
  let c = new JobserverClient({ baseUrl: 'http://h', workerId: 'w', fetch: f });
  const m = await c.getChampionModel();
  check(m.sha256 === sha, 'F1 sha MAJUSCULE serveur → normalisé → accepté');

  // F2 : sha manquant dans /models/current → erreur claire (pas de "30:undefined")
  f = makeFetch(() => resp({ status: 200, json: { version: 7 } }));
  c = new JobserverClient({ baseUrl: 'http://h', workerId: 'w', fetch: f });
  let threw = false;
  try { await c.getChampionModel(); } catch { threw = true; }
  check(threw, 'F2 sha manquant → erreur');

  // F3 : claim 200 sans job_id/model_version → erreur (A.4 ne reçoit pas un job invalide)
  f = makeFetch(() => resp({ status: 200, json: {} }));
  c = new JobserverClient({ baseUrl: 'http://h', workerId: 'w', fetch: f });
  let threw2 = false;
  try { await c.claim(); } catch { threw2 = true; }
  check(threw2, 'F3 claim 200 sans champs → erreur');
}

console.log('G) globalThis.fetch est bindé (régression « Illegal invocation » navigateur)');
{
  // Sans fetch injecté, le client doit prendre globalThis.fetch BINDÉ à globalThis : un fetch
  // global appelé via this._fetch(...) sans binding lèverait « Illegal invocation » en navigateur.
  // Ici on simule : globalThis.fetch vérifie que `this === globalThis`.
  const saved = globalThis.fetch;
  let sawThis = null;
  globalThis.fetch = function fakeGlobalFetch() {
    sawThis = this; // doit être globalThis (grâce au .bind), pas l'instance client
    return resp({ status: 204 });
  };
  try {
    const c = new JobserverClient({ baseUrl: 'http://h', workerId: 'w' }); // PAS de fetch injecté
    const r = await c.claim(); // 204 → null
    check(r === null, 'G1 claim via globalThis.fetch (204 → null)');
    check(sawThis === globalThis, 'G2 fetch appelé avec this===globalThis (bindé, pas d\'Illegal invocation)');
  } finally {
    globalThis.fetch = saved;
  }
}

console.log('H) submitGame — header X-Pseudo conditionnel (STORY-007)');
{
  // 1 sample minimal aux tailles EXACTES attendues par submit-codec (planes 7616, policy 4672) ;
  // submitGame encode+gzip puis POST. On capture l'init pour inspecter les headers. CompressionStream
  // natif (Node 18+ / navigateur). La policy a une masse non-nulle pour rester une distribution valide.
  const planes = new Float32Array(7616);
  const policy = new Float32Array(4672); policy[0] = 1.0;
  const sample = { planes, policy, value: 0.0 };
  const lastInit = {};
  const makeSubmitFetch = () =>
    makeFetch((path, init) => {
      if (path === '/jobs/j1/submit') { lastInit.headers = init.headers; return resp({ status: 200, json: { status: 'ok' } }); }
      return resp({ status: 404 });
    });

  // H1 : pseudo null (défaut) → AUCUN header X-Pseudo (strictement comme aujourd'hui, rétro-compat).
  let f = makeSubmitFetch();
  let c = new JobserverClient({ baseUrl: 'http://h', workerId: 'w', fetch: f }); // pas de pseudo
  check(c.pseudo === null, 'H1a pseudo absent → this.pseudo === null');
  await c.submitGame('j1', [sample]);
  check(!('X-Pseudo' in lastInit.headers), 'H1b submit SANS pseudo → pas de header X-Pseudo');

  // H2 : pseudo défini → header X-Pseudo présent et égal au pseudo.
  f = makeSubmitFetch();
  c = new JobserverClient({ baseUrl: 'http://h', workerId: 'w', pseudo: 'alice', fetch: f });
  await c.submitGame('j1', [sample]);
  check(lastInit.headers['X-Pseudo'] === 'alice', 'H2 submit AVEC pseudo → header X-Pseudo: alice');

  // H3 : pseudo undefined explicite → null (?? null), pas de header.
  f = makeSubmitFetch();
  c = new JobserverClient({ baseUrl: 'http://h', workerId: 'w', pseudo: undefined, fetch: f });
  await c.submitGame('j1', [sample]);
  check(c.pseudo === null && !('X-Pseudo' in lastInit.headers), 'H3 pseudo undefined → null → pas de header');
}

console.log(fail === 0 ? '\nOK — jobserver-client selftest vert' : `\nFAIL — ${fail} check(s)`);
process.exit(fail ? 1 : 0);
