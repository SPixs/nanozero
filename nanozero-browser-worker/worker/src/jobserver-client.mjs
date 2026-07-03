// jobserver-client.mjs — client navigateur du jobserver NanoZero (Story A.3).
//
// Récupère un job (claim) et le .onnx du champion, de façon ANONYME : la gateway
// (Caddy) porte la clé X-API-Key ; le navigateur n'envoie que X-Worker-Id. Le modèle
// est vérifié par sha256 et caché par (version, sha) — un re-register d'une version
// avec un contenu différent (sha différent) invalide le cache (corrige le bug
// ModelCache stale). Natif : fetch + crypto.subtle. Zéro dépendance. PUR (testable
// avec un fake fetch injecté).
//
// Périmètre A.3 = claim + acquisition modèle. submitGame (binaire gzippé) = A.4.

import { encode } from './submit-codec.mjs';

const HEX64 = /^[0-9a-f]{64}$/;
const BINARY_SUBMIT_CONTENT_TYPE = 'application/x-nanozero-submit-v1';

// gzip un Uint8Array via CompressionStream natif (zéro dépendance).
async function gzipBytes(bytes) {
  const stream = new Response(bytes).body.pipeThrough(new CompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// Le sha du serveur est un pass-through du script de register : normaliser (casse/espaces)
// avant comparaison, sinon un sha en MAJUSCULES rejetterait TOUT modèle valide.
function normalizeSha(s) {
  return typeof s === 'string' ? s.trim().toLowerCase() : '';
}

// sha256 hex via crypto.subtle (natif). Absent hors contexte sécurisé (HTTP/file://) → message clair.
async function sha256Hex(bytes) {
  if (!globalThis.crypto || !globalThis.crypto.subtle) {
    throw new Error(
      'jobserver-client : crypto.subtle indisponible — contexte sécurisé (HTTPS ou localhost) '
        + 'requis pour vérifier le sha256 du modèle',
    );
  }
  const digest = await globalThis.crypto.subtle.digest('SHA-256', bytes);
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('');
}

export class JobserverClient {
  /**
   * @param {object} opts
   * @param {string} opts.baseUrl   URL de base de la gateway (ex. "https://host").
   * @param {string} opts.workerId  identifiant volontaire (envoyé en X-Worker-Id).
   * @param {string|null} [opts.pseudo]  STORY-007 — label PUBLIC cosmétique pour les crédits
   *   (« adresse ouverte »). Non-null → envoyé en header `X-Pseudo` sur submitGame ; null/absent →
   *   AUCUN header (= comportement historique, rétro-compat flotte). Le serveur le re-normalise.
   * @param {typeof fetch} [opts.fetch]  fetch injectable (défaut : global) — pour les tests.
   * @param {number} [opts.timeoutMs=30000]  deadline par requête (un serveur muet ne fige pas la boucle).
   * @param {number} [opts.modelCacheMax=2]  nb max d'entrées modèle cachées (évite la fuite mémoire).
   */
  constructor({ baseUrl, workerId, pseudo, contributorToken, fetch: fetchImpl, timeoutMs = 30000, modelCacheMax = 2 } = {}) {
    this.baseUrl = (baseUrl || '').replace(/\/+$/, ''); // sans slash final
    this.workerId = workerId;
    // STORY-007 — null = anonyme (aucun header X-Pseudo). Cosmétique, jamais utilisé pour l'auth.
    this.pseudo = pseudo ?? null;
    // STORY-015 — token contributeur STABLE (1/appareil, dérivé d'un UUID local) pour le décompte
    // serveur. null = pas d'identité stable → aucun header X-Contributor (rétro-compat).
    this.contributorToken = contributorToken ?? null;
    // `globalThis.fetch` DOIT être appelé avec `this === globalThis` : stocké tel quel dans une
    // propriété puis appelé via `this._fetch(...)`, le binding est perdu → « Illegal invocation »
    // (WorkerGlobalScope/Window). On bind donc le fetch global. Un fetch injecté (tests) est pris tel quel.
    if (fetchImpl) {
      this._fetch = fetchImpl;
    } else if (typeof globalThis.fetch === 'function') {
      this._fetch = globalThis.fetch.bind(globalThis);
    }
    if (typeof this._fetch !== 'function') {
      throw new Error('jobserver-client : fetch indisponible (ni opts.fetch ni globalThis.fetch)');
    }
    this.timeoutMs = timeoutMs;
    this._modelCacheMax = modelCacheMax;
    this._modelCache = new Map(); // "${version}:${sha}" -> Uint8Array
  }

  /**
   * STORY-015 — met à jour le pseudo (label crédits) en cours de session, sans recréer le client
   * ni redémarrer le worker. `null`/vide → repasse anonyme (plus de header X-Pseudo).
   * @param {string|null} pseudo
   */
  setPseudo(pseudo) {
    this.pseudo = pseudo ?? null;
  }

  async _req(path, init = {}) {
    return this._fetch(`${this.baseUrl}${path}`, {
      ...init,
      headers: { 'X-Worker-Id': this.workerId, ...(init.headers || {}) },
      signal: AbortSignal.timeout(this.timeoutMs),
    });
  }

  /**
   * Récupère un job. Une file vide / un self-play pausé → 204 → `null`.
   * @returns {Promise<{job_id:string, model_version:number, opening_fen:string|null,
   *   dirichlet_seed:number|null, num_sims:number} | null>}
   */
  async claim() {
    const resp = await this._req('/jobs/claim', { method: 'POST' });
    if (resp.status === 204) return null;
    if (!resp.ok) throw new Error(`claim: HTTP ${resp.status}`);
    let job;
    try {
      job = await resp.json();
    } catch {
      throw new Error('claim: corps 200 non-JSON/vide');
    }
    if (!job || typeof job.job_id !== 'string' || typeof job.model_version !== 'number') {
      throw new Error('claim: réponse 200 sans job_id/model_version valides');
    }
    return job;
  }

  /**
   * Récupère le `.onnx` du champion courant, vérifié par sha256 et caché par (version, sha).
   * Re-télécharge si le champion a changé (nouvelle version) ou si le sha diffère.
   * @returns {Promise<{version:number, sha256:string, bytes:Uint8Array}>}
   */
  async getChampionModel() {
    const curResp = await this._req('/models/current');
    if (!curResp.ok) throw new Error(`models/current: HTTP ${curResp.status}`);
    const cur = await curResp.json();
    const version = cur && cur.version;
    const expectedSha = normalizeSha(cur && cur.sha256_onnx);
    if (!Number.isInteger(version) || version <= 0 || !HEX64.test(expectedSha)) {
      throw new Error(`models/current: réponse invalide (version=${version}, sha=${cur && cur.sha256_onnx})`);
    }

    const key = `${version}:${expectedSha}`;
    const cached = this._modelCache.get(key);
    if (cached) return { version, sha256: expectedSha, bytes: cached };

    const dlResp = await this._req(`/models/${version}/download`);
    if (!dlResp.ok) throw new Error(`models/${version}/download: HTTP ${dlResp.status}`);
    const bytes = new Uint8Array(await dlResp.arrayBuffer());

    const actual = await sha256Hex(bytes);
    if (actual !== expectedSha) {
      // Ne JAMAIS cacher ni retourner un blob non vérifié.
      throw new Error(`sha256 mismatch pour le modèle v${version} (attendu ${expectedSha}, obtenu ${actual})`);
    }

    this._cacheModel(key, bytes);
    return { version, sha256: expectedSha, bytes };
  }

  // Borne le cache (le worker n'a besoin que du champion courant) : éviction FIFO.
  _cacheModel(key, bytes) {
    while (this._modelCache.size >= this._modelCacheMax) {
      this._modelCache.delete(this._modelCache.keys().next().value);
    }
    this._modelCache.set(key, bytes);
  }

  /**
   * Soumet les samples d'une partie au format binaire gzippé (Story A.4).
   * Les samples self-play `{planes, policy, value}` sont mappés vers le format codec
   * `{planes, policy, outcome}` (outcome = z), encodés, gzippés, et POSTés.
   * @param {string} jobId
   * @param {Array<{planes:Float32Array, policy:Float32Array, value:number}>} samples
   * @returns {Promise<object>} la réponse JSON du serveur (positions_stored, ...).
   */
  async submitGame(jobId, samples) {
    const buf = encode(samples.map((s) => ({ planes: s.planes, policy: s.policy, outcome: s.value })));
    const body = await gzipBytes(new Uint8Array(buf));
    // STORY-007 — header X-Pseudo ajouté UNIQUEMENT si un pseudo est défini : sans pseudo (cas par
    // défaut + flotte native), aucun header n'est posé → requête STRICTEMENT identique à aujourd'hui.
    // Le pseudo ne touche PAS le payload binaire (submit-codec versionné, partagé serveur) → le codec
    // reste inchangé. Le serveur re-normalise/valide ; un pseudo invalide est silencieusement ignoré.
    const headers = { 'Content-Type': BINARY_SUBMIT_CONTENT_TYPE, 'Content-Encoding': 'gzip' };
    if (this.pseudo) headers['X-Pseudo'] = this.pseudo;
    // STORY-015 — identité stable de décompte (pseudonyme, non affichée). Absente → pas de header.
    if (this.contributorToken) headers['X-Contributor'] = this.contributorToken;
    const resp = await this._req(`/jobs/${jobId}/submit`, {
      method: 'POST',
      headers,
      body,
    });
    if (!resp.ok) throw new Error(`submit: HTTP ${resp.status}`);
    // Le submit a RÉUSSI (2xx) même si le corps n'est pas lisible (proxy/gateway) → ne pas
    // transformer un succès serveur en échec (qui armerait le circuit-breaker de la boucle).
    try {
      return await resp.json();
    } catch {
      return { status: 'ok' };
    }
  }
}
