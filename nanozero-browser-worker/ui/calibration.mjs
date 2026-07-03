// calibration.mjs — logique PURE de dérivation des presets d'énergie (Éco / Normal / Max) à partir
// des résultats du bench. Aucun accès DOM/navigator/localStorage : importable en Node pour le test
// de déterminisme (V1) ET par app.mjs (thread principal). C'est ici que vit LE fix du bug
// `pool 1 ↔ pool 4` : le départage déterministe par BANDE D'ÉQUIVALENCE.
//
// MODÈLE VALIDÉ PAR MESURE (docs/CALIBRATION-DESIGN.md, « Design FINAL validé (deux backends) ») :
//   • CPU/WASM = inference-overhead-bound. Levier = K (Web Workers parallèles, ×4,3), nt=1, pool~8,
//     wave=1. Modes = niveaux de K : Éco ≈ ¼·K_max, Normal ≈ ½, Max = K_max (cap RAM/onglet).
//   • GPU/WebGPU = feeding/WDDM-bound. Levier = batch (pool×wave, ×5,2 → cap 256) + K=2 (+76%).
//     Modes = batch×K : Éco = petit batch(16-32)×K1, Normal = batch moyen(64-128)×K1, Max = 256×K2.
// Le sweep numThreads a été RETIRÉ (validé inutile au-delà de K=1-2 ; CPU = nt=1).

// Bande d'équivalence ε = 5 %. Deux configs A,B sont ÉQUIVALENTES si |pps_A − pps_B| ≤ ε × peak. Sur
// un plateau (deux configs au même débit), on choisit alors la PLUS LÉGÈRE → départage déterministe.
export const EQUIV_EPS = 0.05;

// Version du schéma nz_calib. v3 = modèle K (CPU) / batch×K (GPU). v1/v2 invalidés par le lecteur.
export const CALIB_VERSION = 3;

// Pool/wave figés par backend pendant le sweep (le design n'a qu'UN axe par backend) :
//   CPU : pool=8, wave=1 (recherche séquentielle ; le levier est K, pas pool/wave).
//   GPU : pool variable (= le batch via pool×wave), wave figé (un arbre remplit le batch).
export const CPU_POOL = 8;
export const CPU_WAVE = 1;
export const GPU_WAVE = 8; // wave de prod en contribution GPU (leaf-batch)

// ── Départage déterministe générique (cœur du fix bug `pool 1↔4`) ──────────────────────────────
// « Charge » d'un point pour ORDONNER les équivalents du plus léger au plus lourd. On minimise
// l'empreinte machine à débit égal. Proxy = K × batch_effectif (numThreads × pool × wave). Tie-break
// stable et déterministe → jamais deux points « égaux » → jamais d'argmax sur du bruit.
export function chargeOf(r) {
  const k = Number.isFinite(r.K) ? r.K : 1;
  const t = Number.isFinite(r.numThreads) ? r.numThreads : 1;
  const p = Number.isFinite(r.pool) ? r.pool : 1;
  const w = Number.isFinite(r.wave) ? r.wave : 1;
  return k * t * p * w;
}

// Comparateur « du plus léger au plus lourd » : charge, puis K, puis pool, puis numThreads, puis wave.
function lighterFirst(a, b) {
  const ca = chargeOf(a), cb = chargeOf(b);
  if (ca !== cb) return ca - cb;
  const ka = a.K || 0, kb = b.K || 0;
  if (ka !== kb) return ka - kb;
  const pa = a.pool || 0, pb = b.pool || 0;
  if (pa !== pb) return pa - pb;
  const ta = a.numThreads || 0, tb = b.numThreads || 0;
  if (ta !== tb) return ta - tb;
  return (a.wave || 0) - (b.wave || 0);
}

// Cœur du fix : départage DÉTERMINISTE par bande d'équivalence ε.
//   peak   = max(pps).
//   bande  = toutes les configs à |pps − peak| ≤ ε × peak (≈ le plateau).
//   normal = la PLUS LÉGÈRE de la bande (= début du plateau, stable — pas la fin).
//   max    = la plus légère atteignant le pic (= band[0] aussi → zéro gaspillage).
//   eco    = la plus légère mesurée.
// Sur 84/181/217/217/196, la bande = {217@4, 217@8} → normal = le plus léger (4) DÉTERMINISTE.
export function pickModes(results) {
  const valid = results.filter((r) => Number.isFinite(r.pps) && r.pps > 0);
  const src = (valid.length ? valid : results).slice();
  if (src.length === 0) return null;

  const peak = src.reduce((mx, r) => Math.max(mx, r.pps), 0);
  const band = src
    .filter((r) => Math.abs(r.pps - peak) <= EQUIV_EPS * peak)
    .sort(lighterFirst);
  const allByCharge = src.slice().sort(lighterFirst);

  return {
    peak,
    eco: allByCharge[0], // la plus légère mesurée (charge mini)
    normal: band[0],     // la plus légère DANS la bande du pic (début du plateau)
    max: band[0],        // la plus légère atteignant le pic (zéro gaspillage)
  };
}

// ── CPU : modes = niveaux de K ─────────────────────────────────────────────────────────────────
// `kResults` = [{K, pps}] (pps = débit AGRÉGÉ des K workers, mesuré par l'orchestrateur). K_max = la
// PLUS GRANDE valeur de K atteignant le pic dans la bande ε (le bench borne K au plateau/OOM en amont).
// Modes sur l'axe K : Éco ≈ ¼·K_max, Normal ≈ ½·K_max, Max = K_max — tous ≥ 1 et bornés à `coresLogical`.
// Déterminisme : sur égalité de débit, la bande retient le K LE PLUS PETIT atteignant le pic (la plus
// légère gagne) → K_max = ce K-de-bande, pas le plus gros K simplement essayé.
export function pickCpuKModes(kResults, coresLogical = null) {
  const valid = (kResults || []).filter((r) => Number.isFinite(r.K) && r.K >= 1 && Number.isFinite(r.pps) && r.pps > 0);
  if (valid.length === 0) return null;
  const peak = valid.reduce((mx, r) => Math.max(mx, r.pps), 0);
  // K_max = le plus PETIT K atteignant le pic (bande ε) → zéro gaspillage (plateau = même débit).
  const kMax = valid
    .filter((r) => Math.abs(r.pps - peak) <= EQUIV_EPS * peak)
    .reduce((mn, r) => Math.min(mn, r.K), Infinity);
  const cap = (Number.isFinite(coresLogical) && coresLogical > 0) ? coresLogical : Infinity;
  const clampK = (k) => Math.max(1, Math.min(Math.round(k), kMax, cap));
  // ppsAt : débit MESURÉ le plus proche d'un K cible (pour peupler peakPps/affichage), sinon 0.
  const ppsAt = (k) => {
    const exact = valid.find((r) => r.K === k);
    return exact ? exact.pps : 0;
  };
  const kEco = clampK(kMax / 4);
  const kNormal = clampK(kMax / 2);
  const kMaxClamped = clampK(kMax);
  return {
    eco: kEco,
    normal: kNormal,
    max: kMaxClamped,
    kMax: kMaxClamped,
    peakPps: peak,
    ppsAt,
  };
}

// ── GPU : modes = batch × K ────────────────────────────────────────────────────────────────────
// `batchResults` = [{pool,wave,pps}] (sweep batch mono-worker). On dérive le batch par mode via la
// bande déterministe (pickModes) : Max = le meilleur batch (band[0] = plus léger atteignant le pic),
// Normal = un batch moyen, Éco = le plus petit batch mesuré. `maxK` = 2 si le K=2 a tenu sans OOM, 1
// sinon (repli). Le mode Max porte alors K=maxK ; Éco/Normal restent K=1 (charge basse/moyenne).
export function pickGpuBatchModes(batchResults, maxK = 1) {
  const picked = pickModes(batchResults || []);
  if (!picked) return null;
  const sorted = (batchResults || [])
    .filter((r) => Number.isFinite(r.pps) && r.pps > 0)
    .sort((a, b) => a.pool * a.wave - b.pool * b.wave);
  const eco = sorted[0] || picked.eco;            // plus petit batch mesuré (charge basse)
  const max = picked.max;                          // meilleur batch (plus léger atteignant le pic)
  // Normal = batch MÉDIAN entre éco et max (charge moyenne) ; à défaut, le pic lui-même.
  const mid = sorted.length ? sorted[Math.floor((sorted.length - 1) / 2)] : max;
  const k = (maxK >= 2) ? 2 : 1;
  return {
    eco: { pool: eco.pool, wave: eco.wave, K: 1 },
    normal: { pool: mid.pool, wave: mid.wave, K: 1 },
    max: { pool: max.pool, wave: max.wave, K: k },
    peakPps: picked.peak,
  };
}

// ── Dérivation du calib v3 complet (par-backend) ───────────────────────────────────────────────
// CPU : à partir des kResults [{K,pps}]. Chaque mode pose K + pool/wave figés + numThreads=1.
// @param {Array<{K,pps}>} kResults  débits agrégés par K.
// @param {object} ctx  { coresLogical:number|null, cores:number|null, peakPps?:number }
export function deriveCpuCalibration(kResults, ctx = {}) {
  const { coresLogical = null, cores = null } = ctx;
  const picked = pickCpuKModes(kResults, coresLogical);
  if (!picked) return null;
  const mode = (K) => ({ K, pool: CPU_POOL, wave: CPU_WAVE, numThreads: 1 });
  return {
    v: CALIB_VERSION,
    ts: Date.now(),
    backend: 'cpu',
    gpuAvailable: false,
    cores: Number.isFinite(cores) ? cores : null,
    coresLogical: Number.isFinite(coresLogical) ? coresLogical : (Number.isFinite(cores) ? cores : null),
    kMax: picked.kMax,
    eco: mode(picked.eco),
    normal: mode(picked.normal),
    max: mode(picked.max),
    peakPps: picked.peakPps,
  };
}

// GPU : à partir des batchResults [{pool,wave,pps}] + maxK (1 ou 2, selon le test K=2/OOM).
// @param {Array<{pool,wave,pps}>} batchResults
// @param {object} ctx  { maxK:1|2, cores:number|null, peakPps?:number }
export function deriveGpuCalibration(batchResults, ctx = {}) {
  const { maxK = 1, cores = null, peakPps = null } = ctx;
  const picked = pickGpuBatchModes(batchResults, maxK);
  if (!picked) return null;
  return {
    v: CALIB_VERSION,
    ts: Date.now(),
    backend: 'gpu',
    gpuAvailable: true,
    cores: Number.isFinite(cores) ? cores : null,
    coresLogical: Number.isFinite(cores) ? cores : null,
    maxK: (maxK >= 2) ? 2 : 1,
    eco: picked.eco,
    normal: picked.normal,
    max: picked.max,
    peakPps: Number.isFinite(peakPps) && peakPps > 0 ? peakPps : picked.peakPps,
  };
}
