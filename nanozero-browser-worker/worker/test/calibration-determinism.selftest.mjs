// calibration-determinism.selftest.mjs — V1 (design §Validation) : PREUVE anti-régression du bug
// `pool 1 ↔ pool 4`. Sur la courbe plateau 84/181/217/217/196, l'ANCIEN départage (argmax brut
// `best = max(pps)` avec `>=`, ordre-dépendant) fait basculer le pic retenu au hasard entre les
// deux jumeaux du plateau (217@4 ≡ 217@8). Le NOUVEAU départage déterministe (bande d'équivalence ε,
// la plus LÉGÈRE gagne — pickModes) donne le mode IDENTIQUE sur ≥ 90 % des runs (ici 100 %).
//
//   ÉCHOUE avec l'ancien argmax (section B) · PASSE avec le nouveau (section A).
//   Lancer : node worker/test/calibration-determinism.selftest.mjs
//
// MODÈLE VALIDÉ (docs/CALIBRATION-DESIGN.md) : modes = niveaux de K (CPU) / batch×K (GPU). Le cœur
// du départage (bande ε, plus léger gagne) est INCHANGÉ — c'est ce que cette preuve protège, sur la
// courbe-axe du backend (K pour CPU via pickCpuKModes ; batch pour GPU via pickModes/pickGpuBatchModes).
//
// MODÈLE DE BRUIT (réaliste). Le bruit dominant = la DÉRIVE THERMIQUE : elle scale TOUTE la courbe
// d'un même facteur run-à-run (le plateau reste un plateau) + un petit jitter par config. Sous ce
// modèle, la bande contient toujours les deux jumeaux 217 → la plus légère gagne TOUJOURS (100 %),
// tandis que l'argmax `>=` dépend de l'ORDRE de parcours → flip 4↔8.

import {
  pickModes, pickCpuKModes, pickGpuBatchModes,
  deriveCpuCalibration, deriveGpuCalibration, CALIB_VERSION,
} from '../../ui/calibration.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };

// Courbe de référence (design §1) : pool 1→84, 4→217, 8→217, 16→196. Pic = PLATEAU (4≡8).
const BASE = [
  { pool: 1, wave: 1, pps: 84 },
  { pool: 2, wave: 1, pps: 181 },
  { pool: 4, wave: 1, pps: 217 },
  { pool: 8, wave: 1, pps: 217 },
  { pool: 16, wave: 1, pps: 196 },
];

// PRNG reproductible (mulberry32) → test déterministe (pas de Math.random).
function mulberry32(seed) {
  return function () {
    seed |= 0; seed = (seed + 0x6D2B79F5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
// Un run bruité : dérive thermique COHÉRENTE (scale ±8 % commun) + jitter par config (±2 %, < ε).
// Mélange aussi l'ordre du tableau → expose l'ordre-dépendance de l'argmax `>=` (le bug).
function noisyRun(base, rnd, scaleFrac = 0.08, jitterFrac = 0.02) {
  const scale = 1 + (rnd() * 2 - 1) * scaleFrac;
  const out = base.map((r) => ({ ...r, pps: Math.max(1, Math.round(r.pps * scale * (1 + (rnd() * 2 - 1) * jitterFrac))) }));
  for (let k = out.length - 1; k > 0; k--) { const j = Math.floor(rnd() * (k + 1)); [out[k], out[j]] = [out[j], out[k]]; }
  return out;
}

// ── ANCIEN départage : argmax brut `best = reduce((b,r)=> r.pps >= b.pps ? r : b)` — ordre-dépendant :
// sur le plateau (217 == 217) le `>=` garde le DERNIER vu.
function oldArgmaxPool(results) {
  const valid = results.filter((r) => r.pps > 0);
  const src = valid.length ? valid : results;
  return src.reduce((b, r) => (r.pps >= b.pps ? r : b), src[0]).pool;
}

const RUNS = 20;

console.log('A) NOUVEAU départage déterministe (pickModes, bande ε=5 %) — pic stable ≥ 90 %');
{
  const rnd = mulberry32(1234);
  const counts = new Map();
  for (let i = 0; i < RUNS; i++) {
    const p = pickModes(noisyRun(BASE, rnd)).normal.pool;
    counts.set(p, (counts.get(p) || 0) + 1);
  }
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
  const stability = top[1] / RUNS;
  console.log(`  distribution normal.pool sur ${RUNS} runs : ${JSON.stringify([...counts.entries()])} → mode=${top[0]} (${top[1]}/${RUNS}, ${Math.round(stability * 100)} %)`);
  check(stability >= 0.9, `A1 pic-de-bande mode-unique ≥ 90 % (mesuré ${Math.round(stability * 100)} %)`);
  check(top[0] === 4, `A2 le mode déterministe = pool 4 (début du plateau, le plus léger de la bande) — mesuré ${top[0]}`);
  check(stability === 1, `A3 idéal atteint : 100 % identique avec le départage déterministe (mesuré ${Math.round(stability * 100)} %)`);
}

console.log('\nB) ANCIEN argmax brut (best = max `>=`) — DOIT être instable (preuve que le test capte le bug)');
{
  const rnd = mulberry32(1234); // MÊME graine → exactement les mêmes courbes bruitées qu'en A
  const counts = new Map();
  for (let i = 0; i < RUNS; i++) {
    const p = oldArgmaxPool(noisyRun(BASE, rnd));
    counts.set(p, (counts.get(p) || 0) + 1);
  }
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
  const stability = top[1] / RUNS;
  console.log(`  distribution pool retenu sur ${RUNS} runs : ${JSON.stringify([...counts.entries()])} → mode=${top[0]} (${top[1]}/${RUNS}, ${Math.round(stability * 100)} %)`);
  check(stability < 0.9, `B1 l'ancien argmax est INSTABLE < 90 % (mesuré ${Math.round(stability * 100)} %) — le bug est reproduit`);
  check(counts.size >= 2, `B2 l'ancien argmax bascule entre ≥ 2 pools (${[...counts.keys()].join(' ↔ ')}) sur le plateau`);
}

// ── CPU : courbe K agrégée (modes = niveaux de K). Bruit thermique → K_max-de-bande STABLE. ──
// Courbe K mesurée (design « Sweep 2D », nt=1) : K1→105, K2→220, K4→384, K6→472, K8→629 (encore en
// hausse) → pas de plateau ici, K_max = le plus grand mesuré. On teste AUSSI un plateau (K6≡K8) → le
// départage retient le K LE PLUS PETIT de la bande (zéro gaspillage).
const KBASE = [
  { K: 1, pps: 105 }, { K: 2, pps: 220 }, { K: 4, pps: 384 }, { K: 6, pps: 472 }, { K: 8, pps: 629 },
];
const KPLATEAU = [
  { K: 1, pps: 105 }, { K: 2, pps: 220 }, { K: 4, pps: 384 }, { K: 6, pps: 620 }, { K: 8, pps: 629 },
];
function noisyK(base, rnd) {
  const scale = 1 + (rnd() * 2 - 1) * 0.08;
  return base.map((r) => ({ ...r, pps: Math.max(1, Math.round(r.pps * scale * (1 + (rnd() * 2 - 1) * 0.02))) }));
}

console.log('\nC) CPU pickCpuKModes — K_max stable (bande ε), modes = ¼/½/plein de K_max');
{
  const rnd = mulberry32(99);
  const counts = new Map();
  for (let i = 0; i < RUNS; i++) {
    const k = pickCpuKModes(noisyK(KBASE, rnd), 20).kMax;
    counts.set(k, (counts.get(k) || 0) + 1);
  }
  const top = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
  const stability = top[1] / RUNS;
  console.log(`  distribution kMax sur ${RUNS} runs : ${JSON.stringify([...counts.entries()])} → mode=${top[0]} (${Math.round(stability * 100)} %)`);
  check(stability >= 0.9, `C1 kMax mode-unique ≥ 90 % (mesuré ${Math.round(stability * 100)} %)`);
  check(top[0] === 8, `C2 kMax = 8 (le pic mesuré, pas de plateau) — mesuré ${top[0]}`);

  // Modes ¼/½/plein, bornés aux cœurs logiques.
  const m = pickCpuKModes(KBASE, 20);
  check(m.eco === 2 && m.normal === 4 && m.max === 8, `C3 modes K = ¼/½/plein de K_max=8 → 2/4/8 (mesuré ${m.eco}/${m.normal}/${m.max})`);

  // Plateau K6≡K8 (619 ε 629) → K_max = le PLUS PETIT atteignant le pic (6, zéro gaspillage).
  const p = pickCpuKModes(KPLATEAU, 20);
  check(p.kMax === 6, `C4 plateau K6≡K8 → kMax=6 (le plus petit de la bande, zéro gaspillage) — mesuré ${p.kMax}`);

  // Borne cœurs logiques : K_max=8 mais 4 cœurs → modes clampés à 4.
  const c = pickCpuKModes(KBASE, 4);
  check(c.max === 4 && c.kMax === 4, `C5 borne cœurs logiques : 4 cœurs → K_max clampé à 4 (mesuré ${c.kMax})`);
}

console.log('\nD) deriveCpuCalibration v3 — schéma K (CPU)');
{
  const calib = deriveCpuCalibration(KBASE, { coresLogical: 20, cores: 20 });
  check(calib.v === CALIB_VERSION && calib.v === 3, 'D1 nz_calib versionné v:3');
  check(calib.backend === 'cpu' && calib.gpuAvailable === false, 'D2 backend cpu, gpuAvailable false');
  check(calib.kMax === 8, 'D3 kMax = 8');
  check(calib.eco.K === 2 && calib.normal.K === 4 && calib.max.K === 8, 'D4 modes portent K = 2/4/8');
  check(calib.normal.numThreads === 1 && calib.normal.pool === 8 && calib.normal.wave === 1, 'D5 CPU : nt=1, pool=8, wave=1 (design validé)');
  check(calib.peakPps === 629, 'D6 peakPps = 629');
}

console.log('\nE) GPU pickGpuBatchModes + deriveGpuCalibration v3 — modes = batch×K');
{
  // Courbe batch GPU (3090, design) : 16→986, 32→2400, 64→3600, 128→4600, 256→5176 (monotone).
  const GBASE = [
    { pool: 2, wave: 8, pps: 986 }, { pool: 4, wave: 8, pps: 2400 }, { pool: 8, wave: 8, pps: 3600 },
    { pool: 16, wave: 8, pps: 4600 }, { pool: 32, wave: 8, pps: 5176 },
  ];
  // maxK=2 (K=2 a tenu sans OOM) → Max porte K=2 ; Éco/Normal = K=1.
  const g = pickGpuBatchModes(GBASE, 2);
  check(g.max.pool === 32 && g.max.wave === 8, 'E1 GPU Max = meilleur batch (pool 32 × wave 8 = 256)');
  check(g.max.K === 2, 'E2 GPU Max porte K=2 (pas d\'OOM)');
  check(g.eco.K === 1 && g.normal.K === 1, 'E3 GPU Éco/Normal restent K=1');
  check(g.eco.pool === 2, 'E4 GPU Éco = plus petit batch mesuré (pool 2 = 16)');

  const calib = deriveGpuCalibration(GBASE, { maxK: 2, cores: 20, peakPps: 7297 });
  check(calib.v === 3 && calib.backend === 'gpu' && calib.gpuAvailable === true, 'E5 calib GPU v3, backend gpu');
  check(calib.maxK === 2, 'E6 maxK = 2');
  check(calib.max.K === 2 && calib.max.pool === 32, 'E7 max = batch 256 × K=2');
  check(calib.peakPps === 7297, 'E8 peakPps = pic agrégé K=2 (7297)');

  // Repli maxK=1 (K=2 OOM) → Max = batch 256 × K=1.
  const g1 = pickGpuBatchModes(GBASE, 1);
  check(g1.max.K === 1, 'E9 repli OOM : maxK=1 → Max porte K=1');
}

console.log(fail === 0 ? '\nOK — calibration determinism (V1) vert' : `\nFAIL — ${fail} check(s)`);
process.exit(fail ? 1 : 0);
