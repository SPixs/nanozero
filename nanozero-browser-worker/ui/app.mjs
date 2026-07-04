// app.mjs — thread principal de la page volontaire (Epic C). Consentement (C.1) + grille de
// mini-échiquiers live (C.3) alimentée par les messages coalescés du Web Worker (C.2).
// Aucun calcul ici : tout le self-play tourne dans le Worker.
import { createConsentController } from './consent.mjs';
import { fenToCells, algToFileRank } from './board-render.mjs';
import { gpuStatus } from './gpu-status.mjs';
// calibration VALIDÉE — logique PURE de dérivation des presets (départage déterministe bande ε,
// modes = K pour CPU / batch×K pour GPU). Vit dans son propre module → testable en Node hors DOM.
import {
  deriveCpuCalibration, deriveGpuCalibration, CALIB_VERSION,
  CPU_POOL, CPU_WAVE, GPU_WAVE,
} from './calibration.mjs';

// Surbrillance du dernier coup : calque OR translucide sous la pièce (cases from+to) — fil conducteur v3.
const LASTMOVE = 'linear-gradient(rgba(155,199,0,.55), rgba(155,199,0,.55))'; // dernier coup : aplat jaune-vert lichess sur DÉPART + ARRIVÉE (motif SOTA : tint pleine case, from=to). .55 vs .41 lichess pour tenir sur fond sombre + petites cases

// On affiche UN board par SLOT, mais SEULEMENT les `displayCap` premiers slots : le débit veut
// beaucoup de parties (amortir le plancher WDDM), l'affichage n'en montre que quelques-unes.
const MAX_BOARDS = 128; // garde-fou dur
let displayCap = 4; // configurable (champ « Échiquiers affichés »), lu au démarrage — 4 par défaut (STORY-013, était 3)
// FEN de départ standard — placeholder des échiquiers À L'INSTANT du démarrage (STORY-013 : boards
// visibles dès Démarrer, sans attendre le 1er coup ; remplacé par les vrais updates worker).
const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const $ = (id) => document.getElementById(id);

let workers = [];                 // K Web Workers parallèles (worker.bundle.js) ; index 0 pousse les boards affichés
let contributedByWorker = [];     // parties soumises par worker → somme = total de la session
let perfByWorker = [];            // dernière perf par worker → agrégée pour l'affichage (pps/inférence/GPU%)
let ppsEma = null;                // EMA du pos/s agrégé : lisse l'oscillation 0↔x de la fenêtre glissante 2s
let endedCount = 0;               // workers terminés (done/error) → on finalise quand tous ont fini
let anyError = false;             // au moins un worker a fini en erreur
// Total cumulé à travers les sessions/reloads (le worker recompte de 0 à chaque session ;
// baseTotal = total persisté AVANT cette session ; persisté en localStorage → survit au reload).
let baseTotal = 0;
const boardEls = new Map(); // slotIndex -> {card, grid, label, ...} (boards AFFICHÉS)
const lastState = new Map(); // slotIndex -> dernier {slot,gameId,fen,ply,lastMove,wdl} (TOUS les slots, même non affichés)

let runStarted = false; // 1er board reçu → on passe le statut « Chargement » à « En cours »
let gpuOk = false; // adaptateur WebGPU réellement disponible (sinon repli CPU)

// ---- Machine d'états du worker (STORY-010 T2) ------------------------------------------------
// IDLE     : pas de contribution en cours (état initial).
// RUNNING  : les workers tournent (parties + soumissions).
// PAUSED   : halte TEMPORAIRE — mêmes workers arrêtés que pour Arrêter (le worker n'a pas de
//            message 'pause' ; on réutilise {type:'stop'}) MAIS la page reste, et Reprendre
//            relance une session identique. Le worker ne calcule/soumet donc plus → AC-5 honnête.
// STOPPED  : halte TERMINALE (« Arrêté »), distincte de Pause (AC-7).
const WorkerState = { IDLE: 'idle', RUNNING: 'running', PAUSED: 'paused', STOPPED: 'stopped' };
let currentWorkerState = WorkerState.IDLE;
// Cible terminale d'une halte en cours : PAUSED (clic Pause) ou STOPPED (clic Arrêter). Lu par
// finalizeStopped() quand TOUS les workers ont fini, pour trancher l'état d'arrivée. Pause et
// Arrêter empruntent le MÊME chemin worker ({type:'stop'}) — seul l'état d'UI final diffère.
let pendingHaltState = WorkerState.STOPPED;
// Config de la dernière session lancée → permet à Reprendre de relancer à l'identique (AC-6).
// onStart lit les champs Avancé au démarrage ; on garde la même intention au resume.

// Applique un état + déclenche le rendu des contrôles (et du footer STORY-011 si présent).
function setWorkerState(state) {
  currentWorkerState = state;
  renderWorkerControls(state);
  // Couplage STORY-011 (footer contextuel) NON faite : appel défensif, ne crashe pas si absent.
  if (typeof renderFooter === 'function') {
    try { renderFooter(state); } catch { /* footer optionnel — ne jamais casser le worker */ }
  }
}

// Pilote l'affichage/activation des boutons selon l'état (STORY-010 T3). Le bouton Pause sert
// AUSSI de bouton Reprendre en état PAUSED (libellé + icône commutés). Aucun « Démarrer ma
// contribution » visible en RUNNING ou PAUSED (AC-8).
function renderWorkerControls(state) {
  const start = $('btn-start'), pause = $('btn-pause'), stop = $('btn-stop');
  const lbl = $('btn-pause-lbl'), pauseIcon = pause && pause.querySelector('.icon use');
  const setPauseAs = (resume) => {
    if (lbl) lbl.textContent = resume ? 'Reprendre' : 'Pause';
    if (pauseIcon) pauseIcon.setAttribute('href', resume ? '#i-play' : '#i-pause');
    if (pause) pause.title = resume ? 'Reprendre la contribution' : 'Suspendre le calcul (la page reste ouverte)';
  };
  switch (state) {
    case WorkerState.RUNNING:
      if (start) { start.disabled = true; start.hidden = true; }       // AC-8 : pas de « Démarrer » en cours
      if (pause) { pause.disabled = false; pause.hidden = false; }
      if (stop) stop.disabled = false;
      setPauseAs(false);
      break;
    case WorkerState.PAUSED:
      if (start) { start.disabled = true; start.hidden = true; }       // AC-8 : pas de « Démarrer » en pause
      if (pause) { pause.disabled = false; pause.hidden = false; }      // devient « ▶ Reprendre » (AC-6)
      if (stop) stop.disabled = false;
      setPauseAs(true);
      break;
    case WorkerState.STOPPED:
    case WorkerState.IDLE:
    default:
      if (start) { start.disabled = false; start.hidden = false; }     // re-proposer « Démarrer » (T3.4)
      if (pause) { pause.disabled = true; }
      if (stop) stop.disabled = true;
      setPauseAs(false);
      break;
  }
  // Verrou du groupe « charge » (modes + parties/leaf-batch/K) : ces knobs sont FIGÉS à la session
  // → éditables UNIQUEMENT à l'arrêt (sinon fausse affordance : on toucherait sans effet réel).
  // #boards reste actif (vraiment live). À l'arrêt, la note reflète preset reconnu vs personnalisé.
  const chargeLocked = (state === WorkerState.RUNNING || state === WorkerState.PAUSED);
  setChargeLocked(chargeLocked);
  setModeNote(chargeLocked ? 'réglable à l\'arrêt' : (currentMode ? '' : 'personnalisé'));
  renderCalibControls(state); // « Re-calibrer ma machine » : actif uniquement à l'arrêt (IDLE/STOPPED)
}

// ---- Calibration MESURÉE (validée) — presets dérivés du bench ---------------------------------
// nz_calib v3 = { v:3, ts, backend:'cpu'|'gpu', gpuAvailable, cores, coresLogical,
//   kMax|maxK, eco, normal, max, peakPps }.
//   • CPU : eco/normal/max = {K, pool, wave, numThreads:1} (modes = niveaux de K).
//   • GPU : eco/normal/max = {pool, wave, K} (modes = batch × K ; Max porte K=2 si pas d'OOM).
// Au 1er lancement (absent), le bench mesure le débit AGRÉGÉ réel → eco/normal/max. « Passer » =
// défauts conservateurs (pas de bench). Invalidation : version, âge > 30 j, ou backend qui diffère.
const CALIB_KEY = 'nz_calib';
const CALIB_MAX_AGE_MS = 30 * 24 * 3600 * 1000; // 30 jours → re-proposer (machine/navigateur évolue)
// nz_calib est VERSIONNÉ (v:3). On invalide proprement les schémas antérieurs (v1/v2, ou tout `v` ≠
// courant) → null = « pas de calib » → re-proposer le bench / défauts sûrs. La validité PAR-BACKEND
// (gpuAvailable ≠ gpuOk) et l'âge sont contrôlés au boot (calibStale), une fois gpuOk résolu.
const loadCalib = () => {
  try {
    const r = localStorage.getItem(CALIB_KEY);
    if (!r) return null;
    const c = JSON.parse(r);
    return (c && c.v === CALIB_VERSION) ? c : null; // v1/v2/legacy → ignoré (sera recalibré)
  } catch { return null; }
};
// Calib présente mais PÉRIMÉE : backend différent (GPU↔CPU), cœurs changés, ou trop vieille (> 30 j).
// → on re-propose le panneau (sans effacer : si l'utilisateur Passe, on garde l'ancienne en repli).
function calibStale(calib) {
  if (!calib) return false; // absente ≠ périmée (le 1er lancement gère l'absence séparément)
  if (!!calib.gpuAvailable !== !!gpuOk) return true; // backend a changé (ex. GPU activé/désactivé)
  const cores = navigator.hardwareConcurrency || null;
  if (Number.isFinite(calib.cores) && cores && calib.cores !== cores) return true;
  if (Number.isFinite(calib.ts) && (Date.now() - calib.ts) > CALIB_MAX_AGE_MS) return true;
  return false;
}
const saveCalib = (c) => { try { localStorage.setItem(CALIB_KEY, JSON.stringify(c)); } catch { /* localStorage off */ } };
const clearCalib = () => { try { localStorage.removeItem(CALIB_KEY); } catch { /* localStorage off */ } };

// ── Plan de balayage du bench (design validé) ──────────────────────────────────────────────────
// GPU : sweep BATCH (pool × wave figé GPU_WAVE) mono-worker {16,32,64,128,256 effectif}. wave=8 →
//   pool ∈ {2,4,8,16,32} donne batch ∈ {16,32,64,128,256} ; le batch effectif est plafonné à 256
//   par BatchedEvaluator. Puis K=2 au meilleur batch (test OOM → repli K=1).
const GPU_BATCH_POOLS = [2, 4, 8, 16, 32]; // × GPU_WAVE(8) = batch effectif {16,32,64,128,256}
const gpuBatchConfigs = () => GPU_BATCH_POOLS.map((pool) => ({ pool, wave: GPU_WAVE }));

// CPU : sweep K (Web Workers parallèles), nt=1, pool=CPU_POOL(8), wave=CPU_WAVE(1). On balaie K
// jusqu'au PLATEAU (bande ε) ou jusqu'à l'OOM/crash (détecté par l'orchestrateur → borne K_max).
// Échelle {1,2,4,6,8,…} bornée aux cœurs LOGIQUES (l'effondrement d'oversubscription est aux logiques,
// pas physiques — sweep 2D §CORRECTION). Cap dur K_HARD_CAP pour ne jamais exploser la RAM de l'onglet.
const K_HARD_CAP = 12;
function cpuKLadder() {
  const cores = navigator.hardwareConcurrency || 4;
  const cap = Math.max(1, Math.min(K_HARD_CAP, cores)); // jamais > cœurs logiques ni > cap dur
  const ladder = [1, 2, 4, 6, 8, 10, 12].filter((k) => k <= cap);
  if (!ladder.includes(cap)) ladder.push(cap); // garantit qu'on teste le cap (K_max candidat)
  return [...new Set(ladder)].sort((a, b) => a - b);
}

// Défauts CONSERVATEURS (AC-7) quand pas de calibration (« Passer » ou bench impossible). Le design
// fixe le DÉFAUT GPU à K=2 (le K=1 d'avant laissait +76% sur la table ; K=2 ne fait pas OOM sur 3090
// — la calib testera K=2→repli sur GPU faible). CPU défaut conservateur K=2 (sûr partout).
const CONSERVATIVE = {
  cpu: {
    eco: { K: 1, pool: CPU_POOL, wave: CPU_WAVE, numThreads: 1 },
    normal: { K: 2, pool: CPU_POOL, wave: CPU_WAVE, numThreads: 1 },
    max: { K: 2, pool: CPU_POOL, wave: CPU_WAVE, numThreads: 1 },
  },
  gpu: {
    eco: { pool: 4, wave: GPU_WAVE, K: 1 },   // batch 32 × K1 (charge basse)
    normal: { pool: 16, wave: GPU_WAVE, K: 1 }, // batch 128 × K1 (charge moyenne)
    max: { pool: 32, wave: GPU_WAVE, K: 2 },  // batch 256 × K2 (débit max — défaut K=2)
  },
};

// ---- Mode énergie (STORY-010) — presets RÉELS de charge --------------------------------------
// Les modes mappent sur les knobs de charge EXISTANTS (« Parties simultanées » #pool + « Leaf-batch »
// #wave), figés à la session → un changement s'applique au prochain (re)démarrage (Pause→Reprendre
// ou Arrêter→Démarrer). Aucun comportement simulé (canon §5) : on écrit de vraies valeurs dans les
// champs Avancés — le volontaire VOIT ce que le mode règle et peut affiner (→ « personnalisé »).
const MODE_BTN = { eco: 'btn-eco', normal: 'btn-normal', max: 'btn-max' };
let currentMode = 'normal';

// MODE_PRESETS = source de vérité des 3 modes : valeurs MESURÉES si nz_calib v3 présent, sinon défauts
// conservateurs selon le backend (gpuOk). Chaque preset porte {K, pool, wave[, numThreads]}. Recalculé
// car gpuOk est async ET nz_calib peut apparaître pendant la session (fin du bench). buildModePresets()
// lit l'état courant ; rebuildModePresets() le fige et réapplique le mode si à l'arrêt.
//   • CPU : K = niveau de parallélisme (Web Workers), pool=8, wave=1, numThreads=1.
//   • GPU : K (≤2), pool×wave = batch (Max porte K=2 si pas d'OOM).
let MODE_PRESETS = CONSERVATIVE.cpu; // valeur initiale prudente avant résolution de gpuOk (réécrite au boot)
function buildModePresets() {
  const calib = loadCalib();
  // On n'utilise une calib que si son backend correspond au backend courant (gpuOk) — sinon elle est
  // périmée (calibStale le re-proposera) et on retombe sur les défauts conservateurs du bon backend.
  if (calib && calib.eco && calib.normal && calib.max && (!!calib.gpuAvailable === !!gpuOk)) {
    return { eco: calib.eco, normal: calib.normal, max: calib.max };
  }
  return gpuOk ? CONSERVATIVE.gpu : CONSERVATIVE.cpu;
}

// K (Web Workers) à spawn au démarrage = K du mode courant (mesuré), sinon lecture du champ #workers
// (édition manuelle « personnalisée »), sinon défaut conservateur. Borné : CPU au cap dur (RAM onglet)
// & cœurs logiques ; GPU à 2 (chaque worker = 1 session WebGPU → K>2 = OOM, cf. AGENTS.md). Toujours ≥1.
function startK() {
  const preset = currentMode && MODE_PRESETS[currentMode];
  let k = (preset && Number.isFinite(preset.K) && preset.K > 0)
    ? preset.K
    : (parseInt($('workers') && $('workers').value, 10) || 1);
  const cap = gpuOk ? 2 : Math.max(1, Math.min(K_HARD_CAP, navigator.hardwareConcurrency || K_HARD_CAP));
  return Math.max(1, Math.min(Math.round(k), cap));
}

// numThreads (CPU/WASM) à envoyer au worker au démarrage. Design validé : CPU = nt=1 (le levier est K,
// pas numThreads). On lit le preset (qui porte numThreads:1) ; défaut 1. En GPU on n'envoie RIEN (null).
function startNumThreads() {
  const preset = currentMode && MODE_PRESETS[currentMode];
  if (preset && Number.isFinite(preset.numThreads) && preset.numThreads > 0) return preset.numThreads;
  return 1;
}

// Recalcule MODE_PRESETS puis, si AUCUNE session ne tourne (knobs non verrouillés), réécrit les champs
// #pool/#wave/#workers depuis le mode courant — sinon on respecte la session figée (effet au redémarrage).
function rebuildModePresets() {
  MODE_PRESETS = buildModePresets();
  // Sur CPU, le mode Max peut poser un K > 2 → on relève le plafond du champ #workers en conséquence
  // (sinon le navigateur clamperait la valeur écrite par setMode à l'ancien max=2). GPU reste à 2.
  const wEl = $('workers');
  if (wEl) {
    const cap = gpuOk ? 2 : Math.max(2, Math.min(K_HARD_CAP, navigator.hardwareConcurrency || K_HARD_CAP));
    wEl.max = String(cap);
  }
  const running = (currentWorkerState === WorkerState.RUNNING || currentWorkerState === WorkerState.PAUSED);
  if (!running && currentMode) setMode(currentMode);
  else syncModeFromFields();
}

function highlightMode(mode) {
  for (const [m, id] of Object.entries(MODE_BTN)) {
    const b = $(id); if (!b) continue;
    const on = (m === mode);
    b.classList.toggle('on', on);            // mode null → aucune surbrillance (= personnalisé)
    b.setAttribute('aria-pressed', on ? 'true' : 'false');
  }
}
function setModeNote(text) { const n = $('mode-note'); if (n) n.textContent = text || ''; }
// Verrou du groupe « charge » : désactive les 3 modes + les 3 champs session-fixés (#pool/#wave/
// #workers) quand une session tourne — ils ne prennent effet qu'au (re)démarrage (cf. infobulles).
// Appelé par renderWorkerControls (piloté par l'état). #boards exclu (appliqué en direct).
function setChargeLocked(locked) {
  for (const id of ['btn-eco', 'btn-normal', 'btn-max', 'pool', 'wave', 'workers']) {
    const el = $(id); if (el) el.disabled = locked;
  }
}

// « Re-calibrer ma machine » : ACTIF uniquement à l'arrêt (IDLE/STOPPED). En RUNNING/PAUSED il est
// DÉSACTIVÉ + une note « Arrête la contribution pour re-calibrer » (même pattern que setChargeLocked).
// Re-calibrer relance le bench multi-worker → impossible pendant une contribution (sessions concurrentes).
function renderCalibControls(state) {
  const btn = $('btn-recalib'), note = $('recalib-note');
  const locked = (state === WorkerState.RUNNING || state === WorkerState.PAUSED);
  if (btn) btn.disabled = locked;
  if (note) note.textContent = locked ? 'Arrête la contribution pour re-calibrer' : '';
}

// Clic sur un mode (AC-3, feedback < 500 ms) : écrit les knobs (pool/wave/K=workers) + surligne. Effet
// réel au (re)démarrage. K est posé sur #workers (le nb de Web Workers spawné à onStart).
function setMode(mode) {
  const preset = MODE_PRESETS[mode], btn = $(MODE_BTN[mode]);
  if (!preset || !btn || btn.disabled) return;
  currentMode = mode;
  if ($('pool') && Number.isFinite(preset.pool)) $('pool').value = preset.pool;
  if ($('wave') && Number.isFinite(preset.wave)) $('wave').value = preset.wave;
  if ($('workers') && Number.isFinite(preset.K)) $('workers').value = preset.K;
  if ($('kwarn')) $('kwarn').hidden = (parseInt($('workers') && $('workers').value, 10) || 1) < 2; // warn OOM si K≥2
  highlightMode(mode);
  setModeNote('');                          // setMode ne s'exécute qu'à l'arrêt (boutons verrouillés sinon)
  renderModeHeatWarning();                  // note « chaleur » visible seulement en Max (data-neutre)
}

// Réconcilie la surbrillance avec les valeurs RÉELLES des champs (édition manuelle Avancé → custom).
// Un mode « match » exige pool+wave+K identiques au preset (K = le champ #workers).
function syncModeFromFields() {
  const pool = parseInt($('pool') && $('pool').value, 10);
  const wave = parseInt($('wave') && $('wave').value, 10);
  const k = parseInt($('workers') && $('workers').value, 10) || 1;
  let match = null;
  for (const [m, p] of Object.entries(MODE_PRESETS)) {
    const pK = Number.isFinite(p.K) ? p.K : 1;
    if (p.pool === pool && p.wave === wave && pK === k) { match = m; break; }
  }
  currentMode = match;
  highlightMode(match);
  setModeNote(match ? '' : 'personnalisé');
  renderModeHeatWarning();                  // édition manuelle Avancé alignée sur Max → note chaleur
}

// ---- Bench de calibration (validée) — jauge speedtest + orchestration MULTI-WORKER --------------
// Au 1er lancement (pas de nz_calib v3), on MESURE le débit AGRÉGÉ réel. L'orchestration vit dans le
// THREAD PRINCIPAL (ici) car le levier validé = K Web Workers PARALLÈLES : on spawn K bundles en mode
// 'bench' mono-config et on SOMME leurs pos/s (le débit d'1 worker ne reflète PAS la machine).
//   • GPU : sweep batch (mono-worker) → meilleur batch ; puis K=2 à ce batch (OOM → repli K=1).
//   • CPU : sweep K {1,2,4,6,8,…} (chaque K = K workers spawnés) jusqu'au plateau OU OOM/crash → K_max.
// Dérivation déterministe (bande ε, la plus légère/petit-K gagne) = ui/calibration.mjs.
const CALIB_CIRC = 452.39; // circonférence de l'anneau (2π·72) — doit matcher stroke-dasharray du SVG
const CALIB_BURST_MS = 1500; // durée d'un burst de mesure par config (≈ budget historique)
let calibBusy = false;       // une calibration tourne (anti double-clic / re-entrée)
let calibAborted = false;    // « Passer » pendant le sweep → on coupe proprement
let calibLiveWorkers = [];   // bench workers VIVANTS (terminés en cas d'abandon) — borne la RAM
let calibPeak = 0;           // max des pos/s agrégés reçus → gros chiffre central qui grimpe

// Met l'anneau (dashoffset) + le % de progression. reduced-motion : la transition CSS est désactivée
// (le fill saute à la valeur), on ne fait donc pas d'animation continue — juste une MAJ discrète.
function setCalibProgress(frac) {
  const pct = Math.round(Math.max(0, Math.min(1, frac)) * 100);
  const fill = $('calib-fill');
  if (fill) fill.setAttribute('stroke-dashoffset', String(CALIB_CIRC * (1 - frac)));
  const prog = $('calib-progress');
  if (prog) prog.textContent = `Calibrage… ${pct}%`;
}
function setCalibPps(pps) {
  const el = $('calib-pps');
  if (el) el.textContent = pps > 0 ? nfFRcompact.format(pps) : '—';
}

// Affiche/masque le panneau #calib (révèle aussi qu'il n'est PAS dans .live → visible avant Démarrer).
function showCalibPanel(show) {
  const p = $('calib'); if (p) p.hidden = !show;
}

// Tue tous les bench workers encore vivants (abandon, fin de phase, erreur) → libère la RAM/sessions.
function killCalibWorkers() {
  for (const w of calibLiveWorkers) { try { w.terminate(); } catch { /* déjà mort */ } }
  calibLiveWorkers = [];
}

// Spawn UN bench worker mono-config et résout avec son pos/s mesuré (bench-done). C'est l'unité de
// mesure : K de ces workers en parallèle = le débit agrégé. Un worker qui CRASHE (onerror = OOM
// renderer) rejette → l'orchestrateur K interprète ça comme « K trop grand » et borne K_max.
// Garde-fou : timeout dur (le burst + warmup + download ne devraient jamais dépasser ~burst+25 s).
function runBenchWorker(baseUrl, config, numThreads) {
  return new Promise((resolve, reject) => {
    let settled = false;
    const w = new Worker('worker.bundle.js');
    calibLiveWorkers.push(w);
    const done = (fn, arg) => {
      if (settled) return; settled = true;
      clearTimeout(timer);
      try { w.terminate(); } catch { /* déjà mort */ }
      calibLiveWorkers = calibLiveWorkers.filter((x) => x !== w);
      fn(arg);
    };
    const timer = setTimeout(() => done(reject, new Error('timeout bench worker')), CALIB_BURST_MS + 40000);
    w.onmessage = (e) => {
      const m = e.data || {};
      if (m.type === 'bench-done') {
        const r = (m.results || [])[0] || { pps: 0, avgBatch: 0 };
        done(resolve, { pps: r.pps || 0, avgBatch: r.avgBatch || 0, modelVersion: m.modelVersion });
      } else if (m.type === 'bench-error') {
        done(reject, new Error(m.message || 'bench-error'));
      }
      // logs des bench workers silencieux (sinon flood en K élevé).
    };
    // onerror = le renderer du worker a crashé (OOM probable à K élevé) → rejet = signal « K trop grand ».
    w.onerror = (ev) => done(reject, new Error('worker crash (OOM ?) : ' + ((ev && ev.message) || 'F12')));
    w.postMessage({ type: 'bench', baseUrl, configs: [config], numThreads: numThreads ?? null, burstMs: CALIB_BURST_MS });
  });
}

// Mesure le débit AGRÉGÉ de K workers parallèles (même config) → somme des pos/s. Si UN worker
// rejette (OOM/crash/timeout), on remonte l'échec (l'orchestrateur K en déduit le plafond K_max).
async function measureAggregate(baseUrl, K, config, numThreads) {
  const perWorker = await Promise.all(
    Array.from({ length: K }, () => runBenchWorker(baseUrl, config, numThreads)),
  );
  return {
    agg: perWorker.reduce((s, r) => s + (r.pps || 0), 0),
    perWorker: perWorker.map((r) => r.pps || 0),
    modelVersion: perWorker[0] && perWorker[0].modelVersion,
  };
}

// ── Sweep CPU : K ∈ ladder, jusqu'au plateau (bande ε) OU OOM/crash → K_max + courbe agrégée. ──
async function sweepCpuK(baseUrl) {
  const ladder = cpuKLadder();
  const config = { pool: CPU_POOL, wave: CPU_WAVE };
  const kResults = [];
  let peak = 0, modelVersion = null;
  let plateauStreak = 0; // nb de K consécutifs sans gain net > ε → early-stop (on tient le plateau)
  for (let idx = 0; idx < ladder.length; idx++) {
    if (calibAborted) break;
    const K = ladder[idx];
    setCalibProgress(idx / ladder.length);
    const prog = $('calib-progress'); if (prog) prog.textContent = `Calibrage… K=${K} (${kResults.length}/${ladder.length})`;
    let m;
    try {
      m = await measureAggregate(baseUrl, K, config, 1); // CPU nt=1 (design validé)
    } catch (err) {
      // OOM/crash à ce K → on BORNE K_max au dernier K qui a TENU (les précédents sont dans kResults).
      log(`[calib] K=${K} a échoué (${(err && err.message) || err}) → K_max borné à ${kResults.length ? kResults[kResults.length - 1].K : 1}`);
      break;
    }
    kResults.push({ K, pps: m.agg });
    if (m.modelVersion != null) modelVersion = m.modelVersion;
    if (m.agg > calibPeak) { calibPeak = m.agg; setCalibPps(calibPeak); }
    log(`[calib] K=${K} → ${m.agg} pos/s agrégé (${m.perWorker.join(' + ')})`);
    // Early-stop : si le débit ne progresse plus (≤ ε au-dessus du pic) sur 2 K consécutifs, on tient
    // le plateau → inutile de continuer à empiler des workers (RAM) pour zéro gain.
    if (m.agg <= peak * (1 + 0.05)) { plateauStreak += 1; } else { plateauStreak = 0; }
    peak = Math.max(peak, m.agg);
    if (plateauStreak >= 2) { log('[calib] plateau K atteint → arrêt du sweep'); break; }
  }
  return { kResults, modelVersion };
}

// ── Sweep GPU : batch (mono-worker) → meilleur batch ; puis K=2 à ce batch (OOM → repli K=1). ──
async function sweepGpuBatch(baseUrl) {
  const configs = gpuBatchConfigs();
  const batchResults = [];
  let modelVersion = null;
  // Phase 1 : sweep batch mono-worker (K=1).
  for (let i = 0; i < configs.length; i++) {
    if (calibAborted) break;
    const cfg = configs[i];
    setCalibProgress((i / (configs.length + 1)));
    const prog = $('calib-progress'); if (prog) prog.textContent = `Calibrage… batch ${cfg.pool * cfg.wave} (${i + 1}/${configs.length})`;
    let m;
    try { m = await measureAggregate(baseUrl, 1, cfg, null); }
    catch (err) { log(`[calib] batch ${cfg.pool * cfg.wave} échec: ${(err && err.message) || err}`); continue; }
    batchResults.push({ pool: cfg.pool, wave: cfg.wave, pps: m.agg });
    if (m.modelVersion != null) modelVersion = m.modelVersion;
    if (m.agg > calibPeak) { calibPeak = m.agg; setCalibPps(calibPeak); }
    log(`[calib] batch ${cfg.pool * cfg.wave} → ${m.agg} pos/s`);
  }
  // Meilleur batch (le plus de pos/s) → test K=2 dessus.
  let maxK = 1;
  const best = batchResults.slice().sort((a, b) => b.pps - a.pps)[0];
  if (best && !calibAborted) {
    const prog = $('calib-progress'); if (prog) prog.textContent = 'Calibrage… K=2 (test)';
    setCalibProgress(configs.length / (configs.length + 1));
    try {
      const k1 = best.pps; // débit K=1 déjà mesuré au meilleur batch
      const m2 = await measureAggregate(baseUrl, 2, { pool: best.pool, wave: best.wave }, null);
      // K=2 retenu seulement s'il APPORTE (> +10 % vs K=1) ET n'a pas crashé. Sinon repli K=1.
      if (m2.agg > k1 * 1.1) { maxK = 2; if (m2.agg > calibPeak) { calibPeak = m2.agg; setCalibPps(calibPeak); } }
      log(`[calib] GPU K=2 @batch ${best.pool * best.wave} → ${m2.agg} pos/s (K=1=${k1}) → maxK=${maxK}`);
    } catch (err) {
      log(`[calib] GPU K=2 a échoué (${(err && err.message) || err}) → repli K=1`);
      maxK = 1;
    }
  }
  return { batchResults, maxK, modelVersion };
}

// Lance la calibration : choisit le backend (gpuOk), orchestre le sweep multi-worker, dérive le calib
// v3 par-backend, persiste, applique. NE soumet RIEN (self-play local). Idempotent (anti re-entrée).
// Pendant la calibration (bench en cours), on VERROUILLE les actions qui la fausseraient ou la
// perturberaient : Démarrer (lancerait une contribution concurrente qui vole des cœurs/le GPU à la
// mesure), le sélecteur Éco/Normal/Max, les champs de charge, et Re-calibrer. Levé à finish/skip.
function setCalibratingLock(locked) {
  const start = $('btn-start');
  if (start) { start.disabled = locked; if (locked) start.title = 'Calibrage en cours — démarrage indisponible'; else start.removeAttribute('title'); }
  const recalib = $('btn-recalib'); if (recalib) recalib.disabled = locked;
  setChargeLocked(locked); // modes Éco/Normal/Max + #pool/#wave/#workers
}

async function startCalibration() {
  if (calibBusy) return;
  calibBusy = true; calibAborted = false; calibPeak = 0;
  setCalibratingLock(true); // bloque Démarrer/modes/champs le temps de la mesure
  killCalibWorkers();
  setCalibPps(0); setCalibProgress(0);
  const start = $('calib-start'); if (start) start.disabled = true;
  const redo = $('calib-redo'); if (redo) redo.hidden = true;
  const skip = $('calib-skip'); if (skip) { skip.hidden = false; skip.disabled = false; } // « Passer » coupe le sweep
  const title = $('calib-title'); if (title) title.textContent = 'Mesure en cours…';
  const result = $('calib-result'); if (result) result.hidden = true;
  const prog = $('calib-progress'); if (prog) prog.textContent = 'Calibrage… 0%';

  const gpu = gpuOk;
  const baseUrl = ($('baseUrl') && $('baseUrl').value || '').trim();
  try {
    let calib = null;
    if (gpu) {
      const { batchResults, maxK, modelVersion } = await sweepGpuBatch(baseUrl);
      if (!calibAborted) {
        calib = deriveGpuCalibration(batchResults, { maxK, cores: navigator.hardwareConcurrency || null, peakPps: calibPeak });
      }
    } else {
      const { kResults } = await sweepCpuK(baseUrl);
      if (!calibAborted) {
        calib = deriveCpuCalibration(kResults, { coresLogical: navigator.hardwareConcurrency || null, cores: navigator.hardwareConcurrency || null });
      }
    }
    if (calibAborted) { return; } // « Passer » pendant le sweep → skipCalibration a déjà fait le ménage
    finishCalibration(calib, gpu);
  } catch (err) {
    log('[calib] ERREUR bench: ' + ((err && err.message) || err));
    skipCalibration(); // ne bloque jamais (AC-7) → défauts conservateurs
  } finally {
    killCalibWorkers();
    calibBusy = false;
  }
}

// Fin du bench : persiste nz_calib v3, applique aux presets, fige la jauge + le résumé. `calib` est
// déjà dérivé (null = rien d'exploitable → filet conservateur).
function finishCalibration(calib, gpu) {
  if (!calib) { skipCalibration(); return; }
  setCalibratingLock(false); // calibration finie → réactive Démarrer/modes/Re-calibrer
  saveCalib(calib);
  rebuildModePresets(); // MODE_PRESETS = valeurs mesurées + réapplique le mode courant (à l'arrêt)
  setCalibProgress(1);
  setCalibPps(calib.peakPps);
  const title = $('calib-title'); if (title) title.textContent = 'Ta machine est calibrée';
  const prog = $('calib-progress'); if (prog) prog.textContent = '';
  const result = $('calib-result');
  if (result) {
    // Résumé en unités tangibles : le pic + ce que Normal/Max règlent (K pour CPU, batch×K pour GPU).
    const tail = gpu
      ? `Normal = batch ${calib.normal.pool * calib.normal.wave}${calib.max.K >= 2 ? ' · Max = batch ' + (calib.max.pool * calib.max.wave) + ' × 2 workers' : ''}.`
      : `Normal = ${calib.normal.K} worker${calib.normal.K > 1 ? 's' : ''} · Max = ${calib.max.K} (jusqu'à K=${calib.kMax}).`;
    result.innerHTML = `Ta machine : <b>${nfFRcompact.format(calib.peakPps)} pos/s</b> · ${tail} `
      + `Réglages Éco / Normal / Max calés sur cette mesure.`;
    result.hidden = false;
  }
  const start = $('calib-start'); if (start) start.hidden = true;
  const skip = $('calib-skip'); if (skip) skip.hidden = true;
  const redo = $('calib-redo'); if (redo) { redo.hidden = false; redo.disabled = false; }
}

// « Passer » (AC-1/AC-7) : coupe un sweep en cours + garde les défauts conservateurs et masque le
// panneau. nz_calib reste ABSENT → réapparaîtra au prochain chargement (l'utilisateur n'a pas calibré).
function skipCalibration() {
  calibAborted = true;
  killCalibWorkers();
  calibBusy = false;
  setCalibratingLock(false); // « Passer »/erreur → réactive les contrôles
  rebuildModePresets(); // défauts conservateurs selon gpuOk (nz_calib absent)
  showCalibPanel(false);
}

// « Re-calibrer ma machine » : efface nz_calib + relance le bench (panneau révélé). Le garde-fou
// IDLE/STOPPED-only (bouton désactivé en RUNNING/PAUSED) est posé par renderCalibControls.
function recalibrate() {
  if (currentWorkerState === WorkerState.RUNNING || currentWorkerState === WorkerState.PAUSED) return; // sécurité
  clearCalib();
  showCalibPanel(true);
  startCalibration();
}

// ---- Footer contextuel (STORY-011) -----------------------------------------------------------
// Piloté par WorkerState (appelé par setWorkerState, AC-4 < 100 ms). Jamais de CTA contradictoire :
// en cours → « Partager » + « Ouvrir le tableau de bord » (PAS « Démarrer »). Le CTA de démarrage
// reste dans le hero (Dev Notes : ne pas dupliquer). Liens permanents = #footer-links (stables).
const DISCORD_URL = '#'; // TODO: URL du serveur Discord du projet (placeholder, comme #promo-discord)

function shareNanozero() {                          // AC T4 : partage natif (mobile) sinon copie
  const url = window.location.href;
  if (navigator.share) navigator.share({ title: 'NanoZero', url }).catch(() => copyLink(url));
  else copyLink(url);
}
function copyLink(url) {
  if (navigator.clipboard) navigator.clipboard.writeText(url).then(showFootCopied).catch(() => {});
}
function showFootCopied() {                          // confirmation discrète « Lien copié ✓ », auto-revert
  const a = $('foot-share'); if (!a) return;
  const prev = a.textContent; a.textContent = 'Lien copié ✓';
  setTimeout(() => { const s = $('foot-share'); if (s) s.textContent = prev; }, 2000);
}

function renderFooter(state) {
  const status = $('footer-status'), actions = $('footer-actions');
  if (!status || !actions) return;
  if (state === WorkerState.RUNNING) {
    status.textContent = '● Contribution en cours';
    status.className = 'foot-status running';
    actions.innerHTML = '<a href="#" id="foot-share">Partager NanoZero</a>'
      + '<a href="#" id="foot-dash" title="Tableau de bord communautaire — bientôt">Ouvrir le tableau de bord</a>';
  } else if (state === WorkerState.PAUSED) {
    status.textContent = '● En pause';
    status.className = 'foot-status';
    actions.innerHTML = '<a href="#" id="foot-resume">▶ Reprendre</a>';
  } else {                                           // STOPPED / IDLE : footer neutre, CTA dans le hero
    status.textContent = '';
    status.className = 'foot-status';
    actions.innerHTML = '';
  }
  const share = $('foot-share'); if (share) share.onclick = (e) => { e.preventDefault(); shareNanozero(); };
  const resume = $('foot-resume'); if (resume) resume.onclick = (e) => { e.preventDefault(); resumeWorker(); };
  const dash = $('foot-dash'); if (dash) dash.onclick = (e) => e.preventDefault();
}

const loadTotal = () => { try { return parseInt(localStorage.getItem('nz_total') || '0', 10) || 0; } catch { return 0; } };
const saveTotal = (n) => { try { localStorage.setItem('nz_total', String(n)); } catch { /* localStorage off */ } };

// ---- Pseudo « adresse ouverte » (STORY-007 T1) -----------------------------------------------
// Label PUBLIC cosmétique pour les crédits, persisté en localStorage (nz_pseudo) ; AUCUN compte,
// AUCUN token (canon §3). null = anonyme → aucun header X-Pseudo envoyé (rétro-compat). La SAISIE
// du pseudo (nudge + UI) est STORY-008 ; aujourd'hui nz_pseudo reste null → null transmis au worker.
const loadPseudo = () => { try { return localStorage.getItem('nz_pseudo') || null; } catch { return null; } };
const savePseudo = (p) => { try { localStorage.setItem('nz_pseudo', p); } catch { /* localStorage off */ } };

// ---- Identité contributeur stable (STORY-015) ------------------------------------------------
// `nz_client_id` = UUID d'installation persisté en localStorage : STABLE (survit aux redémarrages
// de worker), PARTAGÉ par les K Web Workers d'un onglet. Sert UNIQUEMENT à dériver un token
// pseudonyme ; le BRUT ne quitte jamais le navigateur (canon §3/§10, RGPD). Anonyme-friendly :
// aucun pseudo public, juste une clé de décompte côté serveur (table contributors).
const CLIENT_ID_KEY = 'nz_client_id';
const loadClientId = () => {
  try {
    let id = localStorage.getItem(CLIENT_ID_KEY);
    if (!id) { id = crypto.randomUUID(); localStorage.setItem(CLIENT_ID_KEY, id); }
    return id;
  } catch { return null; } // localStorage/crypto off → pas d'identité stable (non compté, dégradation propre)
};

// token = sha256(nz_client_id) tronqué à 16 hex. Déterministe → tous les K workers calculent le
// même. `crypto.subtle` exige un contexte sécurisé (HTTPS) — vrai en prod (Caddy TLS) ; sinon null.
async function computeContributorToken(clientId) {
  if (!clientId || !globalThis.crypto || !globalThis.crypto.subtle) return null;
  const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(clientId));
  return [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, '0')).join('').slice(0, 16);
}

// Précalcul au chargement (digest async) → prêt avant tout clic « Démarrer ». null = pas encore
// prêt / contexte non sécurisé → le worker n'enverra simplement pas X-Contributor (rétro-compat).
let contributorTokenCached = null;
(async () => { try { contributorTokenCached = await computeContributorToken(loadClientId()); } catch { /* non-bloquant */ } })();

// Suggestion de pseudo « rigolote » dérivée du clientId (STORY-008) : SEULEMENT un placeholder de
// nudge (jamais pré-rempli, jamais envoyé, jamais affiché publiquement — canon §4/§9). ASCII pur
// pour rester un pseudo valide côté serveur (`^[a-z0-9_-]{1,24}$`).
const SUGGEST_NOUN = ['cavalier', 'fou', 'tour', 'pion', 'gambit', 'roque', 'echec', 'finale', 'centre', 'flanc'];
const SUGGEST_ADJ = ['rapide', 'vaillant', 'ruse', 'agile', 'fute', 'tenace', 'vif', 'noble', 'hardi', 'serein'];
function defaultPseudoSuggestion(clientId) {
  let h = 0; const s = clientId || 'anon';
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return `${SUGGEST_NOUN[h % SUGGEST_NOUN.length]}-${SUGGEST_ADJ[(h >>> 8) % SUGGEST_ADJ.length]}-${(h >>> 16) % 100}`;
}

// ---- États d'identité (STORY-009) — 3 états NON mélangés (canon §3, SPEC §4) -------------------
// Le contributeur est dans EXACTEMENT un des 3 états, jamais mélangé (AC-4) :
//   identified_contributor : nz_pseudo non vide → pseudo affiché dans le header.
//   anonymous_contributor  : nz_total > 0 (a déjà contribué) mais pas de pseudo → invite DISCRÈTE.
//   public                 : ni l'un ni l'autre → CTA Démarrer seul, AUCUN pseudo, AUCUNE invite.
// ⚠️ Distinct de STORY-008 : le nudge est l'invitation INTRUSIVE engagement-gated ; #anon-invite est
// l'invite PERMANENTE discrète des anonymes actifs. Ni l'un ni l'autre n'affiche un pseudo inventé (AC-5).
const IdentityState = { PUBLIC: 'public', ANON: 'anonymous_contributor', IDENTIFIED: 'identified_contributor' };

// 3 branches EXCLUSIVES (AC-4) : retourne TOUJOURS une des 3 valeurs, jamais un état intermédiaire.
// Priorité : un pseudo (identifié) l'emporte sur le volume (anonyme), qui l'emporte sur le public.
function getIdentityState() {
  if (loadPseudo()) return IdentityState.IDENTIFIED;     // nz_pseudo non vide (loadPseudo → null si vide/absent)
  if (loadTotal() > 0) return IdentityState.ANON;        // a déjà soumis (nz_total persisté), mais anonyme
  return IdentityState.PUBLIC;                            // jamais contribué, pas de pseudo
}

// Reflète l'état d'identité dans le DOM (T2). Les zones d'identité PERMANENTES sont :
//   - #identity-pseudo (header) : visible UNIQUEMENT en identifié, avec la valeur de nz_pseudo.
//   - #anon-invite (sous le flux) : visible UNIQUEMENT en anonyme (texte discret, sans pseudo inventé AC-5).
// #btn-start reste géré par la machine WorkerState (renderWorkerControls) ; on ne le force pas ici pour
// ne pas écraser l'état RUNNING/PAUSED — en état public/IDLE il est déjà visible/actif (AC-1).
// state optionnel → recalculé si absent (appel défensif STORY-008 sans argument).
function renderIdentityState(state) {
  const s = state || getIdentityState();
  const pseudoEl = $('identity-pseudo');
  const anonEl = $('anon-invite');
  const identified = (s === IdentityState.IDENTIFIED);
  const anon = (s === IdentityState.ANON);
  if (pseudoEl) {
    if (identified) { pseudoEl.textContent = loadPseudo(); pseudoEl.hidden = false; } // pseudo RÉEL only (AC-5)
    else { pseudoEl.textContent = ''; pseudoEl.hidden = true; }                        // public/anonyme : aucun pseudo
  }
  if (anonEl) anonEl.hidden = !anon;   // invite discrète UNIQUEMENT pour l'anonyme actif (AC-2) ; jamais en public (AC-1)
}

// ---- Nudge pseudo (STORY-008) — état module ---------------------------------------------------
// Génération cible (gen-N+1) pour la copy du nudge, alimentée par /stats/season (applySeasonStats).
// null = endpoint dégradé/non encore reçu → fallback « la prochaine génération » dans la copy.
let currentTargetGen = null;
// Soumissions cumulées dans CETTE session (somme des K workers), mise à jour dans onWorkerMessage.
// Réutilise le compteur de flux existant (STORY-004) — pas de tracking parallèle.
let submittedCount = 0;

// Bannière inline (au-dessus de la grille) : erreur (rouge) ou info (cyan). Visible, pas dans le log.
function showAlert(msg, info) {
  $('alert-msg').textContent = msg;
  $('alert').classList.toggle('info', !!info);
  $('alert').hidden = false;
}

// Dernier comptage « vérifié » connu (server-authoritative). null = jamais reçu (mode dégradé).
// Sert à n'animer le flash or QUE sur une AUGMENTATION réelle (STORY-004 AC-3 / STORY-005 cas 2-5).
let lastValidated = null;

const nfFRcompact = new Intl.NumberFormat('fr-FR'); // séparateurs de milliers pour le flux

// Met à jour la composition « Ma contribution » (STORY-004 + STORY-005).
//   submitted = parties soumises dans CETTE session (compteur worker, somme des K workers).
//   grandTotal = cumul persisté toutes sessions (localStorage nz_total) — affiché au pied.
//   validated  = comptage VÉRIFIÉ par session renvoyé par le serveur, ou null si non câblé.
// MODE DÉGRADÉ (validated === null) : seul #flow-sent visible, libellé « soumis · en attente de
//   vérification », ton neutre, AUCUN flash or — on ne célèbre jamais un chiffre qui peut chuter.
// MODE COMPLET (validated >= 0) : envoyées → vérifiées (or) → en cours ; flash or si vérifiées ↑.
function updateContributionFlow(submitted, grandTotal, validated) {
  const flow = $('flow');
  if (validated == null) {
    // ── Dégradé (STORY-005 AC-1/AC-2) — l'état shippable tant que le gate D.2 n'est pas câblé. ──
    flow.classList.remove('full');
    $('flow-sent').textContent = nfFRcompact.format(submitted);
    $('flow-sent-lbl').textContent = 'soumis · en attente de vérification';
    // pied : cumul navigateur en bleu (AC-6), libellé « soumises » (jamais « vérifiées »), sans impact or.
    $('flow-total').textContent = nfFRcompact.format(grandTotal);
    $('flow-impact').textContent = '';
    lastValidated = null;
    return;
  }
  // ── Complet (STORY-004) — s'active automatiquement dès que le serveur fournit un comptage. ──
  flow.classList.add('full');
  const pending = Math.max(0, submitted - validated);
  $('flow-sent-lbl').textContent = 'envoyées';
  $('flow-sent').textContent = nfFRcompact.format(submitted);
  $('flow-verified').textContent = nfFRcompact.format(validated);
  $('flow-pending').textContent = nfFRcompact.format(pending);
  // Flash or UNIQUEMENT si « vérifiées » a augmenté (AC-3 ; jamais au chargement ni en boucle).
  // MÊME déclencheur pour le flash vert du board primaire (STORY-006 T3) : board 0 « relié au
  // feedback ». La 1re mesure (lastValidated == null) n'anime jamais (init silencieuse, canon §2).
  // ⚠️ Dégradé attendu : validated vient de m.sessionVerified, null aujourd'hui (gate D.2 absent)
  // → on n'atteint jamais cette branche, donc AUCUN flash pour l'instant — VOULU, comme le flash or.
  if (lastValidated != null && validated > lastValidated) {
    flashGold('flow-verified');
    flashPrimaryBoard();
  }
  lastValidated = validated;
  // pied : cumul navigateur (bleu) + impact tangible en menthe (AC-5) — jamais en % du total collectif.
  $('flow-total').textContent = nfFRcompact.format(grandTotal);
  const games = Math.round(validated / 60); // ≈ parties explorées (constante affinable, STORY-004 T3.4)
  $('flow-impact').textContent = games > 0 ? ` · ≈ ${nfFRcompact.format(games)} partie${games === 1 ? '' : 's'} explorée${games === 1 ? '' : 's'}` : '';
}

// Flash or 300 ms sur le compteur « vérifiées » (réutilise le keyframe CSS, neutralisé si reduced-motion).
function flashGold(id) {
  const el = $(id);
  el.classList.remove('flash-gold'); void el.offsetWidth; // re-trigger l'animation si vérif consécutives
  el.classList.add('flash-gold');
  setTimeout(() => el.classList.remove('flash-gold'), 320);
}

// ---- Flash de vérification sur l'échiquier primaire (STORY-006 T2/T4) -------------------------
// Le board 0 est « relié au feedback » du volontaire (canon §2). Quand une position produite ici
// vient d'être vérifiée (validated ↑), on : (a) anneau VERT MENTHE ~1,4 s (.board-verified) ;
// (b) label sous le board → « Une position produite ici vient d'être vérifiée » pendant le flash,
// puis remise du texte par défaut (coup + éval, via le dernier état connu). Le flash s'applique au
// DOM d'une card DÉJÀ créée — il ne déclenche AUCUN re-render et ne touche PAS la position affichée.
// reduced-motion : on saute l'anneau (AC-4), seul le texte change ; le label revient ensuite.
// Réentrant : un nouveau flash relance la fenêtre proprement (cas de vérifs consécutives).
const VERIFIED_COPY = 'Une position produite ici vient d\'être vérifiée';
const FLASH_MS = 1400;
let primaryFlashTimer = null;

function reducedMotion() {
  try { return window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches; }
  catch { return false; }
}

// Remet le label du board 0 à son texte par défaut (coup + éval) en re-rendant depuis le dernier
// état connu si on l'a (n'altère pas la position : renderBoard est idempotent), sinon vide le marqueur.
function restorePrimaryLabel() {
  const entry = boardEls.get(0);
  if (!entry) return;
  entry.label.classList.remove('verified');
  const st = lastState.get(0);
  if (st) renderBoard(0, st.gameId, st.fen, st.ply, st.lastMove, st.wdl); // réécrit le label par défaut
}

function flashPrimaryBoard() {
  const entry = boardEls.get(0);
  if (!entry) return; // board primaire pas encore créé (aucun coup reçu) → no-op silencieux
  const card = entry.card, label = entry.label;
  // Label de vérification (toujours, même en reduced-motion — c'est le feedback textuel, AC-4).
  label.classList.add('verified');
  label.textContent = VERIFIED_COPY;
  // Anneau menthe : sauté si reduced-motion (AC-4) ; sinon re-trigger propre si flash consécutif.
  if (!reducedMotion()) {
    card.classList.remove('board-verified'); void card.offsetWidth;
    card.classList.add('board-verified');
  }
  clearTimeout(primaryFlashTimer);
  primaryFlashTimer = setTimeout(() => {
    card.classList.remove('board-verified');
    restorePrimaryLabel(); // remet « Coup N · favori % » (texte par défaut)
  }, FLASH_MS);
}

function log(msg) {
  const el = $('log');
  el.textContent += msg + '\n';
  el.scrollTop = el.scrollHeight;
}

function ensureBoard(slot) {
  if (boardEls.has(slot)) return boardEls.get(slot);
  if (slot >= displayCap || boardEls.size >= MAX_BOARDS) return null; // n'affiche que les premiers slots
  const card = document.createElement('div'); card.className = 'card';
  // Le board 0 est l'échiquier PRIMAIRE « relié au feedback » (STORY-006 T1) : cible du flash vert
  // de vérification (flashPrimaryBoard). On l'identifie/marque ici, à la création de la card.
  if (slot === 0) { card.id = 'board-0'; card.classList.add('primary-board'); }
  card.style.animationDelay = (boardEls.size * 0.04) + 's'; // apparition décalée (N3)
  const label = document.createElement('div'); label.className = 'label';
  const grid = document.createElement('div'); grid.className = 'board';
  for (let i = 0; i < 64; i++) { const c = document.createElement('div'); c.className = 'sq'; grid.appendChild(c); }
  // barre d'éval W/D/L à GAUCHE, orientée Blancs-en-bas (ordre DOM ci-dessous : L haut · D · W bas)
  const row = document.createElement('div'); row.className = 'boardrow';
  const evalbar = document.createElement('div'); evalbar.className = 'evalbar';
  const ew = document.createElement('div'); ew.className = 'ev ev-w';
  const ed = document.createElement('div'); ed.className = 'ev ev-d';
  const el = document.createElement('div'); el.className = 'ev ev-l';
  // ordre haut→bas : L (Noirs, en haut = côté noir) · D · W (Blancs, en bas = côté blanc), comme lichess
  evalbar.append(el, ed, ew); row.append(evalbar, grid);
  // Barre de « réflexion » SOUS l'échiquier : progression de la recherche du coup courant (done/sims).
  // Montre que ça travaille pendant la recherche (longue sur CPU) SANS écraser le label du haut.
  const thinkBar = document.createElement('div'); thinkBar.className = 'think-bar';
  const thinkFill = document.createElement('div'); thinkFill.className = 'think-fill'; thinkBar.appendChild(thinkFill);
  card.appendChild(label); card.appendChild(row); card.appendChild(thinkBar); $('grid').appendChild(card);
  const entry = { card, grid, label, ew, ed, el, thinkBar, thinkFill }; boardEls.set(slot, entry); return entry;
}

function renderBoard(slot, gameId, fen, ply, lastMove, wdl) {
  const entry = ensureBoard(slot);
  if (!entry) return;
  if (entry.thinkFill) entry.thinkFill.style.width = '0%'; // coup joué → la barre repart de 0 pour le prochain
  let cells;
  try { cells = fenToCells(fen); } catch { return; }
  // dernier coup : aplat jaune-vert lichess sur DÉPART + ARRIVÉE (motif SOTA — tint pleine case,
  // from=to, sous la pièce). Le jaune-vert tranche sur les deux teintes de bois crème/marron (l'or
  // précédent s'y fondait) et reste lisible autour de la pièce qui couvre ~70 % de la case d'arrivée.
  let fromIdx = -1, toIdx = -1;
  if (lastMove && lastMove.from && lastMove.to) {
    const fr = algToFileRank(lastMove.from); const to = algToFileRank(lastMove.to);
    if (fr) fromIdx = (7 - fr.rank) * 8 + fr.file;
    if (to) toIdx = (7 - to.rank) * 8 + to.file;
  }
  const sqs = entry.grid.children;
  for (let i = 0; i < 64; i++) {
    const c = cells[i];
    const piece = c.piece ? `url(pieces/${c.piece}.svg)` : '';
    const hi = (i === fromIdx || i === toIdx) ? LASTMOVE : '';
    // pièce AU-DESSUS de l'aplat (1ʳᵉ couche listée = dessus) ; '' = case nue
    sqs[i].style.backgroundImage = [piece, hi].filter(Boolean).join(', ');
    sqs[i].className = 'sq ' + (c.dark ? 'dark' : 'light');
  }
  // Pendant le flash de vérification (STORY-006), le board 0 affiche la copy « …vient d'être
  // vérifiée » : on met à jour la POSITION mais on NE touche PAS au label (restorePrimaryLabel le
  // remettra à la fin du flash). Marqueur = classe .verified posée par flashPrimaryBoard().
  const labelLocked = entry.label.classList.contains('verified');
  if (wdl) {
    const wp = (wdl.w || 0) * 100, dp = (wdl.d || 0) * 100, lp = (wdl.l || 0) * 100;
    entry.ew.style.height = wp + '%'; entry.ed.style.height = dp + '%'; entry.el.style.height = lp + '%';
    // libellé grand public : le camp favori + son %, détail W/D/L en tooltip survol
    const mx = Math.max(wp, dp, lp);
    const fav = mx === wp ? `Blancs ${Math.round(wp)}%` : mx === lp ? `Noirs ${Math.round(lp)}%` : `Nulle ${Math.round(dp)}%`;
    if (!labelLocked) entry.label.textContent = `Coup ${ply} · ${fav}`;
    entry.card.title = `Blancs ${Math.round(wp)}% · Nulle ${Math.round(dp)}% · Noirs ${Math.round(lp)}%`;
  } else if (!labelLocked) {
    entry.label.textContent = `Coup ${ply}`;
  }
}

// Applique un nouveau nb d'échiquiers affichés EN DIRECT : retire ceux au-delà du cap ; ceux
// en-deçà réapparaissent au prochain update (ensureBoard les recrée). Map : delete pendant
// for...of est sûr.
function applyDisplayCap(n) {
  displayCap = n;
  // retire les boards au-delà du cap
  for (const [slot, entry] of boardEls) {
    if (slot >= n) { entry.card.remove(); boardEls.delete(slot); }
  }
  // affiche IMMÉDIATEMENT les slots désormais visibles dont on a déjà un état caché (sans attendre un coup)
  for (const [slot, st] of lastState) {
    if (slot < n && !boardEls.has(slot)) renderBoard(slot, st.gameId, st.fen, st.ply, st.lastMove, st.wdl);
  }
}

const consent = createConsentController({
  onStart: () => {
    showCalibPanel(false); // démarrage du self-play → masque le panneau de calibration (son job est fait)
    baseTotal = loadTotal(); // la session repart de 0 ; on cumule sur le total persisté
    submittedCount = 0;      // STORY-008 : compteur de session remis à 0 à chaque (re)démarrage
    markSessionStart();      // STORY-008 T6.1 : horodate cette session (détection du retour au prochain chargement)
    runStarted = false; anyError = false; endedCount = 0;
    displayCap = Math.max(1, parseInt($('boards').value, 10) || 4); // nb d'échiquiers à afficher (défaut 4, STORY-013)
    const poolVal = Math.max(1, parseInt($('pool').value, 10) || 32);
    // K (Web Workers parallèles) = le K du mode courant (mesuré, BORNÉ au K_max calibré → jamais d'OOM
    // en contribution réelle) ou le champ #workers. Plafond : GPU=2 (chaque worker = 1 session WebGPU →
    // K>2 = OOM onglet) ; CPU = K_HARD_CAP & cœurs logiques (le levier validé est K, cf. CALIBRATION-DESIGN).
    const K = startK();
    const waveVal = Math.max(1, parseInt($('wave').value, 10) || 1);
    // numThreads (CPU/WASM only). Design validé : CPU = nt=1 (le levier est K, pas numThreads). En GPU
    // on n'envoie RIEN (null) → le worker laisse ORT (WASM only).
    const numThreads = gpuOk ? null : startNumThreads();
    contributedByWorker = new Array(K).fill(0);
    perfByWorker = new Array(K).fill(null);
    ppsEma = null; // repart propre à chaque session
    boardEls.clear(); lastState.clear(); $('grid').innerHTML = ''; $('log').textContent = '';
    lastValidated = null; updateContributionFlow(0, baseTotal, null); $('pps').textContent = '—'; // flux remis à 0 (mode dégradé), cumul = total persisté
    $('grid').classList.remove('stopped');
    $('hero').classList.add('compact'); // rétrécit le hero (pitch+trust masqués) → place aux boards
    $('live').classList.add('on'); // révèle la zone live (fondu)
    // STORY-013 : échiquiers visibles DÈS le démarrage (position de départ en placeholder), sans
    // attendre le 1er coup du worker. Remplacés par les vrais updates au 1er message 'boards'.
    for (let slot = 0; slot < displayCap; slot++) renderBoard(slot, '', START_FEN, 0, null, null);
    $('ingame').textContent = `${K} workers × ${poolVal} parties = ${K * poolVal} en cours · ${displayCap} affichées`;
    // bannière info si pas de GPU (repli CPU) — message POSITIF, pas une erreur
    if (gpuOk) $('alert').hidden = true;
    else showAlert('GPU non disponible — calcul en mode CPU (plus lent). Tu contribues quand même 👍', true);
    setWorkerState(WorkerState.RUNNING); // start caché, pause+stop actifs (AC-8) + verrou du groupe charge
    scheduleNudgeTimer();                // STORY-008 condition A : arme le timer 10 min (une seule fois)
    $('status').textContent = 'Chargement du modèle…';
    workers.forEach((w) => w.terminate()); workers = []; // défensif : repart propre
    const base = 'browser-' + Math.random().toString(36).slice(2, 8);
    for (let k = 0; k < K; k++) {
      const w = new Worker('worker.bundle.js'); // bundle esbuild (classic worker) — 1 session ORT/WebGPU par worker
      w.onmessage = (e) => onWorkerMessage(e, k);
      w.onerror = (e) => log(`[w${k}] Erreur worker: ` + (e.message || '(F12)'));
      w.postMessage({
        type: 'start',
        baseUrl: $('baseUrl').value.trim(),
        workerId: base + '-w' + k, // identité distincte par worker (claim/submit indépendants côté jobserver)
        poolSize: poolVal,
        // sims : NON envoyé — dicté par le serveur (job.num_sims), appliqué par coup dans run-loop.mjs.
        boards: k === 0 ? displayCap : 0, // SEUL le worker 0 calcule l'éval W/D/L + pousse des boards (échantillon)
        waveSize: waveVal, // leaf-batching K (session, figé au start)
        workerIndex: k,
        pseudo: loadPseudo(), // STORY-007 — label crédits opt-in (null = anonyme → aucun header X-Pseudo)
        contributorToken: contributorTokenCached, // STORY-015 — identité stable (décompte), partagée par les K workers
        numThreads, // calibration incr.1 — CPU/WASM only (null en GPU → worker laisse ORT)
      });
      workers.push(w);
    }
  },
  onStop: () => {
    // ARRÊT RAPIDE (STORY-013) : terminate IMMÉDIAT des workers — kill le coup MCTS en cours (~1200
    // sims) sans l'attendre. La partie en cours n'est de toute façon pas soumise (run-loop : shouldStop
    // pendant la partie → pas de submit). On vide `workers` pour que le handler 'done' ne re-déclenche
    // pas finalizeStopped (la garde workers.length>0 devient fausse). pendingHaltState décide PAUSED/STOPPED.
    workers.forEach((w) => { try { w.terminate(); } catch { /* déjà mort */ } });
    workers = [];
    const pausing = pendingHaltState === WorkerState.PAUSED;
    $('status').textContent = pausing ? 'Mise en pause…' : 'Arrêté';
    $('btn-pause').disabled = true; $('btn-stop').disabled = true;
    // Finalisation UI (consent.reset + état final) au tick suivant : hors du call stack consent.stop.
    setTimeout(finalizeStopped, 0);
  },
});

// Suspend la contribution (AC-4/AC-5) : marque la cible PAUSED puis emprunte le chemin d'arrêt
// worker existant (consent.stop → onStop → finalizeStopped). La page reste, Reprendre relance.
function pauseWorker() {
  if (currentWorkerState !== WorkerState.RUNNING) return;
  pendingHaltState = WorkerState.PAUSED;
  consent.stop();
}

// Reprend après une pause (AC-6) : relance une session à l'identique via le MÊME chemin que
// Démarrer (consent.start → onStart). onStart relit les champs Avancé (config inchangée).
function resumeWorker() {
  if (currentWorkerState !== WorkerState.PAUSED) return;
  consent.start();
}

// Arrêt terminal (AC-7) : cible STOPPED puis chemin d'arrêt worker existant.
function stopWorker() {
  if (currentWorkerState !== WorkerState.RUNNING && currentWorkerState !== WorkerState.PAUSED) return;
  if (currentWorkerState === WorkerState.PAUSED) {
    // Déjà à l'arrêt côté worker (pas de session vivante) → transition d'UI directe vers STOPPED.
    setWorkerState(WorkerState.STOPPED);
    $('status').textContent = 'Arrêté';
    return;
  }
  pendingHaltState = WorkerState.STOPPED;
  consent.stop();
}

function onWorkerMessage(e, k) {
  const m = e.data || {};
  if (m.type === 'progress') {
    contributedByWorker[k] = m.contributed; // recompte par worker → somme = total session
    const submitted = contributedByWorker.reduce((a, b) => a + (b || 0), 0);
    submittedCount = submitted; // STORY-008 : compteur de session lu par checkNudgeTrigger (condition A)
    const grand = baseTotal + submitted;
    const wasZeroTotal = loadTotal() === 0; // STORY-009 T4.3 : capture l'AVANT pour détecter le passage 0→>0
    saveTotal(grand); // persiste → survit au reload (les workers, eux, ne survivent pas à un reload)
    // STORY-009 T4.3 : 1er batch (nz_total passe de 0 à > 0) → public devient anonymous_contributor.
    // On ne re-rend que sur la TRANSITION (pas à chaque progress) ; sans pseudo, l'invite #anon-invite
    // apparaît. Avec un pseudo, getIdentityState reste identifié → no-op visible (header déjà posé).
    if (wasZeroTotal && grand > 0) renderIdentityState();
    // « vérifié » (or) = comptage server-authoritative par session (gate D.2) — PAS encore câblé : le
    // worker n'envoie pas ce champ → m.sessionVerified est undefined → null → MODE DÉGRADÉ (STORY-005).
    // On ne célèbre JAMAIS un « envoyé » (canon §1). Quand le serveur le fournira, le mode complet
    // s'activera tout seul (envoyées → vérifiées → en cours) sans toucher cette page.
    const validated = (m.sessionVerified == null) ? null : Number(m.sessionVerified);
    updateContributionFlow(submitted, grand, validated);
    $('mv').textContent = 'v' + m.modelVersion;
  } else if (m.type === 'perf') {
    perfByWorker[k] = m;
    // Agrégat des K workers : pps SOMMÉ (débit total) ; inférence = vrai session.run moyen (~30-40 ms,
    // ≈ constant) ; GPU% = somme des occupations réelles (streams concurrents) plafonnée à 100. La
    // « fenêtre » (mur absorbant le JS) n'est PAS affichée en multi-workers (cf. browser-worker-gpu-metric-jsbound).
    const live = perfByWorker.filter(Boolean);
    const pps = live.reduce((a, p) => a + (p.pps || 0), 0);
    // EMA fortement amortie : le débit PULSE réellement (leaf-batching = vagues de feuilles → flush
    // → accalmie) et la fenêtre 2s est bruitée. alpha=0.1 ≈ constante de temps ~19 s → l'affichage
    // se pose vraiment, tout en s'adaptant si le débit change pour de bon. 1er point = valeur brute.
    ppsEma = ppsEma == null ? pps : 0.1 * pps + 0.9 * ppsEma;
    const real = live.filter((p) => p.realMsPerCall != null);
    const inf = real.length ? real.reduce((a, p) => a + p.realMsPerCall, 0) / real.length : null;
    const gpu = live.reduce((a, p) => a + (p.gpuPct || 0), 0);
    let t = Math.round(ppsEma) + ' pos/s';
    if (workers.length > 1) t += ` (${workers.length}w)`;
    // « GPU % » n'a de sens qu'avec un VRAI GPU. En CPU/WASM mono-thread, gpuPct = runMs/mur ≈ 100 %
    // (l'inférence EST le temps mur) → trompeur et contredit le bandeau « GPU non disponible ». Masqué.
    if (gpu && gpuOk) t += ` · GPU ~${Math.min(100, Math.round(gpu))}%`;
    if (inf != null) t += ` · inférence ${inf.toFixed(1)} ms`;
    $('pps').textContent = t;
  } else if (m.type === 'model') {
    $('mv').textContent = 'v' + m.version; // affiché dès le téléchargement, sans attendre le 1er submit
    if (!runStarted) $('status').textContent = `Modèle v${m.version} chargé — démarrage des parties…`;
  } else if (m.type === 'boards') {
    if (!runStarted) { runStarted = true; $('status').textContent = 'En cours'; } // 1er coup joué (worker 0)
    for (const b of m.batch) { lastState.set(b.slot, b); renderBoard(b.slot, b.gameId, b.fen, b.ply, b.lastMove, b.wdl); }
  } else if (m.type === 'thinking') {
    // « Réflexion » live : la recherche du coup courant progresse (utile sur CPU où le 1er coup est long).
    // On remplit la BARRE SOUS l'échiquier (pas le label du haut, qui garde « Coup X · fav »).
    const entry = boardEls.get(m.slot);
    if (entry && entry.thinkFill) {
      const pct = m.total ? Math.min(100, (m.done / m.total) * 100) : 0;
      entry.thinkFill.style.width = pct + '%';
    }
  } else if (m.type === 'log') {
    log(`[w${k}] ` + m.message);
  } else if (m.type === 'done' || m.type === 'error') {
    if (m.type === 'error') { anyError = true; log(`[w${k}] ERREUR: ` + m.message); showAlert('Problème : ' + (m.message || 'erreur inconnue'), false); }
    const s = m.stats;
    if (s) log(`[w${k}] stats: parties=${s.submitted} avgBatch=${(s.avgBatch || 0).toFixed(1)} maxBatch=${s.maxBatch} idle=${s.idle} skip=${s.skipped} err=${s.errors}`);
    endedCount += 1;
    if (endedCount >= workers.length && workers.length > 0) finalizeStopped(); // tous les workers ont fini
  }
}

// Tous les workers ont terminé (Pause, Stop, fin naturelle ou erreur) → finalise l'UI.
// pendingHaltState distingue une PAUSE (page gardée, Reprendre possible) d'un ARRÊT terminal.
// Une erreur force toujours l'état terminal STOPPED (pas de reprise d'une session cassée).
function finalizeStopped() {
  workers.forEach((w) => w.terminate()); workers = [];
  consent.reset(); // libère la machine consent (sinon un futur start serait un no-op)
  $('grid').classList.add('stopped'); // parties figées → grisées (lève l'ambiguïté « en cours »)
  const paused = pendingHaltState === WorkerState.PAUSED && !anyError;
  if (paused) {
    // Pause : la page reste telle quelle (hero compact conservé), Reprendre relancera (AC-5/AC-6).
    setWorkerState(WorkerState.PAUSED);
    $('status').textContent = 'En pause';
  } else {
    $('hero').classList.remove('compact'); // ré-étend le hero (pitch+trust de nouveau lisibles)
    setWorkerState(WorkerState.STOPPED);   // AC-7 : état de fin « Arrêté »
    $('status').textContent = anyError ? 'Arrêté (erreur)' : 'Arrêté';
  }
  pendingHaltState = WorkerState.STOPPED; // réinitialise pour la prochaine halte (défaut = arrêt)
}

// ---- Nudge pseudo (STORY-008) — invitation à choisir un pseudo sur l'engagement --------------
// Modèle « adresse ouverte » (canon §3) : le pseudo est un LABEL public cosmétique, sans compte ni
// token. Le nudge se déclenche sur l'engagement RESSENTI (canon §4), jamais sur un seuil de volume :
//   (A) session ≥ 10 min ET ≥ 1 soumission, OU (B) 1er retour (nz_last_session ancien > RETURN_GAP_MS).
// Affiché UNE seule fois par appareil (nz_nudge_shown) ; jamais si nz_pseudo existe déjà (AC-7).
const NUDGE_SHOWN_KEY = 'nz_nudge_shown';
const LAST_SESSION_KEY = 'nz_last_session';
const NUDGE_TIMER_MS = 10 * 60 * 1000; // condition A : ≥ 10 min de session (AC-1)
const RETURN_GAP_MS = 4 * 60 * 60 * 1000; // condition B : retour si la dernière session date de > 4 h (T6.2)
// Charset autorisé APRÈS normalisation (lowercase only) — MIROIR EXACT du serveur (STORY-007
// _PSEUDO_RE = ^[a-z0-9_-]{1,24}$). On valide la forme NORMALISÉE (post-toLowerCase), pas la saisie
// brute : un pseudo en MAJUSCULES est lowercasé côté serveur, donc on teste la même chose ici.
const PSEUDO_RE = /^[a-z0-9_-]{1,24}$/; // validation pseudo (T4.1 ; identique au plumbing STORY-007)
// Blocklist MIROIR du serveur (nanozero_jobserver/config.py → PSEUDO_BLOCKLIST). À GARDER
// SYNCHRONISÉE : un pseudo qui CONTIENT (sous-chaîne) l'un de ces 14 termes est rejeté côté client
// AUSSI, sinon le serveur stockerait une forme normalisée différente de ce que l'utilisateur croit.
const PSEUDO_BLOCKLIST = [
  'nigger', 'nigga', 'faggot', 'fuck', 'shit', 'cunt', 'bitch',
  'rape', 'nazi', 'hitler', 'retard', 'slut', 'whore', 'pedo',
];
// Reproduit EXACTEMENT le pipeline serveur normalize_pseudo : NFC → strip → lowercase → troncature
// 24 → regex [a-z0-9_-]{1,24} → blocklist (sous-chaîne). Retourne la chaîne NORMALISÉE (lowercase =
// ce que le serveur stockera) ou null si vide / hors charset / offensant. JAMAIS d'exception.
function normalizePseudo(raw) {
  const s = (raw || '').normalize('NFC').trim().toLowerCase().slice(0, 24);
  if (!PSEUDO_RE.test(s)) return null;
  if (PSEUDO_BLOCKLIST.some((bad) => s.includes(bad))) return null;
  return s;
}
let nudgeTimerScheduled = false; // évite d'armer plusieurs timers de 10 min (un seul par chargement)

const lsGet = (k) => { try { return localStorage.getItem(k); } catch { return null; } };
const lsSet = (k, v) => { try { localStorage.setItem(k, v); } catch { /* localStorage off */ } };

// T6.1 : horodate le démarrage d'une session → permet de détecter un retour au prochain chargement.
function markSessionStart() { lsSet(LAST_SESSION_KEY, String(Date.now())); }

// Le nudge a-t-il déjà été montré OU l'utilisateur a-t-il déjà un pseudo ? (AC-2/AC-7)
// → dans les deux cas, on ne le montre JAMAIS (retour immédiat de checkNudgeTrigger).
function nudgeBlocked() {
  return !!loadPseudo() || lsGet(NUDGE_SHOWN_KEY) === 'true';
}

// ⚠️ DÉVIATION DE COPY ASSUMÉE (honnêteté, invariant canon §0) :
// La copy canonique AC-5 dit « Tes premières contributions SONT VÉRIFIÉES. … ». Or le gate de
// vérification (Epic D.2) n'est PAS actif : aucune position n'est « vérifiée » aujourd'hui (tout
// est en quarantaine, cf. updateContributionFlow mode dégradé). L'invariant §0 interdit de célébrer
// un chiffre qui peut chuter → on n'emploie « vérifiées » QUE quand le comptage vérifié sera actif.
// Aujourd'hui : variante HONNÊTE « Tes premières contributions comptent déjà. ». Le reste de la copy
// (crédits gen-N+1 + multi-machine) est identique à AC-5. Même esprit que la dégradation STORY-004/005.
// Bascule automatique : dès que lastValidated devient non-null (serveur fournit le comptage vérifié),
// la copy canonique « sont vérifiées » s'affiche — sans toucher cette fonction.
function nudgeLeadCopy() {
  return lastValidated != null
    ? 'Tes premières contributions sont vérifiées.'
    : 'Tes premières contributions comptent déjà.';
}

// Construit/réinjecte la copy du nudge dans #nudge-msg (gen-N+1 ou fallback honnête).
function renderNudgeCopy() {
  const msg = $('nudge-msg');
  if (!msg) return;
  // gen-N+1 = target_gen (STORY-001) ; fallback honnête si endpoint dégradé (currentTargetGen null).
  const genLabel = (typeof currentTargetGen === 'number')
    ? `<b>gen-${currentTargetGen}</b>` : 'la prochaine génération';
  msg.innerHTML = `${nudgeLeadCopy()} Choisis un pseudo pour apparaître dans les crédits de ${genLabel} — utilise-le sur toutes tes machines.`;
}

// Si le nudge est DÉJÀ visible quand /stats/season arrive (cas condition B au chargement, avant
// la réponse saison), ré-injecte la gen cible — le fallback « la prochaine génération » devient gen-N.
function refreshNudgeGen() {
  const nudge = $('nudge');
  if (nudge && !nudge.hidden) renderNudgeCopy();
}

// T3 : injecte la gen cible dans la copy puis révèle le nudge. Idempotent (no-op si déjà bloqué).
// ⚠️ Le nudge vit DANS la section .live (placement story : après .flow, avant #grid), masquée par
// défaut (display:none) tant qu'une contribution n'a pas démarré. La condition A (10 min) survient
// EN cours → .live est déjà révélée. Mais la condition B (RETOUR) se déclenche au CHARGEMENT, avant
// tout démarrage → on doit révéler .live, sinon le nudge resterait invisible (0×0). On révèle donc
// la zone live (même geste que onStart) pour que le visiteur de retour VOIE le nudge + son flux.
function showNudge() {
  if (nudgeBlocked()) return;
  const nudge = $('nudge');
  if (!nudge) return;
  renderNudgeCopy();
  // STORY-015 — suggestion (placeholder grisé, JAMAIS pré-rempli : canon §4). Offre, pas étiquette.
  const input = $('nudge-input');
  if (input && !input.value) input.placeholder = defaultPseudoSuggestion(loadClientId());
  const live = $('live'); if (live) live.classList.add('on'); // garantit la visibilité (condition B au load)
  nudge.hidden = false;
}

// T2 : évalue les déclencheurs. reason='timer' vient du setTimeout 10 min (condition A) ; sinon
// appel au chargement (condition B = retour). Retour immédiat si déjà montré ou pseudo présent.
function checkNudgeTrigger(reason) {
  if (nudgeBlocked()) return; // AC-7/AC-2 : ni si pseudo déjà choisi, ni si déjà montré une fois
  if (reason === 'timer') {
    // Condition A : 10 min écoulées ET au moins une soumission dans la session.
    if (submittedCount >= 1) showNudge();
    return;
  }
  // Condition B (au chargement) : un retour est détecté si la dernière session date d'il y a > 4 h.
  const last = parseInt(lsGet(LAST_SESSION_KEY) || '', 10);
  if (Number.isFinite(last) && (Date.now() - last) > RETURN_GAP_MS) showNudge();
}

// Arme le timer de la condition A (une seule fois). Appelé au 1er démarrage de contribution.
function scheduleNudgeTimer() {
  if (nudgeTimerScheduled || nudgeBlocked()) return;
  nudgeTimerScheduled = true;
  setTimeout(() => checkNudgeTrigger('timer'), NUDGE_TIMER_MS);
}

// Affiche/efface l'erreur inline de validation du pseudo (T4.1).
function setNudgeError(text) {
  const err = $('nudge-error'), input = $('nudge-input');
  if (err) { err.textContent = text || ''; err.hidden = !text; }
  if (input) input.classList.toggle('invalid', !!text);
}

// T4 : « Choisir mon pseudo » — valide, persiste, confirme. T5 : « Plus tard » — marque vu, cache.
function confirmNudgePseudo() {
  const input = $('nudge-input');
  // Validation MIROIR du serveur (normalizePseudo) : rejette tôt ce que le serveur rejetterait ET
  // persiste la forme NORMALISÉE (lowercase) — donc le pseudo sauvegardé == ce que le serveur stocke.
  const pseudo = normalizePseudo(input && input.value);
  if (!pseudo) {
    setNudgeError('Pseudo invalide : 1 à 24 caractères (lettres, chiffres, « _ » ou « - »), sans terme offensant.');
    return;
  }
  setNudgeError('');
  savePseudo(pseudo);            // T4.2 — STORY-007 (label crédits)
  // STORY-015 AC-6 — applique le pseudo aux workers DÉJÀ en cours (plus besoin de redémarrer).
  workers.forEach((w) => { try { w.postMessage({ type: 'set-pseudo', pseudo }); } catch { /* worker mort */ } });
  lsSet(NUDGE_SHOWN_KEY, 'true'); // T4.3 — une seule fois par appareil
  const nudge = $('nudge'); if (nudge) nudge.hidden = true; // T4.4
  const conf = $('nudge-confirm-msg');
  if (conf) {
    // AC-6 : copy EXACTE de confirmation (label public + multi-machine + ré-saisie après effacement).
    conf.textContent = 'Ton pseudo est un label public. Utilise-le où tu veux pour cumuler tes contributions ; si tu effaces tes données, retape-le simplement pour continuer.';
    conf.hidden = false;
  }
  // T4.5 : mettre à jour le header (STORY-009 PAS faite) — appel DÉFENSIF, ne crashe pas si absent.
  if (typeof renderIdentityState === 'function') {
    try { renderIdentityState(); } catch { /* header optionnel — ne jamais casser le nudge */ }
  }
}

function dismissNudge() {
  lsSet(NUDGE_SHOWN_KEY, 'true');                 // T5.1 — ne réapparaît plus (AC-2)
  const nudge = $('nudge'); if (nudge) nudge.hidden = true;
}

// ---- Jauge collective « Ensemble vers gen-N » (STORY-002) ------------------------------------
// Poll public de /stats/season (même origine). DÉGRADATION GRACIEUSE (AC-4) : tant que l'endpoint
// n'existe pas (404 aujourd'hui) ou échoue, la section #collective reste MASQUÉE — jamais d'UI
// cassée, de spinner infini ni de message d'erreur. Elle n'apparaît que sur une réponse valide.
const SEASON_BASE_MS = 30_000;   // cadence nominale (AC-3)
const SEASON_MAX_MS = 120_000;   // cap du backoff exponentiel (30 → 60 → 120 s)
let seasonDelay = SEASON_BASE_MS;
let seasonTimer = null;
let seasonPctShown = 0; // la jauge ne régresse JAMAIS visuellement → on garde le max affiché
const nfFR = new Intl.NumberFormat('fr-FR'); // séparateurs de milliers (« 19 500 000 »)

function applySeasonStats(d) {
  // Validation défensive : sans les champs attendus, on traite comme un échec (reste masquée).
  if (!d || typeof d.collective_pct !== 'number' || typeof d.target_gen !== 'number') throw new Error('payload invalide');
  const pct = Math.max(0, Math.min(100, d.collective_pct));
  seasonPctShown = Math.max(seasonPctShown, pct); // anti-régression visuelle
  const shown = seasonPctShown;
  currentTargetGen = d.target_gen; // STORY-008 : gen-N+1 pour la copy du nudge (gen cible saison)
  refreshNudgeGen();               // si le nudge est DÉJÀ affiché (condition B au load, avant la saison), ré-injecte gen-N
  $('target-gen').textContent = String(d.target_gen);
  $('season-pct').textContent = shown.toFixed(shown >= 99.95 ? 0 : 1);
  $('season-bar').style.width = shown + '%';
  const verified = Number(d.verified_positions || 0);
  // STORY-016 — deux nombres COMBINÉS (fleet + browser) : cumul (au passé, « contributeurs ») +
  // live (« actifs maintenant » = ont soumis dans la dernière heure). Sans seuil (décision user).
  // Rétro-compat : si l'endpoint ne renvoie pas encore les nouveaux champs, repli sur active_contributors.
  const total = Number(d.contributors_total ?? d.active_contributors ?? 0);
  const online = Number(d.contributors_online ?? d.active_contributors ?? 0);
  // « positions vérifiées » jamais abrégé (canon §1) ; jamais « personnes » (canon §10) ; « actifs
  // maintenant » plutôt que « en ligne » (évite la connotation réseau social).
  $('season-sub').innerHTML =
    `<b>${nfFR.format(verified)}</b> positions vérifiées · `
    + `<b>${nfFR.format(total)}</b> contributeur${total === 1 ? '' : 's'} · `
    + `<b>${nfFR.format(online)}</b> actif${online === 1 ? '' : 's'} maintenant`;
  $('collective').classList.add('on'); // révèle la section (1re réponse valide)
  maybeShowPromoBanner(d); // STORY-003 : bandeau « ce qui a changé » (même flux, pas de 2e fetch)
}

// ---- Bandeau de promotion « ce qui a changé pendant ton absence » (STORY-003) ----------------
// Affiché UNIQUEMENT si une génération a été promue DEPUIS la dernière visite : on compare
// last_promoted_gen (server-authoritative) à nz_last_seen_gen (localStorage). Sinon → silence
// (jamais « rien de nouveau »). Dégradation : si /stats/season est indispo, applySeasonStats()
// lève AVANT d'arriver ici → le bandeau reste masqué (état live actuel attendu).
const PROMO_SEEN_KEY = 'nz_last_seen_gen';
const loadSeenGen = () => { try { return localStorage.getItem(PROMO_SEEN_KEY); } catch { return null; } };
const saveSeenGen = (g) => { try { localStorage.setItem(PROMO_SEEN_KEY, String(g)); } catch { /* localStorage off */ } };
let promoDismissed = false; // fermé pendant CETTE session → ne pas le faire réapparaître au refresh suivant

function maybeShowPromoBanner(d) {
  const banner = $('promo-banner');
  if (!banner) return;
  const gen = d && d.last_promoted_gen;
  // Aucune promotion connue (champ null/absent) → masqué (AC-1).
  if (gen == null || typeof gen !== 'number') { banner.hidden = true; return; }
  const seen = loadSeenGen();
  if (seen == null) {
    // Premier visiteur : on initialise SILENCIEUSEMENT la valeur courante, sans afficher (AC-7).
    saveSeenGen(gen);
    banner.hidden = true;
    return;
  }
  // Promotion survenue depuis la dernière visite ? (comparaison sur last_promoted_gen, pas current_gen)
  const isNew = gen > parseInt(seen, 10);
  if (!isNew || promoDismissed) { banner.hidden = true; return; }
  // Copy : Elo si renseigné, sinon copy neutre (jamais « +null Elo ») — AC-3.
  const eloGain = d.last_promoted_elo_gain;
  const hasElo = typeof eloGain === 'number' && Number.isFinite(eloGain);
  const tail = hasElo
    ? `+${nfFR.format(eloGain)} Elo pendant ton absence.`
    : `une nouvelle génération est active.`;
  $('promo-text').innerHTML =
    `<b>gen-${gen} promue</b> — ${tail} <small>Tes contributions y sont créditées.</small>`;
  banner.hidden = false;
}

async function pollSeasonStats() {
  try {
    const base = ($('baseUrl') && $('baseUrl').value.trim()) || '';
    const res = await fetch((base ? base.replace(/\/$/, '') : '') + '/stats/season', { cache: 'no-store' });
    if (!res.ok) throw new Error('HTTP ' + res.status); // 404 (pas encore déployé) → masquée
    applySeasonStats(await res.json());
    seasonDelay = SEASON_BASE_MS; // succès → retour à la cadence nominale
  } catch {
    // Échec silencieux : on NE touche PAS le DOM (la section reste dans son état précédent —
    // masquée si jamais affichée). Backoff exponentiel borné.
    seasonDelay = Math.min(seasonDelay * 2, SEASON_MAX_MS);
  } finally {
    clearTimeout(seasonTimer);
    seasonTimer = setTimeout(pollSeasonStats, seasonDelay);
  }
}

// ---- Avertissements énergie (edge-cases laptop) — discrets, NON bloquants, data-neutres ----------
// Une seule zone #energy-warn (aria-live) sous le sélecteur de mode ; chaque note se montre/cache
// indépendamment. AUCUNE n'affecte la recherche ni les données soumises. Toutes les API optionnelles
// (getBattery, hardwareConcurrency) sont gardées (?./try-catch) → jamais de crash au boot si absentes.
let batteryNoticeShown = false; // batterie : une seule fois par session (mémoire, PAS localStorage)

// Montre/masque une note de la zone #energy-warn (no-op défensif si l'élément n'existe pas).
function showEnergyNote(id, show) {
  const el = $(id);
  if (el) el.hidden = !show;
}

// Note « chaleur » du mode Max : visible UNIQUEMENT quand Max est le mode courant, masquée sinon
// (Éco/Normal/personnalisé). Appelée par setMode/syncModeFromFields (le handler de sélection de mode).
function renderModeHeatWarning() {
  showEnergyNote('warn-max', currentMode === 'max');
}

// Évalue au boot les edge-cases batterie + CPU faible (le cas « GPU absent » est géré ailleurs).
function initEnergyWarnings() {
  // CPU faible : très peu de cœurs → contribution lente (le GPU reste utilisé s'il est disponible).
  try {
    const cores = navigator.hardwareConcurrency;
    if (Number.isFinite(cores) && cores < 2) showEnergyNote('warn-cpu', true);
  } catch { /* hardwareConcurrency absent → on ignore (pas de note) */ }
  // Batterie : sur batterie ET non en charge → suggère le mode Éco (une fois/session). API optionnelle
  // (getBattery absent sur certains navigateurs) → garde ?. + promesse défensive.
  try {
    const gb = navigator.getBattery && navigator.getBattery();
    if (gb && typeof gb.then === 'function') {
      gb.then((battery) => {
        try {
          if (battery && !battery.charging && !batteryNoticeShown) {
            batteryNoticeShown = true;
            showEnergyNote('warn-battery', true);
          }
        } catch { /* lecture battery échoue → on ignore */ }
      }).catch(() => { /* refus/erreur de la promesse → on ignore */ });
    }
  } catch { /* getBattery absent ou jette → on ignore */ }
}

// Note GPU : le flag Chrome rendu en <code> (identifiable / copiable), le reste en texte. `hint` est
// statique (issu de gpuStatus) → construction par nœuds, jamais d'innerHTML.
function renderGpuHint(el, hint) {
  el.textContent = '';
  if (!hint) { el.hidden = true; return; }
  // Un seul enfant (span) → un seul flex-item : le texte s'écoule normalement avec le <code> inline.
  const span = document.createElement('span');
  const FLAG = 'chrome://flags/#enable-vulkan';
  const i = hint.indexOf(FLAG);
  if (i < 0) {
    span.textContent = hint;
  } else {
    span.append(hint.slice(0, i));
    const code = document.createElement('code');
    code.textContent = FLAG;
    span.append(code, hint.slice(i + FLAG.length));
  }
  el.append(span);
  el.hidden = false;
}

// Transparence (FR8) : cœurs + WebGPU RÉEL. navigator.gpu peut exister SANS adaptateur utilisable
// (machine sans GPU, ou Linux/Chrome avec Vulkan désactivé) → on demande vraiment un adaptateur pour
// ne pas afficher « oui » à tort, et on explique le repli CPU au lieu de le subir en silence.
(async () => {
  let adapter = null;
  if (navigator.gpu) { try { adapter = await navigator.gpu.requestAdapter(); } catch { adapter = null; } }
  const platform = navigator.userAgentData?.platform || navigator.platform || '';
  const status = gpuStatus({ hasWebGpuApi: !!navigator.gpu, hasAdapter: !!adapter, platform });
  gpuOk = status.gpuOk;
  $('hw').textContent = `${navigator.hardwareConcurrency || '?'} cœurs · ${status.hw}`;
  const gpuNote = $('warn-gpu');
  if (gpuNote) renderGpuHint(gpuNote, status.hint);
  // gpuOk est async → on (re)calcule MODE_PRESETS avec le bon backend une fois connu (défauts
  // conservateurs CPU/GPU si pas de nz_calib valide, valeurs mesurées sinon) et on réapplique le mode
  // courant (à l'arrêt, knobs non verrouillés). Au CHARGEMENT, aucune session ne tourne.
  rebuildModePresets();
  renderCalibControls(currentWorkerState); // état initial du bouton « Re-calibrer » (actif à l'arrêt)
  const calib = loadCalib();
  if (!calib) {
    // 1er lancement (pas de nz_calib) → on affiche le panneau ET on DÉMARRE automatiquement le bench
    // (jauge speedtest). « Passer » reste visible (→ défauts conservateurs). Une seule fois.
    showCalibPanel(true);
    startCalibration();
  } else if (calibStale(calib)) {
    // Calib présente mais PÉRIMÉE (backend changé / cœurs changés / > 30 j) → on RE-PROPOSE le panneau
    // (sans auto-démarrer : l'utilisateur clique « Calibrer » ou « Passer » pour garder l'ancienne).
    const title = $('calib-title'); if (title) title.textContent = 'Ta machine a changé — re-calibrer ?';
    const prog = $('calib-progress'); if (prog) prog.textContent = 'On a détecté un changement (matériel / GPU / ancienneté). Re-mesure pour recaler Éco / Normal / Max.';
    const startBtn = $('calib-start'); if (startBtn) { startBtn.hidden = false; startBtn.disabled = false; }
    const skip = $('calib-skip'); if (skip) { skip.hidden = false; skip.disabled = false; }
    showCalibPanel(true);
  }
  // Sinon (calib valide & fraîche) : panneau masqué, presets reflètent déjà les valeurs mesurées.
})();
updateContributionFlow(0, loadTotal(), null); // flux au chargement : 0 soumises cette session, cumul persisté (mode dégradé)
// Le serveur est en MÊME ORIGINE que cette page (la gateway sert la page ET proxifie le
// jobserver). Deux topologies de déploiement :
//   - dev/DevSrv direct : page ET API à la racine de l'origine → base = origin.
//   - prod (VPS nanozero.org) : page volontaire servie sous /worker/, API proxifiée sous
//     /api/v1 (frontière versionnée) → base = origin + /api/v1.
// Discriminant = le chemin /worker/ (sinon on taperait nanozero.org/jobs/claim → landing HTML).
// Éditable pour un usage avancé (cross-origin/dev).
$('baseUrl').value = window.location.pathname.startsWith('/worker')
  ? window.location.origin + '/api/v1'
  : window.location.origin;
$('btn-start').addEventListener('click', () => consent.start());
// #btn-pause sert Pause (RUNNING) ET Reprendre (PAUSED) — le libellé/icône commutent (AC-6).
$('btn-pause').addEventListener('click', () => {
  if (currentWorkerState === WorkerState.PAUSED) resumeWorker();
  else pauseWorker();
});
$('btn-stop').addEventListener('click', () => stopWorker());
// Sélecteur de mode (STORY-010) : Normal seul fonctionnel ; Éco/Max désactivés (clics ignorés).
$('btn-eco').addEventListener('click', () => setMode('eco'));
$('btn-normal').addEventListener('click', () => setMode('normal'));
$('btn-max').addEventListener('click', () => setMode('max'));
// « Comment ça marche ? » (AC-9) : placeholder href="#" comme les autres liens (Discord/source).
$('howto').addEventListener('click', (e) => { e.preventDefault(); });
// nb d'échiquiers affichés : appliqué EN DIRECT (ajoute/retire les boards sans redémarrer).
// On prévient AUSSI le worker → il aligne l'éval W/D/L (1 inférence/coup) sur le nouveau préfixe affiché.
$('boards').addEventListener('input', () => {
  const n = Math.max(1, parseInt($('boards').value, 10) || 3);
  applyDisplayCap(n);
  if (workers[0]) workers[0].postMessage({ type: 'config', boards: n }); // seul le worker 0 affiche
});
// (plus de réglage 'sims' côté client : les sims/coup sont dictés par le serveur, cf. run-loop.mjs)
// Warning OOM quand on passe K à 2 : chaque worker = 1 session WebGPU → +mémoire GPU dans l'onglet.
$('workers').addEventListener('input', () => { $('kwarn').hidden = (parseInt($('workers').value, 10) || 1) < 2; });
// Édition manuelle de la charge (Avancé) → réconcilie le mode énergie (preset reconnu ou « personnalisé »).
$('pool').addEventListener('input', syncModeFromFields);
$('wave').addEventListener('input', syncModeFromFields);
$('alert-x').addEventListener('click', () => { $('alert').hidden = true; });
// Bench de calibration : « Calibrer » lance le bench, « Passer » garde les défauts + coupe le sweep,
// « Re-calibrer » (dans le panneau ET près du sélecteur de mode) efface nz_calib + relance. Appels
// défensifs (panneau optionnel). #btn-recalib est gated IDLE/STOPPED par renderCalibControls.
if ($('calib-start')) $('calib-start').addEventListener('click', startCalibration);
if ($('calib-skip')) $('calib-skip').addEventListener('click', skipCalibration);
if ($('calib-redo')) $('calib-redo').addEventListener('click', recalibrate);
if ($('btn-recalib')) $('btn-recalib').addEventListener('click', recalibrate);
// Note batterie : « Passer en Éco » réutilise la sélection de mode existante (setMode) puis retire la
// note (suggestion suivie). setMode est no-op si les modes sont verrouillés (session en cours) — sûr.
if ($('warn-battery-eco')) $('warn-battery-eco').addEventListener('click', (e) => {
  e.preventDefault();
  setMode('eco');
  showEnergyNote('warn-battery', false);
});
// « Voir les 2 autres parties » (STORY-006 T5.2) : toggle CSS PUR (.is-expanded sur #grid), pas de
// navigation. Visibilité du bouton = 100 % CSS (caché en desktop, affiché < 560 px). On met juste à
// jour son libellé + aria-expanded. Pas de listener resize : le CSS gère la bascule desktop/mobile.
if ($('seemore')) {
  $('seemore').addEventListener('click', () => {
    const expanded = $('grid').classList.toggle('is-expanded');
    $('seemore').setAttribute('aria-expanded', expanded ? 'true' : 'false');
    $('seemore').textContent = expanded ? 'Masquer les autres parties' : 'Voir les 2 autres parties';
  });
}
// Bandeau de promotion (STORY-003) : fermer met à jour nz_last_seen_gen avec la gen courante
// (AC-4) et marque la session comme « vu » → ne réapparaît pas au prochain rafraîchissement.
$('promo-close').addEventListener('click', () => {
  const banner = $('promo-banner');
  const txt = $('promo-text').textContent || '';
  const m = txt.match(/gen-(\d+)/); // récupère la gen affichée pour persister nz_last_seen_gen
  if (m) saveSeenGen(m[1]);
  promoDismissed = true;
  banner.hidden = true;
});
setWorkerState(WorkerState.IDLE); // état initial : Démarrer activé, Pause/Arrêter désactivés (T3.1) + footer neutre
setMode('normal');                // mode par défaut = Normal (AC-1) — masque aussi la note « chaleur » Max
initEnergyWarnings();             // edge-cases laptop (batterie / CPU faible) — discrets, data-neutres
renderIdentityState();            // STORY-009 T4.1 : reflète l'état d'identité au chargement (public/anonyme/identifié)
// Liens Discord (footer + bannière promo) centralisés sur DISCORD_URL (placeholder tant que non défini).
if ($('footer-discord')) $('footer-discord').href = DISCORD_URL;
if ($('promo-discord')) $('promo-discord').href = DISCORD_URL;
// Nudge pseudo (STORY-008) : câblage des boutons + évaluation de la condition B (retour) au chargement.
if ($('nudge-confirm')) $('nudge-confirm').addEventListener('click', confirmNudgePseudo);
if ($('nudge-later')) $('nudge-later').addEventListener('click', dismissNudge);
// Entrée = valider (ergonomie clavier) ; saisie = efface l'erreur inline dès qu'on retape.
if ($('nudge-input')) {
  $('nudge-input').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); confirmNudgePseudo(); } });
  $('nudge-input').addEventListener('input', () => setNudgeError(''));
}
checkNudgeTrigger(); // condition B (retour) évaluée au chargement, AVANT d'horodater la session courante
// Jauge collective : poll au chargement puis toutes les 30 s (backoff sur erreur). Indépendant de
// Démarrer/Arrêter : la progression collective s'affiche même sans contribuer.
pollSeasonStats();
