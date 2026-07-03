// apps/sts/probe.mjs — bench STS web (thread principal). Parse l'EPD STS1-15, pilote le worker
// (meilleur coup de Nano au budget du niveau choisi), score live (rank c9 → points c8) + Elo Dann
// Corbit (44.523·% − 242.85) + par catégorie. Bundlé en dist/sts.bundle.js.

import { STRENGTH_LEVELS, levelById } from '../play/strength-levels.mjs';

const API = `${location.origin}/api/v1`;
const EPD_URL = 'STS1-STS15_LAN_v3.epd'; // servi sous /sts/
const IS_MOBILE = (matchMedia && matchMedia('(pointer:coarse)').matches) || /Mobi|Android/i.test(navigator.userAgent);
const BACKEND = IS_MOBILE ? 'wasm' : 'webgpu';
const $ = (id) => document.getElementById(id);

let worker = null, modelReady = false, positions = [], sample = [], running = false;

// ---- Parse EPD : fen (4 champs) + c9 (coups UCI) + c8 (poids, best=10) + catégorie (id) ----
function parseEpd(text) {
  const out = [];
  for (const raw of text.split('\n')) {
    const s = raw.trim();
    if (!s) continue;
    const c9 = s.match(/c9\s+"([^"]+)"/);
    const c8 = s.match(/c8\s+"([^"]+)"/);
    const idm = s.match(/id\s+"([^"]+)"/);
    if (!c9 || !c8) continue;
    const p = s.split(/\s+/);
    const fen = `${p[0]} ${p[1]} ${p[2]} ${p[3]} 0 1`;
    const moves = c9[1].trim().split(/\s+/);
    const weights = c8[1].trim().split(/\s+/).map(Number);
    let cat = '?';
    if (idm) { const cm = idm[1].match(/\)\s+(.+?)\.\d+\s*$/); if (cm) cat = cm[1].trim(); }
    out.push({ fen, moves, weights, cat, max: weights[0] || 10 });
  }
  return out;
}

function setStatus(t) { $('status').textContent = t; }

// ---- Scoring live ----
let total = 0, maxTotal = 0, done = 0, n = 0, t0 = 0, byCat = {};

function resetScore(count) {
  total = 0; maxTotal = 0; done = 0; n = count; byCat = {}; t0 = performance.now();
  $('bar-fill').style.width = '0%';
  $('cat-table').innerHTML = '';
}

function onPos(idx, uci) {
  const p = sample[idx];
  const rank = uci ? p.moves.indexOf(uci) : -1;
  const pts = rank >= 0 ? p.weights[rank] : 0;
  total += pts; maxTotal += p.max; done += 1;
  const c = (byCat[p.cat] ||= { s: 0, m: 0, n: 0 });
  c.s += pts; c.m += p.max; c.n += 1;
  // live
  const pct = maxTotal ? (100 * total / maxTotal) : 0;
  const elo = Math.round(44.523 * pct - 242.85);
  const elapsed = (performance.now() - t0) / 1000;
  const rate = done / elapsed;
  const eta = rate > 0 ? (n - done) / rate : 0;
  $('bar-fill').style.width = `${Math.round(100 * done / n)}%`;
  $('score').textContent = `${total} / ${maxTotal}`;
  $('pct').textContent = `${pct.toFixed(1)} %`;
  $('elo').textContent = `~${elo}`;
  $('prog').textContent = `${done} / ${n}`;
  $('rate').textContent = `${rate.toFixed(1)} pos/s · ETA ${(eta / 60).toFixed(1)} min`;
}

function renderCategories() {
  const rows = Object.entries(byCat).sort((a, b) => a[0].localeCompare(b[0]))
    .map(([cat, c]) => {
      const pct = c.m ? (100 * c.s / c.m).toFixed(0) : 0;
      return `<tr><td>${cat}</td><td>${c.s}/${c.m}</td><td>${pct}%</td></tr>`;
    }).join('');
  $('cat-table').innerHTML = `<tr><th>Catégorie</th><th>Score</th><th>%</th></tr>${rows}`;
}

function onWorkerMsg(m) {
  if (m.type === 'model-ready') { modelReady = true; setStatus(''); $('run-btn').disabled = false; }
  else if (m.type === 'pos') { onPos(m.idx, m.uci); }
  else if (m.type === 'done') {
    running = false;
    renderCategories();
    const pct = maxTotal ? (100 * total / maxTotal) : 0;
    setStatus(`Terminé : ${total}/${maxTotal} (${pct.toFixed(1)} %) → STS Elo ~${Math.round(44.523 * pct - 242.85)}`);
    $('run-btn').disabled = false; $('run-btn').textContent = 'Relancer';
  } else if (m.type === 'error') { setStatus('Erreur : ' + m.message); }
}

async function run() {
  if (running) return;
  if (!modelReady) { setStatus('Modèle pas encore prêt…'); return; }
  const level = levelById(document.querySelector('input[name=level]:checked').value);
  const count = Math.min(positions.length, Math.max(1, parseInt($('count').value, 10) || positions.length));
  // Échantillon RÉPARTI sur les 15 catégories (les N premières positions = 1 seule catégorie → biais).
  if (count >= positions.length) sample = positions.slice();
  else { sample = []; const step = positions.length / count; for (let i = 0; i < count; i++) sample.push(positions[Math.floor(i * step)]); }
  running = true;
  $('run-btn').disabled = true; $('run-btn').textContent = 'En cours…';
  resetScore(sample.length);
  setStatus(`STS « ${level.label} » (${level.sims} sims) sur ${sample.length} positions — backend ${BACKEND}…`);
  worker.postMessage({ type: 'run', sims: level.sims, fens: sample.map((p) => p.fen) });
}

function renderLevels() {
  $('level-opts').innerHTML = STRENGTH_LEVELS.map((l, i) => `
    <label class="opt"><input type="radio" name="level" value="${l.id}" ${i === STRENGTH_LEVELS.length - 1 ? 'checked' : ''}>
      <span class="box"><span class="t">${l.label}</span><span class="d">${l.sims} sims</span></span></label>`).join('');
}

async function init() {
  renderLevels();
  $('run-btn').disabled = true;
  setStatus(BACKEND === 'wasm' ? 'Mode CPU. Chargement du modèle…' : 'Chargement du modèle (WebGPU)…');
  worker = new Worker('dist/sts-worker.bundle.js');
  worker.onmessage = (e) => onWorkerMsg(e.data || {});
  worker.onerror = (e) => setStatus('Erreur worker : ' + (e.message || '(F12)'));
  worker.postMessage({ type: 'start', apiBase: API, backend: BACKEND });
  try {
    const txt = await (await fetch(EPD_URL)).text();
    positions = parseEpd(txt);
    $('count').value = positions.length;
    $('count').max = positions.length;
    $('epd-info').textContent = `${positions.length} positions STS chargées`;
  } catch (err) {
    setStatus('Erreur chargement EPD : ' + err);
  }
  $('run-btn').addEventListener('click', run);
  $('stop-btn').addEventListener('click', () => { if (worker) worker.postMessage({ type: 'stop' }); });
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
