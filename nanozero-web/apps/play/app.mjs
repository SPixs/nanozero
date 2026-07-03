// apps/play/app.mjs — contrôleur de la surface « Jouer » (J1 config + J2 boucle de jeu).
// Bundlé (dist/app.bundle.js). Thread principal : UI + échiquier (core/board-interactive) + machine
// d'états ; l'inférence (ORT + MCTS) tourne dans dist/worker.bundle.js. 100 % client.

import { BoardInteractive, sideToMoveFromFen } from '../../core/board-interactive.mjs';
import { GameState } from '../../core/game-state.mjs';
import { STRENGTH_LEVELS, levelById } from './strength-levels.mjs';
import { buildShareUrl, copyShareLink, readShareParams } from './share-link.mjs';

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const API = `${location.origin}/api/v1`;
const SITE_API = `${location.origin}/api/site`; // backend site (persistance des parties, B1)
// Backend device-aware (spike F4) : mobile → WASM (WebGPU plus lent sur le 8×96) ; desktop → WebGPU.
const IS_MOBILE = (matchMedia && matchMedia('(pointer:coarse)').matches) || /Mobi|Android/i.test(navigator.userAgent);
const BACKEND = IS_MOBILE ? 'wasm' : 'webgpu';
const ST = { IDLE: 0, HUMAN: 1, THINKING: 2, OVER: 3 };
const BADGE_LABEL = { calm: 'Position nette', tense: 'Position tendue', sharp: 'Position tranchante' };

const $ = (id) => document.getElementById(id);

// ---- État ----
let worker = null, modelReady = false, calibVersion = 1, thresholds = { calm: 0.10, tense: 0.25 };
let state = ST.IDLE, gs = null, board = null, playerColor = 'w', level = null, plies = [], pendingEngine = false;
// Persistance (B1) : version réseau (du backend site), diagnostic, lien de partie, temps de réflexion.
let networkVersion = null, engineBackend = BACKEND, wasmThreads = 0;
let gameShareId = null, savedThisGame = false, thinkTotalMs = 0, thinkCount = 0, thinkStartMs = 0;

// ---- Worker d'inférence (créé + pré-chauffé dès le chargement) ----
function initWorker() {
  worker = new Worker('dist/worker.bundle.js?backend=' + BACKEND);
  worker.onmessage = (e) => onWorkerMsg(e.data || {});
  worker.onerror = (e) => setLoad('Problème côté moteur : ' + (e.message || '(F12)'));
  worker.postMessage({ type: 'start', apiBase: API, backend: BACKEND, thresholds });
}

function onWorkerMsg(m) {
  if (m.type === 'model-ready') {
    modelReady = true;
    if (m.backend) engineBackend = m.backend;            // EP demandé (TODO : remonter l'EP effectif)
    if (typeof m.wasmThreads === 'number') wasmThreads = m.wasmThreads;
    setLoad('');
    $('start-btn').disabled = false;
    if (pendingEngine) { pendingEngine = false; engineThink(); }
  } else if (m.type === 'progress') {
    if (m.total) $('bar-fill').style.width = `${Math.round((m.done / m.total) * 100)}%`;
  } else if (m.type === 'move-result') {
    applyEngineMove(m);
  } else if (m.type === 'error') {
    setLoad('Problème : ' + m.message);
  }
}

// ---- Helpers UI ----
function setLoad(t) { const el = $('load-status'); if (el) el.textContent = t; }

// Icône de difficulté par niveau : feuille (doux) → bouclier (solide) → flamme (à fond),
// dégradé menthe→or→corail = échelle easy→hard intuitive.
const LEVEL_ICON = {
  chill: { accent: 'mint',  svg: '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/><path d="M2 21c0-3 1.85-5.36 5.08-6"/>' },
  club:  { accent: 'gold',  svg: '<path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/>' },
  full:  { accent: 'coral', svg: '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>' },
};

function renderLevels() {
  $('level-opts').innerHTML = STRENGTH_LEVELS.map((l, i) => {
    const ic = LEVEL_ICON[l.id] || { accent: 'gold', svg: '' };
    const elo = l.measuredElo ? `<span class="elo">≈ ${l.measuredElo} Elo</span>` : '';
    return `<label class="opt"><input type="radio" name="level" value="${l.id}" ${i === 0 ? 'checked' : ''}>
      <span class="box"><span class="ic badge" data-accent="${ic.accent}"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${ic.svg}</svg></span><span class="t">${l.label}</span><span class="d">${l.desc}</span>${elo}</span></label>`;
  }).join('');
}

function sanFor(from, to, promotion) {
  const m = gs.legalMoves().find((x) => x.cjs.from === from && x.cjs.to === to
    && (x.cjs.promotion || undefined) === (promotion || undefined));
  return m ? m.san : `${from}${to}`;
}

function pushMove(san) {
  plies.push(san);
  let html = '';
  for (let i = 0; i < plies.length; i += 2) {
    html += `<li class="num">${i / 2 + 1}.</li><li>${plies[i]}</li><li>${plies[i + 1] || ''}</li>`;
  }
  const ml = $('move-list'); ml.innerHTML = html; ml.scrollTop = ml.scrollHeight;
}

function setBadge(cx) {
  const b = $('badge');
  // Badge MASQUÉ tant que la calibration ML n'est pas validée (calibration.json version >= 2).
  // La métrique brute ne discrimine PAS calme/tranchant avec le réseau 8×96 actuel (sonde 2026-06-29 :
  // mat-en-1 β≈0.40 ≈ ouverture calme β≈0.32) → elle induisait en erreur. Réactivation AUTOMATIQUE
  // dès que les seuils calibrés (pipeline Lichess) sont déployés. Le calcul (worker) reste en place.
  if (!cx || calibVersion < 2) { b.hidden = true; return; }
  b.dataset.level = cx.level;
  b.innerHTML = BADGE_LABEL[cx.level];
  b.hidden = false;
}

// ---- Sizer board (fit deux axes, façon Lichess) ----
function fitBoard() {
  const area = document.querySelector('.board-area'), wrap = document.querySelector('.board-wrap');
  if (!area || !wrap) return;
  const top = area.getBoundingClientRect().top;
  const size = Math.max(160, Math.floor(Math.min(area.clientWidth, window.innerHeight - top - 24, 800)));
  wrap.style.width = `${size}px`; wrap.style.height = `${size}px`; wrap.style.maxWidth = 'none';
}

// ---- Machine d'états ----
function startGame(color, lvl) {
  level = lvl; playerColor = color; plies = []; state = ST.IDLE;
  gameShareId = null; savedThisGame = false; thinkTotalMs = 0; thinkCount = 0; thinkStartMs = 0;
  gs = new GameState(START_FEN);
  $('config').hidden = true; $('game').hidden = false;
  $('level-name').textContent = lvl.label;
  $('move-list').innerHTML = ''; $('result').hidden = true; setBadge(null); $('thinking').hidden = true;
  board = new BoardInteractive($('board'), { flip: color === 'b', onMove: onHumanMove });
  board.updatePosition(gs.fen());
  fitBoard();
  if (color === 'b') engineThink(); else state = ST.HUMAN;
}

function onHumanMove(from, to, promotion) {
  if (state !== ST.HUMAN) return;
  const san = sanFor(from, to, promotion);
  gs.applyMove({ from, to, promotion });
  board.updatePosition(gs.fen(), { from, to });
  pushMove(san);
  if (checkEnd()) return;
  engineThink();
}

function engineThink() {
  state = ST.THINKING;
  setBadge(null); // pas de badge pendant la réflexion (AC-6)
  $('bar-fill').style.width = '0%';
  $('thinking').hidden = false;
  if (!modelReady) { pendingEngine = true; return; } // attend le pré-chargement
  thinkStartMs = performance.now(); // mesure du temps de réflexion (diagnostic avgThinkMs)
  worker.postMessage({
    type: 'move', fen: gs.fen(), sims: level.sims, // cap mobile retiré → sims = niveau, partout
    temperature: level.temperature, addNoise: level.addNoise,
  });
}

function applyEngineMove(m) {
  if (state !== ST.THINKING) return;
  if (thinkStartMs) { thinkTotalMs += performance.now() - thinkStartMs; thinkCount++; thinkStartMs = 0; }
  const { from, to, promotion } = m.move;
  const san = sanFor(from, to, promotion);
  gs.applyMove(m.move);
  board.updatePosition(gs.fen(), { from, to }, { animate: true }); // anime la réponse de Nano
  pushMove(san);
  $('thinking').hidden = true;
  if (checkEnd()) return;
  state = ST.HUMAN;
  setBadge(m.complexity); // complexité affichée entre les coups (tour humain)
}

// ---- Persistance d'une partie (B1) : sauvegarde AUTO anonyme + lien partageable ----
// Best-effort : tout échec est silencieux, la partie locale reste valide. Données ANONYMES
// (en-têtes PGN neutres, diagnostic GROSSIER, aucune IP/UA brut/GPU — cf. backend-persistance-parties.md).
function fetchInfo() {
  fetch(`${SITE_API}/info`).then((r) => (r.ok ? r.json() : null))
    .then((d) => { if (d && d.networkVersion) networkVersion = d.networkVersion; })
    .catch(() => {});
}

function gameResultStr() { // appelé uniquement quand gs.isGameOver()
  if (gs.isCheckmate()) return sideToMoveFromFen(gs.fen()) === 'w' ? '0-1' : '1-0';
  return '1/2-1/2'; // pat / nulle
}

function buildPgn(sans, result) {
  const nano = `NanoZero ${networkVersion || ''}`.trim();
  const white = playerColor === 'w' ? 'Joueur' : nano; // en-têtes NEUTRES (pas de nom réel)
  const black = playerColor === 'b' ? 'Joueur' : nano;
  let body = '';
  for (let i = 0; i < sans.length; i++) body += (i % 2 === 0 ? `${i / 2 + 1}. ` : '') + sans[i] + ' ';
  return `[Event "NanoZero"]\n[White "${white}"]\n[Black "${black}"]\n[Result "${result}"]\n\n${(body + result).trim()}\n`;
}

const bucketCores = (n) => (!n ? undefined : n <= 4 ? '1-4' : n <= 8 ? '5-8' : '9+');
const bucketMem = (g) => (g == null ? undefined : g <= 4 ? '≤4' : '>4');
function browserFamily() {
  const ua = navigator.userAgent;
  if (/Edg\//.test(ua)) return 'edge';
  if (/Firefox\//.test(ua)) return 'firefox';
  if (/Chrome\//.test(ua)) return 'chrome';   // après Edge (UA Edge contient "Chrome")
  if (/Safari\//.test(ua)) return 'safari';   // après Chrome (UA Chrome contient "Safari")
  return 'other';
}

async function persistGame() {
  if (savedThisGame || !networkVersion) return; // /info pas chargé → on s'abstient (best-effort)
  savedThisGame = true;
  const result = gameResultStr();
  const client = { cores: bucketCores(navigator.hardwareConcurrency), memGb: bucketMem(navigator.deviceMemory), browser: browserFamily(), wasmThreads };
  if (thinkCount) client.avgThinkMs = Math.round(thinkTotalMs / thinkCount);
  try {
    const res = await fetch(`${SITE_API}/games`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        pgn: buildPgn(plies, result), playerColor, levelId: level.id, result,
        plyCount: plies.length, networkVersion,
        deviceClass: IS_MOBILE ? 'mobile' : 'desktop', backend: engineBackend,
        effectiveSims: level.sims, client,
      }),
    });
    if (res.ok) { const d = await res.json(); gameShareId = d.shareId; markGameSaved(); }
  } catch { /* best-effort */ }
}

function markGameSaved() { // le bouton partage copie le LIEN DE LA PARTIE ; le bouton Analyser passe en lien propre
  const b = $('share-link');
  if (b) { b.textContent = 'Partager la partie'; b.dataset.label = 'Partager la partie'; }
  const a = $('analyse-game');
  if (a && gameShareId) a.href = `/analyse/?game=${gameShareId}`; // URL propre + cross-onglet
}

// J3 — fin de partie : message non-jugeant (centré sur la position, pas sur le jeu du joueur)
// + CTA Rejouer / Changer les réglages. « Analyser » présent mais désactivé tant que la surface
// Analyse n'existe pas (jamais de lien /analyse/ cassé). Pas de compteur « positions tranchantes »
// (badge complexité masqué) ni d'opt-in W/D/L (value head 8×96 trop bruité — cohérent avec le badge).
function checkEnd() {
  if (!gs.isGameOver()) return false;
  state = ST.OVER;
  persistGame(); // B1 : sauvegarde automatique anonyme (best-effort, non bloquant)
  setBadge(null);
  const full = Math.ceil(plies.length / 2);
  const sub = `${full} coup${full > 1 ? 's' : ''} joué${full > 1 ? 's' : ''}`;
  let title;
  if (gs.isCheckmate()) {
    const loser = sideToMoveFromFen(gs.fen());
    title = loser !== playerColor ? "Échec et mat — tu l'emportes !" : "Échec et mat — NanoZero l'emporte";
  } else if (gs.legalMoves().length === 0) {
    title = 'Pat — aucun coup légal, partie nulle';
  } else {
    title = 'Partie nulle';
  }
  showGameOver(title, sub);
  return true;
}

function showGameOver(title, sub) {
  const r = $('result');
  // Handoff vers l'analyse : PGN dans sessionStorage → URL PROPRE « /analyse/ » (pas de ?pgn= géant).
  // Si la partie est déjà persistée, on a un lien propre cross-onglet ?game=<id> (sinon upgrade au save).
  try { sessionStorage.setItem('nz_current_game', buildPgn(plies, gameResultStr())); } catch { /* indispo */ }
  const analyseHref = gameShareId ? `/analyse/?game=${gameShareId}` : '/analyse/';
  r.innerHTML = `<div class="r">${title}</div><div class="r-sub">${sub}</div>`
    + `<div class="r-actions">`
    + `<button class="r-btn r-btn--primary" id="replay">Rejouer</button>`
    + `<a class="r-btn" id="analyse-game" href="${analyseHref}">Analyser cette partie</a>`
    + `<button class="r-btn" id="reconfig">Changer les réglages</button>`
    + `</div>`
    + `<p class="r-legal">Parties sauvegardées anonymement · <a href="/mentions-legales/">en savoir plus</a></p>`;
  r.hidden = false;
  r.scrollIntoView({ block: 'nearest' }); // révèle les CTA sur mobile (no-op si déjà visible)
  $('replay').addEventListener('click', () => { if (board) board.destroy(); startGame(playerColor, level); });
  $('reconfig').addEventListener('click', backToConfig);
}

function backToConfig() {
  if (board) board.destroy();
  $('game').hidden = true; $('config').hidden = false;
}

// ---- Mode VISUALISEUR (B2/B3) : /play/?game=<shareId> → partie en lecture seule + pont Analyse ----
let viewer = null; // { positions:[{fen,san,from,to}], idx }

// Extrait les SAN d'un PGN (nos PGN sont générés proprement ; on tolère en-têtes/commentaires/NAGs).
function extractSans(pgn) {
  const body = pgn.replace(/\[[^\]]*\]/g, ' ').replace(/\{[^}]*\}/g, ' ').replace(/\$\d+/g, ' ');
  const out = [];
  for (let tok of body.split(/\s+/)) {
    if (!tok) continue;
    tok = tok.replace(/^\d+\.+/, ''); // "12." ou "12...e4" collé → retire le numéro
    if (!tok || /^\d+$/.test(tok) || tok === '1-0' || tok === '0-1' || tok === '1/2-1/2' || tok === '*') continue;
    out.push(tok);
  }
  return out;
}

// Rejoue les SAN depuis la position de départ via le cœur (pas de chessops dans ce bundle).
function buildPositions(sans) {
  const g2 = new GameState(START_FEN);
  const pos = [{ fen: g2.fen(), san: null, from: null, to: null }];
  for (const san of sans) {
    const lm = g2.legalMoves().find((m) => m.san === san)
      || g2.legalMoves().find((m) => m.san.replace(/[+#]/g, '') === san.replace(/[+#]/g, ''));
    if (!lm) break;
    const from = lm.cjs.from, to = lm.cjs.to;
    g2.applyMove(lm);
    pos.push({ fen: g2.fen(), san, from, to });
  }
  return pos;
}

async function enterViewer(shareId) {
  $('config').hidden = true; $('game').hidden = false;
  $('thinking').hidden = true; $('badge').hidden = true; $('result').hidden = true;
  $('new-game').hidden = true;                 // le viewer a son propre « Jouer une partie »
  $('viewer-panel').hidden = false;
  $('level-name').textContent = 'Partie partagée';
  gameShareId = shareId; markGameSaved();      // le bouton « partage » copie l'URL de cette partie
  let g;
  try {
    const res = await fetch(`${SITE_API}/games/${encodeURIComponent(shareId)}`);
    if (!res.ok) { showViewerError(); return; }
    g = await res.json();
  } catch { showViewerError(); return; }
  viewer = { positions: buildPositions(extractSans(g.pgn)), idx: 0 };
  board = new BoardInteractive($('board'), { flip: g.playerColor === 'b', readOnly: true });
  renderViewerSummary(g);
  renderViewerMoves();
  $('v-analyse').href = `/analyse/?game=${encodeURIComponent(shareId)}`; // B3 : lien PROPRE vers l'analyse
  viewerGoTo(viewer.positions.length - 1);      // ouvre sur la position finale (le résultat)
  fitBoard();
}

function showViewerError() {
  $('vp-summary').innerHTML = '<div class="vps-title">Partie introuvable</div>'
    + '<div class="vps-sub">Ce lien est invalide ou la partie n’existe plus.</div>';
  $('v-first').disabled = $('v-prev').disabled = $('v-next').disabled = $('v-last').disabled = true;
  $('v-analyse').removeAttribute('href');
}

function renderViewerSummary(g) {
  const lvl = (STRENGTH_LEVELS.find((l) => l.id === g.levelId) || {}).label || g.levelId;
  const youWhite = g.playerColor === 'w';
  const phrase = g.result === '1/2-1/2' ? 'Partie nulle'
    : ((g.result === '1-0') === youWhite ? 'Victoire du joueur' : 'NanoZero l’emporte');
  let date = '';
  try { if (g.createdAt) date = new Date(g.createdAt).toLocaleDateString('fr-FR'); } catch { /* ignore */ }
  $('vp-summary').innerHTML = `<div class="vps-title">${phrase}</div>`
    + `<div class="vps-sub">${youWhite ? 'Joueur (Blancs)' : 'Joueur (Noirs)'} vs Nano · ${lvl} · ${g.result} · ${g.networkVersion}${date ? ' · ' + date : ''}</div>`;
}

function renderViewerMoves() {
  const pos = viewer.positions;
  let html = '';
  for (let i = 1; i < pos.length; i += 2) {
    const w = `<li class="vmove" data-i="${i}">${pos[i].san}</li>`;
    const b = pos[i + 1] ? `<li class="vmove" data-i="${i + 1}">${pos[i + 1].san}</li>` : '<li></li>';
    html += `<li class="num">${(i + 1) / 2}.</li>${w}${b}`;
  }
  const ml = $('move-list'); ml.innerHTML = html;
  ml.querySelectorAll('.vmove').forEach((el) => el.addEventListener('click', () => viewerGoTo(+el.dataset.i)));
}

function viewerGoTo(i) {
  const pos = viewer.positions;
  viewer.idx = Math.max(0, Math.min(pos.length - 1, i));
  const p = pos[viewer.idx];
  board.updatePosition(p.fen, (p.from && p.to) ? { from: p.from, to: p.to } : null, { animate: false });
  const ml = $('move-list');
  ml.querySelectorAll('.vmove').forEach((el) => el.classList.toggle('active', +el.dataset.i === viewer.idx));
  const act = ml.querySelector('.vmove.active'); // défilement INTERNE à la liste (jamais la page)
  if (act) { const a = act.getBoundingClientRect(), l = ml.getBoundingClientRect();
    if (a.top < l.top) ml.scrollTop -= (l.top - a.top) + 8;
    else if (a.bottom > l.bottom) ml.scrollTop += (a.bottom - l.bottom) + 8; }
  $('v-first').disabled = $('v-prev').disabled = viewer.idx === 0;
  $('v-last').disabled = $('v-next').disabled = viewer.idx === pos.length - 1;
}

function initViewerControls() {
  $('v-first').addEventListener('click', () => viewerGoTo(0));
  $('v-prev').addEventListener('click', () => viewerGoTo(viewer.idx - 1));
  $('v-next').addEventListener('click', () => viewerGoTo(viewer.idx + 1));
  $('v-last').addEventListener('click', () => viewerGoTo(viewer.positions.length - 1));
  document.addEventListener('keydown', (e) => {
    if (!viewer) return;
    if (e.key === 'ArrowLeft') { viewerGoTo(viewer.idx - 1); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { viewerGoTo(viewer.idx + 1); e.preventDefault(); }
    else if (e.key === 'Home') { viewerGoTo(0); e.preventDefault(); }
    else if (e.key === 'End') { viewerGoTo(viewer.positions.length - 1); e.preventDefault(); }
  });
  window.addEventListener('resize', fitBoard);
  window.addEventListener('orientationchange', fitBoard);
}

// ---- Init ----
function init() {
  const sharedGame = new URLSearchParams(location.search).get('game');
  if (sharedGame) { initViewerControls(); enterViewer(sharedGame); return; } // mode viewer : pas de moteur

  renderLevels();
  $('start-btn').disabled = true;
  setLoad('Chargement du modèle…');
  if (BACKEND === 'wasm') setLoad('Mode CPU — le moteur sera un peu plus lent. Chargement…');
  initWorker();
  fetchInfo(); // B1 : version réseau du backend site (pour annoter les parties persistées)
  // calibration (seuils complexité + version pour la mention « expérimental »)
  fetch(`${location.origin}/calibration/calibration.json`).then((r) => r.json()).then((c) => {
    if (c && typeof c.version === 'number') calibVersion = c.version;
    if (c && c.beta_thresholds) thresholds = c.beta_thresholds;
  }).catch(() => {});

  $('config-form').addEventListener('submit', (e) => {
    e.preventDefault();
    const colorSel = document.querySelector('input[name=color]:checked').value;
    const color = colorSel === 'random' ? (Math.random() < 0.5 ? 'w' : 'b') : (colorSel === 'black' ? 'b' : 'w');
    const lvl = levelById(document.querySelector('input[name=level]:checked').value);
    startGame(color, lvl);
  });
  $('new-game').addEventListener('click', backToConfig);

  // J4 — lien défi : copie le réglage courant (couleur + niveau). Visible en jeu ET en fin de partie.
  $('share-link').addEventListener('click', async () => {
    const btn = $('share-link');
    // Si la partie a été sauvegardée → lien de PARTIE ; sinon → lien de config (défi).
    const url = gameShareId
      ? `${location.origin}/play/?game=${gameShareId}`
      : buildShareUrl({ color: playerColor === 'b' ? 'black' : 'white', level: level ? level.id : null });
    const ok = await copyShareLink(url);
    const orig = btn.dataset.label || btn.textContent;
    btn.dataset.label = orig;
    btn.textContent = ok ? 'Lien copié !' : 'Copie impossible';
    btn.classList.toggle('copied', ok);
    setTimeout(() => { btn.textContent = btn.dataset.label; btn.classList.remove('copied'); }, 2000);
  });

  // J4 — pré-remplir la config depuis un lien défi `/play/?color=&level=`.
  const shared = readShareParams();
  if (shared) {
    if (shared.color) { const el = document.querySelector(`input[name=color][value="${shared.color}"]`); if (el) el.checked = true; }
    if (shared.level) { const el = document.querySelector(`input[name=level][value="${shared.level}"]`); if (el) el.checked = true; }
  }
  window.addEventListener('resize', fitBoard);
  window.addEventListener('orientationchange', fitBoard);
  const area = document.querySelector('.board-area');
  if (area && typeof ResizeObserver !== 'undefined') new ResizeObserver(fitBoard).observe(area);
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
