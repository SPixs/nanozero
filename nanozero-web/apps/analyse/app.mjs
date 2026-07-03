// app.mjs — surface Analyse (thread principal). MVP client (S-02/03/04) : charge PGN/FEN, affiche
// l'échiquier, navigue coup par coup (boutons + clavier + liste cliquable). 100 % navigateur, aucun
// backend. L'analyse MCTS + jauge W/D/L (S-07) et le handoff depuis « Jouer » (S-13) viendront ensuite.

import { BoardInteractive } from '../../core/board-interactive.mjs';
import { GameState } from '../../core/game-state.mjs';
import { positionsFromPgn, positionFromFen } from './pgn-loader.mjs';
import { copyShareLink } from '../play/share-link.mjs';

const $ = (id) => document.getElementById(id);
const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const API = `${location.origin}/api/v1`;
const SITE_API = `${location.origin}/api/site`; // backend site (handoff /analyse/?game=<shareId>)
const IS_MOBILE = (typeof matchMedia !== 'undefined' && matchMedia('(pointer:coarse)').matches) || /Mobi|Android/i.test(navigator.userAgent);
const BACKEND = IS_MOBILE ? 'wasm' : 'webgpu';
const SIMS_STEPS = [64, 128, 256, 512, 1024, 2048];
let ANALYZE_SIMS = 512; // profondeur ARRIÈRE-PLAN (graphe), réglée par le curseur
const FG_PACKET = 256;  // sims par paquet d'AVANT-PLAN (approfondissement continu de la position courante)
const FG_YIELD = 3;     // pendant le remplissage du graphe : 1 paquet d'avant-plan sur N (le reste remplit le graphe)

let worker = null, modelReady = false;
let evals = [], queue = [];          // evals[i] = {pw,pd,pl,fm,pv?,n?} (Blancs) | 'terminal' | null
let analysisGen = 0, busy = false, tick = 0, paused = false; // ordonnanceur worker unique

// Partie d'exemple (Morphy — « Opéra », 1858), courte et célèbre.
const SAMPLE_PGN = `[Event "Paris Opera"]
[White "Morphy"]
[Black "Allies"]
[Result "1-0"]

1. e4 e5 2. Nf3 d6 3. d4 Bg4 4. dxe5 Bxf3 5. Qxf3 dxe5 6. Bc4 Nf6 7. Qb3 Qe7
8. Nc3 c6 9. Bg5 b5 10. Nxb5 cxb5 11. Bxb5+ Nbd7 12. O-O-O Rd8 13. Rxd7 Rxd7
14. Rd1 Qe6 15. Bxd7+ Nxd7 16. Qb8+ Nxb8 17. Rd8# 1-0`;

let board = null;
let positions = [{ fen: START_FEN, san: null, from: null, to: null }];
let index = 0;
// Variante (S-08) : 1 niveau, éphémère. branchIdx = index mainline d'où part la variante.
let variant = null; // { branchIdx, moves:[{fen,san,from,to}], idx }
let live = null;    // { fen, data } — éval de la position AFFICHÉE (data = {pw,pd,pl,fm,pv,n} | 'terminal')

function setErr(msg) { $('err').textContent = msg || ''; }

// ---- Position courante (mainline OU variante) ----
function currentFen() { return variant ? variant.moves[variant.idx].fen : positions[index].fen; }
function currentPos() { return variant ? variant.moves[variant.idx] : positions[index]; }
function currentEval() {
  if (live && live.fen === currentFen()) return live.data;
  if (!variant) return evals[index]; // {..} | 'terminal' | null
  return null; // variante pas encore analysée
}

// Jouer un coup depuis la position affichée. À la FIN de la ligne → prolonge la partie principale
// (construction d'une partie à la main, façon Lichess) ; en DÉVIANT au milieu → entre/étend une
// variante ; si c'est le coup réel suivant → simple avance.
function playMove(from, to, promotion) {
  let san, fen2;
  try {
    const gs = new GameState(currentFen());
    const lm = gs.legalMoves().find((m) => m.cjs.from === from && m.cjs.to === to
      && ((m.cjs.promotion || '') === (promotion || '')));
    if (!lm) return;
    san = lm.san;                 // SAN lu AVANT applyMove : getter paresseux calculé sur l'état pré-coup
    gs.applyMove(lm); fen2 = gs.fen();
  } catch { return; }
  const mv = { fen: fen2, san, from, to, uci: lm.uci }; // uci OBLIGATOIRE : sans lui le chemin envoyé au
  // worker est vide → même clé que la position précédente → PV/éval de l'ANCIENNE position (bug 2026-07-02)
  if (!variant) {
    const nextMain = positions[index + 1]; // si c'est le coup réel suivant → simple avance, pas de variante
    if (nextMain && nextMain.from === from && nextMain.to === to) { goTo(index + 1); return; }
    if (index === positions.length - 1) { // fin de la mainline → PROLONGER la partie (et le graphe)
      positions.push(mv); evals.push(null); index = positions.length - 1; queue.push(index);
      $('meta').textContent = `${positions.length - 1} demi-coups`;
      render(); return;
    }
    variant = { branchIdx: index, moves: [mv], idx: 0, evals: [null] }; // déviation au milieu → variante
  } else {
    if (variant.idx < variant.moves.length - 1) {
      variant.moves = variant.moves.slice(0, variant.idx + 1);
      variant.evals = variant.evals.slice(0, variant.idx + 1);
    }
    variant.moves.push(mv); variant.evals.push(null); variant.idx = variant.moves.length - 1;
  }
  render();
}

function exitVariant() {
  if (!variant) return;
  index = variant.branchIdx; variant = null; render();
}

function variantBlockHtml() {
  const f = positions[variant.branchIdx].fen.split(' ');
  let fm = parseInt(f[5], 10) || 1, white = f[1] === 'w', inner = '';
  for (let k = 0; k < variant.moves.length; k++) {
    const pre = white ? `${fm}. ` : (k === 0 ? `${fm}… ` : '');
    inner += `<span class="vm${k === variant.idx ? ' active' : ''}" data-vi="${k}">${pre}${figurine(variant.moves[k].san)}</span> `;
    if (!white) fm++;
    white = !white;
  }
  return `<li class="variant-block"><span class="vlead">↳</span>${inner}</li>`;
}

function render() {
  const p = currentPos();
  const last = (p.from && p.to) ? { from: p.from, to: p.to } : null;
  board.updatePosition(p.fen, last, { animate: false });
  $('fen-input').value = p.fen; // la FEN courante remplit le champ (copiable)
  renderMoveList(); // mainline + bloc variante + surlignage
  scrollActiveMoveIntoList(); // défilement INTERNE à la liste, jamais la page (sinon saut/scroll sur mobile)
  $('variant-bar').hidden = !variant;
  // état des boutons (en variante : tous actifs — prev sort au bord, first/last reviennent à la partie)
  $('first').disabled = $('prev').disabled = (!variant && index === 0);
  $('last').disabled = $('next').disabled = (!variant && index === positions.length - 1);
  // analyse : ordonnanceur (avant-plan = position courante, arrière-plan = graphe) + rendu
  pump();
  renderTimeline();
  renderGauge();
  renderMultiPV();
}

// Garde le coup actif visible DANS la liste, en ajustant uniquement son scrollTop interne.
// (Ne PAS utiliser Element.scrollIntoView : il fait défiler tous les conteneurs scrollables, dont
//  la page → sur mobile l'échiquier saute sous le doigt à chaque coup. Cf. retour utilisateur.)
function scrollActiveMoveIntoList() {
  const list = $('moves');
  const active = list && list.querySelector('.active');
  if (!active) return;
  const a = active.getBoundingClientRect(), l = list.getBoundingClientRect();
  if (a.top < l.top) list.scrollTop -= (l.top - a.top) + 8;
  else if (a.bottom > l.bottom) list.scrollTop += (a.bottom - l.bottom) + 8;
}

// ---- Analyse (ordonnanceur sur worker UNIQUE) ----
// PRIORITÉ AU GRAPHE : tant que des positions du graphe restent à analyser (file non vide), l'ARRIÈRE-PLAN
// est prioritaire (profondeur fixe = curseur) pour remplir vite la courbe W/D/L de toute la partie, avec
// 1 paquet d'AVANT-PLAN sur FG_YIELD pour donner rapidement une valeur à la jauge de la position courante.
// Graphe rempli → AVANT-PLAN en continu (tree-reuse, paquets FG_PACKET, sans cap) sur la position affichée.
// Un seul worker → on arbitre dans pump(). idx+gen dans chaque message → mapping robuste, périmés ignorés.
function resetAnalysis() {
  if (worker) worker.postMessage({ type: 'reset' }); // vide le cache arrière-plan + aborte la recherche en cours
  evals = new Array(positions.length).fill(null);
  queue = positions.map((_, i) => i);
  analysisGen++; // les résultats en vol de la passe précédente seront ignorés
  pump();
}
function orientWdl(fen, wdl) {
  if (!wdl) return 'terminal';
  let pw = wdl[0], pd = wdl[1], pl = wdl[2];
  if (fen.split(' ')[1] === 'b') { const t = pw; pw = pl; pl = t; } // orienter du POV Blancs
  return { pw, pd, pl, fm: parseInt(fen.split(' ')[5], 10) || 1 };
}
let evalRenderPending = false;
function scheduleEvalRender() { // coalesce les MAJ partielles → au plus ~4 rendus/s (anti-flicker)
  if (evalRenderPending) return;
  evalRenderPending = true;
  setTimeout(() => { evalRenderPending = false; renderGauge(); renderMultiPV(); }, 250);
}
// Chemin de coups (UCI) menant à une position → le worker rejoue depuis startFen pour l'HISTORIQUE NN
// (sinon éval hors-distribution). startFen = position de départ de la partie (startpos ou header FEN).
function gameStartFen() { return positions[0].fen; }
function mainMoves(i) { const m = []; for (let k = 1; k <= i && k < positions.length; k++) m.push(positions[k].uci || '?'); return m; }
function variantMoves(vk) { const m = mainMoves(variant.branchIdx); for (let k = 0; k <= vk; k++) m.push(variant.moves[k].uci || '?'); return m; }
// '?' défensif : un uci manquant ne doit JAMAIS produire un chemin vide (collision de clé avec la
// position précédente) ; le worker retombe alors proprement sur la FEN nue.

function pump() {
  if (busy || paused || !modelReady || !positions.length) return;
  tick++;
  const fgWanted = currentEval() !== 'terminal'; // ne pas approfondir une position terminale
  const sf = gameStartFen();
  if (variant) {
    // En variante : avant-plan = position de variante affichée (live) ; arrière-plan = compléter le
    // préfixe mainline [0..branchIdx] PUIS les positions de variante → remplit le graphe de la variante.
    let bgFen = null, bgIdx = -1, bgVk = -1;
    for (let i = 0; i <= variant.branchIdx && i < positions.length; i++) {
      if (evals[i] == null) { bgFen = positions[i].fen; bgIdx = i; break; } // trou du préfixe
    }
    if (bgFen == null) {
      const k = variant.evals.findIndex((e, j) => e == null && variant.moves[j].fen !== currentFen());
      if (k >= 0) { bgFen = variant.moves[k].fen; bgVk = k; } // position de variante (résultat routé par FEN)
    }
    if (bgFen != null && (!fgWanted || tick % FG_YIELD !== 0)) {
      busy = true;
      const moves = bgIdx >= 0 ? mainMoves(bgIdx) : variantMoves(bgVk);
      worker.postMessage({ type: 'analyze', fen: bgFen, startFen: sf, moves, sims: ANALYZE_SIMS, idx: bgIdx, reuse: false, gen: analysisGen });
    } else if (fgWanted) {
      busy = true; // idx:-1 = hors graphe mainline ; résultat routé vers variant.evals par FEN
      worker.postMessage({ type: 'analyze', fen: currentFen(), startFen: sf, moves: variantMoves(variant.idx), sims: FG_PACKET, idx: -1, reuse: true, gen: analysisGen });
    }
    return;
  }
  while (queue.length && (evals[queue[0]] != null || queue[0] === index)) queue.shift(); // courante = avant-plan
  const bgIdx = queue.length ? queue[0] : -1;
  // Graphe incomplet (bgIdx>=0) → arrière-plan prioritaire (le reste du temps), 1 avant-plan sur FG_YIELD.
  // Graphe complet → uniquement avant-plan (approfondissement continu de la position affichée).
  if (bgIdx >= 0 && (!fgWanted || tick % FG_YIELD !== 0)) {
    queue.shift(); busy = true;
    worker.postMessage({ type: 'analyze', fen: positions[bgIdx].fen, startFen: sf, moves: mainMoves(bgIdx), sims: ANALYZE_SIMS, idx: bgIdx, reuse: false, gen: analysisGen });
  } else if (fgWanted) {
    busy = true; // avant-plan sur la position AFFICHÉE (mainline) ; idx = index
    worker.postMessage({ type: 'analyze', fen: currentFen(), startFen: sf, moves: mainMoves(index), sims: FG_PACKET, idx: index, reuse: true, gen: analysisGen });
  }
  // sinon (position terminale + file vide) : idle ; relancé à la navigation.
}
function onWorkerMsg(m) {
  if (m.type === 'model-ready') { modelReady = true; pump(); }
  else if (m.type === 'result') {
    busy = false;
    if (m.gen === analysisGen) { // ignore les résultats périmés
      const o = orientWdl(m.fen, m.wdl); // d'après la FEN analysée ; 'terminal' si wdl null
      if (o && o !== 'terminal') { o.pv = m.topPV || null; o.n = m.n || 0; o.crit = m.criticality || null; }
      if (m.idx >= 0 && m.idx < positions.length) { evals[m.idx] = o; renderTimeline(); } // graphe mainline
      if (variant) { const vk = variant.moves.findIndex((mv) => mv.fen === m.fen); if (vk >= 0) { variant.evals[vk] = o; renderTimeline(); } } // graphe variante (routé par FEN)
      if (m.fen === currentFen()) { live = { fen: m.fen, data: o }; scheduleEvalRender(); } // position affichée
    }
    pump();
  } else if (m.type === 'error') {
    busy = false;
    $('eval').hidden = false; $('eval-status').textContent = 'Analyse indisponible.';
    pump();
  }
}

// Valeur WDL v = P(W)−P(L) ∈ [−1,1] → équivalent-pions type UCI (mapping Lc0), borné à ±15,00. Échelle UNIQUE
// pour l'éval principale ET les variantes (MultiPV) : un seul barème affiché à l'utilisateur.
function vToPawns(v) {
  const c = (x, a, b) => Math.max(a, Math.min(b, x));
  return c(Math.round(111.71 * Math.tan(1.562 * c(v, -0.999, 0.999))), -1500, 1500) / 100;
}
function fmtPawns(p) { return `${p >= 0 ? '+' : '−'}${Math.abs(p).toFixed(2).replace('.', ',')}`; }

function renderGauge() {
  $('eval').hidden = false;
  const e = currentEval();
  if (e == null) { $('eval-status').textContent = 'Analyse…'; return; }
  const setG = (w, d, l) => { $('g-w').style.flexGrow = w; $('g-d').style.flexGrow = d; $('g-l').style.flexGrow = l; };
  if (e === 'terminal') {
    setG(0, 1, 0); $('g-w').textContent = $('g-d').textContent = $('g-l').textContent = '';
    $('eval-v').textContent = 'Position terminale'; $('eval-flag').hidden = true; $('eval-status').textContent = '';
    $('crit').hidden = true;
    return;
  }
  const { pw, pd, pl, fm } = e;
  setG(pw, pd, pl);
  const lbl = (x) => { const k = Math.round(x * 100); return k >= 8 ? `${k}%` : ''; }; // % incrusté (masqué si segment trop fin)
  $('g-w').textContent = lbl(pw); $('g-d').textContent = lbl(pd); $('g-l').textContent = lbl(pl);
  const v = pw - pl;
  const phrase = v > 0.25 ? 'La position favorise les Blancs.' : v < -0.25 ? 'La position favorise les Noirs.' : 'Position équilibrée.';
  $('eval-v').textContent = `${fmtPawns(vToPawns(v))} · ${phrase}`; // équivalent-pions (même échelle que les variantes)
  // jauge de CRITICITÉ (vert→rouge, label incrusté) — computeCriticality (v3e sim-stable, validé Opéra/Byrne-Fischer).
  const c = e.crit;
  if (c && c.level) {
    $('crit').hidden = false;
    const s = Math.max(0, Math.min(1, c.score));
    const hue = Math.max(0, 130 * (1 - s / 0.55)); // vert(130)→rouge(0), rouge dès ~0,55
    $('crit-fill').style.width = `${Math.round(s * 100)}%`;
    $('crit-fill').style.background = `hsl(${hue} 66% 42%)`;
    $('crit-txt').textContent = c.level === 'sharp' ? 'Position critique — à réfléchir'
      : c.level === 'tense' ? 'À surveiller' : 'Position tranquille';
  } else { $('crit').hidden = true; }
  $('eval-flag').hidden = !(fm < 12 && Math.max(pw, pl) > 0.60);
  // compteur de sims : feedback de l'approfondissement continu (s'affine tant que la position est regardée)
  $('eval-status').textContent = e.n ? `${e.n.toLocaleString('fr-FR')} sims` : '';
}

// Notation façon Lichess : figurines (♚♛♜♝♞) + numéros de coups depuis la position courante.
const FIG = { K: '♚', Q: '♛', R: '♜', B: '♝', N: '♞' };
const figurine = (san) => san.replace(/[KQRBN]/g, (c) => FIG[c]);
function numberedPV(sanList, fen) {
  const f = fen.split(' ');
  let fm = parseInt(f[5], 10) || 1, white = f[1] === 'w', out = '';
  for (let i = 0; i < sanList.length; i++) {
    const mv = figurine(sanList[i]);
    if (white) { out += `${fm}. ${mv} `; white = false; }
    else { out += (i === 0 ? `${fm}… ${mv} ` : `${mv} `); fm++; white = true; }
  }
  return out.trim();
}

// MultiPV : 3 variantes principales, orientées Blancs (mini-jauge W/D/L + cp en pions type UCI + ligne SAN).
// Ordre = celui du moteur (plus visité d'abord) : la MEILLEURE variante est TOUJOURS en 1re ligne.
function renderMultiPV() {
  const e = currentEval();
  const pv = (e && e !== 'terminal') ? e.pv : null;
  const card = $('pv-card');
  if (!pv || !pv.length) { card.hidden = true; if (board) board.clearArrows(); return; }
  card.hidden = false;
  const fen = currentFen();
  const blackToMove = fen.split(' ')[1] === 'b';
  const firstOf = (l) => (l.pv && l.pv[0]) || '';
  const lines = pv;                   // ordre moteur, pas de réordonnancement (cf. ci-dessus)
  const bestMove = firstOf(lines[0]); // meilleure variante = 1re ligne
  $('pv-list').innerHTML = lines.map((line) => {
    const vW = blackToMove ? -line.v : line.v; // orienter du POV Blancs (cohérent avec la jauge)
    const cpTxt = fmtPawns(vToPawns(vW)); // même échelle que l'éval principale
    // mini-jauge W/D/L de la variante, orientée Blancs (wdl enfant = POV trait APRÈS le coup)
    let pw, pd, pl;
    if (line.wdl) { const c = line.wdl; const o = blackToMove ? c : [c[2], c[1], c[0]]; pw = o[0]; pd = o[1]; pl = o[2]; }
    else { pw = Math.max(0, vW); pl = Math.max(0, -vW); pd = Math.max(0, 1 - pw - pl); }
    const pct = (x) => { const k = Math.round(x * 100); return k >= 20 ? `${k}%` : ''; }; // % incrusté, masqué si segment trop fin
    const gauge = `<span class="pv-gauge"><i class="gw" style="flex-grow:${pw}">${pct(pw)}</i><i class="gd" style="flex-grow:${pd}">${pct(pd)}</i><i class="gl" style="flex-grow:${pl}">${pct(pl)}</i></span>`;
    const moves = line.pv.length ? (numberedPV(line.pv.slice(0, 10), fen) + (line.pv.length > 10 ? ' …' : '')) : '—';
    const best = firstOf(line) === bestMove ? ' pv-best' : '';
    return `<li class="pv-line${best}" data-from="${line.from || ''}" data-to="${line.to || ''}">${gauge}<span class="pv-moves">${moves}</span><span class="pv-eval">${cpTxt}</span></li>`;
  }).join('');
  // flèches persistantes sur l'échiquier (façon Lichess) : largeur ∝ score (meilleure = plus épaisse/opaque)
  if (board) {
    const vbest = pv[0] ? pv[0].v : 0;
    const arrows = pv.filter((l) => l.from && l.to).map((l, i) => {
      const t = Math.min(1, Math.max(0, vbest - l.v) / 0.4); // 0 = meilleure ; 1 = ≥0,4 moins bonne
      return { from: l.from, to: l.to, width: 0.19 - 0.12 * t, opacity: Math.max(0.4, 0.92 - 0.18 * i) };
    });
    board.setArrows(arrows);
  }
}

// Graphe W/D/L empilé (3 zones) : la PARTIE, ou — en variante — le préfixe mainline + la ligne de variante.
function renderTimeline() {
  // Données : mainline, ou (en variante) evals[0..branchIdx] suivis des evals de la variante.
  const data = variant ? evals.slice(0, variant.branchIdx + 1).concat(variant.evals) : evals;
  const cursor = variant ? variant.branchIdx + 1 + variant.idx : index;
  const label = variant ? 'Analyse de la variante' : 'Analyse de la partie';
  const n = data.length, H = 36;
  if (n < 2) { $('timeline-wrap').hidden = true; return; }
  $('timeline-wrap').hidden = false;
  const X = (i) => (i / (n - 1)) * 100;
  const pts = [];
  for (let i = 0; i < n; i++) {
    const e = data[i];
    if (!e || e === 'terminal') break; // zone continue = préfixe contigu
    pts.push({ x: X(i), b1: H * (1 - e.pw), b2: H * e.pl });
  }
  let svg = '';
  if (pts.length >= 2) {
    const f = pts[0].x.toFixed(2), l = pts[pts.length - 1].x.toFixed(2);
    const top = pts.map((p) => `${p.x.toFixed(2)},${p.b1.toFixed(2)}`);
    const mid = pts.map((p) => `${p.x.toFixed(2)},${p.b2.toFixed(2)}`);
    const white = `${f},${H} ${l},${H} ${top.slice().reverse().join(' ')}`;
    const draw = `${top.join(' ')} ${mid.slice().reverse().join(' ')}`;
    const black = `${mid.join(' ')} ${l},0 ${f},0`;
    // Couleurs W/D/L = convention self-play : Blancs blanc chaud / Nulle cyan / Noirs gris profond.
    svg = `<polygon points="${white}" fill="#ece7da"/>`
        + `<polygon points="${draw}" fill="#6ec9d4"/>`
        + `<polygon points="${black}" fill="#383b4e"/>`;
  }
  $('timeline').innerHTML = svg;
  $('timeline').classList.remove('tl-dim'); // on affiche le vrai graphe (partie OU variante), jamais grisé
  $('tl-overlay').hidden = true;
  $('tl-cursor').style.left = `${X(Math.max(0, Math.min(n - 1, cursor)))}%`;
  const filled = data.reduce((a, e) => a + (e ? 1 : 0), 0);
  $('timeline-prog').textContent = filled >= n ? (variant ? 'Graphe · variante' : '') : `${label}… ${filled}/${n}`;
}

function initWorker() {
  worker = new Worker('dist/worker.bundle.js?backend=' + BACKEND);
  worker.onmessage = (e) => onWorkerMsg(e.data || {});
  worker.onerror = () => { busy = false; $('eval').hidden = false; $('eval-status').textContent = 'Modèle indisponible.'; pump(); };
  worker.postMessage({ type: 'start', apiBase: API, backend: BACKEND });
}

// Hygiène : pause l'analyse continue quand l'onglet est caché (reprise au retour) — pas de calcul invisible.
document.addEventListener('visibilitychange', () => {
  paused = document.visibilityState === 'hidden';
  if (!paused) pump();
});

function renderMoveList() {
  // positions[0] = départ ; positions[i>=1].san = i-ème demi-coup. + bloc variante après le branchement.
  let html = '';
  const block = variant ? variantBlockHtml() : '';
  if (variant && variant.branchIdx === 0) html += block; // variante depuis la position de départ
  for (let i = 1; i < positions.length; i += 2) {
    const num = (i + 1) / 2;
    const act = (j) => (!variant && j === index) ? ' active' : '';
    const w = `<li class="mv${act(i)}" data-i="${i}">${positions[i].san}</li>`;
    const b = positions[i + 1] ? `<li class="mv${act(i + 1)}" data-i="${i + 1}">${positions[i + 1].san}</li>` : '<li></li>';
    html += `<li class="num">${num}.</li>${w}${b}`;
    if (variant && (i === variant.branchIdx || i + 1 === variant.branchIdx)) html += block;
  }
  const ol = $('moves');
  ol.innerHTML = html;
  ol.querySelectorAll('.mv').forEach((el) => el.addEventListener('click', () => goTo(Number(el.dataset.i))));
  ol.querySelectorAll('.vm').forEach((el) => el.addEventListener('click', () => { variant.idx = Number(el.dataset.vi); render(); }));
}

// Navigation. goTo = mainline (sort de toute variante). next/prev gèrent la variante.
function goTo(i) { variant = null; index = Math.max(0, Math.min(positions.length - 1, i)); render(); }
function next() {
  if (variant) { if (variant.idx < variant.moves.length - 1) { variant.idx++; render(); } return; }
  goTo(index + 1);
}
function prev() {
  if (variant) {
    if (variant.idx > 0) { variant.idx--; render(); } else exitVariant(); // au 1er coup → retour partie
    return;
  }
  goTo(index - 1);
}

function loadSequence(seq, label) {
  positions = seq.positions;
  index = 0;
  variant = null; live = null; // nouvelle partie : aucune variante en cours
  resetAnalysis();
  renderMoveList();
  render();
  const meta = label || '';
  $('meta').textContent = positions.length > 1 ? `${meta}${positions.length - 1} demi-coups` : (meta || 'Position chargée');
  setErr('');
}

function headersLabel(headers) {
  if (!headers || !headers.get) return '';
  const w = headers.get('White'), b = headers.get('Black');
  return (w || b) ? `${w || '?'} – ${b || '?'} · ` : '';
}

// Handoff : lien propre `?game=<shareId>` (récupère le PGN du backend), `?pgn=`/`?fen=`, ou partie
// reçue depuis « Jouer » via sessionStorage['nz_current_game']. Priorité game > pgn > sessionStorage > FEN.
async function tryHandoff() {
  const params = new URLSearchParams(location.search);
  const gameParam = params.get('game');
  const pgnParam = params.get('pgn');
  const fenParam = params.get('fen');
  let stored = null;
  try { stored = sessionStorage.getItem('nz_current_game'); } catch { /* sessionStorage indispo */ }
  try {
    if (gameParam) { // lien propre /analyse/?game=<shareId>
      const res = await fetch(`${SITE_API}/games/${encodeURIComponent(gameParam)}`);
      if (res.ok) { const g = await res.json(); loadSequence(positionsFromPgn(g.pgn)); return; }
      setErr('Partie introuvable.'); return;
    }
    if (pgnParam) { loadSequence(positionsFromPgn(pgnParam)); return; }
    if (stored) { loadSequence(positionsFromPgn(stored)); return; }
    if (fenParam) { loadSequence(positionFromFen(fenParam)); return; }
  } catch (e) { setErr(e.message || 'Lien invalide.'); }
}

function onLoad() {
  const pgn = $('pgn-input').value.trim();
  const fen = $('fen-input').value.trim();
  try {
    if (pgn) {
      const r = positionsFromPgn(pgn);
      loadSequence(r, headersLabel(r.headers));
    } else if (fen) {
      loadSequence(positionFromFen(fen));
    } else {
      setErr('Colle un PGN ou un FEN.');
    }
  } catch (e) {
    setErr(e.message || 'Chargement impossible.');
  }
}

function init() {
  board = new BoardInteractive($('board'), { flip: false, onMove: playMove }); // jouer un coup → variante
  board.updatePosition(START_FEN);

  // Mobile : la carte Évaluation (jauge W/D/L + criticité) remonte SOUS l'échiquier — dans l'ordre
  // naturel du flux elle arrivait en dernier (après Variantes ET Charger), enterrée sous 2 écrans de scroll.
  if (IS_MOBILE) {
    const ba = document.querySelector('.board-area');
    if (ba && $('eval') && $('pv-card')) ba.insertBefore($('eval'), $('pv-card'));
  }

  $('load-btn').addEventListener('click', onLoad);
  $('sample-btn').addEventListener('click', () => {
    $('pgn-input').value = SAMPLE_PGN;
    $('fen-input').value = '';
    onLoad();
  });
  $('first').addEventListener('click', () => goTo(0));
  $('prev').addEventListener('click', prev);
  $('next').addEventListener('click', next);
  $('last').addEventListener('click', () => goTo(positions.length - 1));

  document.addEventListener('keydown', (e) => {
    const t = e.target;
    if (t && (t.tagName === 'TEXTAREA' || t.tagName === 'INPUT')) return; // pas en cours de saisie
    if (e.key === 'ArrowLeft') { prev(); e.preventDefault(); }
    else if (e.key === 'ArrowRight') { next(); e.preventDefault(); }
    else if (e.key === 'Home') { goTo(0); e.preventDefault(); }
    else if (e.key === 'End') { goTo(positions.length - 1); e.preventDefault(); }
    else if (e.key === 'Escape' && variant) { exitVariant(); e.preventDefault(); }
  });

  $('vb-exit').addEventListener('click', exitVariant); // retour à la partie
  // Clic sur une ligne MultiPV → jouer son 1er coup (entre/poursuit la variante), façon Lichess.
  $('pv-list').addEventListener('click', (e) => {
    const li = e.target.closest('.pv-line');
    if (li && li.dataset.from && li.dataset.to) playMove(li.dataset.from, li.dataset.to);
  });

  // Partage de position (S-13) : lien /analyse/?fen=<courant>.
  $('share-btn').addEventListener('click', async () => {
    const btn = $('share-btn');
    const url = `${location.origin}/analyse/?fen=${encodeURIComponent(positions[index].fen)}`;
    const ok = await copyShareLink(url);
    const orig = btn.dataset.label || btn.textContent; btn.dataset.label = orig;
    btn.textContent = ok ? 'Lien copié !' : 'Copie impossible';
    setTimeout(() => { btn.textContent = btn.dataset.label; }, 2000);
  });

  // Clic sur le graphe → navigation vers le coup correspondant (mainline OU variante).
  $('timeline').addEventListener('click', (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    if (variant) {
      const n = variant.branchIdx + 1 + variant.moves.length;
      const j = Math.round(frac * (n - 1));
      if (j <= variant.branchIdx) { variant = null; index = j; render(); }       // clic dans le préfixe → retour partie
      else { variant.idx = j - (variant.branchIdx + 1); render(); }              // clic dans la variante
      return;
    }
    if (positions.length < 2) return;
    goTo(Math.round(frac * (positions.length - 1)));
  });

  // Curseur de profondeur (S-14) : input = MAJ libellé ; change (relâché) = relance la passe au nouveau budget.
  $('depth').addEventListener('input', () => { $('depth-val').textContent = `${SIMS_STEPS[+$('depth').value]} sims`; });
  $('depth').addEventListener('change', () => {
    ANALYZE_SIMS = SIMS_STEPS[+$('depth').value] || 512;
    $('depth-val').textContent = `${ANALYZE_SIMS} sims`;
    resetAnalysis(); renderTimeline(); renderGauge(); renderMultiPV();
  });

  initWorker();
  resetAnalysis();
  tryHandoff(); // lien partagé (?pgn/?fen) ou partie reçue depuis « Jouer » (sessionStorage)

  // Sizer board + graphe (fit largeur dispo, plafond 620, façon /play/).
  const fit = () => {
    const area = document.querySelector('.board-area');
    const wrap = document.querySelector('.board-wrap');
    if (!area || !wrap) return;
    const size = Math.max(160, Math.floor(Math.min(area.clientWidth, window.innerHeight - 120, 620)));
    wrap.style.width = `${size}px`; wrap.style.height = `${size}px`;
    const tl = document.querySelector('.timeline-wrap');
    if (tl) tl.style.width = `${size}px`;
    const pvc = document.querySelector('.pv-card');
    if (pvc) pvc.style.width = `${size}px`;
  };
  window.addEventListener('resize', fit);
  if (typeof ResizeObserver !== 'undefined') new ResizeObserver(fit).observe(document.querySelector('.board-area'));
  fit();
  render();
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init); else init();
