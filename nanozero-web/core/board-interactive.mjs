// board-interactive.mjs — échiquier interactif « board-pro » (drag + clic-clic + feedbacks Lichess-like).
// Composant PARTAGÉ (core/), réutilisé par Jouer (Lot P) et Analyse (Lot A). Sans framework.
// Délègue TOUTES les règles au cœur (game-state/fast-board) — l'UI ne recalcule jamais la légalité.
//
// Feedbacks (parité Lichess/Chess.com) : clic-clic + drag, points sur cases vides + anneaux sur
// captures, surlignage dernier coup, case sélectionnée, roi en échec (rouge), coordonnées inline,
// promotion verticale sur la file d'arrivée, animation de glissement ~200 ms, curseur grab au survol.
//
// API : new BoardInteractive(containerEl, { flip?, onMove?, piecesBase? })
//   onMove(from, to, promotion?) — from/to algébriques ('e2','e4'), promotion 'q'|'r'|'b'|'n'|undefined.
//   updatePosition(fen, lastMove?, { animate? }?) · setFlip(bool) · destroy()

import { fenToCells } from './board-render.mjs';
import { GameState } from './game-state.mjs';

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const PROMO_ORDER = ['q', 'n', 'r', 'b']; // ordre Lichess : Dame, Cavalier, Tour, Fou
const DRAG_THRESHOLD = 3; // px avant de passer du clic au drag (Chessground = 3)
const ANIM_MS = 200;

// ---- Helpers PURS (testables sans DOM) -------------------------------------------------------

// Cellule i de fenToCells (ordre a8..h1) → case algébrique. i=0→a8, i=7→h8, i=63→h1.
export function algForCell(i) {
  return String.fromCharCode(97 + (i % 8)) + (8 - Math.floor(i / 8));
}

// Trait depuis le FEN (2e champ : 'w'|'b'). Évite de dépendre du retour exact de sideToMove().
export function sideToMoveFromFen(fen) {
  const f = String(fen).trim().split(/\s+/)[1];
  return f === 'b' ? 'b' : 'w';
}

/**
 * Coups légaux partant de `fromAlg`, regroupés par case d'arrivée.
 * @returns {{ targets: string[], promo: Record<string,string[]> }}
 */
export function legalTargetsFrom(gs, fromAlg) {
  const targets = new Set();
  const promo = {};
  for (const m of gs.legalMoves()) {
    const c = m.cjs;
    if (c.from !== fromAlg) continue;
    targets.add(c.to);
    if (c.promotion) (promo[c.to] ||= []).push(c.promotion);
  }
  return { targets: [...targets], promo };
}

// Case algébrique du roi du trait (pour le surlignage d'échec). null si introuvable.
export function kingSquareForSide(cells, side) {
  const king = side + 'K';
  for (let i = 0; i < 64; i++) if (cells[i] && cells[i].piece === king) return algForCell(i);
  return null;
}

// ---- Feuille de style injectée une fois ------------------------------------------------------

const STYLE_ID = 'nz-board-style';
const CSS = `
.nz-board{display:grid;grid-template-columns:repeat(8,1fr);grid-template-rows:repeat(8,1fr);
  aspect-ratio:1/1;max-width:100%;min-width:0;position:relative;user-select:none;touch-action:none;}
.nz-sq{position:relative;min-width:0;min-height:0;}
.nz-sq.d{background:var(--board-dark,#8B6914);}
.nz-sq.l{background:var(--board-light,#F5E6C0);}
.nz-sq img{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;pointer-events:none;
  z-index:2;}
/* surlignages (dernier coup / sélection / échec) — couche sous la pièce */
.nz-sq.last::before,.nz-sq.sel::before,.nz-sq.chk::before{content:"";position:absolute;inset:0;z-index:1;}
.nz-sq.last::before{background:var(--nz-last,rgba(255,206,84,.42));}
.nz-sq.sel::before{background:var(--nz-sel,rgba(255,206,84,.55));}
.nz-sq.chk::before{background:radial-gradient(circle,var(--nz-check,rgba(220,54,54,.9)) 0%,
  rgba(220,54,54,.45) 45%,transparent 72%);}
/* marqueurs de coups légaux — au-dessus de la pièce, non cliquables */
.nz-sq.tgt::after,.nz-sq.cap::after{content:"";position:absolute;inset:0;z-index:3;pointer-events:none;}
.nz-sq.tgt::after{background:radial-gradient(circle,var(--nz-dot,rgba(40,28,8,.32)) 0 22%,transparent 24%);}
.nz-sq.cap::after{background:radial-gradient(circle,transparent 0 78%,var(--nz-dot,rgba(40,28,8,.32)) 79% 90%,transparent 91%);}
/* flèches de variante (PV) — overlay AU-DESSUS des pièces, non cliquable */
.nz-arrows{position:absolute;inset:0;width:100%;height:100%;pointer-events:none;z-index:4;overflow:visible;}
/* coordonnées inline (coins) */
.nz-coord{position:absolute;font:600 max(9px,2.1cqmin)/1 var(--font-disp,system-ui),sans-serif;
  z-index:2;pointer-events:none;opacity:.85;}
.nz-file{bottom:2px;right:3px;}
.nz-rank{top:2px;left:3px;}
.nz-sq.d .nz-coord{color:var(--board-light,#F5E6C0);}
.nz-sq.l .nz-coord{color:var(--board-dark,#8B6914);}
/* survol desktop : curseur grab sur les pièces jouables */
@media (pointer:fine){.nz-sq.own{cursor:grab;}.nz-sq.own:hover::before{content:"";position:absolute;
  inset:0;z-index:1;background:rgba(255,206,84,.16);}}
.nz-ghost{position:fixed;z-index:9999;pointer-events:none;opacity:.9;will-change:transform;}
.nz-promo{position:absolute;z-index:50;display:flex;flex-direction:column;box-shadow:0 6px 24px rgba(0,0,0,.6);
  border-radius:6px;overflow:hidden;background:var(--surface,#1A1F2B);}
.nz-promo button{width:100%;aspect-ratio:1/1;border:0;background:var(--board-light,#F5E6C0);cursor:pointer;padding:0;}
.nz-promo button:nth-child(even){background:var(--board-dark,#8B6914);}
.nz-promo img{width:100%;height:100%;object-fit:contain;display:block;}
.nz-promo-veil{position:absolute;inset:0;z-index:49;background:rgba(0,0,0,.35);}
@media (prefers-reduced-motion:reduce){.nz-anim{animation:none!important;}}
`;

function ensureStyle() {
  if (typeof document === 'undefined' || document.getElementById(STYLE_ID)) return;
  const s = document.createElement('style');
  s.id = STYLE_ID;
  s.textContent = CSS;
  document.head.appendChild(s);
}

// ---- Composant DOM ---------------------------------------------------------------------------

export class BoardInteractive {
  constructor(containerEl, { flip = false, onMove = () => {}, piecesBase = '/pieces/', readOnly = false } = {}) {
    ensureStyle();
    this.el = containerEl;
    this.flip = !!flip;
    this.onMove = onMove;
    this.readOnly = !!readOnly; // lecture seule (viewer) : aucune interaction de coup
    this.piecesBase = piecesBase;
    this.fen = START_FEN;
    this.gs = null;
    this._lastMove = null;   // {from,to} — surligné jusqu'au prochain coup
    this._selected = null;   // case sélectionnée (clic-clic)
    this._targets = null;    // {set:Set,promo:{}} pour la sélection courante
    this._drag = null;       // {from,sx,sy,moved,ghost}
    this._squares = [];
    this._buildGrid();
    this.updatePosition(START_FEN);
  }

  _buildGrid() {
    const el = this.el;
    el.classList.add('nz-board');
    el.style.containerType = 'inline-size'; // pour les coordonnées en cqmin
    el.innerHTML = '';
    this._squares = [];
    for (let d = 0; d < 64; d++) {
      const sq = document.createElement('div');
      sq.className = 'nz-sq';
      sq.addEventListener('pointerdown', (e) => this._onPointerDown(e, d));
      el.appendChild(sq);
      this._squares.push(sq);
    }
    el.addEventListener('pointermove', (e) => this._onPointerMove(e));
    el.addEventListener('pointerup', (e) => this._onPointerUp(e));
    // overlay SVG pour les flèches de variante (coordonnées 0..8 = une unité par case)
    this._arrows = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    this._arrows.setAttribute('class', 'nz-arrows');
    this._arrows.setAttribute('viewBox', '0 0 8 8');
    el.appendChild(this._arrows);
  }

  // Centre d'une case (en unités viewBox 0..8), orientation prise en compte.
  _center(alg) { const s = this._slotForAlg(alg); return { x: (s % 8) + 0.5, y: Math.floor(s / 8) + 0.5 }; }

  // Affiche PLUSIEURS flèches (variantes MultiPV). arrows = [{from, to, width?, opacity?}].
  // La meilleure (dessinée en dernier = au-dessus) est plus épaisse/opaque (façon Lichess).
  setArrows(arrows) {
    if (!this._arrows) return;
    if (!arrows || !arrows.length) { this._arrows.innerHTML = ''; return; }
    let html = '';
    const OUT = 0.028; // contour fin, unique
    const P = (x, y) => `${x.toFixed(3)},${y.toFixed(3)}`;
    for (const ar of [...arrows].reverse()) { // worst d'abord → best au-dessus
      if (!ar.from || !ar.to) continue;
      const a = this._center(ar.from), b = this._center(ar.to);
      const dx = b.x - a.x, dy = b.y - a.y, len = Math.hypot(dx, dy) || 1;
      const ux = dx / len, uy = dy / len, px = -uy, py = ux;
      const w = ar.width || 0.15;
      const sw = w / 2;                              // demi-largeur du corps
      const hw = w * 1.45;                            // demi-largeur de la tête
      const hl = Math.min(len * 0.55, w * 2.6);       // longueur de la tête (bornée pour coups courts)
      const cx = b.x - ux * hl, cy = b.y - uy * hl;   // base de la tête
      // UNE SEULE forme corps+tête → contour continu, pas de raccord
      const pts = [
        P(a.x + px * sw, a.y + py * sw),
        P(cx + px * sw, cy + py * sw),
        P(cx + px * hw, cy + py * hw),
        P(b.x, b.y),
        P(cx - px * hw, cy - py * hw),
        P(cx - px * sw, cy - py * sw),
        P(a.x - px * sw, a.y - py * sw),
      ].join(' ');
      const op = ar.opacity != null ? ar.opacity : 0.9;
      html += `<polygon points="${pts}" fill="rgba(201,169,110,${op})" stroke="rgba(92,68,34,${Math.min(1, op * 0.85)})" stroke-width="${OUT}" stroke-linejoin="round"/>`;
    }
    this._arrows.innerHTML = html;
  }

  clearArrows() { if (this._arrows) this._arrows.innerHTML = ''; }

  _logicalForSlot(d) { return this.flip ? 63 - d : d; }
  _slotForAlg(alg) {
    const file = alg.charCodeAt(0) - 97;
    const rank = alg.charCodeAt(1) - 49;
    const logical = (7 - rank) * 8 + file;
    return this.flip ? 63 - logical : logical;
  }

  setFlip(flip) { this.flip = !!flip; this._render(); }

  updatePosition(fen, lastMove = null, { animate = false } = {}) {
    const prevFen = this.fen;
    this.fen = fen || START_FEN;
    this._lastMove = lastMove || null;
    try { this.gs = new GameState(this.fen); } catch { this.gs = null; }
    this._clearSelection();
    this.clearArrows();
    this._render();
    if (animate && lastMove && prevFen !== this.fen) this._animate(lastMove);
  }

  _cells() {
    try { return fenToCells(this.fen); }
    catch { return Array.from({ length: 64 }, () => ({ piece: '', dark: false })); } // FEN invalide → vide, pas de crash
  }

  _render() {
    const cells = this._cells();
    const side = sideToMoveFromFen(this.fen);
    const checkSq = (this.gs && this.gs.isInCheck()) ? kingSquareForSide(cells, side) : null;
    const lm = this._lastMove;
    const sel = this._selected;
    const tset = this._targets ? this._targets.set : null;
    for (let d = 0; d < 64; d++) {
      const L = this._logicalForSlot(d);
      const cell = cells[L];
      const alg = algForCell(L);
      const sq = this._squares[d];
      const cls = ['nz-sq', cell.dark ? 'd' : 'l'];
      if (lm && (alg === lm.from || alg === lm.to)) cls.push('last');
      if (sel === alg) cls.push('sel');
      if (checkSq === alg) cls.push('chk');
      if (tset && tset.has(alg)) cls.push(cell.piece ? 'cap' : 'tgt');
      if (this.gs && cell.piece && cell.piece[0] === side) cls.push('own');
      sq.className = cls.join(' ');
      // contenu : coordonnées (bords) + pièce
      let html = '';
      const file = L % 8, rankTop = Math.floor(L / 8);
      const onBottom = this.flip ? rankTop === 0 : rankTop === 7;
      const onLeft = this.flip ? file === 7 : file === 0;
      if (onBottom) html += `<span class="nz-coord nz-file">${alg[0]}</span>`;
      if (onLeft) html += `<span class="nz-coord nz-rank">${alg[1]}</span>`;
      if (cell.piece) html += `<img src="${this.piecesBase}${cell.piece}.svg" alt="${cell.piece}" draggable="false">`;
      sq.innerHTML = html;
    }
  }

  // ---- Sélection / coups ----
  _select(alg) {
    if (!this.gs) return false;
    const { targets, promo } = legalTargetsFrom(this.gs, alg);
    if (!targets.length) return false;
    this._selected = alg;
    this._targets = { set: new Set(targets), promo };
    this._render();
    return true;
  }

  _clearSelection() {
    this._selected = null;
    this._targets = null;
    if (this._drag && this._drag.ghost) this._drag.ghost.remove();
    this._drag = null;
  }

  _commit(toAlg) {
    const from = this._selected;
    const promoChars = this._targets ? this._targets.promo[toAlg] : null;
    this._clearSelection();
    this._render();
    if (promoChars && promoChars.length) this._openPromotion(from, toAlg, promoChars);
    else this.onMove(from, toAlg);
  }

  _ownPieceAt(alg) {
    const file = alg.charCodeAt(0) - 97, rank = alg.charCodeAt(1) - 49;
    const logical = (7 - rank) * 8 + file; // inverse de algForCell
    const c = this._cells()[logical];
    return c && c.piece && c.piece[0] === sideToMoveFromFen(this.fen) ? c.piece : null;
  }

  _onPointerDown(e, slot) {
    if (this.readOnly || !this.gs) return;
    const alg = algForCell(this._logicalForSlot(slot));
    // 2e clic sur une cible légale → joue le coup (clic-clic).
    if (this._selected && this._targets && this._targets.set.has(alg)) { this._commit(alg); return; }
    // sinon : (re)sélection d'une pièce du trait + amorce de drag.
    if (this._ownPieceAt(alg)) {
      if (this._selected !== alg) this._select(alg);
      this._drag = { from: alg, sx: e.clientX, sy: e.clientY, moved: false, ghost: null, slot };
      try { this.el.setPointerCapture(e.pointerId); } catch { /* ignore */ }
    } else if (this._selected) {
      this._clearSelection(); this._render(); // clic ailleurs → annule la sélection
    }
  }

  _onPointerMove(e) {
    const dg = this._drag;
    if (!dg) return;
    if (!dg.moved) {
      if (Math.hypot(e.clientX - dg.sx, e.clientY - dg.sy) < DRAG_THRESHOLD) return;
      dg.moved = true;
      const img = this._squares[dg.slot].querySelector('img');
      if (img) {
        const r = this._squares[dg.slot].getBoundingClientRect();
        const g = img.cloneNode(true);
        g.className = 'nz-ghost';
        g.style.width = `${r.width}px`; g.style.height = `${r.height}px`;
        document.body.appendChild(g);
        dg.ghost = g;
      }
    }
    if (dg.ghost) {
      const w = parseFloat(dg.ghost.style.width);
      dg.ghost.style.left = `${e.clientX - w / 2}px`;
      dg.ghost.style.top = `${e.clientY - w / 2}px`;
    }
  }

  _onPointerUp(e) {
    const dg = this._drag;
    if (!dg) return;
    this._drag = null;
    if (dg.ghost) dg.ghost.remove();
    if (!dg.moved) return; // simple clic : la sélection reste (clic-clic) — géré au prochain pointerdown
    const dropEl = document.elementFromPoint(e.clientX, e.clientY);
    const sqEl = dropEl && dropEl.closest ? dropEl.closest('.nz-sq') : null;
    const toAlg = sqEl ? algForCell(this._logicalForSlot(this._squares.indexOf(sqEl))) : null;
    if (toAlg && this._targets && this._targets.set.has(toAlg)) this._commit(toAlg);
    else { this._clearSelection(); this._render(); }
  }

  // ---- Promotion (overlay vertical sur la file d'arrivée, façon Lichess) ----
  _openPromotion(from, to, available) {
    const color = sideToMoveFromFen(this.fen);
    const slot = this._slotForAlg(to);
    const col = slot % 8;
    const rowTop = Math.floor(slot / 8); // 0 = haut du board
    const order = PROMO_ORDER.filter((c) => available.includes(c));
    const veil = document.createElement('div');
    veil.className = 'nz-promo-veil';
    const panel = document.createElement('div');
    panel.className = 'nz-promo';
    // la pile part de la case d'arrivée et descend vers l'intérieur du board.
    const down = rowTop === 0; // arrivée en haut → on empile vers le bas
    panel.style.left = `${(col / 8) * 100}%`;
    panel.style.width = `${100 / 8}%`;
    if (down) panel.style.top = `${(rowTop / 8) * 100}%`;
    else panel.style.bottom = `${((7 - rowTop) / 8) * 100}%`;
    if (!down) panel.style.flexDirection = 'column-reverse';
    for (const c of order) {
      const btn = document.createElement('button');
      const img = document.createElement('img');
      img.src = `${this.piecesBase}${color}${c.toUpperCase()}.svg`;
      img.alt = c;
      btn.appendChild(img);
      btn.addEventListener('click', () => { veil.remove(); panel.remove(); this.onMove(from, to, c); });
      panel.appendChild(btn);
    }
    veil.addEventListener('pointerdown', () => { veil.remove(); panel.remove(); }); // annuler
    this.el.appendChild(veil);
    this.el.appendChild(panel);
  }

  // ---- Animation de glissement (~200 ms) ----
  _animate({ from, to }) {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const toSlot = this._slotForAlg(to);
    const fromSlot = this._slotForAlg(from);
    const img = this._squares[toSlot] && this._squares[toSlot].querySelector('img');
    if (!img) return;
    const dx = ((fromSlot % 8) - (toSlot % 8)) * (this.el.clientWidth / 8);
    const dy = (Math.floor(fromSlot / 8) - Math.floor(toSlot / 8)) * (this.el.clientHeight / 8);
    if (!dx && !dy) return;
    try {
      img.animate(
        [{ transform: `translate(${dx}px, ${dy}px)` }, { transform: 'translate(0,0)' }],
        { duration: ANIM_MS, easing: 'ease-out' },
      );
    } catch { /* WAAPI absent → pas d'anim */ }
  }

  destroy() {
    this._clearSelection();
    this.el.innerHTML = '';
  }
}
