// game-state.mjs — Story 5 : ADAPTATEUR MINCE sur FastBoard.
//
// ⚠️ SWAP (SPEC FastBoard Story 5) : `GameState` n'EST plus un wrapper chess.js. C'est un
// proxy mince par-dessus `FastBoard` (chessops + make/unmake + Zobrist) qui PRÉSERVE
// EXACTEMENT l'interface publique consommée par le MCTS (mcts.mjs JAMAIS modifié) :
//   constructor(fen?) ; sideToMove() ; toEncoderState() ; legalMoves() ; applyMove(move) ;
//   undoMove() ; isGameOver() ; isCheckmate() ; isDraw() ; fen() ; lastMove().
//
// Le hot path board (legalMoves/applyMove/undoMove/toEncoderState/isGameOver) est
// délégué tel quel à FastBoard — plus AUCUN chess.js, plus AUCUN boardToBitboards ici.
//
// COMPAT-TESTS (au-delà du contrat MCTS, exigée par les gates Story 0-4 + selftests) :
//   - legalMoves()[i] expose .from/.to NUMÉRIQUES (FastBoard) + .type/.promo (NanoZero)
//     ET, en plus, .san (notation algébrique standard) + .cjs/.uci (cases algébriques),
//     reconstruits SANS chess.js depuis FastBoard (oracle SAN validé par gate parité).
//   - applyMove accepte : un move de legalMoves() (from/to numériques + promotion role),
//     un objet {from:'e2',to:'e4',promotion?:'q'} ALGÉBRIQUE (cross-validate / harness),
//     ou une chaîne SAN ('Nf3') — tous résolus contre les coups légaux de FastBoard.
//   - lastMove() retourne {from,to} en ALGÉBRIQUE (strings) pour la flèche UI.

import { FastBoard } from './fast-board.mjs';
import {
  PROMOTION,
  CASTLING,
  EN_PASSANT,
  unpackForApply,
} from './encoding/move-encoding.mjs';

// --- conversions case <-> algébrique (LSB=a1 : file=sq&7, rank=sq>>3) ---
const fileChar = (sq) => String.fromCharCode(97 + (sq & 7));
const algOf = (sq) => fileChar(sq) + ((sq >> 3) + 1);
const sqFromAlg = (alg) => (Number(alg[1]) - 1) * 8 + (alg.charCodeAt(0) - 97);

// chessops role -> lettre de promotion UCI (et l'inverse).
const ROLE_TO_PROMO_CHAR = { knight: 'n', bishop: 'b', rook: 'r', queen: 'q' };
const PROMO_CHAR_TO_ROLE = { n: 'knight', b: 'bishop', r: 'rook', q: 'queen' };
// rôle chessops -> lettre SAN (majuscule), pion = '' (jamais préfixé).
const ROLE_TO_SAN = { pawn: '', knight: 'N', bishop: 'B', rook: 'R', queen: 'Q', king: 'K' };

export class GameState {
  constructor(fen) {
    this._fb = new FastBoard(fen || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    this._lastMove = null; // {from,to} algébrique, ou null avant le 1er coup
  }

  // --------- contrat MCTS : pure délégation au hot path FastBoard ---------
  sideToMove() { return this._fb.sideToMove(); }
  toEncoderState() { return this._fb.toEncoderState(); }
  // Clé COMPACTE du cache d'éval NN (eval-cache.mjs) : `${lo},${hi},${rule50},${rep}`. Lc0-tronquée
  // (zobrist pièces+trait+roque+ep + 50-coups + répétition de la position COURANTE). CHEAP : lit
  // l'état FastBoard sans construire l'encoder-state complet → ~zéro surcoût sur un HIT.
  cacheKey() {
    const k = this._fb._repKey(); // {lo,hi} zobrist convention-Java
    const rule50 = this._fb.position.halfmoves; // compteur 50-coups (chess.js : nul à 100)
    const rep = this._fb._currentRepCount(); // 0/1/2 : occurrences passées de la position courante
    return `${k.lo},${k.hi},${rule50},${rep}`;
  }
  isGameOver() { return this._fb.isGameOver(); }
  isCheckmate() { return this._fb.isCheckmate(); }
  isInCheck() { return this._fb.position.isCheck(); } // trait en échec (UI : roi en rouge) — additif
  isDraw() { return this._fb.isDraw(); }
  fen() { return this._fb.fen(); }
  lastMove() { return this._lastMove; }

  // legalMoves() : coups FastBoard {from,to,type,promo,promotion} ENRICHIS de .san/.cjs/.uci.
  // L'objet retourné reste JOUABLE par applyMove (il porte from/to numériques + promotion role).
  //
  // ⚠️ PERF (cœur du swap) : le MCTS n'accède JAMAIS à .san/.cjs/.uci — uniquement à
  // from/to/type/promo. Calculer le SAN coûte un _make/_unmake par coup (suffixe échec/mat),
  // ce qui ANNULERAIT le gain board s'il était fait à chaque feuille. On les expose donc en
  // GETTERS PARESSEUX (mémoïsés) : zéro coût pour le hot path, payé seulement par les
  // tests/UI qui lisent réellement .san. Le tableau `moves` est passé pour la désambiguïsation.
  legalMoves() {
    const moves = this._fb.legalMoves();
    const self = this;
    for (const m of moves) {
      let cjs, uci, san; // caches
      Object.defineProperty(m, 'cjs', {
        enumerable: false, configurable: true,
        get() {
          if (cjs) return cjs;
          const promoChar = m.promotion ? ROLE_TO_PROMO_CHAR[m.promotion] : '';
          cjs = { from: algOf(m.from), to: algOf(m.to), promotion: promoChar || undefined };
          return cjs;
        },
      });
      Object.defineProperty(m, 'uci', {
        enumerable: false, configurable: true,
        get() {
          if (uci !== undefined) return uci;
          const promoChar = m.promotion ? ROLE_TO_PROMO_CHAR[m.promotion] : '';
          uci = algOf(m.from) + algOf(m.to) + promoChar;
          return uci;
        },
      });
      Object.defineProperty(m, 'san', {
        enumerable: false, configurable: true,
        get() {
          if (san !== undefined) return san;
          san = self._san(m, moves);
          return san;
        },
      });
    }
    return moves;
  }

  // applyMove : résout `move` vers un coup légal FastBoard puis _make(). Mémorise lastMove
  // (cases algébriques) pour la flèche UI. 3 formes acceptées (cf. en-tête).
  applyMove(move) {
    const fb = this._fb;
    let resolved;
    // Coup compact MCTS (int packMove) → objet rejouable {from,to,type,promo,promotion}. `type`
    // défini ⇒ joué tel quel par la branche from-numérique (pas de re-résolution coûteuse).
    if (typeof move === 'number') move = unpackForApply(move);
    if (typeof move === 'string') {
      resolved = this._resolveSan(move);
    } else if (typeof move.from === 'number') {
      // move FastBoard (from/to numériques) — vient de legalMoves() : a déjà type/promo/promotion,
      // donc JOUABLE tel quel. Cas dégénéré {from:int,to:int} brut sans `type` → on le re-résout
      // (notamment pour fixer la promotion implicite en dame).
      resolved = move.type !== undefined
        ? move
        : this._resolveAlg(algOf(move.from), algOf(move.to),
            move.promotion ? ROLE_TO_PROMO_CHAR[move.promotion] : undefined);
    } else {
      // objet algébrique {from:'e2', to:'e4', promotion?:'q'}.
      const src = move.cjs || move;
      resolved = this._resolveAlg(src.from, src.to, src.promotion);
    }
    if (!resolved) throw new Error(`applyMove : coup illégal/introuvable ${JSON.stringify(move)}`);
    this._lastMove = { from: algOf(resolved.from), to: algOf(resolved.to) };
    fb._make(resolved);
  }

  undoMove() {
    this._fb._unmake();
    // lastMove redevient indéterminé après un undo (le MCTS ne le lit jamais en descente).
    this._lastMove = null;
  }

  // ----------------------- résolution de coups -----------------------
  // Résout un (from,to) algébrique (+ lettre de promo optionnelle) vers un coup légal FastBoard.
  _resolveAlg(fromAlg, toAlg, promoChar) {
    const from = sqFromAlg(fromAlg);
    const to = sqFromAlg(toAlg);
    const wantRole = promoChar ? PROMO_CHAR_TO_ROLE[promoChar] : undefined;
    const cands = this._fb.legalMoves().filter((m) => m.from === from && m.to === to);
    if (cands.length === 0) return null;
    if (wantRole) return cands.find((m) => m.promotion === wantRole) || null;
    // pas de promo demandée : s'il y a des candidats de promotion, choisir la DAME par défaut
    // (convention chess.js : promotion omise → dame). Sinon le coup unique.
    const promoCands = cands.filter((m) => m.type === PROMOTION);
    if (promoCands.length) return promoCands.find((m) => m.promotion === 'queen') || promoCands[0];
    return cands[0];
  }

  // Résout une chaîne SAN vers un coup légal FastBoard (match exact sur le .san reconstruit).
  _resolveSan(san) {
    const norm = san.replace(/[+#]/g, '').replace(/=([QRBN])/, '=$1'); // tolère/normalise les suffixes
    const moves = this.legalMoves();
    // match exact d'abord (avec/sans suffixe échec) ; sinon match sur la racine sans +/#.
    return (
      moves.find((m) => m.san === san) ||
      moves.find((m) => m.san.replace(/[+#]/g, '') === norm) ||
      null
    );
  }

  // ----------------------- génération du SAN (sans chess.js) -----------------------
  // Reconstruit la notation algébrique standard d'un coup, depuis l'état FastBoard COURANT.
  // `allMoves` = la liste légale courante (pour la désambiguïsation). Suffixe +/# calculé en
  // jouant le coup (FastBoard _make/_unmake) puis en lisant isCheckmate/échec.
  _san(m, allMoves) {
    const board = this._fb.position.board;
    const piece = board.get(m.from);
    if (!piece) return '';

    // roque : O-O (côté roi) / O-O-O (côté dame) selon le file d'arrivée du roi.
    if (m.type === CASTLING) {
      const kingToFile = m.to & 7;
      const base = kingToFile === 6 ? 'O-O' : 'O-O-O';
      return base + this._checkSuffix(m);
    }

    const toAlg = algOf(m.to);
    const isCapture =
      m.type === EN_PASSANT || !!board.get(m.to) /* pièce occupant la case d'arrivée */;

    let core;
    if (piece.role === 'pawn') {
      // pion : capture → fichier de départ + 'x' ; sinon juste la case d'arrivée.
      core = isCapture ? fileChar(m.from) + 'x' + toAlg : toAlg;
      if (m.type === PROMOTION) core += '=' + ROLE_TO_SAN[m.promotion].toUpperCase();
    } else {
      // pièce : lettre + désambiguïsation minimale + ('x'?) + case d'arrivée.
      const letter = ROLE_TO_SAN[piece.role];
      const disamb = this._disambiguate(m, piece, allMoves);
      core = letter + disamb + (isCapture ? 'x' : '') + toAlg;
    }
    return core + this._checkSuffix(m);
  }

  // Désambiguïsation SAN : si ≥1 autre coup d'une pièce de MÊME rôle/couleur va sur la même
  // case d'arrivée, préciser le fichier (sinon la rangée si les fichiers coïncident, sinon les
  // deux). Convention standard FIDE/chess.js.
  _disambiguate(m, piece, allMoves) {
    const rivals = allMoves.filter(
      (x) =>
        x !== m &&
        x.to === m.to &&
        x.from !== m.from &&
        this._sameKind(x.from, piece),
    );
    if (rivals.length === 0) return '';
    const fromFile = m.from & 7, fromRank = m.from >> 3;
    const sameFile = rivals.some((x) => (x.from & 7) === fromFile);
    const sameRank = rivals.some((x) => (x.from >> 3) === fromRank);
    if (!sameFile) return fileChar(m.from);          // fichier suffit
    if (!sameRank) return String(fromRank + 1);      // rangée suffit
    return fileChar(m.from) + (fromRank + 1);        // les deux
  }

  _sameKind(sq, piece) {
    const p = this._fb.position.board.get(sq);
    return !!p && p.role === piece.role && p.color === piece.color;
  }

  // Suffixe +/# : joue le coup, lit échec/mat, annule. Mat='#', échec simple='+', sinon ''.
  _checkSuffix(m) {
    const fb = this._fb;
    fb._make(m);
    let suffix = '';
    try {
      if (fb.isCheckmate()) suffix = '#';
      else if (fb.position.isCheck()) suffix = '+';
    } finally {
      fb._unmake();
    }
    return suffix;
  }
}
