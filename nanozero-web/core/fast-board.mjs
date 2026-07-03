// fast-board.mjs — Story 1 : skeleton FastBoard + make/unmake INCRÉMENTAL.
//
// Wrapper MINCE par-dessus chessops `Position.play()` (PAS de clone() de la position :
// le clone alloue board+castles+SquareSets à chaque descente MCTS → pression GC qui tue
// le gain board-bound, cf. SPEC §Décision 1). `_make` sauve un undo-record minimal AVANT
// play ; `_unmake` restaure les cases touchées + epSquare/halfmoves/fullmoves/turn/castles
// sans réallouer le board.
//
// PÉRIMÈTRE Story 1 (strict) : constructeur FEN, _make/_unmake, sideToMove(), legalMoves()
// MINIMAL (format interne {from,to,promotion} suffisant pour le perft). PAS de Zobrist
// (Story 2), PAS de toEncoderState (Story 3), PAS du format NanoZero {type,promo} (Story 4).
//
// Pièges adressés (silent-poison, vérifiés contre la lecture de chessops Position.play) :
//   - roque déplace roi ET tour → unmake remet les deux (king from kingCastlesTo, rook from
//     rookCastlesTo) sur leurs cases d'origine ;
//   - promotion → la pièce postée sur `to` est une dame/etc., mais l'original sur `from`
//     était un pion → on restaure le `mover` capturé (rôle pion) tel quel ;
//   - capture en passant → la pièce capturée n'est PAS sur `to` mais sur `to ∓ 8` ;
//   - droits de roque après prise de tour (et discardRook/discardColor) → `castles` est
//     MUTABLE et muté par play() ; on en sauve un clone léger et on le réassigne en unmake ;
//   - epSquare/halfmoves/fullmoves/turn → réécrits inconditionnellement par play(), restaurés
//     depuis l'undo-record.

import { Chess } from 'chessops/chess';
import { parseFen, makeFen } from 'chessops/fen';
import { kingCastlesTo, rookCastlesTo, opposite } from 'chessops/util';
import {
  WHITE,
  BLACK,
  NORMAL,
  PROMOTION,
  CASTLING,
  EN_PASSANT,
} from './encoding/move-encoding.mjs';

// Story 4 — format NanoZero {from,to,type,promo} : promo figé 0=N 1=B 2=R 3=Q
// (= PROMO de game-state.mjs ; l'ORDRE pilote l'index sous-promo de move-encoding).
const PROMO_BY_ROLE = { knight: 0, bishop: 1, rook: 2, queen: 3 };
// Ordre d'émission des 4 enfants de promotion (N/B/R/Q). L'ensemble seul compte
// pour la parité (cross-validate compare par uci) ; l'ordre est purement cosmétique.
const PROMO_ROLES_ORDERED = ['knight', 'bishop', 'rook', 'queen'];

// ===========================================================================
// Story 2 — Zobrist {lo:int32, hi:int32} incrémental + clé de répétition
// convention-Java (ep TOUJOURS après double-poussée, indépendant de
// position.epSquare qui filtre capturable-only, cf. SPEC Décision 2 + piège
// « zobrist ep absent de la clé »).
//
// Pourquoi {lo,hi} et pas BigInt/Number : BigInt est lent en hot-path MCTS ;
// un Number 64-bit perd les bits > 2^53. Deux int32 = arithmétique exacte 64-bit
// via XOR (associatif/commutatif, et SON PROPRE INVERSE → unmake trivial).
//
// Layout des cases : LSB=a1 (convention SquareSet chessops). file = sq & 7,
// rank = sq >> 3. L'ep dans la clé porte le FILE (0-7) du pion qui vient de
// faire la double-poussée — pas la case ep entière (la convention Java/_key()
// de game-state.mjs met l'algébrique « e3 » dont seul le file discrimine deux
// positions par ailleurs identiques ; un slot EP[file] suffit et suit la
// sémantique de l'encodeur de plans rép12/13).
// ===========================================================================

// Index pièce-couleur 0-11 : pawn..king (blanc 0-5) puis pawn..king (noir 6-11).
const ROLE_IDX = { pawn: 0, knight: 1, bishop: 2, rook: 3, queen: 4, king: 5 };
const pcIndex = (color, role) => (color === 'white' ? 0 : 6) + ROLE_IDX[role];

// Cases-coins porteuses des droits de roque (LSB=a1) — pour l'index roque 4-bit.
const SQ_A1 = 0,
  SQ_H1 = 7,
  SQ_A8 = 56,
  SQ_H8 = 63;

// --- Tables Zobrist, initialisées UNE fois au chargement du module ---
// PIECE[64][12] = {lo,hi} ; SIDE ; CASTLING[16] ; EP[9] (file 0-7 + slot « aucun »=8).
// Math.random()|0 donne un int32 signé uniforme (les 32 bits sont aléatoires).
const rnd32 = () => Math.random() * 0x100000000 | 0;

const ZP = []; // ZP[sq][pc] = {lo,hi}
for (let sq = 0; sq < 64; sq++) {
  const row = [];
  for (let pc = 0; pc < 12; pc++) row.push({ lo: rnd32(), hi: rnd32() });
  ZP.push(row);
}
const ZSIDE = { lo: rnd32(), hi: rnd32() };
const ZCASTLE = [];
for (let i = 0; i < 16; i++) ZCASTLE.push({ lo: rnd32(), hi: rnd32() });
const ZEP = []; // 0-7 = file, 8 = pas d'ep
for (let i = 0; i < 9; i++) ZEP.push({ lo: rnd32(), hi: rnd32() });

// Index roque 4-bit (0-15) depuis le SquareSet castlingRights : bit0=a1(Q blanc),
// bit1=h1(K blanc), bit2=a8(q noir), bit3=h8(k noir). play() mute castlingRights
// (discardRook/discardColor) → l'index change automatiquement après une prise de
// tour ou un mouvement de roi, donc la clé reste correcte sans cas particulier.
function castleIndex(castlingRights) {
  let idx = 0;
  if (castlingRights.has(SQ_A1)) idx |= 1;
  if (castlingRights.has(SQ_H1)) idx |= 2;
  if (castlingRights.has(SQ_A8)) idx |= 4;
  if (castlingRights.has(SQ_H8)) idx |= 8;
  return idx;
}

// Clone LÉGER de l'objet Castles : castlingRights/path sont des SquareSet IMMUABLES
// (réassignés, jamais mutés en place) → on partage les refs ; seule la table `rook`
// (4 cases | undefined) est dupliquée car play() écrit dedans (discardRook/discardColor).
function cloneCastles(c) {
  return {
    castlingRights: c.castlingRights,
    rook: {
      white: { a: c.rook.white.a, h: c.rook.white.h },
      black: { a: c.rook.black.a, h: c.rook.black.h },
    },
    path: c.path, // jamais muté par play()
  };
}

// Restaure (en place) les 3 champs de pos.castles depuis un clone léger, SANS réallouer
// l'objet Castles (préserve son prototype/méthodes discardRook…).
function restoreCastles(c, saved) {
  c.castlingRights = saved.castlingRights;
  c.rook.white.a = saved.rook.white.a;
  c.rook.white.h = saved.rook.white.h;
  c.rook.black.a = saved.rook.black.a;
  c.rook.black.h = saved.rook.black.h;
  // path partagé (immuable) — saved.path === c.path, rien à faire.
}

// ===========================================================================
// Story 3 — bitboards LSB=a1 (ABSOLU, aucun pré-flip) depuis SquareSet {lo,hi}.
//
// chessops SquareSet = {lo:int32, hi:int32} couvrant les cases 0-63 (LSB=a1) :
// lo = cases 0-31, hi = cases 32-63. Le bit i de lo = case i ; le bit j de hi =
// case 32+j. On reconstruit le BigInt 64-bit attendu par l'encodeur (qui itère
// les bits set, sq = rank*8+file, LSB=a1) :
//     bb = BigInt(lo>>>0) | (BigInt(hi>>>0) << 32n)
// `>>> 0` force l'interprétation NON SIGNÉE des int32 (sinon le bit de signe
// pollue les 32 bits hauts du BigInt). AUCUN flip vertical ici : reverseBytes()
// vit dans plane-encoding.mjs et s'applique seulement si side==BLACK.
// ===========================================================================
function ssToBigInt(ss) {
  return BigInt(ss.lo >>> 0) | (BigInt(ss.hi >>> 0) << 32n);
}

// 6 rôles dans l'ordre pawn..king (= TYPE de game-state.mjs : p,n,b,r,q,k).
const ROLES_ORDERED = ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king'];

// Board chessops → {white:bigint[6], black:bigint[6]} (LSB=a1, absolu).
// pieces(color, role) = board[color].intersect(board[role]) (cf. board.js:114).
function boardToBitboards(board) {
  const white = new Array(6);
  const black = new Array(6);
  for (let r = 0; r < 6; r++) {
    const roleSS = board[ROLES_ORDERED[r]];
    white[r] = ssToBigInt(board.white.intersect(roleSS));
    black[r] = ssToBigInt(board.black.intersect(roleSS));
  }
  return { white, black };
}

export class FastBoard {
  constructor(fen) {
    const setup = parseFen(fen).unwrap();
    this.position = Chess.fromSetup(setup).unwrap();
    // Pile d'undo-records (réutilisée tout le long de la descente/remontée MCTS).
    this._undo = [];

    // --- ep convention-Java : file 0-7 (ou -1) parsé DIRECTEMENT du FEN ---
    // PIÈGE CENTRAL : on NE lit PAS pos.epSquare (validEpSquare l'a filtré à
    // capturable-only à la construction, cf. chess.js:130 → undefined sur f3
    // isolé). La convention Java garde l'ep dès qu'il est présent dans le FEN.
    const epField = (fen.trim().split(/\s+/)[3] || '-');
    this._epFile = epField === '-' ? -1 : (epField.charCodeAt(0) - 97); // 'a'..'h' → 0..7

    // --- Zobrist {lo,hi} calculé from-scratch ---
    this._zobrist = this._computeZobrist();

    // ----- Story 3 : pile de snapshots (miroir de GameState.snapshots) -----
    // snapshots[0] = position initiale ... snapshots[last] = position COURANTE.
    // Chaque entrée : {repKey:{lo,hi}, white:bigint[6], black:bigint[6]}.
    // Le repKey EST le zobrist convention-Java (pièces+trait+roque+ep-Java) — c'est
    // la clé de comparaison pour countRep (équivalente au _key() string de GameState).
    // Le constructeur pousse la position initiale (comme GameState appelle _snapshot()
    // dans son constructeur) → le snapshot COURANT se compte lui-même dans countRep.
    this.snapshots = [];
    this._snapshot();
  }

  // Capture l'état COURANT (clé + bitboards) au sommet de la pile.
  _snapshot() {
    const { white, black } = boardToBitboards(this.position.board);
    this.snapshots.push({
      repKey: { lo: this._zobrist.lo, hi: this._zobrist.hi },
      white,
      black,
    });
  }

  // Recalcul COMPLET du hash depuis l'état courant (oracle pour Gate 2, et base
  // initiale). XOR de : chaque pièce sur sa case, le trait, l'index roque, le
  // slot ep convention-Java.
  _computeZobrist() {
    const pos = this.position;
    let lo = 0,
      hi = 0;
    for (const [sq, piece] of pos.board) {
      const z = ZP[sq][pcIndex(piece.color, piece.role)];
      lo ^= z.lo;
      hi ^= z.hi;
    }
    if (pos.turn === 'black') {
      lo ^= ZSIDE.lo;
      hi ^= ZSIDE.hi;
    }
    const zc = ZCASTLE[castleIndex(pos.castles.castlingRights)];
    lo ^= zc.lo;
    hi ^= zc.hi;
    const ze = ZEP[this._epFile < 0 ? 8 : this._epFile];
    lo ^= ze.lo;
    hi ^= ze.hi;
    return { lo: lo | 0, hi: hi | 0 };
  }

  // _repKey : LE zobrist courant EST la clé de répétition (pièces+trait+roque+
  // ep convention-Java). Retour {lo,hi} (copie défensive : l'appelant ne doit
  // pas muter l'état interne).
  _repKey() {
    return { lo: this._zobrist.lo, hi: this._zobrist.hi };
  }

  sideToMove() {
    return this.position.turn === 'white' ? WHITE : BLACK;
  }

  // ===========================================================================
  // Story 3 — toEncoderState() BIT-IDENTIQUE à GameState.toEncoderState().
  //
  // Reproduit la sémantique exacte de game-state.mjs (snapshots, fenêtre
  // halfmove, snapshot-courant-se-compte-lui-même), mais :
  //   - les bitboards viennent des snapshots déjà calculés (LSB=a1 absolu) ;
  //   - countRep compare le repKey {lo,hi} (zobrist convention-Java) au lieu de
  //     la string FEN+ep — même clé d'équivalence (pièces+trait+roque+ep-Java) ;
  //   - castling lu depuis castlingRights (coins), pas depuis le champ FEN.
  //
  // Le flip vertical N'EST PAS fait ici : plane-encoding.mjs l'applique si
  // side==BLACK. On retourne donc les bitboards en absolu.
  // ===========================================================================
  toEncoderState() {
    const pos = this.position;
    const cr = pos.castles.castlingRights;
    const hm = pos.halfmoves;
    const snaps = this.snapshots;
    const histSize = snaps.length - 1; // positions PASSÉES (hors courante)
    const limit = Math.max(0, histSize - hm); // fenêtre bornée par le compteur 50-coups

    // countRepetitions(Java) : occurrences de `key` dans les positions [limit..histSize-1].
    // Pour t≥1, le snapshot lui-même est dans cette plage → il se compte (rep≥1).
    const eqKey = (a, b) => a.lo === b.lo && a.hi === b.hi;
    const countRep = (key) => {
      let c = 0;
      for (let i = histSize - 1; i >= limit; i--) if (eqKey(snaps[i].repKey, key)) c++;
      return c;
    };

    const history = [];
    for (let t = 0; t < 8; t++) {
      const idx = histSize - t; // t=0 → position courante (idx=histSize)
      if (idx < 0) break; // profondeur insuffisante → timestamps suivants à 0 (encodeur)
      const s = snaps[idx];
      const rep = countRep(s.repKey);
      history.push({ white: s.white, black: s.black, rep1: rep >= 1, rep2: rep >= 2 });
    }

    return {
      sideToMove: this.sideToMove(),
      castling: {
        wk: cr.has(SQ_H1), // blanc kingside  → coin h1
        wq: cr.has(SQ_A1), // blanc queenside → coin a1
        bk: cr.has(SQ_H8), // noir  kingside  → coin h8
        bq: cr.has(SQ_A8), // noir  queenside → coin a8
      },
      fullmoveNumber: pos.fullmoves,
      halfmoveClock: hm,
      history,
    };
  }

  // ===========================================================================
  // Story 4 — coups légaux au FORMAT NanoZero {from,to,type,promo} (+ `promotion`
  // role chessops conservé pour rester JOUABLE par _make/pos.play()).
  //   type ∈ NORMAL | CASTLING | EN_PASSANT | PROMOTION (move-encoding.mjs)
  //   promo ∈ 0=N 1=B 2=R 3=Q
  // L'index policy (move-encoding.encode) ne dépend du `type` que pour la
  // SOUS-PROMOTION (plans 64-72) ; roque/ep/promo-dame sont classés par géométrie.
  // MAIS on émet le `type` complet car expandNode/encode reçoivent ces objets en
  // Story 5 et GameState.legalMoves() expose déjà ce contrat → parité d'ensemble.
  //
  // CLASSIFICATION (les 3 pièges Story 4) :
  //   - CASTLING : roi & |fileOf(to)-fileOf(from)| === 2 (post-normalisation roque,
  //     to = case du roi g1/c1 → fileDiff=2) ;
  //   - EN_PASSANT : pion & to === case-ep CONVENTION-JAVA (reconstruite depuis
  //     this._epFile, PAS pos.epSquare qui est filtré capturable-only) ;
  //   - PROMOTION : pion & rang d'arrivée 7(blanc)/0(noir) → 4 ENFANTS (N/B/R/Q) ;
  //   - NORMAL : tout le reste.
  //
  // ⚠️ NORMALISATION ROQUE (requise dès Story 3 pour la parité UCI/index policy) :
  // chessops `allDests()` émet le roque en convention « roi-prend-sa-tour » (e1→h1/a1,
  // le `to` étant la case de la TOUR). Java/MoveEncoding attend la convention standard
  // « roi-vers-sa-case-d'arrivée » (e1→g1/c1). On remplace donc le `to` du roque par
  // kingCastlesTo(color,side). C'est SANS RISQUE pour le moteur :
  //   - chessops play()/castlingSide acceptent les DEUX conventions (delta===2 OU
  //     board[turn].has(to)) → play() pose toujours roi+tour correctement ;
  //   - _make détecte le roque via |delta|===2 (g1 : delta=2) et lit rookFrom depuis
  //     castles.rook (indépendant du `to`) ;
  //   - _unmake (branche roque) n'utilise pas `to`.
  // L'index policy d'un roque devient ainsi un coup queen-style dist=2 (roi qui bouge),
  // exactement comme côté Java.
  legalMoves() {
    const pos = this.position;
    const out = [];
    const turn = pos.turn;
    const promoRank = turn === 'white' ? 7 : 0;
    // Case ep CONVENTION-JAVA reconstruite depuis _epFile (file 0-7) + le côté au trait :
    // ep blanc (un pion NOIR vient de pousser 7→5) → rang 6, idx 5 ; ep noir → rang 3, idx 2.
    // PIÈGE : on N'UTILISE PAS pos.epSquare (filtré capturable-only, undefined sur ep isolé).
    // -1 si pas d'ep → aucun `to` ne pourra matcher (cohérent : pas de coup ep).
    const epSquareJava =
      this._epFile < 0 ? -1 : this._epFile + 8 * (turn === 'white' ? 5 : 2);
    for (const [from, dests] of pos.allDests()) {
      const isPawn = pos.board.pawn.has(from);
      const isKing = pos.board.king.has(from);
      for (const to0 of dests) {
        // roque : `to0` est la case de la propre tour → normaliser vers la case du roi.
        let to = to0;
        let isCastle = false;
        if (isKing && pos.board[turn].has(to0)) {
          const side = to0 - from > 0 ? 'h' : 'a';
          to = kingCastlesTo(turn, side);
          isCastle = true;
        }
        // Détection roque robuste : roi & écart de file === 2 (post-normalisation).
        if (isKing && Math.abs((to & 7) - (from & 7)) === 2) isCastle = true;

        if (isPawn && (to >> 3) === promoRank) {
          // PROMOTION : 4 coups distincts N/B/R/Q (la policy/perft les comptent à part).
          // L'index sous-promo (move-encoding plans 64-72) dépend de `promo` 0/1/2 ;
          // la promo-dame (promo 3) reste queen-style → tout écart de promo casse l'index.
          for (const role of PROMO_ROLES_ORDERED) {
            out.push({ from, to, type: PROMOTION, promo: PROMO_BY_ROLE[role], promotion: role });
          }
        } else if (isCastle) {
          out.push({ from, to, type: CASTLING, promo: 0, promotion: undefined });
        } else if (isPawn && to === epSquareJava) {
          out.push({ from, to, type: EN_PASSANT, promo: 0, promotion: undefined });
        } else {
          out.push({ from, to, type: NORMAL, promo: 0, promotion: undefined });
        }
      }
    }
    return out;
  }

  // ===========================================================================
  // Story 5 — fin de partie / nulle : PARITÉ EXACTE avec GameState (chess.js).
  //
  // ⚠️ OVERRIDE de la SPEC Story 4 (qui restreignait à la sémantique terminale
  // locale et déléguait 3-fold/50-coups au MCTS). Pour un swap TRANSPARENT,
  // FastBoard DOIT reproduire le comportement EXACT de GameState/chess.js, sinon
  // la longueur des parties et la data self-play changeraient en silence (la
  // 3-fold est FRÉQUENTE dans ce moteur) :
  //   isGameOver  = checkmate | stalemate | insufficient-material | 3-fold | 50-coups
  //   isDraw      =             stalemate | insufficient-material | 3-fold | 50-coups
  // - 3-fold      : la position COURANTE est la 3e occurrence ⇒ son repKey apparaît
  //                 ≥ 2 fois dans les positions PASSÉES de la fenêtre → réutilise
  //                 EXACTEMENT le countRep de toEncoderState() (même clé, même
  //                 fenêtre bornée halfmove). C'est la « rep2 » de l'encodeur.
  // - 50-coups    : halfmove ≥ 100 (seuil chess.js, vérifié v1.4.0).
  //
  // On calcule le `ctx` chessops UNE fois et on le passe aux prédicats locaux
  // (évite le recalcul kingAttackers/checkers dans le hot path MCTS).
  // ===========================================================================

  // 3-fold convention-Java : la position COURANTE (sommet de la pile) est-elle la
  // 3e occurrence ? Compte les occurrences PASSÉES du repKey courant dans la fenêtre
  // [limit..histSize-1] (bornée par le compteur halfmove, comme toEncoderState).
  // ≥ 2 occurrences passées ⇒ 3e occurrence ⇒ 3-fold. RÉUTILISE la logique de
  // répétition existante (même clé zobrist convention-Java, même fenêtre).
  _isThreefold() {
    const snaps = this.snapshots;
    const histSize = snaps.length - 1; // positions PASSÉES (hors courante)
    const limit = Math.max(0, histSize - this.position.halfmoves);
    const cur = snaps[histSize].repKey; // repKey de la position COURANTE
    let c = 0;
    for (let i = histSize - 1; i >= limit; i--) {
      const k = snaps[i].repKey;
      if (k.lo === cur.lo && k.hi === cur.hi && ++c >= 2) return true;
    }
    return false;
  }

  // 50-coups : seuil chess.js (halfmove ≥ 100). Demi-coup réversible compté par chessops.
  _isFiftyMove() {
    return this.position.halfmoves >= 100;
  }

  isGameOver() {
    const ctx = this.position.ctx();
    // mat / pat = absence de coup légal (départagés par l'échec) → !hasDests
    if (!this.position.hasDests(ctx)) return true;
    return (
      this.position.isInsufficientMaterial() ||
      this._isThreefold() ||
      this._isFiftyMove()
    );
  }

  isCheckmate() {
    return this.position.isCheckmate();
  }

  isDraw() {
    const ctx = this.position.ctx();
    return (
      this.position.isStalemate(ctx) ||
      this.position.isInsufficientMaterial() ||
      this._isThreefold() ||
      this._isFiftyMove()
    );
  }

  // _make : sauvegarde l'undo-record minimal AVANT position.play(). On capture les pièces
  // présentes sur les cases qui peuvent changer (mover sur `from`, capture sur `to`, capture
  // ep sur `to∓8`, tour de roque) AVANT toute mutation, plus l'état scalaire.
  _make(move) {
    const pos = this.position;
    const board = pos.board;
    const turn = pos.turn; // côté qui joue (avant play)
    const { from, to } = move;

    // --- pièce qui bouge (rôle d'origine — pion avant promo) ---
    const mover = board.get(from); // {color, role, promoted} | undefined

    // --- détection roque : roi qui saute de 2 cases (ou capture sa propre tour, FRC-style) ---
    const isKing = mover && mover.role === 'king';
    let castling; // 'a' | 'h' | undefined
    if (isKing) {
      const delta = to - from;
      if (Math.abs(delta) === 2 || board[turn].has(to)) {
        castling = delta > 0 ? 'h' : 'a';
      }
    }

    // --- capture en passant : pion qui va sur epSquare alors que `to` est vide ---
    const isPawn = mover && mover.role === 'pawn';
    const isEp = isPawn && to === pos.epSquare;
    const epCapSq = isEp ? to + (turn === 'white' ? -8 : 8) : -1;

    // --- pièces capturées (à restaurer en unmake) ---
    // capture normale = pièce sur `to` (undefined si vide / roque) ; capture ep = pièce sur epCapSq.
    const capturedAtTo = castling ? undefined : board.get(to);
    const capturedEp = isEp ? board.get(epCapSq) : undefined;

    // Case d'origine de la tour de roque, lue AVANT play() (qui mute castles).
    const rookFrom = castling ? pos.castles.rook[turn][castling] : undefined;

    this._undo.push({
      from,
      to,
      mover, // rôle d'origine (pion si promo)
      castling, // 'a'|'h'|undefined
      castleColor: turn, // couleur du roque (pour kingCastlesTo/rookCastlesTo)
      rookFrom,
      capturedAtTo, // pièce reprise par capture normale (ou undefined)
      isEp,
      epCapSq, // case réelle de la pièce ep-capturée (ou -1)
      capturedEp, // pièce ep-capturée (ou undefined)
      epPrev: pos.epSquare, // epSquare AVANT play (remis à undefined par play)
      castlesPrev: cloneCastles(pos.castles),
      halfmovesPrev: pos.halfmoves,
      fullmovesPrev: pos.fullmoves,
      turnPrev: turn,
      // Story 2 : état Zobrist/ep convention-Java AVANT play, pour _unmake.
      epFilePrev: this._epFile,
      zobristPrev: { lo: this._zobrist.lo, hi: this._zobrist.hi },
    });

    // --- Zobrist incrémental : on capture l'index roque AVANT play (il sera muté). ---
    const castleIdxBefore = castleIndex(pos.castles.castlingRights);

    pos.play(move);

    // ----- mise à jour incrémentale du hash (XOR out l'ancien, XOR in le nouveau) -----
    // L'ordre des XOR est SANS IMPORTANCE (commutatif), mais on couvre EXACTEMENT
    // les mêmes contributions que _computeZobrist sinon Gate 2 (incrémental ==
    // from-scratch) casse. _unmake n'inverse rien : il restaure zobristPrev.
    let lo = this._zobrist.lo,
      hi = this._zobrist.hi;
    const xorPiece = (sq, color, role) => {
      const z = ZP[sq][pcIndex(color, role)];
      lo ^= z.lo;
      hi ^= z.hi;
    };

    if (castling) {
      // Roque : roi from→kingCastlesTo, tour rookFrom→rookCastlesTo. Aucune capture.
      const color = turn;
      const kTo = kingCastlesTo(color, castling);
      const rTo = rookCastlesTo(color, castling);
      xorPiece(from, color, 'king'); // out roi départ
      xorPiece(kTo, color, 'king'); // in roi arrivée
      xorPiece(rookFrom, color, 'rook'); // out tour départ (lue avant play)
      xorPiece(rTo, color, 'rook'); // in tour arrivée
    } else {
      // out : mover sur from (rôle d'ORIGINE — pion si promo).
      xorPiece(from, mover.color, mover.role);
      // out : capture éventuelle.
      if (isEp && capturedEp) xorPiece(epCapSq, capturedEp.color, capturedEp.role);
      else if (capturedAtTo) xorPiece(to, capturedAtTo.color, capturedAtTo.role);
      // in : pièce ARRIVÉE sur to (rôle promu si promotion, sinon rôle mover).
      const arrived = board.get(to); // lue APRÈS play → reflète la promotion
      xorPiece(to, arrived.color, arrived.role);
    }

    // trait : alterne à chaque coup.
    lo ^= ZSIDE.lo;
    hi ^= ZSIDE.hi;

    // index roque : XOR out l'ancien, XOR in le nouveau (play() a pu le réduire).
    const castleIdxAfter = castleIndex(pos.castles.castlingRights);
    if (castleIdxAfter !== castleIdxBefore) {
      const zb = ZCASTLE[castleIdxBefore];
      const za = ZCASTLE[castleIdxAfter];
      lo ^= zb.lo ^ za.lo;
      hi ^= zb.hi ^ za.hi;
    }

    // ep convention-Java : file de la nouvelle double-poussée (indépendant de
    // pos.epSquare). Détection auto-suffisante |from-to|===16 sur un pion.
    const wasDoublePush = mover.role === 'pawn' && Math.abs(from - to) === 16;
    const newEpFile = wasDoublePush ? (to & 7) : -1;
    if (newEpFile !== this._epFile) {
      const ze = ZEP[this._epFile < 0 ? 8 : this._epFile];
      const zn = ZEP[newEpFile < 0 ? 8 : newEpFile];
      lo ^= ze.lo ^ zn.lo;
      hi ^= ze.hi ^ zn.hi;
      this._epFile = newEpFile;
    }

    this._zobrist = { lo: lo | 0, hi: hi | 0 };

    // Story 3 : pousse le snapshot de la NOUVELLE position courante (après play +
    // zobrist mis à jour) — comme GameState appelle _snapshot() à la fin d'applyMove.
    this._snapshot();
  }

  // _unmake : pop l'undo-record et restaure l'état EXACT d'avant _make, sans réallocation.
  _unmake() {
    if (this._undo.length === 0) return; // garde de borne (D3) : rien à défaire → no-op (pas de pop sur pile vide)
    const u = this._undo.pop();
    const pos = this.position;
    const board = pos.board;

    if (u.castling) {
      // Roque : play() a posé le roi sur kingCastlesTo et la tour sur rookCastlesTo.
      // On vide ces deux cases puis on remet roi sur `from` et tour sur `rookFrom`.
      const color = u.castleColor;
      board.take(kingCastlesTo(color, u.castling));
      board.take(rookCastlesTo(color, u.castling));
      board.set(u.from, u.mover); // roi
      board.set(u.rookFrom, { color, role: 'rook', promoted: false });
    } else {
      // Coup normal / capture / promo / ep.
      board.take(u.to); // enlève la pièce arrivée (dame promue, ou mover déplacé)
      board.set(u.from, u.mover); // remet le mover d'origine (pion si promo) sur `from`
      if (u.isEp) {
        // ep : la case `to` est vide, la pièce capturée revient sur epCapSq.
        if (u.capturedEp) board.set(u.epCapSq, u.capturedEp);
      } else if (u.capturedAtTo) {
        // capture normale : remet la pièce reprise sur `to`.
        board.set(u.to, u.capturedAtTo);
      }
    }

    // État scalaire + castles (mutés inconditionnellement par play()).
    pos.epSquare = u.epPrev;
    pos.halfmoves = u.halfmovesPrev;
    pos.fullmoves = u.fullmovesPrev;
    pos.turn = u.turnPrev;
    restoreCastles(pos.castles, u.castlesPrev);

    // Story 2 : restauration directe du Zobrist + ep convention-Java d'avant _make.
    // Le XOR est son propre inverse → restaurer zobristPrev EST l'inverse exact du
    // delta appliqué en _make (Gate 1 self-inverse garanti, zéro recalcul fragile).
    this._epFile = u.epFilePrev;
    this._zobrist = { lo: u.zobristPrev.lo, hi: u.zobristPrev.hi };

    // Story 3 : dépile le snapshot de la position qu'on vient d'annuler — la pile
    // revient à l'état d'avant _make (comme GameState.undoMove appelle snapshots.pop()).
    this.snapshots.pop();
  }

  // Accès lecture pour les invariants/tests (board chessops sous-jacent).
  get board() {
    return this.position.board;
  }

  // Story 5 — FEN courant (pour l'UI/onPly + le smoke self-play). Délègue à makeFen(chessops)
  // sur la position COURANTE : l'ep émis est capturable-only (comme chess.js l'était), donc
  // parité de comportement avec l'ancien GameState.fen() côté affichage. PAS dans le hot path
  // MCTS (la clé de répétition/encodeur passe par le zobrist convention-Java, jamais par ce FEN).
  fen() {
    return makeFen(this.position.toSetup());
  }
}

// silence un import non utilisé selon les linters (opposite pourra servir Story 2/4).
void opposite;
