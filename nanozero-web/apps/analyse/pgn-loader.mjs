// pgn-loader.mjs — charge une partie (PGN) ou une position (FEN) en SÉQUENCE de positions
// navigables, via chessops (parsing robuste : headers, commentaires, variantes ignorées → mainline).
// Retourne { positions, headers } où positions = [{ fen, san, from, to }] :
//   - index 0 = position de départ (san/from/to = null) ;
//   - from/to = cases algébriques du coup menant à cette position (pour la flèche du dernier coup).
// Tolère la notation FRANÇAISE (R/D/T/F/C) et les figurines unicode : chessops n'accepte que la
// notation anglaise (K/Q/R/B/N), or l'utilisateur colle souvent une partie en français. ⚠️ La
// conversion doit se faire sur le TEXTE BRUT : parsePgn supprime lui-même toute lettre de pièce
// inconnue (« Cf3 » → san « f3 ») avant qu'on puisse la lire.

import { parsePgn } from 'chessops/pgn';
import { parseSan } from 'chessops/san';
import { parseFen, makeFen } from 'chessops/fen';
import { makeSquare } from 'chessops/util';
import { Chess } from 'chessops/chess';

// Position de départ d'une partie : header FEN si présent (parties à position custom), sinon startpos.
function startPos(headers) {
  const fen = headers && headers.get && headers.get('FEN');
  if (fen) return Chess.fromSetup(parseFen(fen).unwrap()).unwrap();
  return Chess.default();
}

// Normalisations du TEXTE PGN, sûres pour un PGN déjà anglais (appliquées en passe 1) :
// figurines unicode → lettres, et roque avec des zéros → lettres O.
const UNI = { '♔': 'K', '♚': 'K', '♕': 'Q', '♛': 'Q', '♖': 'R', '♜': 'R', '♗': 'B', '♝': 'B', '♘': 'N', '♞': 'N' };
function baseNorm(text) {
  return text.replace(/0-0-0/g, 'O-O-O').replace(/0-0/g, 'O-O').replace(/[♔♚♕♛♖♜♗♝♘♞]/g, (c) => UNI[c]);
}
// Notation française → anglaise (passe 2). Ordre critique : R(Roi)→K AVANT T(Tour)→R, sinon les R
// issus de T seraient re-convertis. D/T/F/C sont absentes de l'anglais (conversion sûre) ; seul R est
// ambigu (Roi FR vs Tour EN) — d'où la double passe : on ne tente le français QUE si l'anglais tronque,
// et on garde l'interprétation qui parse le PLUS de coups (un PGN anglais n'est donc jamais corrompu).
// ⚠️ Mange aussi les R/D/T/F/C des en-têtes/commentaires — c'est pourquoi on relit headers + FEN sur le
// parse PROPRE, et qu'on n'utilise ce texte QUE pour extraire les coups.
function frenchNorm(text) {
  return baseNorm(text)
    .replace(/R/g, 'K').replace(/T/g, 'R').replace(/D/g, 'Q').replace(/F/g, 'B').replace(/C/g, 'N');
}

function mainlineSans(game) {
  const sans = [];
  for (let node = game.moves; node.children.length; ) { node = node.children[0]; sans.push(node.data.san); }
  return sans;
}

// UCI STANDARD du coup (pour rejouer l'historique sur le core GameState). Piège : chessops encode le
// roque roi→TOUR (from=roi, to=tour de même couleur) → on normalise en roi→case g/c (UCI standard).
// Robuste aux deux conventions : si chessops donnait déjà roi→case, la détection roque échoue et on
// retombe sur from+to (= déjà standard).
const PROMO = { queen: 'q', rook: 'r', bishop: 'b', knight: 'n' };
function uciOf(pos, move) {
  if (!('from' in move)) return null;
  const fromSq = makeSquare(move.from), toSq = makeSquare(move.to);
  const piece = pos.board.get(move.from), target = pos.board.get(move.to);
  if (piece && piece.role === 'king' && target && target.role === 'rook' && target.color === piece.color) {
    return fromSq + (move.to > move.from ? 'g' : 'c') + fromSq[1]; // roque → roi vers case g/c (ex. e1c1)
  }
  return fromSq + toSq + (move.promotion ? PROMO[move.promotion] : '');
}

/** PGN (string) → séquence de positions de la ligne principale. Lève si illisible. */
export function positionsFromPgn(pgnText) {
  const text = pgnText || '';
  const clean = parsePgn(text);
  if (!clean.length || !clean[0].moves.children.length && !(clean[0].headers && clean[0].headers.get('FEN'))) {
    throw new Error('Aucun coup trouvé dans ce PGN.');
  }
  const headers = clean[0].headers;                 // en-têtes (et FEN) lus sur le parse NON normalisé
  const nodeCount = mainlineSans(clean[0]).length;  // nb de coups dans le PGN (pour détecter une troncature)
  let basePos;
  try { basePos = startPos(headers); } catch { throw new Error('Position de départ du PGN invalide.'); }

  // Rejoue les coups d'un texte normalisé depuis basePos (position de départ propre).
  const build = (normText) => {
    const g = parsePgn(normText)[0];
    const pos = basePos.clone();
    const out = [{ fen: makeFen(pos.toSetup()), san: null, from: null, to: null, uci: null }];
    if (!g) return out;
    for (const san of mainlineSans(g)) {
      const move = parseSan(pos, san);
      if (!move) break; // coup illégal/ambigu → on s'arrête proprement sur ce qui est valide
      const from = ('from' in move) ? makeSquare(move.from) : null;
      const to = ('to' in move) ? makeSquare(move.to) : null;
      const uci = uciOf(pos, move); // AVANT pos.play : pos = position pré-coup
      pos.play(move);
      out.push({ fen: makeFen(pos.toSetup()), san, from, to, uci });
    }
    return out;
  };

  // Passe 1 = anglais (+ unicode/zéros). Si elle tronque, passe 2 = français ; on garde la plus complète.
  let out = build(baseNorm(text));
  if (out.length - 1 < nodeCount) {
    const fr = build(frenchNorm(text));
    if (fr.length > out.length) out = fr;
  }
  // Le PGN contenait des coups mais AUCUN n'a pu être interprété : ne pas charger silencieusement
  // une « partie » d'une seule position (graphe masqué, liste vide).
  if (out.length === 1 && nodeCount) {
    throw new Error('PGN illisible : le 1er coup n’a pas pu être interprété.');
  }
  return { positions: out, headers };
}

/** FEN (string) → séquence d'une seule position. Lève si la FEN est invalide/illégale. */
export function positionFromFen(fenText) {
  let pos;
  try {
    const setup = parseFen((fenText || '').trim()).unwrap(); // lève si FEN malformée
    pos = Chess.fromSetup(setup).unwrap();                   // lève si position illégale
  } catch {
    throw new Error('FEN invalide ou position illégale.');
  }
  return { positions: [{ fen: makeFen(pos.toSetup()), san: null, from: null, to: null, uci: null }], headers: new Map() };
}
