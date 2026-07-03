// move-encoding.mjs
// Port fidèle de org.nanozero.nn.MoveEncoding (nanozero-nn).
// Encodage AlphaZero : policyIndex = fromSquare * 73 + planeIndex, dans [0, 4672).
//   plans 0-55  : queen-style (8 directions × 7 distances)
//   plans 56-63 : cavalier (8 deltas figés)
//   plans 64-72 : sous-promotion (3 directions fichier × 3 pièces N/B/R)
// Perspective P1 (côté au trait) : flip vertical (sq ^= 56) si BLACK.
// ⚠️ Toute divergence vs Java désaligne la policy → données empoisonnées.

export const POLICY_PLANES = 73;
export const POLICY_INDICES = 4672;

// Couleurs (Color.java)
export const WHITE = 0;
export const BLACK = 1;

// Types de coup — valeurs internes JS (l'index policy n'en dépend QUE via PROMOTION).
// ⚠️ NON garanties identiques aux int de MoveType.java ; à n'utiliser qu'en interne JS.
export const NORMAL = 0;
export const PROMOTION = 1;
export const CASTLING = 2;
export const EN_PASSANT = 3;

// Promotions (figé) : 0=Cavalier, 1=Fou, 2=Tour, 3=Dame
export const PROMO_KNIGHT = 0;
export const PROMO_BISHOP = 1;
export const PROMO_ROOK = 2;
export const PROMO_QUEEN = 3;

// Types de pièce (PieceType.java) — pour la détermination du type au decode
export const PAWN = 0, KNIGHT = 1, BISHOP = 2, ROOK = 3, QUEEN = 4, KING = 5, NONE = -1;

// [dRank, dFile] — N, NE, E, SE, S, SW, W, NW (ordre FIGÉ MoveEncoding.java:58-63)
const QUEEN_DELTAS = [
  [+1, 0], [+1, +1], [0, +1], [-1, +1],
  [-1, 0], [-1, -1], [0, -1], [+1, -1],
];
// Cavalier — ordre normatif FIGÉ (MoveEncoding.java:66-69)
const KNIGHT_DELTAS = [
  [+2, +1], [+1, +2], [-1, +2], [-2, +1],
  [-2, -1], [-1, -2], [+1, -2], [+2, -1],
];
// Sous-promo : direction fichier capture-gauche / avance / capture-droite
const UNDERPROMO_FILE_DELTAS = [-1, 0, +1];

// LUTs pré-calculées (comme le static init Java)
const KNIGHT_DELTA_LUT = new Int8Array(25).fill(-1); // index (dRank+2)*5+(dFile+2)
for (let i = 0; i < 8; i++) {
  const [dr, df] = KNIGHT_DELTAS[i];
  KNIGHT_DELTA_LUT[(dr + 2) * 5 + (df + 2)] = i;
}
const QUEEN_DIR_LUT = new Int8Array(9).fill(-1); // index (signR+1)*3+(signF+1)
for (let d = 0; d < 8; d++) {
  const [dr, df] = QUEEN_DELTAS[d];
  QUEEN_DIR_LUT[(Math.sign(dr) + 1) * 3 + (Math.sign(df) + 1)] = d;
}

// Convention cases : LSB=a1 (sq 0) ... MSB=h8 (sq 63). file=sq&7, rank=sq>>3.
export const rankOf = (sq) => sq >> 3;
export const fileOf = (sq) => sq & 7;

function isKnightShape(dRank, dFile) {
  const aR = Math.abs(dRank), aF = Math.abs(dFile);
  return (aR === 1 && aF === 2) || (aR === 2 && aF === 1);
}

/**
 * Encode un coup en index policy [0, 4672).
 * @param {{from:number,to:number,type:number,promo:number}} move  cases 0-63, type/promo internes
 * @param {number} sideToMove  WHITE|BLACK
 * @returns {number} policyIndex
 */
export function encode(move, sideToMove) {
  let from = move.from, to = move.to;
  const type = move.type, promo = move.promo;
  if (sideToMove === BLACK) { from ^= 56; to ^= 56; }

  const dRank = rankOf(to) - rankOf(from);
  const dFile = fileOf(to) - fileOf(from);

  let planeIndex;
  if (type === PROMOTION && promo !== PROMO_QUEEN) {
    // sous-promotion (plans 64-72)
    const dirIdx = dFile + 1; // -1,0,+1 -> 0,1,2
    if (dirIdx < 0 || dirIdx > 2) throw new Error(`underpromo dFile invalide: ${dFile}`);
    planeIndex = 64 + dirIdx * 3 + promo;
  } else if (isKnightShape(dRank, dFile)) {
    // cavalier (plans 56-63)
    const kIdx = KNIGHT_DELTA_LUT[(dRank + 2) * 5 + (dFile + 2)];
    planeIndex = 56 + kIdx;
  } else {
    // queen-style (plans 0-55) : inclut promo-dame, EP, roque, normal
    const dir = QUEEN_DIR_LUT[(Math.sign(dRank) + 1) * 3 + (Math.sign(dFile) + 1)];
    if (dir < 0) throw new Error(`direction queen invalide: dRank=${dRank} dFile=${dFile}`);
    const dist = Math.max(Math.abs(dRank), Math.abs(dFile));
    if (dist < 1 || dist > 7) throw new Error(`distance queen invalide: ${dist}`);
    planeIndex = dir * 7 + (dist - 1);
  }
  return from * POLICY_PLANES + planeIndex;
}

/**
 * Décode un index policy en coup. Pour les coups queen-style, le type (roque/EP/
 * promo-dame/normal) dépend de la position : fournir `pieceTypeAt(sq)->ptype` et
 * `epSquare`. Sans eux, les coups queen-style sont rendus comme NORMAL (suffisant
 * pour le round-trip d'index, le type n'influence pas l'index).
 */
export function decode(policyIndex, sideToMove, pieceTypeAt = null, epSquare = -1) {
  if (policyIndex < 0 || policyIndex >= POLICY_INDICES) throw new Error(`index hors borne: ${policyIndex}`);
  const fromP1 = Math.floor(policyIndex / POLICY_PLANES);
  const plane = policyIndex % POLICY_PLANES;

  let dRankP1, dFileP1, type, promo;
  const isUnderpromo = plane >= 64;
  const isKnight = !isUnderpromo && plane >= 56;

  if (isUnderpromo) {
    const dirIdx = Math.floor((plane - 64) / 3);
    const pieceIdx = (plane - 64) % 3;
    dFileP1 = UNDERPROMO_FILE_DELTAS[dirIdx];
    dRankP1 = +1; // P1 avance toujours vers le haut
    type = PROMOTION; promo = pieceIdx; // 0=N,1=B,2=R
  } else if (isKnight) {
    const d = KNIGHT_DELTAS[plane - 56];
    dRankP1 = d[0]; dFileP1 = d[1];
    type = NORMAL; promo = 0;
  } else {
    const dir = Math.floor(plane / 7);
    const dist = (plane % 7) + 1;
    const d = QUEEN_DELTAS[dir];
    dRankP1 = d[0] * dist; dFileP1 = d[1] * dist;
    type = -1; promo = 0;
  }

  const toP1 = fromP1 + dRankP1 * 8 + dFileP1;
  let from, to;
  if (sideToMove === BLACK) { from = fromP1 ^ 56; to = toP1 ^ 56; }
  else { from = fromP1; to = toP1; }

  if (!isUnderpromo && !isKnight) {
    const mpt = pieceTypeAt ? pieceTypeAt(from) : NONE;
    const absFileDiff = Math.abs(fileOf(to) - fileOf(from));
    if (mpt === KING && absFileDiff === 2) { type = CASTLING; }
    else if (mpt === PAWN && to === epSquare) { type = EN_PASSANT; }
    else if (mpt === PAWN && (rankOf(to) === 0 || rankOf(to) === 7)) { type = PROMOTION; promo = PROMO_QUEEN; }
    else { type = NORMAL; promo = 0; }
  }
  return { from, to, type, promo };
}

// ---------------------------------------------------------------------------------------------
// MOVE COMPACT — int 15 bits : from(6) | to(6) | promo(2) | isPromo(1).
// Stocké dans les `children` du MCTS À LA PLACE de l'objet move FastBoard (économise l'objet + ses
// 3 getters san/cjs/uci × ~90k coups/arbre × pool → corrige l'OOM renderer du mode Max).
// LOSSLESS, prouvé par move-pack-parity.selftest : pour encode() (qui ne lit que from/to/type/promo,
// et type ne compte QUE pour PROMOTION) ET pour rejouer (FastBoard._make ne lit que from/to + le
// rôle de promotion ; roque/EP sont auto-détectés depuis le board, donc `type` est superflu au make).
export const PACKED_PROMO_ROLE = ['knight', 'bishop', 'rook', 'queen']; // promo 0..3 -> rôle chessops

/** Objet move FastBoard {from,to,type,promo,promotion?} -> int compact. */
export function packMove(m) {
  const isP = (m.promotion != null || m.type === PROMOTION) ? 1 : 0;
  const promo = isP ? (m.promo & 3) : 0;
  return (m.from & 63) | ((m.to & 63) << 6) | (promo << 12) | (isP << 14);
}

/** int compact -> {from,to,type,promo} (suffisant pour encode()). */
export function unpackMove(p) {
  const isP = (p >> 14) & 1;
  return { from: p & 63, to: (p >> 6) & 63, type: isP ? PROMOTION : NORMAL, promo: (p >> 12) & 3 };
}

/** Index policy depuis un coup compact — équivaut bit-à-bit à encode(objet, side). */
export function encodePacked(p, side) { return encode(unpackMove(p), side); }

/** int compact -> objet REJOUABLE par GameState.applyMove. `type` défini → joué tel quel (pas de
 * re-résolution coûteuse) ; `promotion` = rôle chessops (roque/EP auto-détectés par _make). */
export function unpackForApply(p) {
  const isP = (p >> 14) & 1;
  return {
    from: p & 63, to: (p >> 6) & 63, type: isP ? PROMOTION : NORMAL,
    promo: (p >> 12) & 3, promotion: isP ? PACKED_PROMO_ROLE[(p >> 12) & 3] : undefined,
  };
}
