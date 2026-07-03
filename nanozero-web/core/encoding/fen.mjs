// fen.mjs — parseur FEN -> état pour le plane-encoder (1 position, sans historique).
// Sert aux tests et de référence ; à l'exécution, l'adaptateur chess.js produira le
// même format (avec historique réel).

import { WHITE, BLACK } from './move-encoding.mjs';

// type index : P=0,N=1,B=2,R=3,Q=4,K=5 (ordre des plans 0-5)
const TYPE = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };

/**
 * @param {string} fen  FEN complet
 * @returns {{sideToMove,castling,epSquare,halfmoveClock,fullmoveNumber,history}}
 */
export function parseFen(fen) {
  const [board, side, cast, ep, half, full] = fen.trim().split(/\s+/);
  const white = [0n, 0n, 0n, 0n, 0n, 0n];
  const black = [0n, 0n, 0n, 0n, 0n, 0n];
  const ranks = board.split('/'); // ranks[0] = rang 8

  for (let r = 0; r < 8; r++) {
    const rankIdx = 7 - r; // FEN rang 8 (r=0) -> rankIdx 7 ; rang 1 (r=7) -> rankIdx 0
    let file = 0;
    for (const ch of ranks[r]) {
      if (ch >= '1' && ch <= '8') { file += Number(ch); continue; }
      const sq = rankIdx * 8 + file;
      const t = TYPE[ch.toLowerCase()];
      const bit = 1n << BigInt(sq);
      if (ch === ch.toUpperCase()) white[t] |= bit;
      else black[t] |= bit;
      file++;
    }
  }

  const sideToMove = side === 'w' ? WHITE : BLACK;
  const castling = {
    wk: cast.includes('K'), wq: cast.includes('Q'),
    bk: cast.includes('k'), bq: cast.includes('q'),
  };
  let epSquare = -1;
  if (ep && ep !== '-') {
    epSquare = (Number(ep[1]) - 1) * 8 + (ep.charCodeAt(0) - 97);
  }
  return {
    sideToMove, castling, epSquare,
    halfmoveClock: half ? Number(half) : 0,
    fullmoveNumber: full ? Number(full) : 1,
    history: [{ white, black, rep1: false, rep2: false }],
  };
}
