// board-render.mjs — FEN → 64 cellules {piece, dark} pour l'affichage (Story C.3). Pur (testable).
// `piece` = code à 2 lettres couleur+type ('wK','bQ',… ) servant à charger pieces/<code>.svg
// (jeu cburnett vendorisé localement, identique à lichess par défaut) ; '' = case vide.
// Ordre a8..h1 (rang 8 en haut). a1 = case sombre.

const TYPE = { p: 'P', n: 'N', b: 'B', r: 'R', q: 'Q', k: 'K' };

/**
 * @param {string} fen
 * @returns {Array<{piece:string, dark:boolean}>} 64 cellules ; piece='' = case vide.
 */
export function fenToCells(fen) {
  const board = String(fen).trim().split(/\s+/)[0]; // 1er champ = position
  const ranks = board.split('/');
  if (ranks.length !== 8) throw new Error(`FEN invalide : ${ranks.length} rangées`);
  const cells = [];
  for (let r = 0; r < 8; r++) {
    const rankIndex = 7 - r; // r=0 → rang 8 → rankIndex 7 (a1 sombre : (0+0)%2===0)
    let file = 0;
    for (const ch of ranks[r]) {
      if (ch >= '1' && ch <= '8') {
        const n = ch.charCodeAt(0) - 48;
        for (let k = 0; k < n; k++) { cells.push({ piece: '', dark: (file + rankIndex) % 2 === 0 }); file++; }
      } else {
        const type = TYPE[ch.toLowerCase()];
        if (!type) throw new Error(`FEN : pièce inconnue '${ch}'`);
        const color = ch === ch.toUpperCase() ? 'w' : 'b'; // majuscule = blanc
        cells.push({ piece: color + type, dark: (file + rankIndex) % 2 === 0 }); file++;
      }
    }
    if (file !== 8) throw new Error(`FEN : rangée ${8 - r} de largeur ${file} (≠ 8)`);
  }
  return cells;
}

/**
 * Case algébrique ('e2') → {file 0-7, rank 0-7}. Pour positionner la flèche du dernier coup.
 * @returns {{file:number, rank:number}|null}
 */
export function algToFileRank(sq) {
  if (typeof sq !== 'string' || sq.length < 2) return null;
  const file = sq.charCodeAt(0) - 97; // 'a'→0
  const rank = sq.charCodeAt(1) - 49; // '1'→0
  if (file < 0 || file > 7 || rank < 0 || rank > 7) return null;
  return { file, rank };
}
