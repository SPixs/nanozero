// gen-openings.mjs — génère un livre d'ouvertures en COUPS (SAN, depuis startpos) pour fastchess.
// Nécessaire car TSCP ne gère pas le FEN (startpos+moves only) ET parce qu'à temp 0 les moteurs sont
// déterministes → il faut BEAUCOUP d'ouvertures distinctes pour décorréler les parties.
// Variété saine : à chaque demi-coup, choix aléatoire parmi le top-4 de Stockfish (depth faible).
// Usage : node bench/gen-openings.mjs [nbLignes=150] [plies=8] > bench/book.pgn
import { spawn } from 'node:child_process';
import { Chess } from 'chessops/chess';
import { makeSan } from 'chessops/san';
import { parseUci } from 'chessops/util';

const N = parseInt(process.argv[2] || '150', 10);
const PLIES = parseInt(process.argv[3] || '8', 10);
const SF = process.env.STOCKFISH || '/usr/games/stockfish';
const DEPTH = 6, MULTIPV = 4;

const sf = spawn(SF, [], { stdio: ['pipe', 'pipe', 'inherit'] });
let buf = '';
const waiters = [];
sf.stdout.on('data', (d) => {
  buf += d.toString();
  let i;
  while ((i = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, i); buf = buf.slice(i + 1);
    for (const w of waiters) w(line);
  }
});
const send = (s) => sf.stdin.write(s + '\n');
// collecte les coups multipv jusqu'à bestmove
function goMultiPV() {
  return new Promise((resolve) => {
    const pv = new Map(); // idx -> 1er coup uci
    const w = (line) => {
      const mm = line.match(/multipv (\d+).* pv (\S+)/);
      if (mm) pv.set(+mm[1], mm[2]);
      if (line.startsWith('bestmove')) {
        waiters.splice(waiters.indexOf(w), 1);
        resolve([...pv.entries()].sort((a, b) => a[0] - b[0]).map((e) => e[1]));
      }
    };
    waiters.push(w);
    send(`go depth ${DEPTH}`);
  });
}
const ready = () => new Promise((resolve) => {
  const w = (line) => { if (line.trim() === 'readyok') { waiters.splice(waiters.indexOf(w), 1); resolve(); } };
  waiters.push(w); send('isready');
});

// PRNG déterministe (reproductible) — pas de Math.random pour un livre stable.
let seed = 0x9e3779b9 >>> 0;
const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 0x100000000; };

send('uci'); send(`setoption name MultiPV value ${MULTIPV}`); await ready();

const seen = new Set();
let made = 0, attempts = 0;
while (made < N && attempts < N * 4) {
  attempts++;
  send('ucinewgame'); await ready();
  const pos = Chess.default();
  const uci = [], san = [];
  let ok = true;
  for (let p = 0; p < PLIES; p++) {
    send('position startpos moves ' + uci.join(' '));
    const moves = await goMultiPV();
    if (!moves.length) { ok = false; break; }
    const pick = moves[Math.floor(rnd() * moves.length)];
    const mv = parseUci(pick);
    if (!mv || !pos.isLegal(mv)) { ok = false; break; }
    san.push(makeSan(pos, mv));
    pos.play(mv);
    uci.push(pick);
  }
  if (!ok) continue;
  const key = uci.join(' ');
  if (seen.has(key)) continue;
  seen.add(key);
  made++;
  // PGN : numérotation SAN + résultat ouvert.
  let body = '';
  for (let i = 0; i < san.length; i++) body += (i % 2 === 0 ? `${i / 2 + 1}. ` : '') + san[i] + ' ';
  process.stdout.write(`[Event "NanoZero opening book"]\n[White "?"]\n[Black "?"]\n[Result "*"]\n\n${body.trim()} *\n\n`);
}
send('quit');
process.stderr.write(`généré ${made} ouvertures (${PLIES} demi-coups)\n`);
