// game-over-equality-fb.mjs — Story 5 GATE #1 (CORRECTION CRITIQUE imposée) :
//   FastBoard.isGameOver/isCheckmate/isDraw  ==  GameState/chess.js  EXACTEMENT,
// 3-fold (3e occurrence) et 50-coups (halfmove≥100) INCLUS.
//
// ⚠️ OVERRIDE de la SPEC Story 4 : la SPEC déléguait 3-fold/50-coups au MCTS (rep2),
// ce qui était FAUX pour un swap transparent — ça changerait la longueur des parties
// et la data self-play en silence (la 3-fold est FRÉQUENTE dans ce moteur). FastBoard
// reproduit donc désormais le comportement EXACT de chess.js :
//   isGameOver = checkmate | stalemate | insufficient | 3-fold | 50-coups
//   isDraw     =            stalemate | insufficient | 3-fold | 50-coups
//
// node test/game-over-equality-fb.mjs

import { Chess } from 'chess.js';
import { FastBoard } from '../src/fast-board.mjs';
import { GameState } from '../src/game-state.mjs';

let fail = 0;
let pass = 0;
const err = (msg) => { fail++; console.error(`  ✗ ${msg}`); };
const ok = (msg) => { pass++; console.log(`  ✓ ${msg}`); };

const sqUci = (s) => (Number(s[1]) - 1) * 8 + (s.charCodeAt(0) - 97);
const UCI_PROMO = { q: 'queen', r: 'rook', b: 'bishop', n: 'knight' };
const resolveUci = (fb, uci) => {
  const from = sqUci(uci.slice(0, 2)), to = sqUci(uci.slice(2, 4));
  const promo = uci.length > 4 ? UCI_PROMO[uci[4]] : undefined;
  return fb.legalMoves().find((m) => m.from === from && m.to === to && m.promotion === promo) || null;
};
// joue un UCI EN PARALLÈLE sur FastBoard, GameState et un Chess(chess.js) témoin.
const tripleStep = (fb, gs, c, u) => {
  fb._make(resolveUci(fb, u));
  gs.applyMove({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u[4] });
  c.move({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u[4] });
};
// compare les 3 prédicats FastBoard vs GameState vs chess.js sur une position donnée.
const cmp3 = (label, fb, gs, c) => {
  const v = {
    fbO: fb.isGameOver(), fbM: fb.isCheckmate(), fbD: fb.isDraw(),
    gsO: gs.isGameOver(), gsM: gs.isCheckmate(), gsD: gs.isDraw(),
    cO: c.isGameOver(), cM: c.isCheckmate(), cD: c.isDraw(),
  };
  const over = v.fbO === v.gsO && v.gsO === v.cO;
  const mate = v.fbM === v.gsM && v.gsM === v.cM;
  const draw = v.fbD === v.gsD && v.gsD === v.cD;
  if (over && mate && draw) {
    ok(`${label.padEnd(28)} over=${v.fbO} mate=${v.fbM} draw=${v.fbD}`);
  } else {
    err(`${label} : FB{O=${v.fbO},M=${v.fbM},D=${v.fbD}} GS{O=${v.gsO},M=${v.gsM},D=${v.gsD}} chess.js{O=${v.cO},M=${v.cM},D=${v.cD}}`);
  }
};

// ===========================================================================
// PARTIE A — positions TERMINALES par FEN (mat / pat / matériel) + témoins en cours.
// expect = oracle d'échecs indépendant.
// ===========================================================================
console.log('=== GATE #1 PARTIE A : mat / pat / matériel-insuffisant / en-cours (FB==GS==chess.js + oracle) ===');
const CASES = [
  { label: 'mate-fools', fen: 'rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3',
    expect: { over: true, mate: true, draw: false } },
  { label: 'mate-backrank', fen: 'R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1',
    expect: { over: true, mate: true, draw: false } },
  { label: 'stalemate-Q', fen: '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1',
    expect: { over: true, mate: false, draw: true } },
  { label: 'stalemate-KP', fen: '8/8/8/8/8/5k2/5p2/5K2 w - - 0 1',
    expect: { over: true, mate: false, draw: true } },
  { label: 'insuff-KvK', fen: '8/8/4k3/8/8/3K4/8/8 w - - 0 1',
    expect: { over: true, mate: false, draw: true } },
  { label: 'insuff-KBvK', fen: '8/8/4k3/8/8/3K4/5B2/8 w - - 0 1',
    expect: { over: true, mate: false, draw: true } },
  { label: 'insuff-KNvK', fen: '8/8/4k3/8/8/3K4/5N2/8 w - - 0 1',
    expect: { over: true, mate: false, draw: true } },
  { label: 'ongoing-startpos', fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    expect: { over: false, mate: false, draw: false } },
  { label: 'ongoing-check', fen: '4k3/8/8/8/8/8/8/4R1K1 b - - 0 1',
    expect: { over: false, mate: false, draw: false } },
];
for (const { label, fen, expect } of CASES) {
  const fb = new FastBoard(fen), gs = new GameState(fen), c = new Chess(fen);
  const fbV = { over: fb.isGameOver(), mate: fb.isCheckmate(), draw: fb.isDraw() };
  const gsV = { over: gs.isGameOver(), mate: gs.isCheckmate(), draw: gs.isDraw() };
  const cV = { over: c.isGameOver(), mate: c.isCheckmate(), draw: c.isDraw() };
  const parity = fbV.over === gsV.over && fbV.mate === gsV.mate && fbV.draw === gsV.draw
    && gsV.over === cV.over && gsV.mate === cV.mate && gsV.draw === cV.draw;
  const oracle = fbV.over === expect.over && fbV.mate === expect.mate && fbV.draw === expect.draw;
  if (parity && oracle) ok(`${label.padEnd(20)} over=${fbV.over} mate=${fbV.mate} draw=${fbV.draw}`);
  else {
    if (!parity) err(`${label} : FB=${JSON.stringify(fbV)} GS=${JSON.stringify(gsV)} chess.js=${JSON.stringify(cV)}`);
    if (!oracle) err(`${label} : FB=${JSON.stringify(fbV)} ≠ oracle=${JSON.stringify(expect)}`);
  }
}

// ===========================================================================
// PARTIE B — 3-FOLD (CORRECTION CRITIQUE) : la 3e occurrence DOIT être terminale
// (isGameOver=true, isDraw=true, isCheckmate=false) — EXACTEMENT comme chess.js.
// On vérifie aussi la 2e occurrence (PAS encore terminale) pour borner le seuil.
// ===========================================================================
console.log('\n=== GATE #1 PARTIE B : 3-fold == chess.js (2e occ. NON terminale, 3e occ. TERMINALE) ===');
{
  const start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  const cycle = ['g1f3', 'g8f6', 'f3g1', 'f6g8'];
  const fb = new FastBoard(start), gs = new GameState(start), c = new Chess(start);

  for (const u of cycle) tripleStep(fb, gs, c, u); // → 2e occurrence de startpos
  cmp3('2e-occurrence-startpos', fb, gs, c);
  if (fb.isDraw() === false) ok('2e occurrence : isDraw()=false (pas encore 3-fold) — borne basse OK');
  else err('2e occurrence : isDraw() devrait être false');

  for (const u of cycle) tripleStep(fb, gs, c, u); // → 3e occurrence de startpos
  cmp3('3e-occurrence-startpos', fb, gs, c);
  if (fb.isGameOver() === true && fb.isDraw() === true && fb.isCheckmate() === false) {
    ok('3e occurrence : FastBoard isGameOver()=true isDraw()=true isCheckmate()=false (3-fold terminal)');
  } else {
    err(`3e occurrence : FB over=${fb.isGameOver()} draw=${fb.isDraw()} mate=${fb.isCheckmate()} (attendu true/true/false)`);
  }
  // le rep2 de l'encodeur DOIT aussi être vrai à la 3e occurrence (cohérence interne).
  if (fb.toEncoderState().history[0].rep2 === true) ok('3e occurrence : rep2=true (encodeur cohérent avec isDraw)');
  else err('3e occurrence : rep2 devrait être true');
}

// 3-fold via une AUTRE séquence (manoeuvre de dames) — robustesse hors startpos.
console.log('\n=== GATE #1 PARTIE B2 : 3-fold milieu de partie (manoeuvre de dames) ===');
{
  const start = '4k3/8/8/8/8/8/8/Q3K3 w - - 0 1';
  const cycle = ['a1b1', 'e8d8', 'b1a1', 'd8e8'];
  const fb = new FastBoard(start), gs = new GameState(start), c = new Chess(start);
  for (let r = 0; r < 2; r++) for (const u of cycle) tripleStep(fb, gs, c, u); // 3e occ.
  cmp3('3-fold-dames', fb, gs, c);
  if (fb.isDraw() === c.isDraw() && c.isDraw() === true) ok('3-fold dames : isDraw()=true == chess.js');
  else err(`3-fold dames : FB=${fb.isDraw()} chess.js=${c.isDraw()}`);
}

// ===========================================================================
// PARTIE C — 50-COUPS (CORRECTION CRITIQUE) : halfmove≥100 → terminal == chess.js.
// On part d'un FEN à hm=98 (matériel suffisant), on joue 2 coups réversibles → hm=100.
// ===========================================================================
console.log('\n=== GATE #1 PARTIE C : 50-coups (halfmove≥100) == chess.js ===');
{
  // Q vs Q, manoeuvre réversible, départ hm=98. Aucune 3-fold (cases différentes à chaque pas).
  const fen98 = '4k3/8/8/3q4/3Q4/8/8/4K3 w - - 98 80';
  const fb = new FastBoard(fen98), gs = new GameState(fen98), c = new Chess(fen98);
  cmp3('hm=98', fb, gs, c);
  if (fb.isDraw() === false) ok('hm=98 : isDraw()=false (sous le seuil 50-coups)');
  else err('hm=98 : isDraw() devrait être false');
  tripleStep(fb, gs, c, 'd4c4'); // hm=99
  tripleStep(fb, gs, c, 'd5c5'); // hm=100 → 50-coups
  cmp3('hm=100', fb, gs, c);
  if (fb.isGameOver() === true && fb.isDraw() === true && fb.isCheckmate() === false) {
    ok('hm=100 : FastBoard isGameOver()=true isDraw()=true (50-coups terminal) == chess.js');
  } else {
    err(`hm=100 : FB over=${fb.isGameOver()} draw=${fb.isDraw()} (attendu true/true)`);
  }
}

// ===========================================================================
// PARTIE D — FUZZ : 300 parties aléatoires, comparaison des 3 prédicats à CHAQUE position
// vs chess.js témoin. C'est le vrai filet anti-divergence silencieuse (3-fold/50-coups
// surviennent naturellement). Egalité TOTALE attendue sur des dizaines de milliers de positions.
// ===========================================================================
console.log('\n=== GATE #1 PARTIE D : FUZZ 300 parties — over/mate/draw == chess.js à chaque position ===');
{
  let positions = 0, diffs = 0;
  const samples = [];
  for (let g = 0; g < 300; g++) {
    const c = new Chess();
    const gs = new GameState();
    const fb = new FastBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    for (let ply = 0; ply < 160; ply++) {
      positions++;
      const cO = c.isGameOver(), cM = c.isCheckmate(), cD = c.isDraw();
      const eq =
        fb.isGameOver() === cO && fb.isCheckmate() === cM && fb.isDraw() === cD &&
        gs.isGameOver() === cO && gs.isCheckmate() === cM && gs.isDraw() === cD;
      if (!eq) {
        diffs++;
        if (samples.length < 6) samples.push({
          fen: c.fen(),
          chessjs: { O: cO, M: cM, D: cD },
          fb: { O: fb.isGameOver(), M: fb.isCheckmate(), D: fb.isDraw() },
          gs: { O: gs.isGameOver(), M: gs.isCheckmate(), D: gs.isDraw() },
        });
      }
      if (cO) break;
      const cMoves = c.moves();
      const pick = cMoves[Math.floor(Math.random() * cMoves.length)];
      c.move(pick);
      gs.applyMove(pick);
      // rejoue le même coup sur FastBoard (résolu via SAN→from/to du GameState).
      const m = gs.legalMoves(); // état GS = état FB (même séquence)
      const target = m.find((x) => x.san === pick || x.san.replace(/[+#]/g, '') === pick.replace(/[+#]/g, ''));
      // GS.applyMove a déjà avancé GS ; pour FB on rejoue indépendamment via UCI du coup choisi.
      // On reconstruit l'UCI depuis le dernier coup GS (lastMove algébrique) + promo éventuelle.
      const lm = gs.lastMove();
      const promo = /=([QRBN])/.exec(pick);
      const uci = lm.from + lm.to + (promo ? promo[1].toLowerCase() : '');
      const fbMove = resolveUci(fb, uci);
      if (!fbMove) { err(`FUZZ : UCI ${uci} (SAN ${pick}) introuvable sur FastBoard`); break; }
      fb._make(fbMove);
      void target;
    }
  }
  if (diffs === 0) ok(`FUZZ : ${positions} positions, 0 divergence FB/GS vs chess.js (over/mate/draw)`);
  else { err(`FUZZ : ${diffs}/${positions} divergences`); for (const s of samples) console.error('   ' + JSON.stringify(s)); }
}

if (fail === 0) {
  console.log(`\n✅✅ GATE #1 VERT : isGameOver/isCheckmate/isDraw FastBoard == GameState == chess.js`);
  console.log(`   3-fold (3e occ.) + 50-coups (hm≥100) + mat + pat + matériel-insuffisant + en-cours : ÉGALITÉ TOTALE (${pass} assertions)`);
} else {
  console.error(`\n❌ ${fail} divergence(s) game-over (sur ${pass + fail} assertions)`);
  process.exit(1);
}
