// encoder-state-equality-fb.mjs — Story 3 GATE 2 : égalité DIRECTE
//   FastBoard.toEncoderState()  ==  GameState.toEncoderState()
// sur toutes les FENs + des séquences de répétition (3-fold). Tout est entier/booléen/
// bigint → comparaison EXACTE (pas de tolérance). Couvre rep1/rep2, fenêtre halfmove,
// snapshot-self-count, castling, fullmove/halfmove, et les bitboards (LSB=a1).
//
// FastBoard avance via _make (résolution UCI → move interne) ; GameState via applyMove
// (cases algébriques). Les deux historiques sont construits identiquement → mêmes snapshots.
//
// node test/encoder-state-equality-fb.mjs

import { FastBoard } from '../src/fast-board.mjs';
import { GameState } from '../src/game-state.mjs';
import { ALL_INVARIANT_FENS, THREEFOLD_SEQUENCES } from './fast-board-fens.mjs';

let fail = 0;
const err = (msg) => { fail++; console.error(`  ✗ ${msg}`); };

const sqUci = (s) => (Number(s[1]) - 1) * 8 + (s.charCodeAt(0) - 97);
const UCI_PROMO = { q: 'queen', r: 'rook', b: 'bishop', n: 'knight' };
function resolveUci(fb, uci) {
  const from = sqUci(uci.slice(0, 2));
  const to = sqUci(uci.slice(2, 4));
  const promo = uci.length > 4 ? UCI_PROMO[uci[4]] : undefined;
  for (const m of fb.legalMoves()) {
    if (m.from === from && m.to === to && m.promotion === promo) return m;
  }
  return null;
}

// Compare deux états d'encodeur champ par champ. Retourne la liste des écarts.
function diffStates(a, b) {
  const d = [];
  if (a.sideToMove !== b.sideToMove) d.push(`sideToMove ${a.sideToMove}≠${b.sideToMove}`);
  if (a.fullmoveNumber !== b.fullmoveNumber) d.push(`fullmove ${a.fullmoveNumber}≠${b.fullmoveNumber}`);
  if (a.halfmoveClock !== b.halfmoveClock) d.push(`halfmove ${a.halfmoveClock}≠${b.halfmoveClock}`);
  for (const k of ['wk', 'wq', 'bk', 'bq']) {
    if (a.castling[k] !== b.castling[k]) d.push(`castling.${k} ${a.castling[k]}≠${b.castling[k]}`);
  }
  if (a.history.length !== b.history.length) {
    d.push(`history.length ${a.history.length}≠${b.history.length}`);
  } else {
    for (let t = 0; t < a.history.length; t++) {
      const ha = a.history[t], hb = b.history[t];
      for (let pt = 0; pt < 6; pt++) {
        if (ha.white[pt] !== hb.white[pt]) d.push(`history[${t}].white[${pt}] ${ha.white[pt]}≠${hb.white[pt]}`);
        if (ha.black[pt] !== hb.black[pt]) d.push(`history[${t}].black[${pt}] ${ha.black[pt]}≠${hb.black[pt]}`);
      }
      if (ha.rep1 !== hb.rep1) d.push(`history[${t}].rep1 ${ha.rep1}≠${hb.rep1}`);
      if (ha.rep2 !== hb.rep2) d.push(`history[${t}].rep2 ${ha.rep2}≠${hb.rep2}`);
    }
  }
  return d;
}

// ===========================================================================
// PARTIE A — FENs isolées : un seul snapshot (la position courante).
// ===========================================================================
console.log('=== PARTIE A : FENs isolées (toEncoderState() FastBoard == GameState) ===');
for (const { label, fen } of ALL_INVARIANT_FENS) {
  const a = new FastBoard(fen).toEncoderState();
  const b = new GameState(fen).toEncoderState();
  const d = diffStates(a, b);
  if (d.length === 0) console.log(`  ✓ ${label} : états identiques (side+castling+fullmove+halfmove+history[1])`);
  else err(`${label} : ${d.length} écart(s) → ${d.slice(0, 4).join(' | ')}`);
}

// ===========================================================================
// PARTIE B — séquences avec historique + RÉPÉTITION : on rejoue la même séquence
// dans FastBoard (_make) et GameState (applyMove) et on compare l'état à CHAQUE pas.
// Couvre rep1/rep2 (3-fold), fenêtre halfmove, snapshot-self-count.
// ===========================================================================
console.log('\n=== PARTIE B : séquences avec répétition (état comparé à chaque pas) ===');
for (const seq of THREEFOLD_SEQUENCES) {
  const startFen = seq.startFen || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  const fb = new FastBoard(startFen);
  const gs = new GameState(startFen);
  let stepFail = 0;

  // état initial (avant tout coup)
  let d0 = diffStates(fb.toEncoderState(), gs.toEncoderState());
  if (d0.length) { err(`${seq.label} [init] : ${d0.slice(0, 3).join(' | ')}`); stepFail++; }

  for (let i = 0; i < seq.ucis.length; i++) {
    const uci = seq.ucis[i];
    const m = resolveUci(fb, uci);
    if (!m) { err(`${seq.label} : UCI ${uci} introuvable (FastBoard)`); stepFail++; break; }
    fb._make(m);
    gs.applyMove({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
    const d = diffStates(fb.toEncoderState(), gs.toEncoderState());
    if (d.length) {
      err(`${seq.label} après ${i + 1} coup(s) (${uci}) : ${d.slice(0, 3).join(' | ')}`);
      stepFail++;
    }
  }
  if (stepFail === 0) {
    // Témoin : la position startpos réapparaît → rep1 doit devenir true à la 2e occurrence.
    const st = fb.toEncoderState();
    console.log(`  ✓ ${seq.label} : ${seq.ucis.length} pas identiques ; history[0].rep1=${st.history[0].rep1} rep2=${st.history[0].rep2}`);
  }
}

// ===========================================================================
// PARTIE C — répétition 3-fold construite à la main + sanity rep1/rep2.
// startpos après g1f3 g8f6 f3g1 f6g8 = 2e occurrence de startpos → rep1=true.
// Répéter une seconde fois → 3e occurrence → rep2=true.
// On vérifie l'ÉGALITÉ FastBoard==GameState ET la valeur attendue (oracle).
// ===========================================================================
console.log('\n=== PARTIE C : sanity rep1/rep2 (oracle 3-fold) ===');
{
  const start = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
  const cycle = ['g1f3', 'g8f6', 'f3g1', 'f6g8'];
  const fb = new FastBoard(start);
  const gs = new GameState(start);
  const play = (uci) => {
    fb._make(resolveUci(fb, uci));
    gs.applyMove({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci[4] });
  };

  for (const u of cycle) play(u); // → 2e occurrence de startpos
  const e2 = fb.toEncoderState(), g2 = gs.toEncoderState();
  if (diffStates(e2, g2).length) err('2e occurrence : FastBoard ≠ GameState');
  if (e2.history[0].rep1 === true && e2.history[0].rep2 === false) {
    console.log('  ✓ 2e occurrence startpos : rep1=true rep2=false (snapshot courant inclus)');
  } else {
    err(`2e occurrence : rep1=${e2.history[0].rep1} rep2=${e2.history[0].rep2} (attendu true/false)`);
  }

  for (const u of cycle) play(u); // → 3e occurrence de startpos
  const e3 = fb.toEncoderState(), g3 = gs.toEncoderState();
  if (diffStates(e3, g3).length) err('3e occurrence : FastBoard ≠ GameState');
  if (e3.history[0].rep1 === true && e3.history[0].rep2 === true) {
    console.log('  ✓ 3e occurrence startpos : rep1=true rep2=true');
  } else {
    err(`3e occurrence : rep1=${e3.history[0].rep1} rep2=${e3.history[0].rep2} (attendu true/true)`);
  }
}

if (fail === 0) {
  console.log('\n✅✅ ÉGALITÉ FastBoard.toEncoderState() == GameState.toEncoderState() : TOTALE');
} else {
  console.error(`\n❌ ${fail} écart(s) FastBoard↔GameState`);
  process.exit(1);
}
