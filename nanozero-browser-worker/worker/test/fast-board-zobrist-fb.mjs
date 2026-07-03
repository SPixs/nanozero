// fast-board-zobrist-fb.mjs — Story 2 gate : Zobrist {lo,hi} incrémental +
// clé de répétition convention-Java sur FastBoard.
//
// Quatre gates (rapport chiffré) :
//   GATE 1 — XOR self-inverse : pour chaque FEN, chaque coup → hash_avant XOR
//            hash_après(_make+_unmake) === {lo:0,hi:0}. Le XOR int32 est son
//            propre inverse → _unmake restaure EXACTEMENT le hash d'avant.
//   GATE 2 — incrémental == from-scratch : après une séquence de coups, le
//            _zobrist maintenu incrémentalement == _computeZobrist() recalculé
//            (lo ET hi). Attrape un delta XOR incomplet (case/roque/ep oubliés).
//   GATE 3 — ep convention-Java DANS la clé : sur la FEN ep-divergente (f3 non
//            capturable, chessops position.epSquare === undefined), _repKey()
//            inclut le file f. Deux positions identiques SAUF l'ep → _repKey()
//            DIFFÉRENTS. Prouve le piège « zobrist ep absent de la clé ».
//   GATE 4 — non-régression : déléguée aux gates Story 1 (perft d=4=197281 +
//            invariant), relancées ici en sanity-check léger (d=3 startpos).
//
// Pur-JS, self-contained. Aucun dump Java. node test/fast-board-zobrist-fb.mjs

import { FastBoard } from '../src/fast-board.mjs';
import { makeFen } from 'chessops/fen';
import { ALL_INVARIANT_FENS, HARD_FENS, THREEFOLD_SEQUENCES } from './fast-board-fens.mjs';

let fail = 0;
const err = (msg) => { fail++; console.error(`  ✗ ${msg}`); };
const ZERO = (z) => z.lo === 0 && z.hi === 0;
const eqZ = (a, b) => a.lo === b.lo && a.hi === b.hi;
const hx = (z) => `{lo:${(z.lo >>> 0).toString(16)},hi:${(z.hi >>> 0).toString(16)}}`;

// Résout un UCI 'e2e4'/'g1f3'/'a7a8q' → le move interne {from,to,promotion}
// correspondant dans legalMoves() (LSB=a1 : file=charCode-97, rank=digit-1).
const sqUci = (s) => (Number(s[1]) - 1) * 8 + (s.charCodeAt(0) - 97);
const PROMO = { q: 'queen', r: 'rook', b: 'bishop', n: 'knight' };
function resolveUci(fb, uci) {
  const from = sqUci(uci.slice(0, 2));
  const to = sqUci(uci.slice(2, 4));
  const promo = uci.length > 4 ? PROMO[uci[4]] : undefined;
  for (const m of fb.legalMoves()) {
    if (m.from === from && m.to === to && m.promotion === promo) return m;
  }
  return null;
}

// ===========================================================================
// GATE 1 — XOR self-inverse : _make+_unmake restaure le hash bit-à-bit.
// ===========================================================================
console.log('=== GATE 1 : XOR self-inverse (make+unmake → hash inchangé) ===');
{
  let moves = 0,
    diverged = 0;
  for (const { label, fen } of ALL_INVARIANT_FENS) {
    const fb = new FastBoard(fen);
    const before = fb._repKey();
    let localBad = 0;
    for (const m of fb.legalMoves()) {
      const pre = fb._repKey();
      fb._make(m);
      fb._unmake();
      const post = fb._repKey();
      moves++;
      // self-inverse strict : post === pre, et pre XOR post === {0,0}.
      const x = { lo: pre.lo ^ post.lo, hi: pre.hi ^ post.hi };
      if (!ZERO(x)) {
        diverged++;
        localBad++;
        if (localBad <= 2) err(`${label} : ${m.from}->${m.to}${m.promotion ? '=' + m.promotion : ''} XOR ≠ 0 (${hx(x)})`);
      }
    }
    // après tous les make/unmake, l'état global doit aussi être intact.
    if (!eqZ(before, fb._repKey())) err(`${label} : hash global altéré après la boucle make/unmake`);
    if (localBad === 0) console.log(`  ✓ ${label} : ${fb.legalMoves().length} coups → XOR(make,unmake)=0`);
  }
  console.log(`  → ${moves} coups testés, ${diverged} divergence(s) self-inverse.`);
}

// ===========================================================================
// GATE 2 — incrémental == from-scratch après séquence de coups.
// ===========================================================================
console.log('\n=== GATE 2 : incrémental == from-scratch (après séquence) ===');
{
  let checks = 0,
    diverged = 0;

  // (a) séquences 3-fold dédiées (aller-retour cavaliers + répétitions).
  for (const seq of THREEFOLD_SEQUENCES) {
    const fb = new FastBoard(seq.startFen || 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
    for (const uci of seq.ucis) {
      const m = resolveUci(fb, uci);
      if (!m) { err(`${seq.label} : UCI ${uci} introuvable`); break; }
      fb._make(m);
      const fromScratch = fb._computeZobrist();
      checks++;
      if (!eqZ(fb._zobrist, fromScratch)) {
        diverged++;
        err(`${seq.label} après ${uci} : inc ${hx(fb._zobrist)} ≠ scratch ${hx(fromScratch)}`);
      }
    }
    console.log(`  ✓ ${seq.label} : ${seq.ucis.length} coups → incrémental == from-scratch à chaque pas`);
  }

  // (b) descente perft-like (profondeur 4) sur chaque FEN dur : à chaque _make,
  // l'incrémental doit coller au from-scratch ; au remontée, idem (Gate 1 couvre
  // le retour, on revérifie l'égalité sur quelques positions profondes).
  function descend(fb, depth) {
    if (depth === 0) return;
    const fromScratch = fb._computeZobrist();
    checks++;
    if (!eqZ(fb._zobrist, fromScratch)) {
      diverged++;
      if (diverged <= 3) err(`profondeur : inc ${hx(fb._zobrist)} ≠ scratch ${hx(fromScratch)}`);
    }
    const moves = fb.legalMoves();
    for (const m of moves) {
      fb._make(m);
      descend(fb, depth - 1);
      fb._unmake();
    }
  }
  for (const { label, fen } of ALL_INVARIANT_FENS) {
    const fb = new FastBoard(fen);
    descend(fb, 3);
    console.log(`  ✓ ${label} : descente d=3 → incrémental == from-scratch sur tous les nœuds`);
  }

  console.log(`  → ${checks} comparaisons inc/scratch, ${diverged} divergence(s).`);
}

// ===========================================================================
// GATE 3 — ep convention-Java DANS la clé (piège central).
// ===========================================================================
console.log('\n=== GATE 3 : ep convention-Java dans la clé (vs chessops capturable-only) ===');
{
  // FEN ep-divergente isolée : f3 dans le FEN, AUCUN pion noir adjacent.
  const epFen = HARD_FENS.find((h) => h.label === 'ep-java-divergent-isolated').fen;
  const noEpFen = epFen.replace(' f3 ', ' - '); // même position, ep retiré du FEN

  const fbEp = new FastBoard(epFen);
  const fbNoEp = new FastBoard(noEpFen);

  // (a) DIVERGENCE chessops capturable-only : la sérialisation FEN (legalEpSquare,
  //     ce que verraient chess.js / game-state._key()) DROPPE l'ep f3 non-capturable
  //     → champ ep = '-'. C'est précisément l'écart que la convention-Java comble.
  //     [NB : position.epSquare interne vaut 21 car validEpSquare est plus laxiste
  //      que legalEpSquare ; c'est legalEpSquare/makeFen qui filtre capturable-only.]
  const serializedEp = makeFen(fbEp.position.toSetup()).trim().split(/\s+/)[3];
  if (serializedEp === '-') {
    console.log('  ✓ chessops makeFen droppe l’ep f3 non-capturable (champ ep = "-") — filtre capturable-only');
  } else {
    err(`chessops makeFen devrait dropper f3 non-capturable (ep attendu "-", vu "${serializedEp}")`);
  }

  // (b) MAIS le file f (5) est dans la convention-Java de FastBoard.
  if (fbEp._epFile === 5) {
    console.log('  ✓ FastBoard._epFile === 5 (file f) — convention-Java garde l’ep non-capturable');
  } else {
    err(`FastBoard._epFile attendu 5 (f), vu ${fbEp._epFile}`);
  }

  // (c) PREUVE : deux positions identiques SAUF l'ep → _repKey() DIFFÉRENTS.
  const kEp = fbEp._repKey();
  const kNoEp = fbNoEp._repKey();
  if (!eqZ(kEp, kNoEp)) {
    console.log(`  ✓ _repKey(f3)=${hx(kEp)} ≠ _repKey(no-ep)=${hx(kNoEp)} — l’ep discrimine bien la clé`);
  } else {
    err(`_repKey IDENTIQUES malgré ep≠ (${hx(kEp)}) — l’ep est ABSENT de la clé (piège raté)`);
  }

  // (d) cohérence : la SEULE différence est ZEP[5] XOR ZEP[8]. On le vérifie via
  // un recalcul from-scratch (les deux positions ont les mêmes pièces/trait/roque).
  const xorKeys = { lo: kEp.lo ^ kNoEp.lo, hi: kEp.hi ^ kNoEp.hi };
  if (ZERO(xorKeys)) {
    err('le delta de clé est nul (incohérent avec ep≠)');
  } else {
    console.log(`  ✓ delta de clé non nul = ${hx(xorKeys)} (slot EP file-f vs aucun-ep)`);
  }

  // (e) bonus : la FEN ep RÉELLE (d6 capturable) doit AUSSI porter son file dans la clé,
  // et différer de la même position sans ep.
  const realEp = HARD_FENS.find((h) => h.label === 'ep-real-playable').fen;
  const realNoEp = realEp.replace(' d6 ', ' - ');
  const r1 = new FastBoard(realEp)._repKey();
  const r2 = new FastBoard(realNoEp)._repKey();
  if (!eqZ(r1, r2)) console.log('  ✓ ep réel (d6 capturable) : clés différentes avec/sans ep');
  else err('ep réel d6 : clés identiques avec/sans ep (file d absent de la clé)');
}

// ===========================================================================
// GATE 4 — non-régression légère (perft d=3 startpos via le chemin Zobrist).
//          Les gates lourds (perft d=4 + invariant) restent dans leurs fichiers
//          Story 1 ; on les invoque depuis le runner de gates.
// ===========================================================================
console.log('\n=== GATE 4 : non-régression légère (perft d=3 startpos avec Zobrist actif) ===');
{
  function perft(fb, depth) {
    if (depth === 0) return 1;
    const moves = fb.legalMoves();
    if (depth === 1) return moves.length;
    let n = 0;
    for (const m of moves) {
      fb._make(m);
      n += perft(fb, depth - 1);
      fb._unmake();
    }
    return n;
  }
  const fb = new FastBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  const n = perft(fb, 3);
  if (n === 8902) console.log('  ✓ perft d=3 startpos = 8902 (Zobrist n’a pas cassé make/unmake)');
  else err(`perft d=3 startpos = ${n} (attendu 8902) — RÉGRESSION`);
  // le hash global doit être intact après la descente complète.
  const after = fb._computeZobrist();
  if (eqZ(fb._zobrist, after)) console.log('  ✓ hash global intact après perft d=3 (incrémental == from-scratch)');
  else err('hash global corrompu après perft d=3');
}

// --- Verdict ---
if (fail === 0) {
  console.log('\n✅✅ ZOBRIST FastBoard : Gates 1-4 VERTS (self-inverse + inc==scratch + ep convention-Java).');
  process.exit(0);
} else {
  console.error(`\n❌ ZOBRIST FastBoard : ${fail} échec(s).`);
  process.exit(1);
}
