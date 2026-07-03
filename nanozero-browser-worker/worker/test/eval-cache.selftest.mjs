// eval-cache.selftest.mjs — cache COMPACT d'éval NN (eval-cache.mjs) + son intégration MCTS.
//   1. UNIT EvalCache : store/lookup, miss (absent), garde longueur→miss, éviction FIFO, compteurs.
//   2. PARITÉ MCTS : net déterministe PUR de la position courante (sous-ensemble de la clé tronquée
//      → cache information-lossless) ⇒ search/searchBatched AVEC cache == SANS cache, bit-à-bit.
//   3. CACHE PRIMÉ : une 2ᵉ recherche sur cache rempli n'appelle PLUS net.evaluate (0 éval) et
//      reproduit le MÊME arbre → preuve que le HIT saute bien l'éval + le masquage.
// Aucun réseau ONNX : faux net déterministe (comme leaf-batching.selftest.mjs).
//
// Lancer : node worker/test/eval-cache.selftest.mjs   (depuis nanozero-browser-worker/)

import { GameState } from '../src/game-state.mjs';
import { search, searchBatched, visitPolicy, selectMove } from '../src/mcts.mjs';
import { EvalCache } from '../src/eval-cache.mjs';
import { unpackMove } from '../src/encoding/move-encoding.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };
const A = (s) => 'abcdefgh'[s & 7] + ((s >> 3) + 1);
const uciOf = (p) => { const u = unpackMove(p); return A(u.from) + A(u.to); };

// Égalité élément-à-élément de deux Float32Array (cible policy MCTS).
function arrEq(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

// ───────────────────────── faux net DÉTERMINISTE pur de la position COURANTE ─────────────────────────
// Hash UNIQUEMENT les 12 plans de placement courant (planes[0..768)) : sous-ensemble STRICT de ce que
// la clé tronquée capture (zobrist = placement+trait+roque+ep). Le net ignore donc l'historique 8-ply,
// rep, rule50, fullmove → deux positions de même clé donnent FORCÉMENT le même output ⇒ le cache est
// information-lossless ⇒ parité EXACTE attendue entre AVEC et SANS cache.
function hashCurrentBoard(planes) {
  let h = 2166136261 >>> 0;
  const N = 12 * 64; // 12 plans pièces × 64 cases = placement de la position courante (t=0)
  for (let i = 0; i < N; i++) {
    h ^= (planes[i] > 0.5 ? 0x9e3779b9 : 0x12345) + i;
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}
function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function makeNet() {
  const net = {
    evalCount: 0,
    async evaluate(planes) {
      net.evalCount++;
      const rng = mulberry32(hashCurrentBoard(planes));
      const policy = new Float32Array(4672);
      for (let i = 0; i < 4672; i++) policy[i] = (rng() - 0.5) * 4; // logits
      const value = (rng() - 0.5) * 1.8; // ∈ ~[-0.9, 0.9]
      return { value, policy }; // wdl omis → mcts dérive valueToWDL(value) (déterministe aussi)
    },
  };
  return net;
}

// ═══════════════════════════════════ 1. UNIT EvalCache ═══════════════════════════════════
console.log('— EvalCache (unit) —');
{
  const c = new EvalCache(8192);
  check(c.size === 0 && c.lookups === 0 && c.hits === 0 && c.hitRate === 0, 'cache neuf vide (size/lookups/hits/hitRate=0)');

  const e = { value: 0.5, wdl: [0.6, 0.3, 0.1], priors: Float32Array.from([0.1, 0.2, 0.7]) };
  c.set('k', e);
  const got = c.get('k', 3);
  check(got === e, 'get(k,3) après set → entrée identique (HIT)');
  check(c.lookups === 1 && c.hits === 1, 'compteurs : lookups=1 hits=1 après 1 HIT');

  check(c.get('absent', 3) === undefined, 'get(clé absente) → undefined (MISS)');
  check(c.lookups === 2 && c.hits === 1, 'MISS absent : lookups++ mais pas hits');

  check(c.get('k', 4) === undefined, 'garde longueur : get(k,4) avec priors len 3 → undefined (MISS, collision)');
  check(c.lookups === 3 && c.hits === 1, 'MISS longueur : lookups++ mais pas hits');
  check(Math.abs(c.hitRate - 1 / 3) < 1e-9, 'hitRate = hits/lookups = 1/3');
}
{
  // Éviction FIFO : cap=2, on insère 3 clés → la plus ANCIENNE (a) est évincée.
  const c = new EvalCache(2);
  const mk = (v) => ({ value: v, wdl: [0, 1, 0], priors: Float32Array.from([1]) });
  c.set('a', mk(1)); c.set('b', mk(2)); c.set('c', mk(3));
  check(c.size === 2, 'FIFO : taille plafonnée à cap=2 après 3 insertions');
  check(c.get('a', 1) === undefined, 'FIFO : la plus ancienne (a) est évincée');
  check(c.get('b', 1) && c.get('c', 1), 'FIFO : b et c conservées');
}

// ═══════════════════════════════════ 2+3. INTÉGRATION MCTS ═══════════════════════════════════
const SIMS = 80;
const startSide = new GameState().sideToMove();

console.log('— search() séquentiel : parité + cache primé —');
{
  // SANS cache (oracle).
  const netA = makeNet();
  const rootA = await search(new GameState(), netA, SIMS, { addNoise: false });
  const piA = visitPolicy(rootA, startSide);
  const bestA = uciOf(selectMove(rootA, 0));

  // AVEC cache (rempli au fil de la recherche).
  const netB = makeNet();
  const cache = new EvalCache(8192);
  const gsB = new GameState();
  const rootB = await search(gsB, netB, SIMS, { addNoise: false, evalCache: cache });
  const piB = visitPolicy(rootB, startSide);

  check(rootB.N === SIMS, `root.N == ${SIMS} (sims complets avec cache)`);
  check(arrEq(piA, piB), 'PARITÉ : visitPolicy AVEC cache == SANS cache (bit-à-bit)');
  check(uciOf(selectMove(rootB, 0)) === bestA, `même coup choisi (${bestA})`);
  check(gsB.legalMoves().length === 20, 'gs restauré après search (startpos → 20 coups)');
  check(cache.size > 0 && cache.lookups > 0, `cache exercé (size=${cache.size}, lookups=${cache.lookups})`);

  // CACHE PRIMÉ : 3ᵉ recherche fraîche sur le MÊME cache rempli → 0 appel net.evaluate, arbre identique.
  netB.evalCount = 0;
  const lookupsBefore = cache.lookups, hitsBefore = cache.hits;
  const rootC = await search(new GameState(), netB, SIMS, { addNoise: false, evalCache: cache });
  const piC = visitPolicy(rootC, startSide);
  check(netB.evalCount === 0, `cache primé : 0 appel net.evaluate (a fait ${netB.evalCount})`);
  check(cache.hits - hitsBefore === cache.lookups - lookupsBefore, 'cache primé : 100 % HIT sur cette recherche');
  check(arrEq(piA, piC), 'cache primé : arbre identique à l’oracle (parité)');
}

console.log('— searchBatched() leaf-batché : parité + cache primé —');
{
  const wave = 8;
  // SANS cache (oracle batché — on compare batché-vs-batché, pas vs séquentiel).
  const netA = makeNet();
  const rootA = await searchBatched(new GameState(), netA, SIMS, { addNoise: false, waveSize: wave });
  const piA = visitPolicy(rootA, startSide);

  // AVEC cache.
  const netB = makeNet();
  const cache = new EvalCache(8192);
  const gsB = new GameState();
  const rootB = await searchBatched(gsB, netB, SIMS, { addNoise: false, waveSize: wave, evalCache: cache });
  const piB = visitPolicy(rootB, startSide);

  check(rootB.N === SIMS, `root.N == ${SIMS} (sims complets, invariant vloss OK)`);
  check(arrEq(piA, piB), 'PARITÉ : searchBatched AVEC cache == SANS cache (bit-à-bit)');
  check(gsB.legalMoves().length === 20, 'gs restauré après searchBatched');
  check(cache.size > 0 && cache.lookups > 0, `cache exercé (size=${cache.size}, lookups=${cache.lookups})`);

  // CACHE PRIMÉ.
  netB.evalCount = 0;
  const rootC = await searchBatched(new GameState(), netB, SIMS, { addNoise: false, waveSize: wave, evalCache: cache });
  const piC = visitPolicy(rootC, startSide);
  check(netB.evalCount === 0, `cache primé : 0 appel net.evaluate (a fait ${netB.evalCount})`);
  check(arrEq(piA, piC), 'cache primé : arbre identique à l’oracle batché (parité)');
}

console.log(fail === 0 ? '\n✅ eval-cache : tous les tests verts' : `\n❌ eval-cache : ${fail} échec(s)`);
process.exit(fail === 0 ? 0 : 1);
