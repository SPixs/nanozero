// hardening.selftest.mjs — couvre les correctifs de durcissement (spec-browser-worker-hardening) :
//   A) robustesse pool : une partie qui lève ne tue pas runParallelGames / runRollingPool.
//   B) mémo FEN GameState : cohérent avec chess.js après applyMove / undoMove.
// N'utilise PAS onnxruntime : des fake-nets contrôlent le comportement (throw vs uniforme).

import { runParallelGames, runRollingPool } from '../src/self-play.mjs';
import { GameState } from '../src/game-state.mjs';
import { NeuralNet } from '../src/nn.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };

// Fake-net qui renvoie n résultats `undefined` -> chaque partie lève à la déstructuration
// `const { policy, value } = await net.evaluate(...)` dans le MCTS.
const throwingNet = { evaluateBatch: async (_b, n) => new Array(n).fill(undefined) };
// Fake-net uniforme valide -> les parties se jouent normalement.
const uniformNet = {
  evaluateBatch: async (_b, n) =>
    Array.from({ length: n }, () => ({ policy: new Float32Array(4672).fill(1 / 4672), value: 0 })),
};

const opts = { sims: 5, maxPlies: 6, maxBatch: 4 };

console.log('A) Robustesse du pool');
{
  // A1 : toutes les parties lèvent -> le pool RÉSOUT (ne rejette pas), chaque résultat = {error}.
  const { results } = await runParallelGames(4, throwingNet, opts);
  check(results.length === 4, `A1 runParallelGames: 4 résultats malgré les throws (${results.length})`);
  check(results.every((r) => r && r.error), 'A1 runParallelGames: chaque partie fautive -> {error}');

  // A2 : non-régression happy-path -> 4 parties valides, aucun error, samples présents.
  const ok = await runParallelGames(4, uniformNet, opts);
  check(ok.results.length === 4 && ok.results.every((r) => !r.error && Array.isArray(r.samples)),
    'A2 runParallelGames: happy-path 4 parties valides');

  // A3 : runRollingPool ne se fige pas et capture les erreurs (target borné).
  const rp = await runRollingPool(4, throwingNet, { ...opts, targetGames: 4 });
  check(rp.results.length >= 1 && rp.results.every((r) => r && r.error),
    `A3 runRollingPool: termine + erreurs capturées (${rp.results.length})`);
}

console.log('B) FEN GameState (cohérence make/unmake — Story 5 : adaptateur FastBoard)');
{
  // ⚠️ Story 5 : plus de cache FEN mémoïsé chess.js (`_currentFen`/`gs.game`). On teste la
  // PROPRIÉTÉ PUBLIQUE équivalente via fen() : startpos correct, change après applyMove,
  // restauré à l'identique après undoMove. Délègue à FastBoard.fen() (chessops makeFen).
  const gs = new GameState();
  check(gs.fen() === 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1', 'B1 fen() startpos correct');

  const fenBefore = gs.fen();
  gs.applyMove(gs.legalMoves()[0]);
  check(gs.fen() !== fenBefore, 'B3 fen() change après applyMove');

  gs.undoMove();
  check(gs.fen() === fenBefore, 'B5 fen() restauré à l\'identique après undoMove');

  const st = gs.toEncoderState();
  check(st && Array.isArray(st.history) && st.history.length >= 1, 'B6 toEncoderState OK');
}

console.log('C) Circuit-breaker runRollingPool (anti spin-loop sur net cassé, targetGames=∞)');
{
  // net cassé + PAS de targetGames (=> Infinity) : doit s'arrêter après ~maxConsecutiveFailures
  // au lieu de spin-looper sur les erreurs jusqu'à une deadline.
  const t0 = performance.now();
  const rp = await runRollingPool(2, throwingNet, { sims: 5, maxPlies: 5, maxBatch: 2, maxConsecutiveFailures: 6 });
  const dt = performance.now() - t0;
  check(rp.results.every((r) => r && r.error), 'C1 toutes les parties en erreur');
  check(rp.results.length < 50, `C2 borné par le breaker (${rp.results.length} parties, pas de boucle infinie)`);
  check(dt < 5000, `C3 termine vite via le breaker (${dt.toFixed(0)}ms)`);
}

console.log('D) Aliasing nn.evaluateBatch (la copie .slice survit à la réutilisation du buffer ORT)');
{
  const sharedPol = new Float32Array(4672); // buffer de sortie ORT réutilisé entre runs (IO-binding)
  let runCount = 0;
  const fakeSession = {
    inputNames: ['board'],
    run: async () => {
      runCount += 1;
      sharedPol.fill(runCount === 1 ? 0.5 : 0.9); // contenu écrasé à chaque run
      return { policy_logits: { data: sharedPol }, value: { data: new Float32Array([1, 0, 0]) } };
    },
  };
  const net = new NeuralNet(fakeSession, { Tensor: function () {} }); // input non inspecté par le fake
  const r1 = await net.evaluate(new Float32Array(7616)); // run#1 -> sharedPol = 0.5
  check(r1.policy[0] === 0.5, 'D1 run#1 lit bien 0.5');
  await net.evaluate(new Float32Array(7616)); // run#2 -> sharedPol écrasé = 0.9
  check(r1.policy[0] === 0.5, 'D2 policy de run#1 INCHANGÉE après run#2 (copie détachée, pas une vue)');
}

console.log(fail === 0 ? '\nOK — hardening selftest vert' : `\nFAIL — ${fail} check(s) en échec`);
process.exit(fail ? 1 : 0);
