// self-play.mjs — joue des parties complètes via MCTS et produit des samples d'entraînement,
// avec N parties concurrentes batchées (P3).

import { GameState } from './game-state.mjs';
import { search, searchBatched, selectMove, visitPolicy } from './mcts.mjs';
import { encodePlanes } from './encoding/plane-encoding.mjs';
import { WHITE } from './encoding/move-encoding.mjs';
import { BatchedEvaluator } from './batched-evaluator.mjs';

// résultat du POINT DE VUE des Blancs : +1 Blancs gagnent, −1 Noirs gagnent, 0 nulle.
function whiteResult(gs) {
  if (gs.isCheckmate()) return gs.sideToMove() === WHITE ? -1 : 1; // côté au trait maté = perd
  return 0; // pat / nulle / cap plies
}

/**
 * Joue une partie complète. Produit des samples {planes, policy(4672), value, side, ply}.
 * value = résultat de la partie du POV du côté au trait du sample (z).
 */
export async function playGame(evaluator, opts = {}) {
  // sims peut être une FONCTION (lu à chaque coup → réglage live) ou un nombre fixe.
  const simsOf = () => (typeof opts.sims === 'function' ? opts.sims() : (opts.sims ?? 800));
  // waveSize>1 active le MCTS leaf-batché (un arbre remplit le batch) ; ==1 = séquentiel (oracle).
  const waveSize = opts.waveSize ?? 1;
  const virtualLoss = opts.virtualLoss ?? 1.0;
  const tempMoves = opts.tempMoves ?? 30;
  const maxPlies = opts.maxPlies ?? 512;
  const deadline = opts.deadline ?? Infinity; // arrêt anticipé (bench débit)
  const onPly = opts.onPly; // hook live (Epic C) : appelé après chaque coup, {ply, fen} (optionnel)
  const onSearchProgress = opts.onSearchProgress; // hook « réflexion » : (done, total) pendant la recherche du coup courant
  const shouldStop = opts.shouldStop; // arrêt RAPIDE (Epic C) : abandonne la partie EN COURS (par coup)
  const evalCache = opts.evalCache ?? null; // cache d'éval NN compact (CPU/WASM only) ; null = désactivé
  const gs = new GameState(opts.startFen);
  const samples = [];
  let ply = 0;
  // TREE REUSE (parité self-play Java) : sous-arbre de l'enfant joué, réutilisé d'un coup à l'autre.
  // null au 1ᵉʳ coup → search fresh.
  let reuseRoot = null;

  while (!gs.isGameOver() && ply < maxPlies && performance.now() < deadline && !(shouldStop && shouldStop())) {
    const side = gs.sideToMove();
    const root = waveSize > 1
      ? await searchBatched(gs, evaluator, simsOf(), { addNoise: true, waveSize, virtualLoss, onProgress: onSearchProgress, root: reuseRoot, evalCache })
      : await search(gs, evaluator, simsOf(), { addNoise: true, onProgress: onSearchProgress, root: reuseRoot, evalCache });
    samples.push({
      planes: encodePlanes(gs.toEncoderState()),
      policy: visitPolicy(root, side),
      side,
      ply,
    });
    const mv = selectMove(root, ply < tempMoves ? 1.0 : 0.0);
    // TREE REUSE : garder le sous-arbre de l'enfant joué comme racine du coup suivant (visites
    // préservées → eff_sims ≈ 2×sims). Ne garder QUE ce nœud détache les frères + l'ancienne racine
    // → GC. null si l'enfant n'est pas instancié (température : coup à 0 visite) → search fresh ensuite.
    const reusedChild = root.children && root.children.find((c) => c.move === mv);
    reuseRoot = reusedChild ? reusedChild.node : null;
    gs.applyMove(mv);
    ply++;
    if (onPly) {
      const info = { ply, fen: gs.fen(), lastMove: gs.lastMove() };
      // Barre d'éval = WDL de l'ENFANT joué, AGRÉGÉ PAR LA RECHERCHE (façon Lc0) → l'éval de la
      // position MAINTENANT affichée, tactiquement consciente (la recherche corrige la cécité de
      // l'éval statique), et GRATUITE (vient de l'arbre — plus d'inférence d'éval/coup). POV de
      // l'enfant = nouveau côté au trait (gs.sideToMove() après le coup) → réorienté BLANCS (stable).
      const child = root.children.find((c) => c.move === mv);
      const cw = child && child.node.wdl;
      if (cw) {
        info.wdl = gs.sideToMove() === WHITE ? { w: cw[0], d: cw[1], l: cw[2] } : { w: cw[2], d: cw[1], l: cw[0] };
      }
      onPly(info);
    }
  }

  const wr = whiteResult(gs);
  for (const s of samples) s.value = wr === 0 ? 0 : (s.side === WHITE ? wr : -wr);
  return { samples, plies: ply, result: wr, over: gs.isGameOver() };
}

/**
 * Lance `numGames` parties EN PARALLÈLE partageant un BatchedEvaluator → batch ~numGames.
 * @returns {{results:Array, stats:object, avgBatch:number}}
 */
export async function runParallelGames(numGames, net, opts = {}) {
  const evaluator = new BatchedEvaluator(net, opts.maxBatch ?? 64);
  evaluator.setLive(numGames);
  const games = [];
  for (let i = 0; i < numGames; i++) {
    // Robustesse : une partie qui lève (encode() throw, erreur chess.js) ne doit PAS rejeter
    // le Promise.all et tuer tout l'onglet bénévole. On capture l'erreur par partie et le
    // finally garantit EXACTEMENT un décrément liveWorkers par partie (une partie morte ne
    // gonfle pas le quorum de flush des survivantes — le _maybeFlush de workerDone le rééquilibre).
    games.push(
      playGame(evaluator, opts)
        .catch((err) => ({ error: String((err && err.stack) || err), samples: [], plies: 0, result: 0, over: false }))
        .finally(() => evaluator.workerDone()),
    );
  }
  const results = await Promise.all(games);
  return { results, stats: evaluator.stats, avgBatch: evaluator.avgBatch };
}

/**
 * Pool ROULANT : maintient `poolSize` parties vivantes en permanence (remplace celles
 * terminées) → garde le batch plein (≈poolSize) jusqu'à `targetGames` ou la `deadline`.
 * C'est la stratégie worker recommandée (évite l'effondrement du batch en fin de run).
 */
export async function runRollingPool(poolSize, net, opts = {}) {
  const target = opts.targetGames ?? Infinity;
  const deadline = opts.deadline ?? Infinity;
  // Circuit-breaker : avec targetGames=Infinity (défaut worker réel), un net cassé ferait
  // tourner le catch par-partie en boucle d'erreurs chaude (CPU 100%) jusqu'à la deadline.
  // Au-delà de N échecs CONSÉCUTIFS, on coupe le pool au lieu de spin-looper.
  const maxConsecutiveFailures = opts.maxConsecutiveFailures ?? 16;
  // idleGap : aligne le bench (STORY-014) sur la VRAIE boucle de contribution (run-loop) quand
  // waveSize>1 — sans idle-gap le quorum `queue>=liveWorkers` flusherait avant que la vague de K
  // feuilles ne se coalesce → batch sous-dimensionné, débit GPU sous-mesuré. Rétro-compat : off
  // par défaut, les appelants existants (sans waveSize/idleGap) gardent le comportement séquentiel.
  const idleGap = opts.idleGap ?? ((opts.waveSize ?? 1) > 1);
  const evaluator = new BatchedEvaluator(net, opts.maxBatch ?? poolSize, { idleGap });
  evaluator.setLive(poolSize);
  let started = 0;
  let consecutiveFailures = 0;
  let aborted = false;
  const results = [];

  async function slot() {
    try {
      while (started < target && performance.now() < deadline && !aborted) {
        started++;
        // Robustesse : une partie qui lève est abandonnée (loggée + comptée), le slot
        // enchaîne sur la suivante au lieu de propager et tuer tout le pool.
        try {
          results.push(await playGame(evaluator, opts));
          consecutiveFailures = 0; // un succès réarme le breaker
        } catch (err) {
          results.push({ error: String((err && err.stack) || err), samples: [], plies: 0, result: 0, over: false });
          if (++consecutiveFailures >= maxConsecutiveFailures) aborted = true; // net cassé -> stop
        }
      }
    } finally {
      evaluator.workerDone(); // ce slot n'alimente plus -> liveWorkers-- (garanti même si erreur)
    }
  }

  await Promise.all(Array.from({ length: poolSize }, () => slot()));
  return { results, stats: evaluator.stats, avgBatch: evaluator.avgBatch };
}
