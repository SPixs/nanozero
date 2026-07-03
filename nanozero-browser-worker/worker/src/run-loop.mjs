// run-loop.mjs — boucle de contribution navigateur (Story A.4) : claim → joue → submit.
//
// Assemble A.3 (client : claim + submitGame) et le self-play (playGameBatch injecté).
// Tourne en continu jusqu'à `shouldStop()`, avec backoff (file vide / erreur réseau, pas de
// spin-loop) et circuit-breaker (N échecs consécutifs → stop, ex. net cassé / serveur down).
// `shouldStop()` est vérifié ENTRE deux cycles (un cycle claim→joue→submit en cours va à son terme).
//
// ⚠️ Cette boucle est SÉQUENTIELLE (1 job = 1 partie = 1 submit). La version N-parties-
// concurrentes (le vrai levier de débit WebGPU, cf DESIGN « N parties concurrentes ») est une
// optimisation DIFFÉRÉE : elle claimerait N jobs et les jouerait en pool batché.
//
// La data soumise reste QUARANTINÉE côté serveur (Epic B) et hors entraînement jusqu'au gate
// (Epic D) — la boucle ne fait que produire.

import { BatchedEvaluator } from './batched-evaluator.mjs';
import { playGame as defaultPlayGame } from './self-play.mjs';

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Boucle de contribution.
 * @param {object} args
 * @param {{claim:Function, submitGame:Function}} args.client  client jobserver (A.3).
 * @param {(job:object)=>Promise<Array<{planes:Float32Array, policy:Float32Array, value:number}>>} args.playGameBatch
 *   joue UNE partie pour ce job et retourne ses samples **plats** (l'adaptateur réel doit aplatir
 *   le `{results:[{samples}]}` de runRollingPool). Un retour vide = partie sans sample → skip.
 *   ⚠️ L'adaptateur DOIT honorer `job.num_sims` (sims dicté par le serveur — cf. la boucle concurrente).
 * @param {number} [args.maxConsecutiveFailures=16]  circuit-breaker.
 * @param {number} [args.backoffMs=2000]  attente sur file vide / erreur.
 * @param {()=>boolean} [args.shouldStop]  hook d'arrêt propre (bouton Stop = FR8/Epic C).
 * @returns {Promise<{cycles:number, submitted:number, idle:number, errors:number, aborted?:string}>}
 */
export async function runContributionLoop({
  client,
  playGameBatch,
  maxConsecutiveFailures = 16,
  backoffMs = 2000,
  shouldStop = () => false,
} = {}) {
  let consecutiveFailures = 0;
  const stats = { cycles: 0, submitted: 0, idle: 0, errors: 0 };

  while (!shouldStop()) {
    try {
      const job = await client.claim();
      if (job === null) {
        // file vide / self-play pausé → backoff, on ne joue rien.
        stats.idle += 1;
        await sleep(backoffMs);
        continue;
      }
      const samples = await playGameBatch(job);
      if (!samples || samples.length === 0) {
        // partie sans sample (deadline atteinte / avortée) : rien à soumettre. On NE trip PAS
        // le breaker (ce n'est pas un échec) et on backoff pour ne pas spin-looper.
        stats.skipped = (stats.skipped || 0) + 1;
        await sleep(backoffMs);
        continue;
      }
      await client.submitGame(job.job_id, samples);
      stats.cycles += 1;
      stats.submitted += 1;
      consecutiveFailures = 0; // un succès réarme le breaker
    } catch (err) {
      stats.errors += 1;
      if (++consecutiveFailures >= maxConsecutiveFailures) {
        // net cassé / serveur down → on coupe au lieu de spin-looper.
        stats.aborted = String((err && err.message) || err);
        break;
      }
      await sleep(backoffMs);
    }
  }
  return stats;
}

/**
 * A4-D1 — Boucle de contribution N-PARTIES CONCURRENTES (levier de débit WebGPU).
 *
 * `poolSize` slots tournent en parallèle en PARTAGEANT UN seul `BatchedEvaluator(net)` : les
 * recherches MCTS des N parties concurrentes fusionnent en un batch d'inférence (batch≈N), vs
 * batch=1 de la boucle séquentielle. Chaque slot boucle `claim → joue → submit` indépendamment.
 *
 * Anti-stall : le `BatchedEvaluator` flushe sur `queue.length >= liveWorkers` (pas d'idle-gap).
 * Un slot qui claim/submit/backoff n'alimente PAS l'évaluateur → s'il restait compté dans
 * `liveWorkers`, le quorum des slots actifs ne serait jamais atteint (stall). On maintient donc
 * un compteur `active` (slots EN RECHERCHE) poussé via `setLive(active)` : `+1` juste avant de
 * jouer, `-1` en `finally` après (le `-1` déclenche aussi un flush des survivants). Aucun
 * changement du `BatchedEvaluator`.
 *
 * @param {object} args
 * @param {{claim:Function, submitGame:Function}} args.client  client jobserver (A.3).
 * @param {object} args.net  réseau (evaluateBatch) — injecté ; en prod = NeuralNet du .onnx champion (A4-D2).
 * @param {number} [args.poolSize=16]  nb de parties concurrentes (= nb de slots).
 * @param {number} [args.maxBatch]  cap du batch (défaut = poolSize).
 * @param {(evaluator:object, opts:object)=>Promise<{samples:Array}>} [args.playGame]  joue UNE partie
 *   sur l'évaluateur partagé (défaut = self-play.playGame). Les samples sont déjà plats {planes,policy,value}.
 * @param {object} [args.gameOpts]  options passées à playGame (sims, maxPlies, deadline...). ⚠️ `sims`
 *   est ÉCRASÉ par `job.num_sims` (le serveur fait autorité) ; gameOpts.sims n'est qu'un repli.
 * @param {number} [args.maxConsecutiveFailures=16]  circuit-breaker PARTAGÉ.
 * @param {number} [args.backoffMs=2000]  attente sur file vide / erreur.
 * @param {()=>boolean} [args.shouldStop]  arrêt propre (vérifié entre cycles ; partie en cours finie + soumise).
 * @returns {Promise<{slots:number, claimed:number, submitted:number, idle:number, skipped:number, errors:number, avgBatch:number, aborted?:string}>}
 */
export async function runConcurrentContributionLoop({
  client,
  net,
  poolSize = 16,
  maxBatch,
  playGame = defaultPlayGame,
  gameOpts = {},
  onPly, // hook live (Epic C) : onPly(slotIndex, {ply, fen, wdl, gameId}) par coup (slot stable), agrégé par l'appelant
  onSearchProgress, // hook « réflexion » : onSearchProgress(slotIndex, done, total) pendant la recherche du coup courant
  progressSlots = 0, // ⚡ ne câble onSearchProgress QUE pour les slots AFFICHÉS (slot<progressSlots) → zéro overhead sur les parties invisibles
  onEvaluator, // hook : reçoit le BatchedEvaluator partagé (lecture de stats live : gpuMs, avgBatch…)
  maxConsecutiveFailures = 16,
  backoffMs = 2000,
  shouldStop = () => false,
} = {}) {
  // waveSize>1 (leaf-batching) : un arbre met K leaves en vol par vague → l'évaluateur doit pouvoir
  // accumuler poolSize×waveSize planes et différer le flush (idle-gap) pour coalescer la rafale.
  const waveSize = (gameOpts && gameOpts.waveSize) || 1;
  const evaluator = new BatchedEvaluator(net, (maxBatch ?? poolSize) * waveSize, { idleGap: waveSize > 1 });
  if (onEvaluator) onEvaluator(evaluator); // expose les stats live (gpuMs/avgBatch) à l'appelant
  // Compteur des slots EN RECHERCHE. setLive(active) garde le quorum de flush aligné sur les
  // chercheurs réels (un slot en claim/submit/backoff n'y est PAS compté → pas de stall).
  let active = 0;
  const setActive = (delta) => {
    active += delta;
    evaluator.setLive(active);
  };

  let consecutiveFailures = 0;
  let aborted = null;
  const stats = { slots: poolSize, claimed: 0, submitted: 0, idle: 0, skipped: 0, errors: 0 };

  const fail = (err) => {
    stats.errors += 1;
    if (++consecutiveFailures >= maxConsecutiveFailures) {
      aborted = String((err && err.message) || err);
    }
  };

  async function slot(slotIndex) {
    while (!shouldStop() && !aborted) {
      let job;
      try {
        job = await client.claim();
      } catch (err) {
        fail(err);
        await sleep(backoffMs);
        continue;
      }
      if (job === null) {
        stats.idle += 1; // file vide / pausé → backoff, on ne joue rien
        await sleep(backoffMs);
        continue;
      }
      stats.claimed += 1;

      let result;
      let threw = false;
      // shouldStop passé à playGame → la partie EN COURS s'abandonne par coup (Stop rapide), pas
      // seulement entre deux parties. Per-slot : la clé d'event est l'INDEX DE SLOT (stable
      // 0..poolSize-1, recyclé d'une partie à l'autre) → l'UI affiche exactement les parties EN
      // COURS (pas une par job_id qui gonflerait). onPly(slotIndex, {ply, fen, gameId, lastMove}).
      // Le nombre de sims est DICTÉ PAR LE SERVEUR (job.num_sims) et fait AUTORITÉ sur le défaut
      // client gameOpts.sims : un contributeur honnête joue exactement le budget de recherche demandé.
      // La qualité de la cible policy (visites MCTS normalisées) est fonction directe des sims → un
      // client mal configuré ne peut plus injecter des cibles basse-résolution. Repli sur gameOpts.sims
      // si le job n'en précise pas. ⚠️ NON contraignant pour un client MALVEILLANT (le compute reste
      // non-fiable, il peut ignorer la consigne) : l'enforcement réel = quarantaine source=browser +
      // détection de drift D.2 + arbitrage SPRT (D.3). Ceci ferme le trou ACCIDENTEL, pas l'hostile.
      const slotOpts = { ...gameOpts, shouldStop };
      if (job.num_sims != null) slotOpts.sims = job.num_sims;
      if (onPly) slotOpts.onPly = (info) => onPly(slotIndex, { ...info, gameId: job.job_id });
      // onSearchProgress UNIQUEMENT pour les slots affichés : les parties invisibles ne paient AUCUN
      // coût de progression (pas de callback par-sim dans le hot path MCTS).
      if (onSearchProgress && slotIndex < progressSlots) slotOpts.onSearchProgress = (done, total) => onSearchProgress(slotIndex, done, total);
      setActive(+1); // ce slot va chercher → entre dans le quorum de flush
      try {
        result = await playGame(evaluator, slotOpts);
      } catch (err) {
        threw = true;
        result = { samples: [] };
        fail(err);
      } finally {
        setActive(-1); // ne cherche plus → sort du quorum (et flushe les survivants)
      }
      if (threw) {
        // Échec déjà compté par fail(). Backoff pour NE PAS spin-looper : un net cassé qui ne
        // trippe pas encore le breaker enchaînerait sinon des parties en échec à 100 % CPU.
        if (!aborted) await sleep(backoffMs);
        continue;
      }
      if (shouldStop()) continue; // Stop demandé pendant la partie → partie partielle, on ne soumet PAS

      const samples = (result && result.samples) || [];
      if (samples.length === 0) {
        // Partie sans sample (deadline/avortée) : rien à soumettre, PAS un échec. Backoff pour
        // éviter un spin-loop CPU 100 % si les parties vides s'enchaînent (la boucle A.4
        // séquentielle backoff aussi sur ce chemin) — un spin gèlerait même le flush des autres
        // slots par starvation de la macrotask/timer queue.
        stats.skipped += 1;
        await sleep(backoffMs);
        continue;
      }
      try {
        await client.submitGame(job.job_id, samples);
        stats.submitted += 1;
        consecutiveFailures = 0; // un succès réarme le breaker partagé
      } catch (err) {
        fail(err);
        await sleep(backoffMs);
      }
    }
  }

  await Promise.all(Array.from({ length: poolSize }, (_, i) => slot(i)));
  stats.avgBatch = evaluator.avgBatch;
  stats.maxBatch = evaluator.stats.maxBatch; // plus grand batch fusionné observé (preuve de fusion)
  if (aborted) stats.aborted = aborted;
  return stats;
}
