// batched-evaluator.mjs — agrège les demandes d'inférence de N parties concurrentes
// en un seul batch NN (le levier perf : batch=1 ~100 pos/s -> batch=64 ~1750+ pos/s).
//
// Interface compatible avec NeuralNet (evaluate(planes) -> Promise<{policy,value}>), donc le
// MCTS l'utilise tel quel. Flush quand la file atteint maxBatch OU quand TOUS les workers
// vivants attendent (file.length >= liveWorkers) — pas de deadlock, batch maximal.

import { TENSOR_SIZE } from './encoding/plane-encoding.mjs';

// Plafond absolu du batch GPU. ORT-Web/WebGPU met en cache un jeu de buffers PAR taille
// de batch distincte → un poolSize×waveSize trop grand ferait une shape unique géante (et
// plusieurs Web Workers = plusieurs sessions partageant le GPU). 256 sature déjà le débit
// (bench leaf-batching) → on borne là. Cf. incident OOM W3090 (K=4 × pool32 × wave8).
const MAX_GPU_BATCH = 256;

// Palier de padding : plus petite puissance de 2 ≥ n, plancher 16, plafonné à `cap`. Borne
// le NOMBRE de shapes distinctes vues par ORT-Web (≤ ~5) → mémoire GPU bornée, plus d'OOM.
function bucketSize(n, cap) {
  let b = 16;
  while (b < n) b <<= 1;
  return Math.min(b, cap);
}

// RESPIRATION MACRO-TÂCHE — en WASM mono-thread (numThreads=1), `session.run` est SYNCHRONE : la
// boucle MCTS enchaîne `evaluate → await → evaluate` en MICRO-tâches ininterrompues, qui ont
// toujours priorité sur les MACRO-tâches. Or les timers d'UI (vitesse `setInterval(2s)`, échiquiers
// coalescés `setInterval(300ms)`) SONT des macro-tâches → ils ne se déclenchent JAMAIS → « — » +
// « démarrage… » figés alors que les parties tournent. Rendre la main au boucleur d'événements
// (une vraie macro-tâche, ≠ `queueMicrotask`) le laisse servir ces timers. MessageChannel = macro-
// tâche immédiate, sans le clamp 4 ms des `setTimeout` imbriqués → coût ~nul.
const YIELD_INTERVAL_MS = 50; // cadence de respiration : << 300ms (boards) → les deux timers passent
const _yieldChannel = (typeof MessageChannel !== 'undefined') ? new MessageChannel() : null;
function macrotaskYield() {
  if (_yieldChannel) {
    return new Promise((resolve) => {
      _yieldChannel.port1.onmessage = () => resolve();
      _yieldChannel.port2.postMessage(0);
    });
  }
  return new Promise((resolve) => setTimeout(resolve, 0));
}

export class BatchedEvaluator {
  constructor(net, maxBatch = 64, { idleGap = false } = {}) {
    this.net = net;
    this.maxBatch = Math.min(maxBatch, MAX_GPU_BATCH);
    this.queue = []; // {planes, resolve}
    this.liveWorkers = 0;
    this.flushing = false;
    // idle-gap : en leaf-batching, UNE vague pousse K planes en une rafale SYNCHRONE. Le quorum
    // `queue >= liveWorkers` flusherait à 1 (pool=1) → batch=1, levier perdu. En mode idle-gap on
    // DIFFÈRE le flush d'une microtask : toute la rafale synchrone (et les vagues concurrentes des
    // autres parties, lancées dans la même pile) s'accumule AVANT le flush → batch ≈ Σ vagues.
    // (Analogue navigateur du flushMicros idle-gap du moteur Java, cf. gpu-worker-pipeline-v2-lessons.)
    this.idleGap = idleGap;
    this._scheduled = false;
    // gpuMs = MUR autour de l'await (sur thread unique il absorbe le MCTS JS → ≠ temps GPU) ;
    // runMs = session.run RÉEL (le vrai temps d'inférence). L'écart gpuMs−runMs = le temps JS bloquant.
    this.stats = { flushes: 0, positions: 0, maxBatch: 0, gpuMs: 0, runMs: 0 };
    this._lastYield = 0; // horodatage de la dernière respiration macro-tâche (cf. macrotaskYield)
  }

  // appelé par le MCTS (via "net.evaluate")
  evaluate(planes) {
    return new Promise((resolve) => {
      this.queue.push({ planes, resolve });
      this._maybeFlush();
    });
  }

  setLive(n) { this.liveWorkers = n; this._maybeFlush(); }
  workerDone() { this.liveWorkers = Math.max(0, this.liveWorkers - 1); this._maybeFlush(); }

  _maybeFlush() {
    if (this.flushing || this.queue.length === 0) return;
    if (this.idleGap) {
      if (this.queue.length >= this.maxBatch) { this._flush(); return; } // plafond atteint → flush immédiat
      if (!this._scheduled) {
        this._scheduled = true; // diffère : laisse la rafale synchrone coalescer (idle-gap)
        queueMicrotask(() => {
          this._scheduled = false;
          if (!this.flushing && this.queue.length > 0) this._flush();
        });
      }
      return;
    }
    if (this.queue.length >= this.maxBatch || this.queue.length >= this.liveWorkers) this._flush();
  }

  async _flush() {
    this.flushing = true;
    const batch = this.queue.splice(0, this.maxBatch);
    const n = batch.length;
    this.stats.flushes++;
    this.stats.positions += n;
    if (n > this.stats.maxBatch) this.stats.maxBatch = n;

    // Padding au palier fixe : on évalue `padded` lignes (puissance de 2 ≥ n), les lignes
    // [n..padded) restant à ZÉRO. Réseau batché = lignes indépendantes → les lignes nulles
    // n'altèrent PAS les n vraies positions (parité conservée) ; on ne résout que les n
    // premières. But : ORT-Web ne voit qu'une poignée de shapes → mémoire GPU bornée (anti-OOM).
    const padded = bucketSize(n, this.maxBatch);
    const planesBatch = new Float32Array(padded * TENSOR_SIZE);
    for (let i = 0; i < n; i++) planesBatch.set(batch[i].planes, i * TENSOR_SIZE);
    const t0 = performance.now();
    const results = await this.net.evaluateBatch(planesBatch, padded);
    this.stats.gpuMs += performance.now() - t0; // MUR autour de l'await : absorbe le MCTS JS mono-thread → trompeur (≠ GPU)
    if (this.net.lastRunMs != null) this.stats.runMs += this.net.lastRunMs; // inférence RÉELLE (session.run) = le vrai temps GPU
    for (let i = 0; i < n; i++) batch[i].resolve(results[i]);

    // On garde `flushing = true` PENDANT la respiration : les évaluations qui arrivent durant le yield
    // s'accumulent dans la file (le batch suivant n'en est que meilleur) au lieu de déclencher un flush
    // ré-entrant concurrent. ~toutes les 50 ms seulement → débloque les timers d'UI sans coût notable.
    const now = performance.now();
    if (now - this._lastYield >= YIELD_INTERVAL_MS) {
      this._lastYield = now;
      await macrotaskYield();
    }

    this.flushing = false;
    this._maybeFlush(); // d'autres ont pu s'enfiler pendant l'await
  }

  get avgBatch() { return this.stats.flushes ? this.stats.positions / this.stats.flushes : 0; }
}
