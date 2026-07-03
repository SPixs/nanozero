// Bench A/B leaf-batching sur VRAI WebGPU. Tranche board-bound vs GPU-latency-bound :
// si wave>1 > wave=1 en pos/s réel → la latence GPU co-limite → activer le batching.
// si wave>1 ≈ wave=1 → board-bound → seul le board #4 lève le plafond.
// Setup évaluateur IDENTIQUE à la prod (run-loop) : maxBatch=pool×wave, idle-gap si wave>1.
import { BatchedEvaluator } from '../worker/src/batched-evaluator.mjs';
import { NeuralNet } from '../worker/src/nn.mjs';
import { playGame } from '../worker/src/self-play.mjs';

const TENSOR_SIZE = 119 * 8 * 8;

export async function runWaveBench(ort, modelBytes, opts = {}) {
  const poolSize = opts.poolSize ?? 32;
  const sims = opts.sims ?? 200;
  const durationMs = opts.durationMs ?? 15000;
  const waveSizes = opts.waveSizes ?? [1, 8, 16];
  const log = opts.log || (() => {});

  log(`session create (WebGPU)…`);
  const session = await ort.InferenceSession.create(modelBytes, {
    executionProviders: ['webgpu'],
    graphOptimizationLevel: 'all',
  });
  const net = new NeuralNet(session, { Tensor: ort.Tensor });
  log(`warmup batch=${poolSize}…`);
  await net.evaluateBatch(new Float32Array(poolSize * TENSOR_SIZE), poolSize);

  const out = [];
  for (const waveSize of waveSizes) {
    // Réplique EXACTE du run-loop de prod (run-loop.mjs:121).
    const evaluator = new BatchedEvaluator(net, poolSize * waveSize, { idleGap: waveSize > 1 });
    evaluator.setLive(poolSize);
    log(`run wave=${waveSize} (pool=${poolSize}, sims=${sims}, ${durationMs / 1000}s)…`);
    const t0 = performance.now();
    const deadline = t0 + durationMs;
    await Promise.all(
      Array.from({ length: poolSize }, async () => {
        while (performance.now() < deadline) {
          try {
            await playGame(evaluator, { sims, waveSize, tempMoves: 30, maxPlies: 400, deadline });
          } catch {
            break;
          }
        }
      }),
    );
    const dt = (performance.now() - t0) / 1000;
    const pps = Math.round(evaluator.stats.positions / dt);
    const avgBatch = +(evaluator.avgBatch || 0).toFixed(1);
    out.push({ waveSize, pps, avgBatch, positions: evaluator.stats.positions, dt: +dt.toFixed(1) });
    log(`→ wave=${waveSize} : ${pps} pos/s | avgBatch ${avgBatch} | ${evaluator.stats.positions} pos / ${dt.toFixed(1)}s`);
  }
  return out;
}
