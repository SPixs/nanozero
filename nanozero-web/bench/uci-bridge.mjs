// uci-bridge.mjs — Adaptateur UCI autour du MOTEUR EXACT de /play/ (core/mcts + ORT-Node).
// But : calibrer en Elo (fastchess) le moteur que les joueurs utilisent vraiment (le cœur JS),
// pas le moteur Java. Recherche à NOMBRE DE SIMS FIXE + échantillonnage par température — comme la
// surface Jouer (selectMove). Déterministe côté recherche (addNoise=false) ; la variété vient de la
// température. Options UCI : Sims (spin), Temperature (string/float).
//
// Lancement : NANOZERO_MODEL=/abs/gen-031.onnx node bench/uci-bridge.mjs
// (sinon ./models/gen-031.onnx par défaut). Réutilisable à chaque génération.

import * as ort from 'onnxruntime-node';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { GameState } from '../core/game-state.mjs';
import { NeuralNet } from '../core/nn.mjs';
import { BatchedEvaluator } from '../core/batched-evaluator.mjs';
import { searchBatched, selectMove } from '../core/mcts.mjs';
import { packMove } from '../core/encoding/move-encoding.mjs';
import { TENSOR_SIZE } from '../core/encoding/plane-encoding.mjs';

const START_FEN = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';
const MODEL = process.env.NANOZERO_MODEL
  || fileURLToPath(new URL('./models/gen-031.onnx', import.meta.url));
const WAVE = 8;

let net = null;
const opts = { sims: 800, temperature: 0 };
let gs = new GameState(START_FEN);

const out = (s) => process.stdout.write(s + '\n');

// 1 thread ORT par moteur par défaut → fastchess parallélise N moteurs sans sur-souscrire le CPU
// (onnxruntime-node prend TOUS les cœurs par défaut → 2 moteurs concurrents thrashaient). Cf. sweeps STS.
const THREADS = Math.max(1, parseInt(process.env.NANOZERO_THREADS || '1', 10));

async function ensureModel() {
  if (net) return;
  const session = await ort.InferenceSession.create(MODEL, {
    executionProviders: ['cpu'], graphOptimizationLevel: 'all',
    intraOpNumThreads: THREADS, interOpNumThreads: 1,
  });
  net = new NeuralNet(session, { Tensor: ort.Tensor });
  await net.evaluateBatch(new Float32Array(WAVE * TENSOR_SIZE), WAVE); // pré-chauffage
}

function applyUci(uci) {
  const lm = gs.legalMoves().find((m) => m.uci === uci);
  if (lm) gs.applyMove(lm);
}

function setPosition(tokens) {
  const movesIdx = tokens.indexOf('moves');
  const end = movesIdx === -1 ? tokens.length : movesIdx;
  if (tokens[0] === 'startpos') gs = new GameState(START_FEN);
  else if (tokens[0] === 'fen') gs = new GameState(tokens.slice(1, end).join(' '));
  if (movesIdx !== -1) for (const mv of tokens.slice(movesIdx + 1)) applyUci(mv);
}

async function go() {
  await ensureModel();
  if (gs.isGameOver()) { out('bestmove 0000'); return; }
  const ev = new BatchedEvaluator(net, WAVE, { idleGap: false });
  ev.setLive(1);
  const root = await searchBatched(gs, ev, opts.sims, { addNoise: false, waveSize: WAVE });
  ev.workerDone();
  const packed = selectMove(root, opts.temperature);
  const lm = gs.legalMoves().find((m) => packMove(m) === packed);
  out('bestmove ' + (lm ? lm.uci : '0000'));
}

// Sérialisation STRICTE : toutes les commandes passent par la même chaîne de promesses, donc
// `position` (mutation de gs) et `go` (lecture de gs) s'exécutent dans l'ORDRE d'arrivée, jamais
// entrelacés — même si plusieurs lignes arrivent d'un coup (cf. bug du test : 2 paires position/go
// envoyées simultanément faisaient lire gs déjà écrasé).
let chain = Promise.resolve();

async function step(t) {
  const cmd = t[0];
  if (cmd === 'uci') {
    out('id name NanoZero-JS gen-031');
    out('id author NanoZero');
    out('option name Sims type spin default 800 min 1 max 200000');
    out('option name Temperature type string default 0');
    out('uciok');
  } else if (cmd === 'isready') {
    await ensureModel();
    out('readyok');
  } else if (cmd === 'setoption') {
    const ni = t.indexOf('name'), vi = t.indexOf('value');
    const name = t.slice(ni + 1, vi).join(' ').toLowerCase();
    const value = t.slice(vi + 1).join(' ');
    if (name === 'sims') opts.sims = Math.max(1, parseInt(value, 10) || opts.sims);
    else if (name === 'temperature') opts.temperature = Math.max(0, parseFloat(value) || 0);
  } else if (cmd === 'ucinewgame') {
    gs = new GameState(START_FEN);
  } else if (cmd === 'position') {
    setPosition(t.slice(1));
  } else if (cmd === 'go') {
    await go();
  } else if (cmd === 'quit') {
    process.exit(0);
  }
}

createInterface({ input: process.stdin }).on('line', (line) => {
  const t = line.trim().split(/\s+/);
  chain = chain.then(() => step(t)).catch((e) => process.stderr.write('ERR ' + (e && e.stack || e) + '\n'));
});
