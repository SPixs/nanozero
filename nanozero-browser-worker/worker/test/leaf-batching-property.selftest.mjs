// leaf-batching-property.selftest.mjs — FUZZ : invariants vloss/gs sur N positions aléatoires × K.
// Donne le VOLUME exigé (« beaucoup de tests vloss ») : ~N×3 runs, chacun vérifiant zéro-résidu (P1),
// conservation Σvisites (P2) et invariant gs (P3). Faux net 'flat' (value≡0 → résidu = W non nul).
// Le hasard ne sert QU'À générer les positions (l'éval reste pure → déterministe par position).

import { GameState } from '../src/game-state.mjs';
import { searchBatched } from '../src/mcts.mjs';

let fail = 0;
const POLICY_LEN = 4672;

function makeFlatNet() { // value 0, logits 0 (uniforme) — pur
  return {
    async evaluate() { return { policy: new Float32Array(POLICY_LEN), value: 0, wdl: [0.3, 0.4, 0.3] }; },
  };
}
function walk(node, fn) { fn(node); if (node.children) for (const ch of node.children) walk(ch.node, fn); }

function randomFen(maxPlies) {
  const gs = new GameState();
  const n = 1 + Math.floor(Math.random() * maxPlies);
  for (let i = 0; i < n; i++) {
    if (gs.isGameOver()) break;
    const moves = gs.legalMoves();
    gs.applyMove(moves[Math.floor(Math.random() * moves.length)]);
  }
  return gs.fen();
}

const N = 60, SIMS = 48;
let runs = 0, wChecked = 0, terminalSkipped = 0;
const KS = [1, 8, 16];

for (let t = 0; t < N; t++) {
  const fen = randomFen(24);
  const probe = new GameState(fen);
  if (probe.isGameOver()) continue; // searchBatched ne searche jamais une position terminale (comme search())
  for (const K of KS) {
    runs++;
    const gs = new GameState(fen);
    const fenBefore = gs.fen();
    let root;
    try {
      root = await searchBatched(gs, makeFlatNet(), SIMS, { addNoise: false, waveSize: K });
    } catch (e) {
      console.error(`  ✗ THROW fen="${fen}" K=${K} : ${e.message}`); fail++; continue;
    }
    // P3 — invariant gs : la position passée est strictement rembobinée à la racine (C3)
    if (gs.fen() !== fenBefore) { console.error(`  ✗ P3 gs désync fen="${fen}" K=${K}`); fail++; }
    // P2 — conservation : root.N exact + Σ enfants == sims-1 (robuste collisions)
    if (root.N !== SIMS) { console.error(`  ✗ P2 root.N=${root.N} fen="${fen}" K=${K}`); fail++; }
    let sumN = 0; for (const ch of root.children) sumN += ch.node.N;
    if (Math.abs(sumN - (SIMS - 1)) > 1e-9) { console.error(`  ✗ P2 Σenfants=${sumN} fen="${fen}" K=${K}`); fail++; }
    // P1 — zéro-résidu vloss : avec value≡0 et SANS terminal atteint, tout W doit valoir 0.
    // (Un terminal injecte ±1 → W non nul légitime → on saute le W-check pour ce run.)
    let anyTerminal = false, maxW = 0, nan = false;
    walk(root, (nd) => {
      if (nd.terminal) anyTerminal = true;
      if (Math.abs(nd.W) > maxW) maxW = Math.abs(nd.W);
      if (!Number.isFinite(nd.N) || !Number.isFinite(nd.W)) nan = true;
    });
    if (nan) { console.error(`  ✗ P1 NaN/Inf fen="${fen}" K=${K}`); fail++; }
    if (anyTerminal) { terminalSkipped++; }
    else { wChecked++; if (maxW > 1e-9) { console.error(`  ✗ P1 RÉSIDU vloss W=${maxW} fen="${fen}" K=${K}`); fail++; } }
  }
}

console.log(`property fuzz : ${runs} runs (${N} positions × ${KS.length} K), ${wChecked} W-checks zéro-résidu, ${terminalSkipped} sautés (terminal atteint)`);
if (wChecked < runs * 0.5) console.log(`  ⚠ peu de W-checks (${wChecked}/${runs}) — beaucoup de positions atteignent un terminal ; invariants N/gs couvrent quand même tous les runs`);

if (fail === 0) console.log(`\n✅ leaf-batching property self-test : invariants tenus sur ${runs} runs aléatoires`);
else { console.error(`\n❌ ${fail} échec(s)`); process.exit(1); }
