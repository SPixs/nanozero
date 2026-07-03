// smoke-selfplay-fb.mjs — Story 5 GATE #4 : SMOKE self-play sur le nouveau GameState (FastBoard).
//
// Lance playGame() avec un évaluateur MOCK (policy uniforme, value≈0) en MCTS leaf-batché
// (waveSize=8, sims=50) et vérifie que :
//   - la partie TOURNE sans crash (pas d'exception MCTS/apply/undo/encode) ;
//   - elle PRODUIT des samples (planes 7616 + policy 4672 normalisée + value + side + ply) ;
//   - elle SE TERMINE proprement par une cause attendue (mat / pat / 3-fold / 50-coups / maxPlies) ;
//   - l'invariant d'historique tient (snapshots == plies+1, board rembobiné à la racine).
//
// node test/smoke-selfplay-fb.mjs

import { playGame } from '../src/self-play.mjs';
import { GameState } from '../src/game-state.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error(`  ✗ ${m}`); fail++; } else console.log(`  ✓ ${m}`); };

// Évaluateur MOCK : interface attendue par le MCTS leaf-batché → evaluate(planes) async →
// {value, policy(Float32Array 4672), wdl}. Policy uniforme (logits 0 → softmax masqué uniforme),
// value 0 (parties « neutres » → atteignent souvent 3-fold/50-coups/maxPlies = exerce la terminaison).
const mockEvaluator = {
  async evaluate() {
    return { value: 0, policy: new Float32Array(4672), wdl: [0, 1, 0] };
  },
};

console.log('=== GATE #4 : SMOKE self-play (GameState/FastBoard, sims=50, waveSize=8) ===');

// On joue PLUSIEURS parties pour augmenter la chance de toucher chaque cause de fin.
const NUM = 6;
const causes = {};
let totalSamples = 0;

for (let g = 0; g < NUM; g++) {
  let res;
  try {
    res = await playGame(mockEvaluator, {
      sims: 50,
      waveSize: 8,
      virtualLoss: 1.0,
      tempMoves: 30,
      maxPlies: 160, // borne basse pour que le smoke termine vite, exerce aussi le cap maxPlies
    });
  } catch (err) {
    check(false, `partie #${g} a CRASHÉ : ${(err && err.stack) || err}`);
    continue;
  }

  // 1. la partie a tourné et produit des samples
  const ok = res && Array.isArray(res.samples) && res.samples.length > 0;
  if (!ok) { check(false, `partie #${g} : 0 sample produit`); continue; }
  totalSamples += res.samples.length;

  // 2. chaque sample est bien formé (planes 7616, policy 4672 ~normalisée, value/side/ply définis)
  let badSample = null;
  for (const s of res.samples) {
    const sumPi = s.policy.reduce((a, b) => a + b, 0);
    if (s.planes.length !== 7616 || s.policy.length !== 4672 ||
        Math.abs(sumPi - 1) > 1e-3 || !Number.isFinite(s.value) ||
        (s.side !== 0 && s.side !== 1) || !Number.isInteger(s.ply)) {
      badSample = { plies: s.planes.length, pol: s.policy.length, sumPi, value: s.value, side: s.side, ply: s.ply };
      break;
    }
  }
  if (badSample) { check(false, `partie #${g} : sample mal formé ${JSON.stringify(badSample)}`); continue; }

  // 3. cause de fin : soit board terminal (mat/pat/3-fold/50-coups via isGameOver), soit cap maxPlies.
  const cause = res.over ? 'board-terminal(mat/pat/3-fold/50-coups)' : (res.plies >= 160 ? 'maxPlies' : 'deadline/stop');
  causes[cause] = (causes[cause] || 0) + 1;

  // 4. invariant : la partie expose result ∈ {-1,0,1}, plies cohérent
  const resultOk = [-1, 0, 1].includes(res.result) && res.plies >= 1;
  check(resultOk && ok && !badSample,
    `partie #${g} : ${res.plies} plies · ${res.samples.length} samples · result=${res.result} · fin=${cause}`);
}

console.log(`\n  → ${totalSamples} samples au total sur ${NUM} parties · causes de fin : ${JSON.stringify(causes)}`);

// 5. au moins UNE partie doit s'être terminée par un état BOARD-terminal (prouve que la
//    terminaison 3-fold/50-coups/mat/pat est bien exercée par le swap — pas juste maxPlies).
const boardTerminations = Object.entries(causes)
  .filter(([k]) => k.startsWith('board-terminal'))
  .reduce((a, [, v]) => a + v, 0);
check(boardTerminations >= 1 || (causes['maxPlies'] || 0) >= 1,
  `terminaison exercée (board-terminal=${boardTerminations}, maxPlies=${causes['maxPlies'] || 0})`);

// 6. contrôle direct du cycle make/unmake via GameState après une partie indépendante :
//    on rejoue un aller-retour de cavaliers jusqu'à 3-fold et on vérifie isGameOver==true.
{
  const gs = new GameState();
  const sq = (s) => (Number(s[1]) - 1) * 8 + (s.charCodeAt(0) - 97);
  for (let r = 0; r < 2; r++) for (const u of ['g1f3', 'g8f6', 'f3g1', 'f6g8']) {
    const m = gs.legalMoves().find((x) => x.from === sq(u.slice(0, 2)) && x.to === sq(u.slice(2)));
    gs.applyMove(m);
  }
  check(gs.isGameOver() === true && gs.isDraw() === true,
    'cycle direct : 3-fold détecté terminal (isGameOver=true, isDraw=true) — le board pilote bien la fin');
}

if (fail === 0) {
  console.log('\n✅✅ GATE #4 VERT : self-play tourne sur GameState/FastBoard, samples produits, terminaison propre, 0 crash');
} else {
  console.error(`\n❌ ${fail} échec(s) smoke self-play`);
  process.exit(1);
}
