// leaf-batching.selftest.mjs — correctness du MCTS leaf-batché (searchBatched) : invariants,
// différentiel vs oracle séquentiel search(), cas limites, preuve de batching (avgBatch>1).
// Faux évaluateur DÉTERMINISTE (value/policy = fonction PURE des planes) → zéro réseau ONNX.
// Plan de test : Murat (Test Architect). Correctness des signes/revert : Cloud Dragonborn.

import { GameState } from '../src/game-state.mjs';
import { search, searchBatched, selectMove, visitPolicy } from '../src/mcts.mjs';
import { BatchedEvaluator } from '../src/batched-evaluator.mjs';
import { TENSOR_SIZE } from '../src/encoding/plane-encoding.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };

const POLICY_LEN = 4672;

function planesHash(planes) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < planes.length; i += 13) {
    h ^= ((planes[i] * 1000) | 0) + 0x9e3779b9;
    h = Math.imul(h, 16777619) >>> 0;
  }
  return h >>> 0;
}

// Faux net déterministe. value/policy = fonction PURE des planes (l'ordre de batching ne change
// JAMAIS le résultat → différentiel valide). Implémente evaluate() ET evaluateBatch() à l'identique.
//  'flat'    : logits 0 (uniforme), value 0  → zéro-résidu vloss observable sur W
//  'fenHash' : value=hash(planes)∈[-1,1[, policy biaisée → arbre non-trivial
//  'spike'   : logit = -i (décroît avec l'indice) → le coup légal d'indice mini domine → collisions
function makeFakeNet(mode = 'flat') {
  let calls = 0;
  const evalOne = (planes) => {
    calls++;
    const policy = new Float32Array(POLICY_LEN);
    let value = 0;
    if (mode === 'spike') {
      for (let i = 0; i < POLICY_LEN; i++) policy[i] = -i;
    } else if (mode === 'fenHash') {
      const h = planesHash(planes);
      value = ((h % 2000) / 1000) - 1;
      for (let i = 0; i < POLICY_LEN; i++) policy[i] = Math.sin(i * 0.013 + (h % 997) * 0.001) * 2.5;
    }
    return { policy, value, wdl: [0.3, 0.4, 0.3] };
  };
  return {
    get calls() { return calls; },
    async evaluate(planes) { return evalOne(planes); },
    async evaluateBatch(planesBatch, n) {
      const out = [];
      for (let i = 0; i < n; i++) out.push(evalOne(planesBatch.subarray(i * TENSOR_SIZE, (i + 1) * TENSOR_SIZE)));
      return out;
    },
  };
}

function walk(node, fn) { fn(node); if (node.children) for (const ch of node.children) walk(ch.node, fn); }

// Les 5 invariants (R1/R2/R5). perNode : seulement vrai à K=1 (les collisions K>1 sur-comptent les leaves).
function assertInvariants(label, root, numSims, { perNode = false } = {}) {
  check(root.N === numSims, `${label} root.N == numSims (${root.N})`);
  let sumRoot = 0; for (const ch of root.children) sumRoot += ch.node.N;
  check(Math.abs(sumRoot - (numSims - 1)) < 1e-9, `${label} Σ root.children.N == numSims-1 (${sumRoot})`);
  let finite = true; walk(root, (n) => { if (!Number.isFinite(n.N) || !Number.isFinite(n.W) || !Number.isFinite(n.Q)) finite = false; });
  check(finite, `${label} aucun N/W/Q non-fini`);
  const pi = visitPolicy(root, 0);
  let s = 0, piFinite = true; for (const v of pi) { s += v; if (!Number.isFinite(v)) piFinite = false; }
  check(piFinite && Math.abs(s - 1) < 1e-6, `${label} visitPolicy somme==1 (${s.toFixed(6)})`);
  if (perNode) {
    let ok = true; walk(root, (n) => {
      if (n.children && !n.terminal) { let c = 0; for (const ch of n.children) c += ch.node.N; if (n.N !== 1 + c) ok = false; }
    });
    check(ok, `${label} node.N == 1+Σenfants.N (expansés non-terminaux)`);
  }
}

console.log('T1) zéro-résidu de vloss (FakeNet flat, value≡0 → tout W==0) [R1]');
{
  const net = makeFakeNet('flat');
  const root = await searchBatched(new GameState(), net, 64, { addNoise: false, waveSize: 8 });
  let anyTerminal = false, maxW = 0;
  walk(root, (n) => { if (n.terminal) anyTerminal = true; maxW = Math.max(maxW, Math.abs(n.W)); });
  check(!anyTerminal, 'aucun terminal atteint (startpos peu profond) — pré-requis du test');
  check(maxW < 1e-9, `max|W| == 0 (résidu vloss = ${maxW}) → vloss intégralement retirée`);
  assertInvariants('T1', root, 64);
}

console.log('D1) ⭐ waveSize=1 ≡ search() en régime FP-STABLE (≤48 sims) — mécanisme 3-phases exact [R1/R6]');
{
  // À ≤48 sims, l'arbre batché(K=1) est IDENTIQUE bit-à-bit au séquentiel → prouve que le mécanisme
  // apply-vloss / éval / backprop / revert-vloss est exact et que la vloss s'annule parfaitement.
  // ⚠️ Au-delà de ~48 sims, l'accumulation racine→leaf de backprop() dérive de quelques ULP vs la
  // remontée récursive de simulate() ; cette dérive bascule des ties PUCT serrés → N-diffs structurels
  // (≈86 à 800 sims). Ce N'EST PAS un bug (value/policy justes) → la parité bit-à-bit N'est PAS
  // revendiquée à 800 sims. Le régime prod est couvert par D2 (équivalence statistique).
  const SIMS = 48;
  const seqRoot = await search(new GameState(), makeFakeNet('fenHash'), SIMS, { addNoise: false });
  const batRoot = await searchBatched(new GameState(), makeFakeNet('fenHash'), SIMS, { addNoise: false, waveSize: 1 });
  let diffs = 0, nodes = 0;
  const cmp = (a, b) => {
    nodes++;
    if (a.N !== b.N) diffs++;
    if (Math.abs(a.W - b.W) > 1e-9) diffs++;
    if (!!a.children !== !!b.children) { diffs++; return; }
    if (!a.children) return;
    if (a.children.length !== b.children.length) { diffs++; return; }
    for (let i = 0; i < a.children.length; i++) {
      if (JSON.stringify(a.children[i].move) !== JSON.stringify(b.children[i].move)) diffs++;
      if (Math.abs(a.children[i].P - b.children[i].P) > 1e-9) diffs++;
      cmp(a.children[i].node, b.children[i].node);
    }
  };
  cmp(seqRoot, batRoot);
  check(seqRoot.N === SIMS && batRoot.N === SIMS, `root.N == ${SIMS} des deux côtés`);
  check(diffs === 0, `arbre batché(K=1) IDENTIQUE au séquentiel à ${SIMS} sims (${nodes} noeuds, ${diffs} écarts) → mécanisme exact`);
  assertInvariants('D1', batRoot, SIMS, { perNode: true });
}

console.log('D2/D3) ⭐ qualité de décision + distribution au RÉGIME PROD (800 sims) — K>1 ne dégrade pas [R6]');
{
  // Les seuls tests de QUALITÉ en K>1 (les invariants N/W ne disent rien du choix). Au régime PROD
  // (800 sims, comme le self-play), le coup choisi par la recherche batchée reste dans le top-3 des
  // visites séquentielles (D2) ET la distribution de visites (= cible policy d'entraînement) reste
  // proche en distance L1 (D3) → ni le biais vloss ni la dérive FP n'empoisonnent l'apprentissage.
  // (Mesuré : à 800 sims, K∈{8,16,32} → argmax rank 0, L1 ≤ 0.34. Le ratio K/sims prod est ~1-4 %.)
  const SIMS = 800;
  const seqRoot = await search(new GameState(), makeFakeNet('fenHash'), SIMS, { addNoise: false });
  const top3 = [...seqRoot.children].sort((a, b) => b.node.N - a.node.N).slice(0, 3).map((c) => JSON.stringify(c.move));
  const piSeq = visitPolicy(seqRoot, 0);
  for (const K of [8, 16, 32]) {
    const batRoot = await searchBatched(new GameState(), makeFakeNet('fenHash'), SIMS, { addNoise: false, waveSize: K });
    const mBat = JSON.stringify(selectMove(batRoot, 0));
    check(top3.includes(mBat), `D2 K=${K} : coup batché ∈ top-3 séquentiel${top3.includes(mBat) ? '' : ' (NON: ' + mBat + ')'}`);
    const piBat = visitPolicy(batRoot, 0);
    let l1 = 0; for (let i = 0; i < piSeq.length; i++) l1 += Math.abs(piSeq[i] - piBat[i]);
    check(l1 < 0.5, `D3 K=${K} : L1(visites séq, batché) = ${l1.toFixed(3)} < 0.5 (cible policy non corrompue)`);
    assertInvariants(`D2(K=${K})`, batRoot, SIMS);
  }
}

console.log('T6/T11) la vloss AGIT dans le bon sens : spike → VL=0 collisionne, VL=1 disperse [R4]');
{
  const SIMS = 64, K = 16;
  const net0 = makeFakeNet('spike');
  const root0 = await searchBatched(new GameState(), net0, SIMS, { addNoise: false, waveSize: K, virtualLoss: 0 });
  const net1 = makeFakeNet('spike');
  const root1 = await searchBatched(new GameState(), net1, SIMS, { addNoise: false, waveSize: K, virtualLoss: 1.0 });
  check(net0.calls <= Math.ceil(SIMS / K) + 2, `VL=0 : ~1 leaf unique/vague (${net0.calls} évals) → toutes les descentes collisionnent`);
  check(net1.calls > net0.calls, `VL=1 : PLUS d'évals que VL=0 (${net1.calls} > ${net0.calls}) → la vloss disperse (signe correct)`);
  assertInvariants('T11(VL=0)', root0, SIMS);
  assertInvariants('T6(VL=1)', root1, SIMS);
}

console.log('T7) leaf terminal en milieu de vague (mat-en-1) — résolu sans éval, undo OK [R2]');
{
  const net = makeFakeNet('flat');
  const gs = new GameState('6k1/5ppp/8/8/8/8/8/R6K w - - 0 1'); // Ra8#
  const fenBefore = gs.fen();
  const root = await searchBatched(gs, net, 128, { addNoise: false, waveSize: 8 });
  check(gs.fen() === fenBefore, 'invariant: fen racine inchangée après les vagues (undo du chemin terminal)');
  let term = null; for (const ch of root.children) if (ch.node.terminal) term = ch;
  check(term && term.node.terminalValue === -1, 'enfant terminal (mat) présent, value -1 (côté maté = trait perd)');
  check(term && JSON.stringify(selectMove(root, 0)) === JSON.stringify(term.move), 'le coup le plus visité = le mat');
  assertInvariants('T7', root, 128);
}

console.log('T8) position à 1 seul coup légal — pas de div/0, vloss sur arbre dégénéré [R1]');
{
  const net = makeFakeNet('flat');
  const gs = new GameState('k7/8/2K5/8/8/8/8/1R6 b - - 0 1'); // Ka7 forcé
  const root = await searchBatched(gs, net, 32, { addNoise: false, waveSize: 8 });
  check(root.children.length === 1, `1 seul coup légal (${root.children.length})`);
  assertInvariants('T8', root, 32);
}

console.log('T9) K > nb de leaves disponibles (arbre minuscule) — pas de deadlock, termine [R3]');
{
  const net = makeFakeNet('fenHash');
  const root = await Promise.race([
    searchBatched(new GameState(), net, 16, { addNoise: false, waveSize: 64 }),
    new Promise((_, rej) => setTimeout(() => rej(new Error('TIMEOUT (deadlock?)')), 5000)),
  ]);
  check(root.N === 16, `terminé sans deadlock, root.N==16 (${root.N})`);
  assertInvariants('T9', root, 16);
}

console.log('T10) sims non-multiple de K — dernière vague partielle, root.N exact [R3]');
{
  const net = makeFakeNet('fenHash');
  const root = await searchBatched(new GameState(), net, 50, { addNoise: false, waveSize: 16 });
  check(root.N === 50, `root.N == 50 exact (${root.N}) malgré 50 % 16 != 0`);
  assertInvariants('T10', root, 50);
}

console.log('T12) ⭐ avgBatch>1 à pool=1, K=16 (un seul arbre remplit le batch) — LA raison d’exister [R3]');
{
  const fake = makeFakeNet('fenHash');
  const ev = new BatchedEvaluator(fake, 16, { idleGap: true });
  ev.setLive(1);
  const root = await searchBatched(new GameState(), ev, 128, { addNoise: false, waveSize: 16 });
  check(root.N === 128, `root.N==128 (${root.N})`);
  check(ev.avgBatch > 1, `avgBatch > 1 à pool=1 (avgBatch=${ev.avgBatch.toFixed(1)}) → l’arbre remplit le batch SEUL`);
  check(ev.stats.maxBatch >= 8, `maxBatch >= 8 (${ev.stats.maxBatch}) — fusion intra-arbre prouvée (idle-gap OK)`);
  const ev1 = new BatchedEvaluator(makeFakeNet('fenHash'), 16, { idleGap: true });
  ev1.setLive(1);
  await searchBatched(new GameState(), ev1, 128, { addNoise: false, waveSize: 1 });
  check(ev1.avgBatch < 1.5, `contraste K=1 : avgBatch≈1 (${ev1.avgBatch.toFixed(1)}) — pas de fusion intra-arbre sans vagues`);
}

console.log('TN) invariants tiennent AVEC Dirichlet (addNoise:true) — structurel, non-différentiel');
{
  const net = makeFakeNet('fenHash');
  const root = await searchBatched(new GameState(), net, 64, { addNoise: true, waveSize: 8 });
  assertInvariants('TN', root, 64);
}

console.log('X2) ⭐ fusion cross-game idle-gap (pool>1) — batch≈N×K, zéro deadlock [scénario PROD]');
{
  // Le scénario réel runConcurrentContributionLoop : N searchBatched concurrents partageant UN
  // évaluateur idle-gap. Prouve la fusion cross-game (batch>K) + l'absence de deadlock/flush prématuré.
  const N = 4, K = 8, SIMS = 64;
  const FENS = [
    undefined,
    'rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2',
    'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3',
    'rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2',
  ];
  const ev = new BatchedEvaluator(makeFakeNet('fenHash'), N * K, { idleGap: true });
  ev.setLive(N);
  const roots = await Promise.race([
    Promise.all(FENS.map((f) => searchBatched(new GameState(f), ev, SIMS, { addNoise: false, waveSize: K }))),
    new Promise((_, rej) => setTimeout(() => rej(new Error('TIMEOUT deadlock')), 8000)),
  ]);
  check(roots.length === N && roots.every((r) => r.N === SIMS), `les ${N} parties terminent (pas de deadlock), root.N==${SIMS}`);
  check(ev.avgBatch > K, `fusion cross-game : avgBatch ${ev.avgBatch.toFixed(1)} > K=${K} (vagues de parties DIFFÉRENTES fusionnées)`);
  check(ev.stats.maxBatch > K, `maxBatch ${ev.stats.maxBatch} > K=${K} → preuve de fusion ≈ N×K (idle-gap pool>1 OK)`);
}

console.log('X3) ⭐ résidu-W avec value≠0 (formule analytique) — vloss retirée SANS casser le backprop [R1]');
{
  // Le 'flat' (value 0) rendait le résidu invisible (tout W=0). Ici value CONSTANTE c≠0,≠1 et on
  // recompute W attendu depuis la STRUCTURE : W(n) = c·e(n) − Σ W(enfants), e(n)=N(n)−Σ N(enfants).
  // Un résidu de vloss (+vl sur N et W) ⇒ |W−attendu| = vl·|1−c| ≠ 0 → DÉTECTÉ. Backprop cassé aussi.
  const c = 0.37;
  const net = { async evaluate() { return { policy: new Float32Array(POLICY_LEN), value: c, wdl: [0.3, 0.4, 0.3] }; } };
  for (const K of [1, 8, 16]) {
    const root = await searchBatched(new GameState(), net, 64, { addNoise: false, waveSize: K });
    let anyTerminal = false; walk(root, (n) => { if (n.terminal) anyTerminal = true; });
    let maxErr = 0;
    const expectW = (n) => {
      let sumChN = 0, sumChW = 0;
      if (n.children) for (const ch of n.children) { sumChN += ch.node.N; sumChW += expectW(ch.node); }
      const w = c * (n.N - sumChN) - sumChW; // POV-alterné : c·(visites se terminant à n) − Σ W_enfants
      if (Math.abs(n.W - w) > maxErr) maxErr = Math.abs(n.W - w);
      return w;
    };
    expectW(root);
    check(!anyTerminal, `K=${K} : aucun terminal (pré-requis : value constante c, pas tv) `);
    check(maxErr < 1e-9, `K=${K} : W réel == W analytique, max écart ${maxErr.toExponential(1)} → vloss retirée + backprop intact`);
  }
}

console.log('X4) sur-comptage leaf collisionné — dédup : 1 éval, M backprops (VL=0 + spike) [R5]');
{
  const net = makeFakeNet('spike');
  const root = await searchBatched(new GameState(), net, 17, { addNoise: false, waveSize: 16, virtualLoss: 0 });
  check(net.calls === 2, `1 vague de 16 descentes collisionnées (VL=0+spike) → 2 évals (racine+leaf), PAS 17 (${net.calls})`);
  let collided = null; for (const ch of root.children) if (ch.node.N === 16) collided = ch.node;
  check(collided && collided.children !== null && collided.N === 16, 'leaf collisionné : expansé UNE fois mais N==16 (M backprops) — sur-comptage attendu, dédup OK');
  assertInvariants('X4', root, 17);
}

console.log('X5) partie multi-coups — aucun état gs résiduel entre searches successives [R2]');
{
  const net = makeFakeNet('fenHash');
  const gs = new GameState();
  let moves = 0, allOk = true;
  for (let m = 0; m < 20 && !gs.isGameOver(); m++) {
    const root = await searchBatched(gs, net, 32, { addNoise: false, waveSize: 8 });
    let sumN = 0; for (const ch of root.children) sumN += ch.node.N;
    if (root.N !== 32 || Math.abs(sumN - 31) > 1e-9) allOk = false;
    gs.applyMove(selectMove(root, 0));
    moves++;
  }
  check(allOk && moves > 0, `invariants tenus à CHAQUE coup sur ${moves} coups d'affilée (pas de désync gs cumulée)`);
}

console.log('X7) numSims≤1 → visitPolicy FINIE (pas de NaN) — garde anti-cible-empoisonnée [R1]');
{
  const net = makeFakeNet('fenHash');
  const root = await searchBatched(new GameState(), net, 1, { addNoise: false, waveSize: 8 });
  check(root.N === 1, `root.N==1 (pré-expansion seule, aucune vague) (${root.N})`);
  const pi = visitPolicy(root, 0);
  let s = 0, finite = true; for (const v of pi) { s += v; if (!Number.isFinite(v)) finite = false; }
  check(finite, 'visitPolicy SANS NaN (fallback priors quand Σ visites enfants==0)');
  check(Math.abs(s - 1) < 1e-5, `visitPolicy somme==1 via fallback priors (${s.toFixed(5)})`);
  const r2 = await search(new GameState(), makeFakeNet('fenHash'), 1, { addNoise: false });
  let f2 = true; for (const v of visitPolicy(r2, 0)) if (!Number.isFinite(v)) f2 = false;
  check(f2, 'search() séquentiel à 1 sim aussi finie (garde partagée)');
}

if (fail === 0) console.log('\n✅ leaf-batching self-test : searchBatched correct (invariants + différentiel + batching)');
else { console.error(`\n❌ ${fail} échec(s)`); process.exit(1); }
