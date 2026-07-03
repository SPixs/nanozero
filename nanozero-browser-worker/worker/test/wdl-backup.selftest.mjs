// wdl-backup.selftest.mjs — remontée du WDL dans l'arbre (façon Lc0) : cohérence avec le scalaire,
// somme=1, composante D, signe (swap W↔L), terminaux, et orientation de la jauge (enfant joué).
// Faux net WDL déterministe (value == wdl[0]−wdl[2]) — zéro réseau ONNX.

import { GameState } from '../src/game-state.mjs';
import { search, searchBatched, selectMove } from '../src/mcts.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };
const POLICY_LEN = 4672;

// net WDL : wfn(planes) -> [w,d,l] ; value = w−l (respecte l'invariant scalaire↔WDL).
function wdlNet(wfn) {
  return { async evaluate() { const wdl = wfn(); return { policy: new Float32Array(POLICY_LEN), value: wdl[0] - wdl[2], wdl }; } };
}
function walk(node, fn) { fn(node); if (node.children) for (const ch of node.children) walk(ch.node, fn); }

console.log('1) cohérence (Ww−Wl)≈W + Σwdl==N + signe du swap, sur search ET searchBatched');
for (const [lbl, K] of [['search K=1', 1], ['batched K=8', 8], ['batched K=16', 16]]) {
  const net = wdlNet(() => [0.55, 0.30, 0.15]); // asymétrique → exerce le swap W↔L
  const root = K === 1 ? await search(new GameState(), net, 128, { addNoise: false })
                       : await searchBatched(new GameState(), net, 128, { addNoise: false, waveSize: K });
  let okScalar = true, okSum = true, finite = true, nodes = 0;
  walk(root, (n) => {
    nodes++;
    if (Math.abs(n.W - (n.Ww - n.Wl)) > 1e-9) okScalar = false; // écart O(1) = bug de signe du swap
    if (Math.abs((n.Ww + n.Wd + n.Wl) - n.N) > 1e-9) okSum = false;
    if (![n.Ww, n.Wd, n.Wl].every(Number.isFinite)) finite = false;
  });
  check(okScalar, `${lbl}: (Ww−Wl) ≈ W sur ${nodes} noeuds (swap W↔L correct)`);
  check(okSum, `${lbl}: Ww+Wd+Wl == N (le WDL somme à 1)`);
  check(finite, `${lbl}: aucun WDL non-fini`);
}

console.log('2) composante D : un net nul (drawish, D symétrique au swap) → root.wdl ≈ [0.1, 0.8, 0.1]');
{
  const root = await searchBatched(new GameState(), wdlNet(() => [0.1, 0.8, 0.1]), 128, { addNoise: false, waveSize: 8 });
  const [w, d, l] = root.wdl;
  check(Math.abs(d - 0.8) < 1e-6, `D≈0.8 propagé sans biais (${d.toFixed(3)})`);
  check(Math.abs(w - 0.1) < 1e-6 && Math.abs(l - 0.1) < 1e-6, `W≈L≈0.1 (${w.toFixed(3)}/${l.toFixed(3)})`);
}

console.log('3) signe : root.wdl[0]−root.wdl[2] a le même signe que le scalaire Q (cohérence sélection↔jauge)');
{
  const root = await searchBatched(new GameState(), wdlNet(() => [0.7, 0.2, 0.1]), 128, { addNoise: false, waveSize: 8 });
  const [w, d, l] = root.wdl;
  check(Math.abs((w - l) - root.Q) < 1e-9, `(wdl[0]−wdl[2]) == Q (${(w - l).toFixed(4)} vs ${root.Q.toFixed(4)})`);
  check(w > l, `position gagnante (net [0.7,_,0.1]) → root.wdl[0] > root.wdl[2]`);
}

console.log('4) terminal : mat-en-1 → l\'enfant matant porte wdl=[0,0,1] (côté maté perd), jauge = Blancs gagnent');
{
  const gs = new GameState('6k1/5ppp/8/8/8/8/8/R6K w - - 0 1'); // Ra8#
  const root = await searchBatched(gs, wdlNet(() => [0.0, 1.0, 0.0]), 128, { addNoise: false, waveSize: 8 });
  let term = null; for (const ch of root.children) if (ch.node.terminal) term = ch;
  const tw = term && term.node.wdl;
  check(tw && tw[2] > 0.99, `enfant terminal (mat) : wdl L≈1 du POV du côté maté (${tw ? tw.map((x) => x.toFixed(2)).join(',') : '?'})`);
  // orientation jauge : POV enfant = Noirs (matés) → réorienté Blancs = [W=tw[2], D, L=tw[0]] = gagnant Blancs
  const mv = selectMove(root, 0);
  check(JSON.stringify(mv) === JSON.stringify(term.move), 'le coup joué = le mat (le plus visité)');
  // après Ra8, trait = Noirs → orientation Blancs prend [tw[2], tw[1], tw[0]] = [≈1, 0, ≈0]
  const whiteW = tw[2];
  check(whiteW > 0.99, `jauge orientée Blancs : W≈1 (les Blancs gagnent) (${whiteW.toFixed(2)})`);
}

console.log('5) getter node.wdl : null si N==0, triplet sinon');
{
  const root = await searchBatched(new GameState(), wdlNet(() => [0.4, 0.3, 0.3]), 64, { addNoise: false, waveSize: 8 });
  check(root.wdl !== null && root.wdl.length === 3, 'racine visitée → wdl triplet');
  const fresh = root.children.find((c) => c.node.N === 0);
  check(!fresh || fresh.node.wdl === null, 'enfant jamais visité (N==0) → wdl null (pas de NaN)');
}

if (fail === 0) console.log('\n✅ wdl-backup self-test : WDL agrégé par la recherche, cohérent (signe, somme, terminaux)');
else { console.error(`\n❌ ${fail} échec(s)`); process.exit(1); }
