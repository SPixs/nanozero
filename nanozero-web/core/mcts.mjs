// mcts.mjs — MCTS PUCT AlphaZero (single-game, batch=1).
// Hyperparams alignés sur nanozero (cPuct 2.5, Dirichlet α0.3/ε0.25).
// Valeurs stockées du POV du côté au trait de CHAQUE noeud ; backprop avec négation.

import { encodePlanes } from './encoding/plane-encoding.mjs';
import { encode, packMove, encodePacked } from './encoding/move-encoding.mjs';

export const C_PUCT = 2.5;
export const DIRICHLET_ALPHA = 0.3;
export const DIRICHLET_EPSILON = 0.25;

class Node {
  constructor() {
    this.N = 0;            // visites
    this.W = 0;            // somme des valeurs scalaires (POV side-to-move) — pilote PUCT, INCHANGÉ
    // Sommes W/D/L (POV side-to-move) remontées DANS l'arbre (façon Lc0) → barre d'éval agrégée par
    // la recherche, pas une éval statique 0-ply. PUREMENT ADDITIF : ne touche ni W ni la sélection
    // (parité D1 préservée). Invariant de cohérence : (Ww − Wl) ≈ W (à la dérive flottante près —
    // sommes séparées vs incrémentale ; un écart O(1) trahirait un bug de signe du swap W↔L).
    this.Ww = 0; this.Wd = 0; this.Wl = 0;
    this.children = null;  // [{move, P, node}] après expansion
    this.terminal = false;
    this.terminalValue = 0;
  }
  get Q() { return this.N > 0 ? this.W / this.N : 0; }
  /** WDL moyen (POV side-to-move) agrégé par la recherche, ou null si jamais évalué. */
  get wdl() { return this.N > 0 ? [this.Ww / this.N, this.Wd / this.N, this.Wl / this.N] : null; }
}

// terminalValue → WDL (POV side-to-move) : mat = défaite [0,0,1] ; nulle = [0,1,0] (le +1 défensif).
function terminalWDL(tv) { return tv < 0 ? [0, 0, 1] : (tv > 0 ? [1, 0, 0] : [0, 1, 0]); }
// fallback si le réseau ne fournit pas de wdl : reconstruit un triplet depuis le scalaire v∈[-1,1].
function valueToWDL(v) { const w = Math.max(0, v), l = Math.max(0, -v); return [w, 1 - w - l, l]; }
// remontée d'un niveau : la négation de la value = inversion de perspective → on échange W↔L.
function swapWDL(wdl) { return [wdl[2], wdl[1], wdl[0]]; }

// ---- échantillonnage Dirichlet (Gamma Marsaglia-Tsang) ----
function gaussian() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}
function gamma(alpha) {
  if (alpha < 1) return gamma(alpha + 1) * Math.pow(Math.random(), 1 / alpha);
  const d = alpha - 1 / 3, c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x, v;
    do { x = gaussian(); v = 1 + c * x; } while (v <= 0);
    v = v * v * v;
    const u = Math.random();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}
function sampleDirichlet(alpha, n) {
  const g = new Array(n); let s = 0;
  for (let i = 0; i < n; i++) { g[i] = gamma(alpha); s += g[i]; }
  for (let i = 0; i < n; i++) g[i] /= s;
  return g;
}
function addDirichletNoise(node) {
  const noise = sampleDirichlet(DIRICHLET_ALPHA, node.children.length);
  node.children.forEach((ch, i) => {
    ch.P = (1 - DIRICHLET_EPSILON) * ch.P + DIRICHLET_EPSILON * noise[i];
  });
}

// Crée les enfants d'un noeud depuis la policy NN (softmax masqué sur les coups légaux).
// PARTAGÉ entre `simulate` (séquentiel) et `searchBatched` → expansion strictement IDENTIQUE
// (prérequis de la parité bit-à-bit `waveSize=1` ≡ `search()`, test D1).
function expandNode(node, side, policy, moves) {
  const logits = moves.map((m) => policy[encode(m, side)]);
  const mx = Math.max(...logits);
  let sum = 0;
  const exps = logits.map((l) => {
    const e = Math.exp(l - mx);
    sum += e;
    return e;
  });
  // move stocké en INT compact (packMove) au lieu de l'objet FastBoard + ses 3 getters → mémoire
  // d'arbre ÷~10 (corrige l'OOM Max). encode(m) ci-dessus utilise encore l'objet (prior P) ; le
  // hot-path aval (applyMove/encodePacked) consomme l'int. Parité prouvée (move-pack-parity).
  node.children = moves.map((m, i) => ({ move: packMove(m), P: exps[i] / sum, node: new Node() }));
}

// PUCT : sélectionne l'enfant qui maximise Q + U (Q du POV du parent = −Q(enfant)).
function selectChild(node) {
  const sqrtParent = Math.sqrt(node.N);
  let best = null, bestScore = -Infinity;
  for (const ch of node.children) {
    const q = ch.node.N > 0 ? -ch.node.Q : 0;
    const u = C_PUCT * ch.P * sqrtParent / (1 + ch.node.N);
    const score = q + u;
    if (score > bestScore) { bestScore = score; best = ch; }
  }
  return best;
}

// accumule un triplet WDL (POV du noeud) sur un noeud — additif, n'affecte ni N ni W.
function addWDL(node, wdl) { node.Ww += wdl[0]; node.Wd += wdl[1]; node.Wl += wdl[2]; }

// Une simulation. Retourne {value, wdl} du POV du côté au trait de `node`.
async function simulate(node, gs, net, isRoot) {
  if (node.terminal) {
    node.N++; node.W += node.terminalValue;
    const wdl = terminalWDL(node.terminalValue); addWDL(node, wdl);
    return { value: node.terminalValue, wdl };
  }

  if (node.children === null) {
    if (gs.isGameOver()) {
      node.terminal = true;
      // côté au trait sans coup légal : maté -> perd (−1) ; pat/nulle -> 0.
      node.terminalValue = gs.isCheckmate() ? -1 : 0;
      node.N++; node.W += node.terminalValue;
      const wdl = terminalWDL(node.terminalValue); addWDL(node, wdl);
      return { value: node.terminalValue, wdl };
    }
    const planes = encodePlanes(gs.toEncoderState());
    const ev = await net.evaluate(planes);
    expandNode(node, gs.sideToMove(), ev.policy, gs.legalMoves());
    if (isRoot) addDirichletNoise(node);
    node.N++; node.W += ev.value;
    const wdl = ev.wdl || valueToWDL(ev.value); addWDL(node, wdl);
    return { value: ev.value, wdl };
  }

  const ch = selectChild(node);
  gs.applyMove(ch.move);
  let child;
  try {
    child = await simulate(ch.node, gs, net, false);
  } finally {
    gs.undoMove(); // garde D3 : toujours rééquilibrer gs, même si simulate() throw en descente
  }
  const v = -child.value;
  const wdl = swapWDL(child.wdl); // remontée d'un niveau → échange W↔L
  node.N++; node.W += v; addWDL(node, wdl);
  return { value: v, wdl };
}

/**
 * Lance `numSims` simulations depuis la position courante de `gs`. Retourne le noeud racine.
 *
 * <p>TREE REUSE (parité self-play Java) : passer `root` = sous-arbre de l'enfant joué au coup
 * précédent → la recherche AJOUTE numSims sims NEUVES par-dessus les visites conservées (root.N final
 * = réutilisé + numSims → eff_sims ≈ 2×sims, comme `SearcherCore.runSearch` + `tryReroot` côté Java).
 * Le Dirichlet est RE-échantillonné sur la racine réutilisée (parité `IdleState.onSubmit` qui
 * clear+resample le noise à chaque search — Bug 3 critique self-play v1.1.2). `root=null` = comportement
 * fresh d'origine, strictement inchangé (param opt-in → parité tests préservée).
 */
export async function search(gs, net, numSims, { addNoise = true, onProgress = null, root: reuse = null } = {}) {
  const root = reuse ?? new Node();
  // Racine réutilisée DÉJÀ expansée : re-bruiter ici (simulate() ne ré-expanse pas la racine donc ne
  // poserait jamais le noise). Une racine réutilisée NON expansée (rare : température tire un coup à 0
  // visite) retombe sur le chemin fresh où simulate() pose le noise à la 1ʳᵉ expansion (isRoot).
  const reusedExpanded = reuse != null && root.children !== null;
  if (reusedExpanded && addNoise) addDirichletNoise(root);
  const noiseOnFirstExpand = reusedExpanded ? false : addNoise;
  for (let i = 0; i < numSims; i++) {
    await simulate(root, gs, net, noiseOnFirstExpand);
    if (onProgress) onProgress(i + 1, numSims); // signal « réflexion » live (data-neutre)
  }
  return root;
}

// Backprop d'une value+WDL POV-leaf le long du chemin racine→leaf, avec négation par niveau (POV
// alterné, échange W↔L pour le WDL) + 1 visite/noeud. Identique à la remontée de `simulate`.
function backprop(path, leafValue, leafWDL) {
  const L = path.length - 1;
  for (let i = 0; i <= L; i++) {
    const even = (L - i) % 2 === 0;
    path[i].N += 1;
    path[i].W += (even ? 1 : -1) * leafValue;
    path[i].Ww += even ? leafWDL[0] : leafWDL[2];
    path[i].Wd += leafWDL[1];
    path[i].Wl += even ? leafWDL[2] : leafWDL[0];
  }
}

// Une VAGUE : k descentes synchrones (virtual-loss) → 1 batch d'éval (dédup collisions) →
// expansion + backprop value réelle + RETRAIT vloss. Retourne le nb de descentes (== k).
// Invariants garantis : `gs` rembobiné à la racine après chaque descente (finally, C3) ;
// aucun résidu de vloss après la phase 3 (vérifié en aval par l'assertion Σenfants).
async function runWave(root, gs, net, vl, k) {
  const reqs = []; // {path:[racine..leaf], leaf, terminal, value?, planes?, side?, moves?}
  // ⚠️ Robustesse exception : un throw en phase 1 (selectChild/applyMove/encodePlanes) PROPAGE hors de
  // searchBatched ; la chaîne playGame→slot()/catch JETTE l'arbre entier (jamais soumis). Le résidu de
  // vloss sur l'enfant partiellement descendu est donc INOFFENSIF (arbre orphelin). NE PAS ajouter de
  // try/catch ici pour « sauver » les descentes valides sans aussi retirer la vloss du chemin avorté.
  // PHASE 1 — sélection SYNCHRONE (aucun await → `gs` jamais préempté entre apply/undo).
  for (let d = 0; d < k; d++) {
    const path = [root];
    let depth = 0;
    try {
      let node = root;
      for (;;) {
        if (node.terminal) {
          reqs.push({ path, leaf: node, terminal: true, value: node.terminalValue });
          break;
        }
        if (node.children === null) {
          if (gs.isGameOver()) {
            node.terminal = true;
            node.terminalValue = gs.isCheckmate() ? -1 : 0;
            reqs.push({ path, leaf: node, terminal: true, value: node.terminalValue });
          } else {
            reqs.push({
              path, leaf: node, terminal: false,
              planes: encodePlanes(gs.toEncoderState()), side: gs.sideToMove(), moves: gs.legalMoves(),
            });
          }
          break;
        }
        const ch = selectChild(node); // node.N PROPRE (vloss jamais posée avant selectChild) → sqrt(parent.N) du U non biaisé
        // vloss sur l'ENFANT descendu, signe +vl = pseudo-victoire POV-enfant → Q_enfant ↑ → −Q vu du
        // parent ↓ → branche dépriorisée pour les descentes suivantes de la vague. (Cloud Dragonborn C1.)
        ch.node.N += vl;
        ch.node.W += vl;
        gs.applyMove(ch.move);
        depth++;
        path.push(ch.node);
        node = ch.node;
      }
    } finally {
      for (let i = 0; i < depth; i++) gs.undoMove(); // C3 : rembobinage par profondeur comptée (exception-safe)
    }
  }
  // PHASE 2 — éval batchée des leaves NON-terminaux UNIQUES (dédup : 1 éval / leaf collisionné).
  const uniq = [];
  const seen = new Set();
  for (const r of reqs) if (!r.terminal && !seen.has(r.leaf)) { seen.add(r.leaf); uniq.push(r); }
  const evals = uniq.length ? await Promise.all(uniq.map((r) => net.evaluate(r.planes))) : [];
  const resByLeaf = new Map(); // leaf -> {value, wdl}
  for (let i = 0; i < uniq.length; i++) {
    const r = uniq[i], ev = evals[i];
    resByLeaf.set(r.leaf, { value: ev.value, wdl: ev.wdl || valueToWDL(ev.value) });
    expandNode(r.leaf, r.side, ev.policy, r.moves); // chaque leaf unique expansé UNE fois
  }
  // PHASE 3 — backprop value+WDL réels + retrait vloss. Additif/commutatif → ordre indifférent, et
  // AUCUNE relecture de Q/selectChild ici (on n'additionne que N/W/WDL) → pas d'état transitoire lu.
  // Le retrait vloss touche N et W seulement (le WDL n'a jamais reçu de vloss) → wdl propre après vague.
  for (const r of reqs) {
    const lv = r.terminal ? { value: r.value, wdl: terminalWDL(r.value) } : resByLeaf.get(r.leaf);
    backprop(r.path, lv.value, lv.wdl);
    for (let i = 1; i < r.path.length; i++) { r.path[i].N -= vl; r.path[i].W -= vl; } // retrait vloss (C2 : chemin exact, racine i=0 exclue)
  }
  return reqs.length;
}

/**
 * MCTS LEAF-BATCHÉ : k descentes concurrentes par vague (virtual-loss) → un SEUL arbre remplit un
 * batch d'inférence ≈ k (vs 1 en séquentiel). Levier perf navigateur — bat le plancher WDDM même à
 * 1 partie. ADDITIF : `search()` (séquentiel) reste l'oracle de référence.
 *
 * Virtual-loss (revue correctness Cloud Dragonborn) : `N += vl ; W += +vl` sur chaque enfant
 * descendu (PAS la racine, PAS avant `selectChild`). Signe load-bearing `+vl` : `selectChild` lit
 * `-ch.node.Q`, donc déprioriser = MONTER Q_enfant. Posée en phase 1, RETIRÉE exactement en phase 3
 * (même vl, même chemin) : apply+revert s'annulent à l'ULP près (négations IEEE exactes).
 *
 * ⚠️ PARITÉ vs `search()` : identique dans le régime FP-STABLE (≤ ~48 sims, vérifié bit-à-bit). Au-
 * delà, l'accumulation `racine→leaf` de `backprop()` vs la remontée récursive de `simulate()` dérive
 * de quelques ULP ; cette dérive finit par BASCULER des ties PUCT serrés → des N-diffs structurels
 * apparaissent (≈86 N-diffs à 800 sims). Ce n'est PAS une corruption : value et cible policy restent
 * justes (invariant W==Σ vérifié), seul un départage d'égalité bascule — exactement ce que ferait
 * toute perturbation d'1 ULP entre deux machines. La recherche reste statistiquement équivalente
 * (argmax ∈ top-k séquentiel, cf. test D2). NE PAS s'appuyer sur un golden-master bit-à-bit à 800 sims.
 *
 * @param {object} gs  position (mutée puis rembobinée à chaque descente — invariant racine tenu).
 * @param {object} net  évaluateur batché partagé (BatchedEvaluator idle-gap recommandé).
 * @param {number} numSims  budget (root.N final == numSims exact ; pré-expansion = sim 1).
 * @param {{addNoise?:boolean, waveSize?:number, virtualLoss?:number}} [opts]
 * @returns {Promise<Node>} racine.
 */
export async function searchBatched(gs, net, numSims, { addNoise = true, waveSize = 8, virtualLoss = 1.0, onProgress = null, shouldAbort = null, root: reuse = null } = {}) {
  const root = reuse ?? new Node();
  if (gs.isGameOver()) {
    // position terminale : pas de coup légal (comme search(), playGame ne searche jamais ça).
    root.terminal = true;
    root.terminalValue = gs.isCheckmate() ? -1 : 0;
    root.N = 1;
    root.W = root.terminalValue;
    addWDL(root, terminalWDL(root.terminalValue));
    return root;
  }
  // TREE REUSE : racine réutilisée DÉJÀ expansée → pas de pré-expansion (enfants/priors/visites
  // conservés) ; on re-bruite juste le Dirichlet (parité IdleState Java) et on lance numSims vagues
  // NEUVES par-dessus (done=0). Sinon (fresh OU racine réutilisée non expansée) : pré-expansion =
  // 1ʳᵉ simulation, pose le Dirichlet une seule fois (chemin d'origine, strictement inchangé).
  let done;
  if (reuse != null && root.children !== null) {
    if (addNoise) addDirichletNoise(root);
    done = 0;
  } else {
    const planes = encodePlanes(gs.toEncoderState());
    const ev = await net.evaluate(planes);
    expandNode(root, gs.sideToMove(), ev.policy, gs.legalMoves());
    if (addNoise) addDirichletNoise(root);
    root.N = 1;
    root.W = ev.value;
    addWDL(root, ev.wdl || valueToWDL(ev.value));
    done = 1; // la pré-expansion compte pour 1 sim
  }
  // Baseline garde anti-vloss (forme DELTA, robuste au reuse) : Σenfants.N AVANT les vagues + nb de
  // descentes attendues (chaque descente de vague ajoute +1 à exactement UN enfant de la racine).
  let sumBefore = 0; for (const ch of root.children) sumBefore += ch.node.N;
  const expectedDescents = numSims - done;
  if (onProgress) onProgress(done, numSims); // signal « réflexion » live (data-neutre : n'affecte pas la recherche)
  while (done < numSims) {
    // Annulation (analyse interactive) : abort ENTRE deux vagues → état propre (vloss déjà retirée),
    // on rend la racine partielle sans le contrôle C4 (résultat ignoré par l'appelant via génération).
    if (shouldAbort && shouldAbort()) return root;
    done += await runWave(root, gs, net, virtualLoss, Math.min(waveSize, numSims - done));
    if (onProgress) onProgress(done, numSims);
  }
  // C4 — vloss intégralement retirée ⇔ Σenfants.N a augmenté d'EXACTEMENT `expectedDescents` (chaque
  // descente de vague ajoute +1 à un enfant de la racine). Forme DELTA = robuste au TREE-REUSE : une
  // racine réutilisée peut porter un SUR-COMPTAGE de collision (N > 1+Σenfants quand un nœud fut
  // first-expansé par K descentes collisionnées, cf. test X4) → l'ancien check absolu `Σ==root.N−1`
  // levait à tort sur ces racines. Le delta ignore l'offset de baseline. Chemin fresh : sumBefore=0,
  // expectedDescents=numSims−1 → identique à l'ancien (parité préservée).
  let sumAfter = 0;
  for (const ch of root.children) sumAfter += ch.node.N;
  if (Math.abs((sumAfter - sumBefore) - expectedDescents) > 1e-9) {
    throw new Error(`searchBatched: résidu vloss détecté (ΔΣenfants=${sumAfter - sumBefore} != descentes=${expectedDescents})`);
  }
  return root;
}

/** Distribution de visites normalisée sur les 4672 indices (cible policy d'entraînement). */
export function visitPolicy(root, side) {
  const pi = new Float32Array(4672);
  if (!root.children) return pi; // racine terminale (aucun coup) : playGame ne searche jamais ça
  let total = 0;
  for (const ch of root.children) total += ch.node.N;
  if (total === 0) {
    // numSims ≤ 1 (aucune visite d'enfant) → fallback sur les priors NN (P) au lieu d'une cible
    // NaN (total=0 → 0/0). Protège AUSSI search() séquentiel (même faille). Les P somment à 1.
    for (const ch of root.children) pi[encodePacked(ch.move, side)] = ch.P;
    return pi;
  }
  for (const ch of root.children) pi[encodePacked(ch.move, side)] = ch.node.N / total;
  return pi;
}

/** Choisit un coup : argmax des visites si temperature==0, sinon échantillonne ∝ N^(1/T). */
export function selectMove(root, temperature) {
  const children = root.children;
  if (temperature === 0) {
    let best = children[0];
    for (const ch of children) if (ch.node.N > best.node.N) best = ch;
    return best.move;
  }
  const t = 1 / temperature;
  const w = children.map((ch) => Math.pow(ch.node.N, t));
  const s = w.reduce((a, b) => a + b, 0);
  let r = Math.random() * s;
  for (let i = 0; i < w.length; i++) { r -= w[i]; if (r <= 0) return children[i].move; }
  return children[children.length - 1].move;
}
