// complexity.mjs — indicateur de « complexité » d'une position depuis la racine MCTS (J2).
// PARTAGÉ (core/) : réutilisé par Jouer ET Analyse. Fichier NEUF (pas de déphasage worker).
//
// ⚠️ Heuristique EXPÉRIMENTALE (gate ML AUC≥0,70 non passé → badge affiché « expérimental », cf. J2).
// β = regret moyen pondéré par la policy : « combien un joueur qui suit la policy perd en moyenne ».
//   - position où un seul coup tient (les autres, plausibles par policy, perdent) → β élevé = tranchante ;
//   - position où tous les coups raisonnables se valent → β bas = nette.
// Valeurs ∈ [0..2] (valeurs MCTS ∈ [-1,1]). Seuils défaut (architecture §6.2) : calm<0.10, tense<0.25.
// Spec canonique = nanozero-web/docs/architecture.md §6.2/§6.2.1 (ce fichier en est la source de vérité).
// Forme = "swing" (E[regret] pondéré policy vs PV) ; les variantes candidates (masse-au-seuil β_τ, τ,
// référence PV/max, terme de naturalité) sont tranchées EMPIRIQUEMENT par le gate de validation (§7).
//
// ROBUSTESSE BAS-BUDGET (fix 2026-06-29) : à faible nombre de sims (ex. niveau « Tranquille » = 48),
// la plupart des enfants ne sont PAS visités (N=0 → Q=0 par défaut = valeur INCONNUE, pas nulle) et
// un enfant visité une seule fois a une valeur bruitée. L'ancien calcul (somme sur TOUS les enfants,
// référence = max brut) transformait ce bruit en « tranchante » sur des positions calmes (ex. 1.e4 e5
// 2.Cf3). On corrige : (1) ne garder que les coups réellement explorés (N>0) ; (2) référence = valeur
// de la ligne principale (coup le plus visité = estimation la plus fiable), PAS le max brut.

const DEFAULT_THRESHOLDS = { calm: 0.10, tense: 0.25 };

/**
 * @param {object} root  racine MCTS (children = [{move, P, node}], node.{N,Q}, Q = valeur POV enfant)
 * @param {{calm:number, tense:number}} [thresholds]
 * @returns {{ beta:number, level:'calm'|'tense'|'sharp' }}
 */
export function computeComplexity(root, thresholds = DEFAULT_THRESHOLDS) {
  const all = root && root.children;
  if (!all || all.length < 2) return { beta: 0, level: 'calm' };
  // Ne considérer que les coups réellement évalués par la recherche : la valeur d'un coup non
  // visité (N=0) est inconnue, pas nulle — l'inclure fausse le regret à bas budget.
  const ch = all.filter((c) => c.node && c.node.N > 0);
  if (ch.length < 2) return { beta: 0, level: 'calm' };
  // Référence = valeur de la ligne principale (coup le plus visité), robuste au bruit d'un coup
  // peu visité (un seul rollout chanceux ferait sinon un faux pic via max). Valeur POV trait = -Q.
  let pv = ch[0];
  for (const c of ch) if (c.node.N > pv.node.N) pv = c;
  const refVal = -pv.node.Q;
  let beta = 0, psum = 0;
  for (const c of ch) {
    const p = c.P || 0;
    beta += p * Math.max(0, refVal - (-c.node.Q));
    psum += p;
  }
  if (psum > 0) beta /= psum;
  const level = beta >= thresholds.tense ? 'sharp' : beta >= thresholds.calm ? 'tense' : 'calm';
  return { beta, level };
}

// ─────────────────────────────────────────────────────────────────────────────
// computeCriticality (β-v3, VALIDÉ 2026-07-01) — indice de « criticité objective »
// gate-passé (AUC 0,79 vs netteté Stockfish n_within_50≤2, GroupKFold+placebo, 15k Lichess),
// 100% in-browser, SANS Stockfish. Résout le verrou stockfish.wasm pour la face « criticité ».
// Cadrage HONNÊTE : « position critique / à réfléchir », JAMAIS « tu vas gaffer » (prouvé trompeur).
// Modèle = régression logistique sur 6 features lues dans l'arbre final (zéro snapshot/1-ply).
// Détail : nanozero-human-fidelity/docs/EXP-beta-v3-preregistration.md + complexity_v3_model.json.
// v3c : cible HONNÊTE (peu de bons coups ET pas une recapture forcée) + AND-gate de naturalité
// (score × non_évident(p_pv)). Sans le gate, les recaptures/échanges OBLIGÉS étaient sur-taggés critiques
// (recapture de dame 84 % → 8 %). p_pv = prior policy du coup le plus visité (gratuit dans l'arbre).
const CRIT_MODEL = {
  // v3e SIM-STABLE : les features de VISITES (ent_visits, KL visites‖prior) DÉRIVENT avec les sims (PUCT
  // concentre les visites → ça mesure la convergence de la recherche, pas la criticité de la position).
  // Remplacées par ent_policy (entropie du PRIOR réseau, SIM-INVARIANTE) + les features de VALEUR
  // (regret/variance/decisiveness, qui CONVERGENT). → la criticité se PRÉCISE avec les sims au lieu de
  // dériver (plus besoin de figer le budget). AUC 0,714 (vs 0,721 avec les instables). Gate p_pv + seuils v3d.
  feats: ['decisiveness', 'betaMcts', 'betaVar', 'betaWdl', 'entPol'],
  mean: [0.6945716170921967, 0.09861491333911031, 0.02068979694686996, 0.04930745750283288, 2.174022416505559],
  std: [0.22425375828914235, 0.14392108901977696, 0.051006589647209125, 0.07196054616111408, 1.0117352373805055],
  coef: [0.49386173952453677, 0.4473380161722358, -0.0071752717397561465, 0.44733987072036385, -0.45915152120748987],
  intercept: -0.3240707226778902,
  natA: 0.5, natB: 0.75,
  tense: 0.3154243428431859, sharp: 0.4497706764987849,
};

/**
 * @param {object} root racine MCTS (root.wdl + children[{P, node:{N,Q,wdl}}])
 * @returns {{ score:number, level:'calm'|'tense'|'sharp', features:object|null }}
 *   score = probabilité calibrée de « position critique » ∈ [0,1] ; level = 3 crans (tertiles).
 */
export function computeCriticality(root) {
  const all = root && root.children;
  if (!all || all.length < 2) return { score: 0, level: 'calm', features: null };
  const ch = all.filter((c) => c.node && c.node.N > 0);
  if (ch.length < 2) return { score: 0, level: 'calm', features: null };
  let pv = ch[0]; for (const c of ch) if (c.node.N > pv.node.N) pv = c;
  const sc = (w) => w[0] + 0.5 * w[1];        // espérance de score
  const sw = (w) => [w[2], w[1], w[0]];       // remontée d'un niveau (échange W↔L)
  const refV = -pv.node.Q, refE = sc(sw(pv.node.wdl));
  let psum = 0; for (const c of ch) psum += (c.P || 0);
  let betaMcts = 0, betaWdl = 0, mean = 0;
  for (const c of ch) {
    const p = (c.P || 0) / (psum || 1), vm = -c.node.Q;
    betaMcts += p * Math.max(0, refV - vm);
    betaWdl += p * Math.max(0, refE - sc(sw(c.node.wdl)));
    mean += p * vm;
  }
  let betaVar = 0;
  for (const c of ch) { const p = (c.P || 0) / (psum || 1), vm = -c.node.Q; betaVar += p * (vm - mean) * (vm - mean); }
  // entropie de POLICY : entropie du PRIOR réseau sur TOUS les coups légaux (pas les visites) → SIM-INVARIANTE.
  let entPol = 0; for (const c of all) { const p = c.P || 0; if (p > 0) entPol += -p * Math.log(p) / Math.LN2; }
  const decisiveness = root.wdl ? 1 - root.wdl[1] : 0;
  const f = { decisiveness, betaMcts, betaVar, betaWdl, entPol };
  const m = CRIT_MODEL;
  let logit = m.intercept;
  for (let i = 0; i < m.feats.length; i++) logit += m.coef[i] * ((f[m.feats[i]] - m.mean[i]) / m.std[i]);
  const base = 1 / (1 + Math.exp(-logit));
  // AND-gate de naturalité : un coup ÉVIDENT (policy très confiante = prior du PV élevé) n'est PAS
  // « critique à réfléchir », même si un seul coup tient (recapture forcée). p_pv = prior du coup le plus visité.
  const pPv = pv.P || 0;
  const notObvious = Math.max(0, Math.min(1, 1 - (pPv - m.natA) / (m.natB - m.natA)));
  const score = base * notObvious;
  const level = score >= m.sharp ? 'sharp' : score >= m.tense ? 'tense' : 'calm';
  return { score, level, features: { ...f, pPv, notObvious } };
}
