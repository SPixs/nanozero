// strength-levels.mjs — 3 niveaux de Nano (J1). sims VALIDÉS côté latence (spike F4) ; FORCE en cours
// de calage par le playtest (D2). Forward-compatible : `measuredElo` rempli en P1 (calibration) sans
// changer l'interface ; AUCUN Elo affiché tant que measuredElo === null. Itérer ce tableau, ne pas hardcoder.
//
// ⚠️ addNoise = FALSE partout : le bruit de Dirichlet est un outil de SELF-PLAY (exploration) ; en partie
// réelle il fait jouer des coups au hasard → « cadeaux » (playtest 2026-06-29). On affaiblit par sims
// réduits (profondeur tactique limitée) + température modérée (légère variété), JAMAIS par du hasard.
// Un adversaire faible vraiment « humain » = chantier human-fidelity (P2), hors MVP.

// Calibration fastchess FIGÉE 2026-06-30 (bridge UCI = moteur web exact, gen-031, échelle par voisins
// ancrée TSCP sd6 ≈ 1700 ; ±~110 Elo absolu, espacement relatif solide). Résultat clé : la TEMPÉRATURE
// 0,07–0,30 est quasi gratuite (≤ ~60 Elo, souvent dans le bruit ; confirmé STS) → SEUL le nb de sims
// affaiblit. « Tranquille » trivial = l'ancien 48 sims (~1180), pas la température. Plancher fixé à 120
// sims (~1400) : accessible sans être trivial, + chill plus rapide sur mobile. measuredElo = Elo de JEU
// (échelle moteur/TSCP, PAS Lichess/FIDE). Détail : bench/calibrate-report.mjs sur /tmp/calib/results.tsv.
export const STRENGTH_LEVELS = [
  { id: 'chill', label: 'Tranquille',     desc: 'Battable, mais plus si évident', sims: 120, temperature: 0.15, addNoise: false, measuredElo: 1400 },
  { id: 'club',  label: 'Joueur de club', desc: 'Adversaire solide',              sims: 300, temperature: 0.07, addNoise: false, measuredElo: 1660 },
  { id: 'full',  label: 'Pleine force',   desc: 'Nano à fond',                    sims: 800, temperature: 0,    addNoise: false, measuredElo: 1905 },
];

export function levelById(id) {
  return STRENGTH_LEVELS.find((l) => l.id === id) || STRENGTH_LEVELS[0];
}
