// fast-board-fens.mjs — Single source of truth des FENs durs du filet FastBoard (Story 0+).
// Étend les 6 FENs de cross-validate.mjs avec les positions silent-poison de la SPEC :
//   ep réel jouable, ep-convention-Java divergent (ep dans le FEN mais non-capturable),
//   roque-sous-échec, sous-promotion, séquences 3-fold.
// ⚠️ NE PAS modifier les FENs/UCIs/comptes sans rejouer perft + invariant + cross-validate.

// --- Oracles perft (constantes d'échecs UNIVERSELLES, vérifiables indépendamment) ---
// startpos & "Kiwipete" (Position 2 des standards perft chessprogramming.org).
export const PERFT_CASES = [
  {
    label: 'startpos',
    fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    // chessprogramming.org : 20 / 400 / 8902 / 197281 / 4865609 ...
    expected: { 1: 20, 2: 400, 3: 8902, 4: 197281 },
    gateDepth: 4,
  },
  {
    label: 'pos2-kiwipete',
    fen: 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
    // chessprogramming.org Position 2 : 48 / 2039 / 97862 / 4085603 ...
    expected: { 1: 48, 2: 2039, 3: 97862 },
    gateDepth: 3,
  },
];

// --- FENs durs (silent-poison). expectMoves = nb coups légaux attendu (oracle chess.js indep). ---
export const HARD_FENS = [
  // ep RÉEL jouable : pion blanc e5 peut capturer d6 en passant.
  {
    label: 'ep-real-playable',
    fen: 'rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3',
    note: 'ep d6 réellement jouable (e5xd6 e.p.) — la clé de rép DOIT contenir d6',
    epInKey: 'd6',
    epCapturable: true,
  },
  // ep CONVENTION-JAVA divergent : ep b3 dans le FEN mais AUCUN pion ne peut capturer.
  // chess.js n'émettra aucun coup ep, mais la clé de répétition (convention Java) DOIT
  // toujours contenir b3 (sinon plans rép12/13 faux sur positions ep non-capturable).
  {
    label: 'ep-java-divergent-noncapturable',
    fen: '8/8/8/8/1Pp5/8/8/4K1k1 b - b3 0 1',
    note: 'ep b3 PRÉSENT dans la clé même si non capturable (le pion noir c4 capturerait b3 e.p.!)',
    epInKey: 'b3',
    // NB : ici c4 peut PHYSIQUEMENT capturer b3 e.p. -> on vérifie surtout que la clé contient b3.
  },
  // ep dans le FEN mais STRICTEMENT non capturable (pion isolé loin) — divergence pure.
  {
    label: 'ep-java-divergent-isolated',
    fen: '4k3/8/8/8/5P2/8/8/4K3 b - f3 0 1',
    note: 'ep f3 dans le FEN, AUCUN pion noir adjacent -> chess.js ignore, clé doit garder f3',
    epInKey: 'f3',
    epCapturable: false,
  },
  // roque pseudo-dispo MAIS roi en échec -> roque illégal (roque-sous-échec).
  {
    label: 'castle-while-in-check',
    fen: 'r3k2r/8/8/8/4r3/8/8/R3K2R w KQkq - 0 1',
    note: 'roi blanc e1 cloué par la tour noire e4 -> O-O et O-O-O ILLÉGAUX (sous échec)',
    inCheck: true,
    noCastlingMoves: true,
  },
  // sous-promotion : pion blanc b7, promo b8 (et captures a8/c8). Couvre N/B/R/Q.
  {
    label: 'underpromotion',
    fen: 'n1n1k3/1P6/8/8/8/8/8/4K3 w - - 0 1',
    note: 'b7 promeut en b8 + capture a8/c8 ; teste indices policy sous-promo N/B/R',
    hasUnderpromo: true,
  },
];

// --- Séquences 3-fold (snapshot courant NE doit PAS se compter lui-même à la 1re occurrence) ---
// startpos puis aller-retour cavaliers : la position startpos réapparaît après 4 demi-coups.
export const THREEFOLD_SEQUENCES = [
  {
    label: '3fold-knight-shuffle',
    startFen: null, // startpos
    // g1f3 g8f6 f3g1 f6g8 -> retour startpos (2e occurrence) ; répéter -> 3e occurrence.
    ucis: ['g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1', 'f6g8'],
  },
];

// Toutes les FENs "isolées" (pour le perft léger + invariant make/unmake).
export const ALL_INVARIANT_FENS = [
  // les 6 FENs historiques de cross-validate.mjs (régression)
  { label: 'cv-startpos', fen: 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1' },
  { label: 'cv-kiwipete', fen: 'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1' },
  { label: 'cv-italian', fen: 'rnbq1rk1/pppp1ppp/5n2/2b1p3/2B1P3/5N2/PPPP1PPP/RNBQ1RK1 w - - 6 5' },
  { label: 'cv-promo', fen: 'n3k3/1P6/8/8/8/8/8/4K3 w - - 0 1' },
  { label: 'cv-kk-endgame', fen: '8/8/8/3k4/8/3K4/8/8 w - - 5 30' },
  { label: 'cv-petroff', fen: 'r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 2 2' },
  // les FENs durs Story 0
  ...HARD_FENS.map((h) => ({ label: h.label, fen: h.fen })),
];
