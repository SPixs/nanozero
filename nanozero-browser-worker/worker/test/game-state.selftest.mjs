// Self-test de l'adaptateur chess.js -> encodeurs.

import { GameState } from '../src/game-state.mjs';
import { parseFen } from '../src/encoding/fen.mjs';
import { encodePlanes } from '../src/encoding/plane-encoding.mjs';
import { encode, WHITE, BLACK } from '../src/encoding/move-encoding.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };
const firstDiff = (a, b, n = a.length) => { for (let i = 0; i < n; i++) if (a[i] !== b[i]) return i; return -1; };

// 1. Startpos : 20 coups + planes adaptateur == planes FEN (extraction bitboard identique)
const gs = new GameState();
check(gs.legalMoves().length === 20, 'startpos : 20 coups légaux');
const gsPlanes = encodePlanes(gs.toEncoderState());
const fenPlanes = encodePlanes(parseFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'));
check(firstDiff(gsPlanes, fenPlanes) === -1, 'startpos : planes adaptateur == planes FEN');

// 2. e2e4 -> index 877
const e4mv = gs.legalMoves().find((m) => m.san === 'e4');
check(!!e4mv && encode(e4mv, WHITE) === 877, 'e2e4 -> index 877');

// 3. Après 1.e4 : trait Noirs, t=0 == position e4, t=1 = historique startpos
gs.applyMove(e4mv);
check(gs.sideToMove() === BLACK, 'après e4 : trait aux Noirs');
const e4Planes = encodePlanes(gs.toEncoderState());
const fenE4 = encodePlanes(parseFen('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'));
check(firstDiff(e4Planes, fenE4, 14 * 64) === -1, 'après e4 : t=0 (plans 0-13) == position e4');
const t1NonZero = Array.from(e4Planes.subarray(14 * 64, 28 * 64)).some((x) => x !== 0);
const fenT1Zero = Array.from(fenE4.subarray(14 * 64, 28 * 64)).every((x) => x === 0);
check(t1NonZero && fenT1Zero, 'après e4 : t=1 = historique (adaptateur) vs vide (FEN 1-position)');

// 4. Répétition : Nf3 Nf6 Ng1 Ng8 -> retour startpos -> rep1
const gs2 = new GameState();
for (const san of ['Nf3', 'Nf6', 'Ng1', 'Ng8']) gs2.applyMove(san);
const h0 = gs2.toEncoderState().history[0];
check(h0.rep1 === true, 'répétition : retour startpos (2e occurrence) -> rep1 = true');
check(h0.rep2 === false, 'répétition : pas encore 3e occurrence -> rep2 = false');

// 5. Tous les coups légaux encodent vers des indices valides et distincts
const gs3 = new GameState('r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'); // Kiwipete
const side = gs3.sideToMove();
const idxs = gs3.legalMoves().map((m) => encode(m, side));
check(idxs.every((i) => i >= 0 && i < 4672), 'Kiwipete : tous les indices dans [0,4672)');
check(new Set(idxs).size === idxs.length, `Kiwipete : ${idxs.length} coups -> indices tous distincts`);

if (fail === 0) console.log('\n✅ game-state self-test : TOUT VERT');
else { console.error(`\n❌ ${fail} échec(s)`); process.exit(1); }
