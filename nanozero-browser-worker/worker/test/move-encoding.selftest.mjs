// Self-test du move-encoding : cohérence interne encode∘decode = identité.
// ⚠️ Ne prouve PAS la parité avec Java (étape suivante = vecteurs de référence Java).
// Prouve : la géométrie encode/decode est bien inverse + bornes correctes.

import {
  encode, decode, fileOf, rankOf,
  WHITE, BLACK, NORMAL, PROMOTION,
  PROMO_KNIGHT, POLICY_INDICES, POLICY_PLANES,
} from '../src/encoding/move-encoding.mjs';

let fail = 0;
const check = (cond, msg) => { if (!cond) { console.error('  ✗ ' + msg); fail++; } };

// --- 1. Round-trip sur tous les indices valides (on-board, sans wrap de fichier) ---
for (const side of [WHITE, BLACK]) {
  let valid = 0, ok = 0;
  for (let idx = 0; idx < POLICY_INDICES; idx++) {
    const fromP1 = Math.floor(idx / POLICY_PLANES);
    const plane = idx % POLICY_PLANES;
    // reproduire le delta P1 pour tester la validité géométrique
    let dR, dF;
    if (plane >= 64) { dF = [-1, 0, 1][Math.floor((plane - 64) / 3)]; dR = +1; }
    else if (plane >= 56) {
      const K = [[2,1],[1,2],[-1,2],[-2,1],[-2,-1],[-1,-2],[1,-2],[2,-1]][plane - 56];
      dR = K[0]; dF = K[1];
    } else {
      const Q = [[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]][Math.floor(plane / 7)];
      const dist = (plane % 7) + 1; dR = Q[0]*dist; dF = Q[1]*dist;
    }
    const toFileP1 = fileOf(fromP1) + dF;
    const toRankP1 = rankOf(fromP1) + dR;
    const onBoard = toFileP1 >= 0 && toFileP1 <= 7 && toRankP1 >= 0 && toRankP1 <= 7;
    if (!onBoard) continue;
    valid++;
    const mv = decode(idx, side);          // pas de position -> queen = NORMAL
    const back = encode(mv, side);
    if (back === idx) ok++;
    else console.error(`  ✗ round-trip side=${side} idx=${idx} -> mv=${JSON.stringify(mv)} -> ${back}`);
  }
  check(ok === valid, `round-trip side=${side}: ${ok}/${valid} OK`);
  console.log(`  side=${side} : ${valid} indices valides (on-board), round-trip ${ok}/${valid}`);
}

// --- 2. Vecteurs connus calculés à la main ---
// e2e4 (Blancs) : from=12 (e2), to=28 (e4), queen N dist2 -> plane 1 -> 12*73+1 = 877
check(encode({ from: 12, to: 28, type: NORMAL, promo: 0 }, WHITE) === 877, 'e2e4 blancs = 877');
// Cavalier g1f3 (Blancs) : from=6 (g1), to=21 (f3). dR=+2,dF=-1 -> KNIGHT_DELTAS[7]=[2,-1] -> plane 56+7=63 -> 6*73+63=501
check(encode({ from: 6, to: 21, type: NORMAL, promo: 0 }, WHITE) === 6 * 73 + 63, 'Cg1f3 blancs');
// Sous-promo cavalier e7e8=N (Blancs) : from=52 (e7), to=60 (e8). dF=0 -> dirIdx 1 ; promo N=0 -> plane 64+1*3+0=67 -> 52*73+67
check(encode({ from: 52, to: 60, type: PROMOTION, promo: PROMO_KNIGHT }, WHITE) === 52 * 73 + 67, 'e7e8=N blancs');
// Symétrie Noirs : e7e5 vu côté Noir == e2e4 côté Blanc (POV side-to-move) -> même index 877
// e7e5 noir : from=52(e7), to=36(e5). flip: from^56=12, to^56=28 -> identique e2e4 -> 877
check(encode({ from: 52, to: 36, type: NORMAL, promo: 0 }, BLACK) === 877, 'e7e5 noirs = e2e4 = 877 (POV)');

if (fail === 0) console.log('\n✅ move-encoding self-test : TOUT VERT');
else { console.error(`\n❌ ${fail} échec(s)`); process.exit(1); }
