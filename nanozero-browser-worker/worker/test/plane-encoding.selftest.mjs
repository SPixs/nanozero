// Self-test du plane-encoder : layout + conventions P1/P2 + flip Noirs.
// ⚠️ Cohérence interne ; la parité bit-à-bit vs Java viendra de la cross-val positions réelles.

import { encodePlanes } from '../src/encoding/plane-encoding.mjs';
import { parseFen } from '../src/encoding/fen.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };
const plane = (planes, p) => planes.subarray(p * 64, p * 64 + 64);
const bits = (pl) => { const s = []; for (let i = 0; i < 64; i++) if (pl[i]) s.push(i); return s; };
const all = (pl, v) => pl.every((x) => x === v);

// --- Startpos, Blancs au trait : P1 = blanc, pas de flip ---
const sp = encodePlanes(parseFen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'));
check(bits(plane(sp, 0)).join() === [8, 9, 10, 11, 12, 13, 14, 15].join(), 'startpos P1 pions = rang 2 (8-15)');
check(bits(plane(sp, 5)).join() === '4', 'startpos P1 roi = e1 (4)');
check(bits(plane(sp, 3)).join() === [0, 7].join(), 'startpos P1 tours = a1,h1 (0,7)');
check(bits(plane(sp, 6)).join() === [48, 49, 50, 51, 52, 53, 54, 55].join(), 'startpos P2 pions = rang 7 (48-55)');
check(bits(plane(sp, 11)).join() === '60', 'startpos P2 roi = e8 (60)');
check(all(plane(sp, 112), 1), 'startpos color = 1 (P1 blanc)');
check(all(plane(sp, 114), 1) && all(plane(sp, 115), 1) && all(plane(sp, 116), 1) && all(plane(sp, 117), 1), 'startpos castling 4 plans = 1');
check(Math.abs(plane(sp, 113)[0] - 1 / 99) < 1e-7, 'startpos fullmove = 1/99');
check(all(plane(sp, 118), 0), 'startpos halfmove = 0');
check(all(plane(sp, 14), 0) && all(plane(sp, 111), 0), 'startpos timestamps t1..t7 vides');

// --- Après 1.e4, Noirs au trait : P1 = noir, FLIP vertical ---
const e4 = encodePlanes(parseFen('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'));
check(bits(plane(e4, 0)).join() === [8, 9, 10, 11, 12, 13, 14, 15].join(), 'e4: P1(noir) pions flippés = rang 2 (8-15)');
check(bits(plane(e4, 5)).join() === '4', 'e4: P1(noir) roi e8 flippé = 4');
check(all(plane(e4, 112), 0), 'e4: color = 0 (P1 noir)');
check(plane(e4, 6)[36] === 1, 'e4: P2(blanc) pion e4 flippé -> bit 36 (e5)');
check(plane(e4, 6)[52] === 0 && plane(e4, 6)[51] === 1, 'e4: P2 pion e absent du rang 7-flip, d présent');
check(Math.abs(plane(e4, 118)[0] - 0) < 1e-7, 'e4: halfmove = 0');

// --- Compteur no-progress + fullmove normalisés ---
const mid = encodePlanes(parseFen('8/8/8/4k3/4K3/8/8/8 w - - 40 60'));
check(Math.abs(plane(mid, 118)[0] - 0.4) < 1e-7, 'no-progress 40 -> 0.40');
check(Math.abs(plane(mid, 113)[0] - 60 / 99) < 1e-7, 'fullmove 60 -> 60/99');
check(all(plane(mid, 114), 0) && all(plane(mid, 116), 0), 'pas de roque -> plans castling = 0');

if (fail === 0) console.log('\n✅ plane-encoding self-test : TOUT VERT');
else { console.error(`\n❌ ${fail} échec(s)`); process.exit(1); }
