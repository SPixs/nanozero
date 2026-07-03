// Tests board-render.mjs (F3 AC-8). `node --test` depuis nanozero-web/.
// ⚠️ Réconciliation : le vrai fenToCells renvoie {piece, dark} en ordre a8..h1 (≠ {square,piece,color}
// a1=0 décrit dans l'AC) et THROW sur FEN invalide (≠ 64 cases vides). On ne change PAS sa signature
// (partagée avec le worker live, règle déphasage) ; on teste le comportement RÉEL.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { fenToCells } from './board-render.mjs';

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

test('startpos : 64 cellules, 32 occupées + 32 vides', () => {
  const cells = fenToCells(START);
  assert.equal(cells.length, 64);
  assert.equal(cells.filter((c) => c.piece).length, 32);
  assert.equal(cells.filter((c) => !c.piece).length, 32);
});

test('ordre a8..h1 : coins corrects', () => {
  const cells = fenToCells(START);
  assert.equal(cells[0].piece, 'bR'); // a8
  assert.equal(cells[4].piece, 'bK'); // e8
  assert.equal(cells[60].piece, 'wK'); // e1
  assert.equal(cells[63].piece, 'wR'); // h1
  assert.equal(cells[56].dark, true); // a1 = case sombre
});

test('FEN milieu de partie (1.e4 e5 2.Cf3 Cc6, aucune prise) : 32 pièces', () => {
  const cells = fenToCells('r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1');
  assert.equal(cells.length, 64);
  assert.equal(cells.filter((c) => c.piece).length, 32);
});

test('FEN invalide : throw (comportement réel, géré par board-interactive)', () => {
  assert.throws(() => fenToCells('pas-un-fen'));
  assert.throws(() => fenToCells('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP')); // 7 rangées
});
