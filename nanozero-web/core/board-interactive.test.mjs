// Tests de la logique PURE de board-interactive.mjs (F3 AC-9, sans DOM).
// La partie DOM (pointer events, ghost, modal) est vérifiée en navigateur (AC-10).
// `node --test` depuis nanozero-web/.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { GameState } from './game-state.mjs';
import { fenToCells } from './board-render.mjs';
import {
  algForCell,
  sideToMoveFromFen,
  legalTargetsFrom,
  kingSquareForSide,
} from './board-interactive.mjs';

const START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1';

test('algForCell : ordre a8..h1', () => {
  assert.equal(algForCell(0), 'a8');
  assert.equal(algForCell(7), 'h8');
  assert.equal(algForCell(56), 'a1');
  assert.equal(algForCell(63), 'h1');
});

test('sideToMoveFromFen', () => {
  assert.equal(sideToMoveFromFen(START), 'w');
  assert.equal(sideToMoveFromFen('8/8/8/8/8/8/8/8 b - - 0 1'), 'b');
});

test('startpos : 20 coups légaux au total (AC-9)', () => {
  const gs = new GameState(START);
  let total = 0;
  for (let i = 0; i < 64; i++) total += legalTargetsFrom(gs, algForCell(i)).targets.length;
  assert.equal(total, 20);
});

test('startpos : e2 → {e3, e4}', () => {
  const gs = new GameState(START);
  const { targets } = legalTargetsFrom(gs, 'e2');
  assert.deepEqual(targets.sort(), ['e3', 'e4']);
});

test('promotion : pion a7 → a8 expose 4 lettres (q,r,b,n)', () => {
  const gs = new GameState('4k3/P7/8/8/8/8/8/4K3 w - - 0 1');
  const { targets, promo } = legalTargetsFrom(gs, 'a7');
  assert.ok(targets.includes('a8'));
  assert.ok(promo.a8 && promo.a8.length === 4);
  assert.ok(promo.a8.includes('q'));
});

test('case adverse non sélectionnable : aucun coup depuis une case noire (trait blanc)', () => {
  const gs = new GameState(START);
  // e7 est un pion noir ; depuis le trait blanc, legalTargetsFrom ne renvoie rien partant de e7.
  assert.equal(legalTargetsFrom(gs, 'e7').targets.length, 0);
});

test('kingSquareForSide : rois sur startpos', () => {
  const cells = fenToCells(START);
  assert.equal(kingSquareForSide(cells, 'w'), 'e1');
  assert.equal(kingSquareForSide(cells, 'b'), 'e8');
});

test('GameState.isInCheck : startpos non, position d\'échec oui', () => {
  assert.equal(new GameState(START).isInCheck(), false);
  // Trait noir, roi noir e8 en échec par la dame blanche e2 (colonne e ouverte).
  const inCheck = new GameState('4k3/8/8/8/8/8/4Q3/4K3 b - - 0 1');
  assert.equal(inCheck.isInCheck(), true);
});
