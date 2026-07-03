// Tests complexity.mjs (J2 T2.3). `node --test core/*.test.mjs` depuis nanozero-web/.
// Racine factice : computeComplexity lit children[].P et children[].node.{N,Q} (Q POV enfant/adverse).
// N>0 obligatoire : seuls les coups réellement explorés comptent (fix bas-budget 2026-06-29).

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { computeComplexity } from './complexity.mjs';

test('un seul coup légal → calm (beta 0)', () => {
  const r = { children: [{ P: 1, node: { N: 48, Q: 0 } }] };
  assert.deepEqual(computeComplexity(r), { beta: 0, level: 'calm' });
});

test('tous les coups équivalents → calm', () => {
  const r = { children: [
    { P: 0.5, node: { N: 25, Q: -0.20 } },
    { P: 0.5, node: { N: 23, Q: -0.18 } },
  ] };
  assert.equal(computeComplexity(r).level, 'calm');
});

test('4 coups policy uniforme, 3 perdants (regret élevé) → sharp', () => {
  // val d'un coup (POV trait) = -Q enfant. PV (le plus visité) = +0.5 ; 3 autres = -0.3 → regret 0.8.
  const r = { children: [
    { P: 0.25, node: { N: 30, Q: -0.5 } },
    { P: 0.25, node: { N: 6, Q: 0.3 } },
    { P: 0.25, node: { N: 6, Q: 0.3 } },
    { P: 0.25, node: { N: 6, Q: 0.3 } },
  ] };
  const c = computeComplexity(r);
  assert.ok(c.beta >= 0.25, `beta=${c.beta}`);
  assert.equal(c.level, 'sharp');
});

test('seuils personnalisés respectés', () => {
  const r = { children: [
    { P: 0.5, node: { N: 20, Q: -0.3 } }, // PV (val +0.3)
    { P: 0.5, node: { N: 10, Q: -0.1 } }, // val +0.1 → regret 0.2
  ] }; // beta = 0.5*0.2 = 0.1
  assert.equal(computeComplexity(r, { calm: 0.05, tense: 0.5 }).level, 'tense');
  assert.equal(computeComplexity(r, { calm: 0.2, tense: 0.5 }).level, 'calm');
});

test('bas budget : coups non visités (N=0) + coup peu visité bruité ignorés → pas de faux « tranchante »', () => {
  // Position CALME : 2 coups bien explorés et équivalents. Mais un coup visité 1 seule fois a une
  // valeur bruitée (val +0.6) et 18 coups ne sont pas visités (N=0, Q=0 défaut). L'ancien calcul
  // (max brut + somme sur tous) → faux « sharp ». Le nouveau (PV + visités) → « calm ».
  const children = [
    { P: 0.45, node: { N: 22, Q: -0.04 } }, // PV calme
    { P: 0.30, node: { N: 14, Q: -0.03 } },
    { P: 0.05, node: { N: 1, Q: -0.6 } },   // 1 rollout chanceux → val +0.6 (piège du max)
  ];
  for (let i = 0; i < 18; i++) children.push({ P: 0.0111, node: { N: 0, Q: 0 } });
  const c = computeComplexity({ children });
  assert.equal(c.level, 'calm', `beta=${c.beta}`);
});
