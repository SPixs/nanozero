// build/esbuild-play.mjs — bundles de la surface « Jouer ».
// Usage : `npm run build:play` (depuis nanozero-web/) ou `node build/esbuild-play.mjs`.
// Deux sorties (sorties gitignorées) :
//   - app.bundle.js   : thread PRINCIPAL (UI + échiquier interactif → game-state/chessops). ESM.
//   - worker.bundle.js: Web Worker (MCTS + ORT-Web). IIFE (classic worker), comme le worker de contribution.
// Séparation : le board (légalité des coups) tourne sur le thread principal ; l'inférence dans le worker.

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import * as esbuild from 'esbuild';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const outdir = resolve(root, 'apps/play/dist');

await esbuild.build({
  entryPoints: [resolve(root, 'apps/play/app.mjs')],
  bundle: true,
  format: 'esm',
  outfile: resolve(outdir, 'app.bundle.js'),
  logLevel: 'info',
});

await esbuild.build({
  entryPoints: [resolve(root, 'apps/play/worker-entry.mjs')],
  bundle: true,
  format: 'iife',
  outfile: resolve(outdir, 'worker.bundle.js'),
  logLevel: 'info',
});

console.log('build:play OK → apps/play/dist/{app,worker}.bundle.js');
