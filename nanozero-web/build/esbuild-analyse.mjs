// build/esbuild-analyse.mjs — bundles de la surface « Analyse ». `node build/esbuild-analyse.mjs`.
//   app.bundle.js    : thread principal (UI + échiquier + chargement PGN/FEN via chessops). ESM.
//   worker.bundle.js : Web Worker (MCTS + ORT-Web) — ajouté avec S-07 (analyse W/D/L).

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import { existsSync } from 'node:fs';
import * as esbuild from 'esbuild';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const outdir = resolve(root, 'apps/analyse/dist');

await esbuild.build({
  entryPoints: [resolve(root, 'apps/analyse/app.mjs')],
  bundle: true,
  format: 'esm',
  outfile: resolve(outdir, 'app.bundle.js'),
  logLevel: 'info',
});

// Worker optionnel tant que S-07 n'est pas livré.
const workerEntry = resolve(root, 'apps/analyse/worker-entry.mjs');
if (existsSync(workerEntry)) {
  await esbuild.build({
    entryPoints: [workerEntry],
    bundle: true,
    format: 'iife',
    outfile: resolve(outdir, 'worker.bundle.js'),
    logLevel: 'info',
  });
}

console.log('build:analyse OK → apps/analyse/dist/');
