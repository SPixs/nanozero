// build/esbuild-sts.mjs — bundles de la page bench STS. `npm run build:sts`.
//   sts.bundle.js        : thread principal (parse EPD + score). ESM.
//   sts-worker.bundle.js : Web Worker (ORT + MCTS, meilleur coup). IIFE classic worker.

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import * as esbuild from 'esbuild';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const outdir = resolve(root, 'apps/sts/dist');

await esbuild.build({ entryPoints: [resolve(root, 'apps/sts/probe.mjs')], bundle: true, format: 'esm',
  outfile: resolve(outdir, 'sts.bundle.js'), logLevel: 'info' });
await esbuild.build({ entryPoints: [resolve(root, 'apps/sts/sts-worker.mjs')], bundle: true, format: 'iife',
  outfile: resolve(outdir, 'sts-worker.bundle.js'), logLevel: 'info' });

console.log('build:sts OK → apps/sts/dist/{sts,sts-worker}.bundle.js');
