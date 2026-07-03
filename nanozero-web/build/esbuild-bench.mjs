// build/esbuild-bench.mjs — bundle de la page de bench latence (F4).
// Usage : `npm run build:bench`. ORT-Web vient du CDN (global), donc EXTERNE au bundle.

import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';
import * as esbuild from 'esbuild';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

await esbuild.build({
  entryPoints: [resolve(root, 'apps/bench/probe.mjs')],
  bundle: true,
  format: 'esm',
  outfile: resolve(root, 'apps/bench/dist/probe.bundle.js'),
  logLevel: 'info',
});

console.log('build:bench OK → apps/bench/dist/probe.bundle.js');
