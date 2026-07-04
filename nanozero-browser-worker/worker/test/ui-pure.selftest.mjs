// ui-pure.selftest.mjs — Epic C : cœurs purs (consent SM, coalesce, board-render) + threading onPly.
import { createConsentController } from '../../ui/consent.mjs';
import { createCoalescer } from '../../ui/coalesce.mjs';
import { fenToCells, algToFileRank } from '../../ui/board-render.mjs';
import { gpuStatus } from '../../ui/gpu-status.mjs';
import { runConcurrentContributionLoop } from '../src/run-loop.mjs';

let fail = 0;
const check = (c, m) => { if (!c) { console.error('  ✗ ' + m); fail++; } else console.log('  ✓ ' + m); };

const fakeNet = { evaluateBatch: async (_p, n) => Array.from({ length: n }, () => ({ policy: new Float32Array(4672), value: 0 })) };

console.log('A) consent — pas d\'auto-start, idempotent');
{
  let starts = 0, stops = 0;
  const c = createConsentController({ onStart: () => starts++, onStop: () => stops++ });
  check(c.state === 'idle' && !c.isRunning() && starts === 0, 'A1 état initial idle (RIEN ne démarre)');
  check(c.start() === true && c.state === 'running' && starts === 1, 'A2 start → running + onStart');
  check(c.start() === false && starts === 1, 'A3 double start = no-op');
  check(c.stop() === true && c.state === 'idle' && stops === 1, 'A4 stop → idle + onStop');
  check(c.stop() === false && stops === 1, 'A5 double stop = no-op');
  // reset() : fin naturelle (abort/erreur) → idle SANS onStop, et start() remarche (anti bouton-mort)
  c.start(); // running
  c.reset();
  check(c.state === 'idle' && stops === 1, 'A6 reset → idle sans onStop (stops inchangé)');
  check(c.start() === true && c.state === 'running', 'A7 start() remarche après reset (pas de bouton mort)');
}

console.log('B) coalesce — latest par clé, flush par lot, schedule une fois');
{
  const batches = [];
  let scheduled = 0;
  const co = createCoalescer({ emit: (b) => batches.push(b), schedule: () => { scheduled++; } });
  co.push('a', 1); co.push('a', 2); co.push('b', 3); // a écrasé par 2
  check(scheduled === 1, 'B1 flush planifié UNE seule fois pour la rafale');
  check(co.pendingSize() === 2, 'B2 2 clés en attente (a,b), pas 3');
  co.flush();
  check(batches.length === 1 && JSON.stringify(batches[0]) === '[2,3]', 'B3 flush émet le DERNIER par clé ([2,3])');
  co.flush();
  check(batches.length === 1, 'B4 flush à vide = no-op');
}

console.log('C) board-render — FEN startpos → 64 cellules, damier, codes de pièce cburnett');
{
  const cells = fenToCells('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');
  check(cells.length === 64, 'C1 64 cellules');
  check(cells[0].piece === 'bR' && cells[0].dark === false, 'C2 a8 = tour NOIRE (bR), case claire');
  check(cells[56].dark === true, 'C3 a1 = case sombre');
  check(cells[60].piece === 'wK', 'C4 e1 = roi BLANC (wK)');
  check(cells.filter((c) => c.piece !== '').length === 32, 'C5 32 pièces');
  let threw = false; try { fenToCells('bad/fen'); } catch { threw = true; }
  check(threw, 'C6 FEN invalide → throw');
  // algToFileRank (positionnement de la flèche du dernier coup)
  check(JSON.stringify(algToFileRank('e2')) === '{"file":4,"rank":1}', 'C7 algToFileRank(e2)={4,1}');
  check(JSON.stringify(algToFileRank('a1')) === '{"file":0,"rank":0}', 'C8 algToFileRank(a1)={0,0}');
  check(algToFileRank('z9') === null && algToFileRank('') === null, 'C9 case invalide → null');
}

console.log('D) threading onPly — la boucle préfixe le gameId (job_id) à l\'event per-ply');
{
  const events = [];
  let submitted = 0;
  const client = {
    claim: async () => ({ job_id: 'game-7', model_version: 30 }),
    submitGame: async () => { submitted++; return {}; },
  };
  // fake playGame : émet 2 events per-ply puis rend un sample
  const playGame = async (_ev, opts) => {
    opts.onPly?.({ ply: 1, fen: 'fen-1' });
    opts.onPly?.({ ply: 2, fen: 'fen-2' });
    return { samples: [{ planes: new Float32Array(1), policy: new Float32Array(1), value: 0 }] };
  };
  await runConcurrentContributionLoop({
    client, net: fakeNet, poolSize: 1, playGame,
    onPly: (slot, info) => events.push({ slot, ...info }),
    shouldStop: () => submitted >= 1,
  });
  check(events.length >= 2, 'D1 events per-ply reçus');
  check(events.every((e) => e.slot === 0), 'D2 slotIndex STABLE (0) en clé (recyclé d\'une partie à l\'autre)');
  check(events.every((e) => e.gameId === 'game-7'), 'D2b gameId dans le payload (label)');
  check(events[0].ply === 1 && events[0].fen === 'fen-1', 'D3 payload {ply, fen} transmis');
}

console.log('E) gpu-status — libellé Ressources + aide au repli CPU (Linux/Vulkan)');
{
  const ok = gpuStatus({ hasWebGpuApi: true, hasAdapter: true, platform: 'Linux x86_64' });
  check(ok.gpuOk === true && ok.hw === 'GPU actif (rapide)' && ok.hint === null, 'E1 adaptateur présent → GPU actif, aucun conseil');

  const linux = gpuStatus({ hasWebGpuApi: true, hasAdapter: false, platform: 'Linux x86_64' });
  check(linux.gpuOk === false && linux.hw === 'CPU seulement (plus lent)', 'E2 Linux sans adaptateur → repli CPU');
  check(/enable-vulkan/.test(linux.hint), 'E3 Linux sans adaptateur → conseil chrome://flags/#enable-vulkan');

  const noApi = gpuStatus({ hasWebGpuApi: false, hasAdapter: false, platform: 'Win32' });
  check(!noApi.gpuOk && /WebGPU/.test(noApi.hint) && !/vulkan/i.test(noApi.hint), 'E4 pas d\'API WebGPU → conseil « mettre à jour », pas Vulkan');

  const otherOs = gpuStatus({ hasWebGpuApi: true, hasAdapter: false, platform: 'MacIntel' });
  check(!otherOs.gpuOk && otherOs.hint !== null && !/vulkan/i.test(otherOs.hint), 'E5 non-Linux sans adaptateur → conseil générique (sans Vulkan)');

  const noPlatform = gpuStatus({ hasWebGpuApi: true, hasAdapter: false });
  check(!noPlatform.gpuOk && noPlatform.hint !== null, 'E6 plateforme absente → repli CPU avec conseil générique (pas de throw)');
}

console.log(fail === 0 ? '\nOK — ui-pure selftest vert' : `\nFAIL — ${fail} check(s)`);
process.exit(fail ? 1 : 0);
