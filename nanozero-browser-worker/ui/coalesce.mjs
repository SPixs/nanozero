// coalesce.mjs — coalesceur d'événements per-ply (Story C.2, NFR4).
// Le Web Worker émet un event par COUP × N parties → spam. On garde le DERNIER état par clé
// (gameId) et on flushe par lots à ≤ 2-4 fps → l'UI reste fluide et le postMessage ne vole pas
// de CPU au self-play. Pur (scheduler injectable → testable sans timers réels).

/**
 * @param {object} opts
 * @param {(batch:Array)=>void} opts.emit  reçoit le lot des derniers états (un par clé) à chaque flush.
 * @param {number} [opts.intervalMs=300]  fenêtre de coalescing (~3 fps).
 * @param {(fn:Function, ms:number)=>any} [opts.schedule]  planificateur (défaut setTimeout) — injectable en test.
 * @returns {{push:(key:string, val:any)=>void, flush:()=>void, pendingSize:()=>number}}
 */
export function createCoalescer({ emit, intervalMs = 300, schedule = (fn, ms) => setTimeout(fn, ms) } = {}) {
  const pending = new Map(); // clé -> dernier état (le plus récent écrase)
  let scheduled = false;

  const flush = () => {
    scheduled = false;
    if (pending.size === 0) return;
    const batch = [...pending.values()];
    pending.clear();
    emit(batch);
  };

  const push = (key, val) => {
    pending.set(key, val); // latest-wins par clé
    if (!scheduled) {
      scheduled = true;
      schedule(flush, intervalMs);
    }
  };

  return { push, flush, pendingSize: () => pending.size };
}
