// consent.mjs — machine d'état du consentement volontaire (Story C.1, FR8).
// PAS d'auto-start : la page ne calcule RIEN tant que start() n'est pas appelé (clic « Contribuer »).
// Idempotent : double start / double stop = no-op. Pur (testable sans DOM).

/**
 * @param {object} [hooks]
 * @param {()=>void} [hooks.onStart]  effet de bord au passage idle→running (créer/démarrer le Worker).
 * @param {()=>void} [hooks.onStop]   effet de bord au passage running→idle (arrêter le Worker).
 * @returns {{state:string, start:()=>boolean, stop:()=>boolean, isRunning:()=>boolean}}
 */
export function createConsentController({ onStart, onStop } = {}) {
  let state = 'idle'; // 'idle' | 'running'
  return {
    get state() { return state; },
    isRunning() { return state === 'running'; },
    /** Démarre la contribution (clic explicite). No-op si déjà running. @returns {boolean} a démarré ? */
    start() {
      if (state !== 'idle') return false;
      state = 'running';
      if (onStart) onStart();
      return true;
    },
    /** Arrête proprement (clic Stop). No-op si déjà idle. @returns {boolean} a arrêté ? */
    stop() {
      if (state !== 'running') return false;
      state = 'idle';
      if (onStop) onStop();
      return true;
    },
    /** Repasse à idle SANS effet de bord (onStop) — pour la fin naturelle du worker
     *  (circuit-breaker/erreur sans clic Stop) : sinon l'état resterait 'running' et un
     *  re-clic « Contribuer » serait un no-op (bouton mort jusqu'au reload). */
    reset() {
      state = 'idle';
    },
  };
}
