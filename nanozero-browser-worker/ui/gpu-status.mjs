// Statut GPU affiché à l'utilisateur (transparence FR8) + aide actionnable au repli CPU.
// Pur (aucun accès DOM/navigator) → testable : l'appelant fournit les faits observés.

/**
 * Décide le libellé « Ressources » et, en repli CPU, un conseil actionnable.
 *
 * Le repli CPU le plus courant sur Linux vient de Chrome : WebGPU n'expose un adaptateur « core »
 * que via le backend Vulkan, désactivé par défaut → `requestAdapter()` renvoie null alors même que
 * `chrome://gpu` affiche « WebGPU: Hardware accelerated ». On guide l'utilisateur vers le flag.
 *
 * @param {object} facts
 * @param {boolean} facts.hasWebGpuApi  `!!navigator.gpu` (l'API existe dans ce navigateur).
 * @param {boolean} facts.hasAdapter    un adaptateur a réellement été obtenu via `requestAdapter()`.
 * @param {string}  [facts.platform]    plateforme (`navigator.userAgentData?.platform` ou `navigator.platform`).
 * @returns {{ gpuOk: boolean, hw: string, hint: (string|null) }}
 */
export function gpuStatus({ hasWebGpuApi, hasAdapter, platform = '' }) {
  const gpuOk = !!hasAdapter;
  const hw = gpuOk ? 'GPU actif (rapide)' : 'CPU seulement (plus lent)';
  if (gpuOk) return { gpuOk, hw, hint: null };

  if (!hasWebGpuApi) {
    return { gpuOk, hw, hint: "Ce navigateur ne gère pas WebGPU : la contribution utilise le CPU (plus lent). Un navigateur à jour (Chrome, Edge) permet d'utiliser le GPU." };
  }
  if (/linux/i.test(platform)) {
    return { gpuOk, hw, hint: 'GPU non exposé à WebGPU. Sous Linux (Chrome), active le flag chrome://flags/#enable-vulkan puis relance le navigateur pour contribuer via le GPU.' };
  }
  return { gpuOk, hw, hint: 'Aucun GPU compatible WebGPU détecté : la contribution utilise le CPU (plus lent).' };
}
