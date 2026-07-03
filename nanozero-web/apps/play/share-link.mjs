// share-link.mjs — lien défi partageable (J4). 100 % client : encode le RÉGLAGE de la partie
// (couleur + niveau) dans l'URL `/play/?color=&level=` pour qu'un ami rejoue la même config et
// compare son résultat. Pas de compte, pas de cookie, pas de serveur (AC-7).
//
// Périmètre MVP : partage du RÉGLAGE. Le partage d'une POSITION précise (FEN milieu de partie +
// seed) est une évolution future — il faudra alors faire démarrer startGame() depuis un FEN arbitraire.

const COLORS = new Set(['white', 'black', 'random']);
const LEVELS = new Set(['chill', 'club', 'full']);

/** Construit l'URL de défi. color 'random' (ou absent) → omis = couleur tirée à l'ouverture (AC-5). */
export function buildShareUrl({ color, level }) {
  const u = new URL('/play/', location.origin);
  if (color && color !== 'random' && COLORS.has(color)) u.searchParams.set('color', color);
  if (level && LEVELS.has(level)) u.searchParams.set('level', level);
  return u.toString();
}

/** Lit les params de défi à l'ouverture. Retourne {color, level} (valeurs valides only) ou null. */
export function readShareParams() {
  const p = new URLSearchParams(location.search);
  const color = p.get('color');
  const level = p.get('level');
  if (!color && !level) return null;
  return {
    color: COLORS.has(color) ? color : null,
    level: LEVELS.has(level) ? level : null,
  };
}

/** Copie l'URL dans le presse-papier. Dégrade : clipboard API → textarea+execCommand → prompt (AC-6). */
export async function copyShareLink(url) {
  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(url);
      return true;
    }
  } catch { /* fallback ci-dessous */ }
  try {
    const ta = document.createElement('textarea');
    ta.value = url;
    ta.setAttribute('readonly', '');
    ta.style.position = 'fixed';
    ta.style.opacity = '0';
    document.body.appendChild(ta);
    ta.select();
    const ok = document.execCommand('copy');
    document.body.removeChild(ta);
    if (ok) return true;
  } catch { /* dernier recours */ }
  try { window.prompt('Copie ce lien :', url); return true; } catch { return false; }
}
