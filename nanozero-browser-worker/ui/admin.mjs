// admin.mjs — script de l'espace admin (STORY-012, monitoring beta read-only).
//
// AUTH (AC-1) : cette page DOIT être servie derrière une authentification (Caddy basicauth, T3.1
// Option A). Ce module ne gère AUCUNE auth (pas de saisie de clé, pas de mécanisme maison). Les
// fetch sont RELATIFS à l'origine : le reverse-proxy authentifié sert la page ET proxifie le
// jobserver. Aujourd'hui Caddy ne proxifie pas encore /stats/selfplay ni /stats/browser → ces
// routes renvoient 404 et la page DÉGRADE gracieusement (AC-7) ; le routage+auth viendra ensuite.
//
// READ-ONLY STRICT (AC-4) : ce module n'émet QUE des GET. Aucune action (throttle/ban/purge/toggle).
// ZÉRO PII (AC-5) : on consomme des COMPTEURS (sessions, positions) ; jamais /stats/workers ni un
// quelconque champ pseudo/IP. Les endpoints utilisés n'en exposent pas.

const $ = (id) => document.getElementById(id);

// Cadence de polling (AC-3 : 5 s) avec backoff exponentiel borné en cas d'échec (réseau/404), aligné
// sur le pattern pollSeasonStats de app.mjs.
const BASE_MS = 5000;        // /stats/browser est CACHÉ 60 s côté serveur (STORY-012) → poll fréquent =
                             // surtout des cache-hits instantanés ; un seul recalcul ~20 s par minute.
const MAX_MS = 120000;
let browserDelay = BASE_MS;
let browserTimer = null;
// Repère pour le débit browser dérivé (Δquarantaine / Δt entre mesures distinctes — cf. updateRate).
let ratePrevPb = null, ratePrevT = null;
let everSucceeded = false; // tant qu'aucun succès → afficher « Chargement… » plutôt que « Aucune donnée »

// fetch relatif à l'origine, JSON, sans cache. Rejette sur !res.ok (404 inclus) → la dégradation
// est pilotée par l'appelant. cache:'no-store' = on veut toujours la donnée fraîche.
async function fetchJson(path) {
  const res = await fetch(path, { cache: 'no-store' });
  if (!res.ok) throw new Error('HTTP ' + res.status);
  return res.json();
}

// Formate un entier avec des espaces fines comme séparateurs de milliers (fr-FR), sûr sur null/NaN.
function fmtInt(n) {
  if (n == null || Number.isNaN(Number(n))) return '—';
  return Number(n).toLocaleString('fr-FR');
}

// renderKpi(id, value, delta, status) — met à jour valeur, tendance et COULEUR de statut.
// status ∈ {'ok','warn','bad', null}. Pour la value au format libre (ex. « N/A »), passer une string ;
// pour un compteur, passer un number (formaté). delta = sous-texte optionnel.
// Le coloriage seuils (vert ≥90 / amber 70-90 / rouge <70, AC-6) est porté par statusFromRate()
// et n'est utilisé QUE quand le vrai taux existera (gate D actif) — voir computeVerifKpi().
function renderKpi(id, value, delta, status) {
  const el = $(id);
  if (!el) return;
  const nEl = el.querySelector('[data-n]');
  const dEl = el.querySelector('[data-d]');
  if (nEl) {
    nEl.textContent = typeof value === 'number' ? fmtInt(value) : value;
    // Couleur du grand nombre : on conserve l'accent par défaut de la carte (classe statique dans le
    // HTML : .b/.g) sauf si un statut seuil est fourni (cas taux de vérification réel).
    nEl.classList.remove('ok', 'warn', 'bad');
    if (status) nEl.classList.add(status);
  }
  if (dEl) {
    dEl.textContent = delta == null ? '' : delta;
    dEl.classList.remove('ok', 'warn', 'bad', 'muted');
    dEl.classList.add(status || 'muted');
  }
}

// Coloriage par seuils (AC-6) — prêt pour QUAND le vrai taux de vérification existera (gate D).
function statusFromRate(pct) {
  if (pct == null || Number.isNaN(pct)) return null;
  if (pct >= 90) return 'ok';
  if (pct >= 70) return 'warn';
  return 'bad';
}

// KPI taux de vérification. En MVP le gate anti-poison (D.2) n'est PAS actif → toutes les positions
// browser sont en quarantaine, 0 vérifiée par le gate. On affiche « N/A — gate anti-poison non actif »
// (PAS un 0 % trompeur, cf. Dev Notes), tout en gardant statusFromRate() prêt pour la vraie valeur.
function computeVerifKpi() {
  // Si un jour le serveur expose un vrai ratio vérifiées/soumises, on le branchera ici :
  //   const pct = ...; renderKpi('kpi-verif', pct.toFixed(1) + ' %', 'sain > 90 %', statusFromRate(pct));
  // Tant que le gate n'est pas opérationnel : N/A honnête.
  renderKpi('kpi-verif', 'N/A', 'gate anti-poison non actif', 'warn');
}

// renderCorpusSection — barre browser vs fleet + volume total + ratio (AC-2 d).
function renderCorpusSection(positionsBrowser, positionsFleet, ratio, driftVerdict, effSims) {
  const pb = Number(positionsBrowser) || 0;
  const pf = Number(positionsFleet) || 0;
  const total = pb + pf;

  $('corpus-browser-vol').textContent = fmtInt(pb);

  // Part du browser dans le corpus total (browser vs fleet). Si total nul → barre vide.
  const pct = total > 0 ? (pb / total) * 100 : 0;
  $('corpus-bar').style.width = pct.toFixed(2) + '%';

  $('corpus-detail').textContent = total > 0
    ? `${pct.toFixed(2)} % du corpus total (browser ${fmtInt(pb)} vs flotte ${fmtInt(pf)})`
    : 'Aucune position dans le corpus.';

  // ratio = browser_fleet_ratio renvoyé par le serveur (peut être null) ; sinon dérivé local.
  const r = ratio != null ? ratio : (pf > 0 ? pb / pf : null);
  $('corpus-split').innerHTML =
    `Browser : <b>${fmtInt(pb)}</b> · Flotte native : <b>${fmtInt(pf)}</b>` +
    `<br>Total : <b>${fmtInt(total)}</b> · ratio browser/flotte : <b class="mono">${r != null ? r.toFixed(4) : '—'}</b>`;

  // drift_verdict peut être null → « — » (AC : gérer gracieusement).
  $('drift-verdict').textContent = driftVerdict != null ? driftVerdict : '—';
  $('eff-sims').textContent = effSims != null ? String(effSims) : '—';
}

// Bascule l'indicateur live + le footer selon l'état (connecté / en attente / dégradé).
function setLive(connected, message) {
  const live = $('live');
  if (live) {
    live.classList.toggle('off', !connected);
    live.textContent = connected ? '● live · maj 5 s' : '● en attente';
  }
  if (message != null) $('footer-status').textContent = message;
}

// Affiche l'état dégradé (AC-7) : « Chargement… » au premier essai, « Aucune donnée disponible »
// ensuite, JAMAIS d'exception visible. On ne touche pas aux valeurs déjà affichées si on a déjà eu
// un succès (on garde la dernière vue connue) — sinon placeholders.
function renderDegraded() {
  setLive(false);
  if (everSucceeded) {
    setLive(false, 'Connexion perdue — dernières données affichées. Nouvelle tentative en cours…');
    return;
  }
  const msg = 'Chargement…';
  renderKpi('kpi-sessions', '—', msg, null);
  renderKpi('kpi-rate', '—', msg, null);
  renderKpi('kpi-verif', '—', msg, null);
  renderKpi('kpi-quarantine', '—', msg, null);
  $('corpus-browser-vol').textContent = '—';
  $('corpus-bar').style.width = '0%';
  $('corpus-detail').textContent = msg;
  $('corpus-split').textContent = msg;
  $('drift-verdict').textContent = '—';
  $('eff-sims').textContent = '—';
  $('footer-status').textContent = 'Aucune donnée disponible pour l\'instant.';
}

// Rendu des métriques BROWSER (réponse /stats/browser, rapide) — l'ESSENTIEL pour piloter la beta :
// sessions actives, corpus quarantaine, provenance. Découplé du débit selfplay (lent) → s'affiche en <1 s.
function renderBrowserStats(browser) {
  // (a) Sessions browser actives (24 h) — best-effort côté serveur, c'est un COMPTE (zéro PII).
  renderKpi('kpi-sessions', Number(browser.browser_workers_recent) || 0, browser.paused ? 'ingress en pause' : 'fenêtre 24 h', null);

  // (c) Taux de vérification — N/A tant que le gate D n'est pas actif (coloriage seuils prêt).
  computeVerifKpi();

  // (d) Corpus browser en quarantaine — positions_browser de /stats/browser (provenance fiable).
  const pb = Number(browser.positions_browser) || 0;
  renderKpi('kpi-quarantine', pb, 'en attente du gate', null);

  // (b) Débit browser — DÉRIVÉ du delta de quarantaine (browser-spécifique, sans l'endpoint lourd selfplay).
  updateRate(pb);

  // Section corpus : barre browser vs fleet + total + ratio + drift + eff sims.
  renderCorpusSection(
    browser.positions_browser,
    browser.positions_fleet,
    browser.browser_fleet_ratio,
    browser.drift_verdict,
    browser.median_eff_sims,
  );

  everSucceeded = true;
  setLive(true, 'Données live du jobserver (métriques browser, maj 5 s).');
  $('last-update').textContent = new Date().toLocaleTimeString('fr-FR');
}

// (b) Débit browser DÉRIVÉ : Δquarantaine / Δt entre deux mesures DISTINCTES → positions browser/h.
// Évite l'endpoint lourd /stats/selfplay. Tient compte du cache serveur (la valeur ne change que
// ~toutes les 60 s) : on ne bouge le repère QUE quand le volume croît, donc Δt couvre un vrai incrément.
function updateRate(pb) {
  const now = Date.now();
  if (ratePrevPb == null) { ratePrevPb = pb; ratePrevT = now; renderKpi('kpi-rate', '—', 'calcul du débit…', null); return; }
  if (pb > ratePrevPb && now > ratePrevT) {
    const perHour = (pb - ratePrevPb) / (now - ratePrevT) * 3600000;
    renderKpi('kpi-rate', fmtInt(Math.round(perHour)), 'débit browser (dérivé quarantaine)', null);
    ratePrevPb = pb; ratePrevT = now;
  }
  // pb inchangé (cache-hit serveur) → on garde la dernière valeur affichée, sans bouger le repère.
}

// pollBrowser — /stats/browser (RAPIDE, 5 s) : sessions, quarantaine, corpus. C'est le poll primaire,
// celui qui pilote l'indicateur « live ». Backoff borné ; replanifie toujours (finally).
async function pollBrowser() {
  try {
    renderBrowserStats(await fetchJson('/stats/browser'));
    browserDelay = BASE_MS;
  } catch {
    renderDegraded(); // 404 aujourd'hui / réseau → dégradation gracieuse (garde la dernière vue si déjà eu un succès)
    browserDelay = Math.min(browserDelay * 2, MAX_MS);
  } finally {
    clearTimeout(browserTimer);
    browserTimer = setTimeout(pollBrowser, browserDelay);
  }
}

// Actualisation manuelle (AC-8) : relance immédiate + reset du backoff (sans empiler de timers).
function refreshNow() {
  browserDelay = BASE_MS;
  clearTimeout(browserTimer);
  pollBrowser();
}

const refreshBtn = $('btn-refresh');
if (refreshBtn) refreshBtn.addEventListener('click', refreshNow);

// Démarrage : « Chargement… » immédiat, puis poll /stats/browser (caché 60 s côté serveur). Le débit se
// calcule dès la 2ᵉ mesure distincte de quarantaine (updateRate). Plus d'appel à /stats/selfplay (trop lourd).
renderDegraded();
pollBrowser();

// Export pour les tests headless (injection de mock fetch / appel direct des fonctions de rendu).
// Sans effet sur le navigateur (objet global, aucune UI).
if (typeof window !== 'undefined') {
  window.__admin = { renderBrowserStats, updateRate, renderDegraded, renderCorpusSection, renderKpi, statusFromRate, pollBrowser, refreshNow };
}
