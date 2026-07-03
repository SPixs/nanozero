"""Observability endpoints — per-version breakdown, self-play progress, fleet.

These complement the flat `/stats` counters (in api/replay.py) with the views we
keep needing operationally but had to hand-query in SQLite before:

  GET /stats/by-version  — durable positions + queue depth per generation.
  GET /stats/selfplay    — "où en est le self-play ?" for the in-flight cycle.
  GET /stats/workers      — per-worker attribution across the ~54-64 node fleet.

Authoritative position counts come from the `batches` registry (durable, never
purged) plus the live unflushed tail — NOT from `count_positions`, which
under-reports by the HOT-cache purge volume (cf. delete_flushed_old).
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from nanozero_jobserver.api.jobs import BLACKLISTED_WORKER_PREFIX
from nanozero_jobserver.auth import require_api_key
from nanozero_jobserver.storage.batches import (
    batch_timestamps,
    batches_summary,
    count_batches_by_version,
    sum_positions_by_version,
)
from nanozero_jobserver.storage.contributors import contributor_counts
from nanozero_jobserver.storage.control import get_autorefill, is_selfplay_paused
from nanozero_jobserver.storage.drift import drift_report
from nanozero_jobserver.storage.jobs import (
    count_jobs_by_version_status,
    worker_stats,
)
from nanozero_jobserver.storage.models import current_model, list_models
from nanozero_jobserver.storage.replay_buffer import (
    count_positions_by_source,
    count_unflushed_by_version,
    policy_resolution,
)
from nanozero_jobserver.storage.sprt import latest_accepted_promotion

router = APIRouter(prefix="/stats", tags=["stats"])


def _parse_iso(ts: str) -> datetime | None:
    """Parse an ISO 8601 UTC timestamp ('...Z') stored by the schema. None on failure."""
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _cadence(timestamps: list[str], positions_total: int) -> tuple[float | None, float | None]:
    """Compute (shards_per_hour, positions_per_hour) from a version's batch times.

    Returns (None, None) when fewer than 2 timestamps exist (no measurable span)
    or the span is degenerate. Uses first→last span across the supplied shards.
    """
    parsed = [t for t in (_parse_iso(s) for s in timestamps) if t is not None]
    if len(parsed) < 2:
        return None, None
    span_hours = (parsed[-1] - parsed[0]).total_seconds() / 3600.0
    if span_hours <= 0:
        return None, None
    # n-1 intervals across the span → rate per hour.
    shards_per_hour = (len(parsed) - 1) / span_hours
    positions_per_hour = positions_total / span_hours if positions_total else None
    return round(shards_per_hour, 3), (
        round(positions_per_hour, 1) if positions_per_hour is not None else None
    )


# -----------------------------------------------------------------------------
# /stats/by-version
# -----------------------------------------------------------------------------


class VersionStat(BaseModel):
    """Per-generation rollup combining positions, batches and queue depth."""

    model_version: int
    name: str | None
    positions_durable: int
    positions_live: int
    positions_total: int
    batches: int
    jobs: dict[str, int]
    promoted: bool
    is_current: bool


class ByVersionResponse(BaseModel):
    versions: list[VersionStat]
    current_model_version: int | None


@router.get(
    "/by-version",
    response_model=ByVersionResponse,
    dependencies=[Depends(require_api_key)],
)
def by_version(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500, description="Nb max de générations (les plus récentes)."),
) -> ByVersionResponse:
    """Per-model_version breakdown of positions (durable + live) and job queue.

    Answers "how many positions did gen-N generate?" without a manual DB query.
    Durable counts come from the batches registry so they remain correct after
    the HOT-cache purge removes the position BLOBs. Borné aux `limit` gens les plus
    récentes ; les comptes batches viennent d'UNE requête (plus de N+1).
    """
    durable = sum_positions_by_version()
    live = count_unflushed_by_version()
    jobs = count_jobs_by_version_status()
    batches_by_v = count_batches_by_version()  # 1 requête (fix N+1)
    models = {m.version: m for m in list_models(limit=10_000)}
    current = current_model()
    current_v = current.version if current else None

    versions = sorted(set(durable) | set(live) | set(jobs) | set(models), reverse=True)[:limit]
    rows = [
        VersionStat(
            model_version=v,
            name=models[v].name if v in models else None,
            positions_durable=durable.get(v, 0),
            positions_live=live.get(v, 0),
            positions_total=durable.get(v, 0) + live.get(v, 0),
            batches=batches_by_v.get(v, 0),
            jobs=jobs.get(v, {}),
            promoted=(v in models and models[v].promoted_at is not None),
            is_current=(v == current_v),
        )
        for v in versions
    ]
    return ByVersionResponse(versions=rows, current_model_version=current_v)


# -----------------------------------------------------------------------------
# /stats/selfplay
# -----------------------------------------------------------------------------


class SelfplayResponse(BaseModel):
    """Progress of the in-flight self-play cycle (the current promoted model)."""

    model_version: int | None
    model_name: str | None
    shards: int
    positions_durable: int
    positions_live: int
    positions_total: int
    positions_browser: int  # B.4 : positions browser quarantinées (live, exclues du training)
    first_shard_at: str | None
    last_shard_at: str | None
    shards_per_hour: float | None
    positions_per_hour: float | None
    next_shard_threshold: int
    jobs_pending: int
    jobs_claimed: int
    jobs_completed: int
    paused: bool
    autorefill_enabled: bool
    autorefill_sims: int


@router.get(
    "/selfplay",
    response_model=SelfplayResponse,
    dependencies=[Depends(require_api_key)],
)
def selfplay(request: Request) -> SelfplayResponse:
    """Self-play progress for the current cycle — the canonical 'où en est-on'.

    The cycle is the currently-promoted model version : workers play with it and
    tag every position with it. Volume = durable (flushed NPZ shards) + live tail.
    Cadence is measured from the spacing of this version's NPZ shards.
    """
    cfg = request.app.state.config
    ar_enabled, ar_sims = get_autorefill()

    current = current_model()
    # Fall back to the highest version that has produced positions if nothing is
    # promoted yet (bootstrap / fresh DB).
    if current is not None:
        mv: int | None = current.version
        name = current.name
    else:
        durable_all = sum_positions_by_version()
        mv = max(durable_all) if durable_all else None
        name = None

    if mv is None:
        return SelfplayResponse(
            model_version=None,
            model_name=None,
            shards=0,
            positions_durable=0,
            positions_live=0,
            positions_total=0,
            positions_browser=0,
            first_shard_at=None,
            last_shard_at=None,
            shards_per_hour=None,
            positions_per_hour=None,
            next_shard_threshold=cfg.flush_threshold_positions,
            jobs_pending=0,
            jobs_claimed=0,
            jobs_completed=0,
            paused=is_selfplay_paused(),
            autorefill_enabled=ar_enabled,
            autorefill_sims=ar_sims,
        )

    durable = sum_positions_by_version().get(mv, 0)
    live = count_unflushed_by_version().get(mv, 0)
    browser = count_positions_by_source(model_version=mv).get("browser", 0)
    timestamps = batch_timestamps(mv)
    shards_per_hour, positions_per_hour = _cadence(timestamps, durable)
    jobs = count_jobs_by_version_status().get(mv, {})

    return SelfplayResponse(
        model_version=mv,
        model_name=name,
        shards=len(timestamps),
        positions_durable=durable,
        positions_live=live,
        positions_total=durable + live,
        positions_browser=browser,
        first_shard_at=timestamps[0] if timestamps else None,
        last_shard_at=timestamps[-1] if timestamps else None,
        shards_per_hour=shards_per_hour,
        positions_per_hour=positions_per_hour,
        next_shard_threshold=cfg.flush_threshold_positions,
        jobs_pending=jobs.get("pending", 0),
        jobs_claimed=jobs.get("claimed", 0),
        jobs_completed=jobs.get("completed", 0),
        paused=is_selfplay_paused(),
        autorefill_enabled=ar_enabled,
        autorefill_sims=ar_sims,
    )


# -----------------------------------------------------------------------------
# /stats/workers
# -----------------------------------------------------------------------------


class WorkerStatOut(BaseModel):
    worker_id: str
    completed: int
    in_flight: int
    last_claimed_at: str | None
    idle_seconds: float | None  # secondes depuis le dernier claim → repérer un nœud mort
    blacklisted: bool


class WorkersResponse(BaseModel):
    workers: list[WorkerStatOut]
    count: int
    window_seconds: int | None


@router.get(
    "/workers",
    response_model=WorkersResponse,
    dependencies=[Depends(require_api_key)],
)
def workers(
    request: Request,
    window_seconds: int | None = Query(
        default=None,
        ge=1,
        description="Only count claims newer than now-this (e.g. 3600 = last hour). "
        "Omit for all-time totals.",
    ),
    limit: int = Query(
        default=200,
        ge=1,
        le=1000,
        description="Nb max de workers (les plus récemment actifs). Borne la réponse contre "
        "une prolifération de worker_ids (browser).",
    ),
) -> WorkersResponse:
    """Per-worker job attribution for fleet health — completed, in-flight, last-seen, idle.

    `idle_seconds` (depuis le dernier claim) repère un nœud MORT d'un coup d'œil (trier DESC).
    `blacklisted` flags workers whose X-Worker-Id matches the NPU prefix; the server 204s
    their claims (INT8 labels pollute the replay buffer). Réponse bornée à `limit` workers.
    """
    stats = worker_stats(since_seconds=window_seconds, limit=limit)
    rows = [
        WorkerStatOut(
            worker_id=w.worker_id,
            completed=w.completed,
            in_flight=w.in_flight,
            last_claimed_at=w.last_claimed_at,
            idle_seconds=w.idle_seconds,
            blacklisted=w.worker_id.startswith(BLACKLISTED_WORKER_PREFIX),
        )
        for w in stats
    ]
    return WorkersResponse(workers=rows, count=len(rows), window_seconds=window_seconds)


class DriftResponse(BaseModel):
    """D.2 — rapport de drift de la cohorte browser vs la flotte pour une génération."""

    model_version: int
    verdict: str  # ok | warn | drift | insufficient
    browser: dict[str, float]
    fleet: dict[str, float]
    # Sims MCTS effectifs estimés (résolution policy) par cohorte ; None si la sonde est désactivée.
    resolution: dict[str, dict[str, float]] | None = None
    deltas: dict[str, float]
    reasons: list[str]


@router.get(
    "/drift",
    response_model=DriftResponse,
    dependencies=[Depends(require_api_key)],
)
def drift(
    request: Request,
    model_version: int | None = Query(
        default=None,
        description="Génération à analyser. Omis = champion courant.",
    ),
    fresh: bool = Query(
        default=False,
        description="Recalcule en ignorant le cache TTL (5 min). À utiliser pour un point exact.",
    ),
) -> DriftResponse:
    """D.2 — compare la data ``source=browser`` à la flotte de confiance (drift report).

    Features distributionnelles cheap (équilibre W/D/L des outcomes, longueur de partie) comparées
    browser↔fleet, PLUS la **résolution policy** (``resolution`` : sims MCTS effectifs estimés depuis
    la granularité de la cible — le pendant VÉRIFICATION du num_sims dicté côté serveur). Verdict
    ``ok|warn|drift|insufficient`` : seuils LÂCHES sur les features distributionnelles (le low-sims est
    légitime), MAIS la résolution cible justement le low-sims pour le quantifier/flaguer. Le flag
    ``drift`` quarantine la cohorte pour D.3 (ouverture du robinet). ⚠️ Sonde contournable (pad de π) :
    arbitre de la valeur = SPRT, pas cette mesure.
    """
    mv = model_version
    if mv is None:
        current = current_model()
        mv = current.version if current else 0
    # #6 BMAD : cache TTL — un appel polled (monitoring) ne rescanne pas ~34M lignes.
    cache = request.app.state.drift_cache
    if not fresh:
        cached = cache.get(mv)
        if isinstance(cached, DriftResponse):
            return cached
    # #7 BMAD : model_validate (Pydantic v2) au lieu de **dict[str,object] → mypy clean.
    report = DriftResponse.model_validate(drift_report(mv))
    cache.set(mv, report)
    return report


# -----------------------------------------------------------------------------
# /stats/browser — tableau de bord de la cohorte volontaire (décision D.3)
# -----------------------------------------------------------------------------


class BrowserStatsResponse(BaseModel):
    """Cohorte browser (volontaire) de la gen courante — pour décider d'ouvrir le robinet (D.3)."""

    model_version: int | None
    positions_browser: int  # provenance FIABLE (source='browser', server-authoritative)
    positions_fleet: int
    browser_fleet_ratio: float | None  # part relative de la cohorte browser
    median_eff_sims: float | None  # sims MCTS effectifs estimés (qualité) — None si échantillon vide
    drift_verdict: str | None  # depuis le cache /stats/drift (None si pas encore calculé)
    browser_workers_recent: int  # BEST-EFFORT : worker_id préfixés 'browser-' actifs 24h (client-set)


@router.get(
    "/browser",
    response_model=BrowserStatsResponse,
    dependencies=[Depends(require_api_key)],
)
def browser(request: Request) -> BrowserStatsResponse:
    """Cohorte volontaire (navigateur) : volume, ratio vs fleet, résolution sims, drift.

    Sert la décision du **chantier 3** (ouvrir le robinet D.3 = A/B SPRT) : a-t-on assez de
    VRAIE data volontaire, à quelle qualité (sims effectifs), drift-elle ? Le volume/ratio et
    `median_eff_sims` sont FIABLES (basés sur `source='browser'`). `drift_verdict` vient du cache
    `/stats/drift` (None si non calculé récemment). `browser_workers_recent` est BEST-EFFORT (préfixe
    worker_id posé par le client, non server-authoritative — un vrai compte exigerait `claimed_by`
    sur les positions). ⚠️ measure-first : aucune de ces métriques ≠ proxy de force ; seul le SPRT tranche.
    """
    # STORY-012 : cache TTL 60 s (le SUM par source + policy_resolution + worker_stats coûtent ~20 s
    # sur ~16M lignes). La page admin poll → on sert un résultat ≤ 60 s sans recalcul. Clé unique (None).
    cache = request.app.state.browser_cache
    cached_resp = cache.get(None)
    if isinstance(cached_resp, BrowserStatsResponse):
        return cached_resp

    current = current_model()
    if current is not None:
        mv: int | None = current.version
    else:
        durable_all = sum_positions_by_version()
        mv = max(durable_all) if durable_all else None
    if mv is None:
        resp = BrowserStatsResponse(
            model_version=None,
            positions_browser=0,
            positions_fleet=0,
            browser_fleet_ratio=None,
            median_eff_sims=None,
            drift_verdict=None,
            browser_workers_recent=0,
        )
        cache.set(None, resp)
        return resp
    by_src = count_positions_by_source(model_version=mv)
    pb = by_src.get("browser", 0)
    pf = by_src.get("fleet", 0)
    res = policy_resolution(mv, "browser")
    eff = res["median_eff_sims"] if res["n_sampled"] else None
    cached = request.app.state.drift_cache.get(mv)  # lecture cheap (pas de scan 11M)
    verdict = cached.verdict if isinstance(cached, DriftResponse) else None
    recent = sum(
        1 for w in worker_stats(since_seconds=86400) if w.worker_id.startswith("browser-")
    )
    resp = BrowserStatsResponse(
        model_version=mv,
        positions_browser=pb,
        positions_fleet=pf,
        browser_fleet_ratio=round(pb / pf, 4) if pf else None,
        median_eff_sims=round(eff, 1) if eff is not None else None,
        drift_verdict=verdict,
        browser_workers_recent=recent,
    )
    cache.set(None, resp)
    return resp


# -----------------------------------------------------------------------------
# /stats/batches — rollup des shards NPZ par gen (outillage A/B SPRT D.3)
# -----------------------------------------------------------------------------


class BatchSummaryRow(BaseModel):
    model_version: int
    n_shards: int
    n_positions: int
    earliest: str | None
    latest: str | None


class BatchesSummaryResponse(BaseModel):
    source: str | None
    versions: list[BatchSummaryRow]


@router.get(
    "/batches",
    response_model=BatchesSummaryResponse,
    dependencies=[Depends(require_api_key)],
)
def batches(
    request: Request,
    source: str | None = Query(
        default=None, description="Filtre cohorte : 'browser' / 'fleet'. Omis = toutes."
    ),
) -> BatchesSummaryResponse:
    """Rollup des shards NPZ durables par génération — pour dimensionner un corpus A/B (D.3).

    Combien de shards/positions par gen, par cohorte (browser/fleet), avec la fenêtre temporelle.
    Lit le registre ``batches`` (durable, survit au purge). Sert à décider si le corpus volontaire
    est assez gros pour l'A/B SPRT du chantier 3.
    """
    rows = [BatchSummaryRow.model_validate(r) for r in batches_summary(source=source)]
    return BatchesSummaryResponse(source=source, versions=rows)


# -----------------------------------------------------------------------------
# /stats/season — jauge collective publique (gamification, STORY-001)
# -----------------------------------------------------------------------------


class SeasonResponse(BaseModel):
    """Progression collective vers la prochaine génération — PUBLIC, lu par la page volontaire.

    « Saison » = cycle de self-play de la génération courante (le champion qui joue). La
    jauge collective célèbre les positions VÉRIFIÉES (cf. DECISIONS-CANONIQUES §1) ; en
    l'absence du gate D.2 câblé sur le compteur, on expose le TOTAL server-authoritative de
    la gen (toutes sources, cf. STORY-001 Dev Notes — total inclusif plus lisible pour le
    volontaire), jamais un chiffre « envoyé » qui pourrait chuter.
    """

    current_gen: int
    target_gen: int
    verified_positions: int
    # STORY-016 — décompte contributeurs (fleet + browser), comptés AU SUBMIT (table contributors).
    contributors_total: int  # cumul lifetime (monotone croissant), toutes sources
    contributors_online: int  # actifs sur la dernière heure, toutes sources
    contributors_total_by_source: dict[str, int]  # {'browser': int, 'fleet': int}
    contributors_online_by_source: dict[str, int]
    # Déprécié (rétro-compat consommateurs STORY-001) : alias de contributors_online.
    active_contributors: int
    collective_pct: float
    last_promoted_gen: int | None
    last_promoted_elo_gain: int | None


@router.get("/season", response_model=SeasonResponse)
def season(request: Request) -> SeasonResponse:
    """Jauge collective publique « Ensemble vers gen-N+1 » (gamification).

    PUBLIC (aucune ``require_api_key``) : la page volontaire le poll sans clé. Réponse mise en
    cache 60 s côté serveur (``season_cache``) — la jauge bouge lentement et l'endpoint est
    polled. Aucune PII : que des agrégats (compteurs de positions + nb de sessions distinctes).

    Champs (cf. STORY-001 AC-2) :
      - ``current_gen`` / ``target_gen`` : génération courante (champion) et la suivante (N+1).
      - ``verified_positions`` : total server-authoritative de la gen courante, TOUTES sources
        (native + browser) ; durable + tail live. ⚠️ ce n'est pas le gate D.2 « vérifié » au sens
        anti-poison (pas encore câblé) mais le total honnête, jamais un « envoyé » trompeur.
      - ``contributors_total`` / ``contributors_online`` (STORY-016) : contributeurs distincts
        (fleet + browser) comptés AU SUBMIT via la table ``contributors`` — cumul lifetime et
        actifs sur la dernière heure. Browser = token stable (1/appareil) ; fleet = par machine
        (suffixe de slot retiré). Ventilation dans ``*_by_source``. PAS un compte de personnes
        (1 personne multi-appareils = N tokens). ``active_contributors`` = alias déprécié de
        ``contributors_online`` (rétro-compat).
      - ``collective_pct`` : ``verified_positions / season_target × 100``, borné à 100.
      - ``last_promoted_gen`` / ``last_promoted_elo_gain`` : dernière promo archivée (SPRT
        ``accepted``), ``null`` si aucune génération n'a jamais été promue (AC-4).
    """
    cfg = request.app.state.config

    cache = request.app.state.season_cache
    cached = cache.get(None)
    if isinstance(cached, SeasonResponse):
        return cached

    current = current_model()
    if current is not None:
        mv: int | None = current.version
    else:
        durable_all = sum_positions_by_version()
        mv = max(durable_all) if durable_all else None

    if mv is None:
        # DB vide / pas de champion : tout à zéro, gen courante=0, cible=1 (AC-5 cas DB vide).
        resp = SeasonResponse(
            current_gen=0,
            target_gen=1,
            verified_positions=0,
            contributors_total=0,
            contributors_online=0,
            contributors_total_by_source={"browser": 0, "fleet": 0},
            contributors_online_by_source={"browser": 0, "fleet": 0},
            active_contributors=0,
            collective_pct=0.0,
            last_promoted_gen=None,
            last_promoted_elo_gain=None,
        )
        cache.set(None, resp)
        return resp

    # Total server-authoritative de la gen courante : durable (shards NPZ) + tail live.
    durable = sum_positions_by_version().get(mv, 0)
    live = count_unflushed_by_version().get(mv, 0)
    verified = durable + live

    # STORY-016 — compteurs contributeurs (fleet + browser) depuis la table `contributors`
    # (au submit, par machine pour le fleet). Remplace l'ancien proxy worker_stats biaisé.
    counts = contributor_counts()

    target = cfg.season_target
    pct = min(100.0, verified / target * 100.0) if target > 0 else 0.0

    promo = latest_accepted_promotion()
    last_gen = promo.candidate_version if promo is not None else None
    last_elo = (
        round(promo.elo_estimate)
        if promo is not None and promo.elo_estimate is not None
        else None
    )

    resp = SeasonResponse(
        current_gen=mv,
        target_gen=mv + 1,
        verified_positions=verified,
        contributors_total=counts["total"],
        contributors_online=counts["online"],
        contributors_total_by_source=counts["total_by_source"],
        contributors_online_by_source=counts["online_by_source"],
        active_contributors=counts["online"],
        collective_pct=round(pct, 2),
        last_promoted_gen=last_gen,
        last_promoted_elo_gain=last_elo,
    )
    cache.set(None, resp)
    return resp
