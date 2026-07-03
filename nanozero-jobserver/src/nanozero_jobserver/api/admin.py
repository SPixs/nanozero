"""Admin/ops endpoints — self-play pause switch + DB maintenance (PostgreSQL).

All routes are auth-protected (the API key is the admin boundary):

  POST /selfplay/pause    — refuse new claims (fleet idles, queue preserved).
  POST /selfplay/resume   — clear the pause flag.
  GET  /admin/storage     — footprint (PG catalog) + position counts (read-only).
  POST /admin/purge       — drop flushed positions <= a model_version.
  POST /admin/vacuum      — VACUUM ANALYZE positions (post-purge reclaim).
  POST /admin/checkpoint  — SQLite-only, supprimé en PG → 501.
  POST /admin/vacuum_into — SQLite-only, supprimé en PG → 501.
  POST /admin/compact     — SQLite-only, supprimé en PG → 501.

Les 3 derniers endpoints (checkpoint / vacuum_into / compact) mappaient sur des
opérations SQLite (WAL, VACUUM INTO, rebuild). PostgreSQL gère son WAL seul et
l'autovacuum récupère l'espace en continu ; les routes sont conservées mais
renvoient 501 avec un message explicatif pour ne pas surprendre l'opérateur
(cf. PG-MIGRATION-plan.md §8 D3 — option A retenue).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field

from nanozero_jobserver.auth import require_api_key
from nanozero_jobserver.storage.control import (
    get_autorefill,
    get_selfplay_target,
    is_selfplay_paused,
    set_autorefill,
    set_selfplay_paused,
    set_selfplay_target,
)
from nanozero_jobserver.storage.maintenance import (
    count_purgeable,
    storage_stats,
    vacuum_analyze,
)
from nanozero_jobserver.storage.replay_buffer import delete_flushed_old

router = APIRouter(tags=["admin"])


# -----------------------------------------------------------------------------
# Self-play pause switch
# -----------------------------------------------------------------------------


class PauseResponse(BaseModel):
    selfplay_paused: bool


@router.post(
    "/selfplay/pause",
    response_model=PauseResponse,
    dependencies=[Depends(require_api_key)],
)
def pause_selfplay(request: Request) -> PauseResponse:
    """Pause self-play : /jobs/claim returns 204 until resumed (queue kept)."""
    set_selfplay_paused(True)
    return PauseResponse(selfplay_paused=True)


@router.post(
    "/selfplay/resume",
    response_model=PauseResponse,
    dependencies=[Depends(require_api_key)],
)
def resume_selfplay(request: Request) -> PauseResponse:
    """Resume self-play : clear the pause flag, workers claim again."""
    set_selfplay_paused(False)
    return PauseResponse(selfplay_paused=False)


# -----------------------------------------------------------------------------
# On-demand autorefill switch (never-empty queue)
# -----------------------------------------------------------------------------


class AutorefillRequest(BaseModel):
    """Configure on-demand job generation."""

    enabled: bool
    num_sims: int | None = Field(default=None, ge=1, le=10000)


class AutorefillResponse(BaseModel):
    autorefill_enabled: bool
    autorefill_sims: int


@router.post(
    "/selfplay/autorefill",
    response_model=AutorefillResponse,
    dependencies=[Depends(require_api_key)],
)
def configure_autorefill(
    body: AutorefillRequest, request: Request
) -> AutorefillResponse:
    """Enable/disable on-demand job generation (never-empty queue).

    When enabled, /jobs/claim synthesises a fresh job for the current champion
    whenever the pending queue is empty — the fleet never idles for lack of work
    and no manual re-enqueue is needed. Still gated by pause and by a promoted
    model. `num_sims` (optional) sets the MCTS sims for generated jobs.
    """
    set_autorefill(body.enabled, body.num_sims)
    enabled, sims = get_autorefill()
    return AutorefillResponse(autorefill_enabled=enabled, autorefill_sims=sims)


class TargetRequest(BaseModel):
    """Cible de positions fleet pour la gen courante. 0/null efface la cible."""

    target_positions: int | None = Field(default=None, ge=0)
    action: str = Field(default="pause")  # "pause" (défaut) | "notify"


class TargetResponse(BaseModel):
    target_positions: int | None
    action: str


@router.post(
    "/selfplay/target",
    response_model=TargetResponse,
    dependencies=[Depends(require_api_key)],
)
def configure_target(body: TargetRequest, request: Request) -> TargetResponse:
    """Pose une cible de positions FLEET → **auto-pause** (ou notify) au seuil, puis efface.

    Décharge l'opérateur du « pause manuel au volume » (footgun récurrent : oublier de pauser
    en fin de cycle). Le flusher vérifie à chaque tick ; déclenchement one-shot. `target_positions=0`
    (ou null) efface la cible. La décision de LANCER l'entraînement reste humaine (seul l'arrêt
    du self-play est automatisé — `resume` suffit à reprendre).
    """
    set_selfplay_target(body.target_positions, body.action)
    target, action = get_selfplay_target()
    return TargetResponse(target_positions=target, action=action)


@router.get(
    "/selfplay/target",
    response_model=TargetResponse,
    dependencies=[Depends(require_api_key)],
)
def read_target(request: Request) -> TargetResponse:
    """État courant de la cible self-play (None si aucune)."""
    target, action = get_selfplay_target()
    return TargetResponse(target_positions=target, action=action)


# -----------------------------------------------------------------------------
# Storage dashboard
# -----------------------------------------------------------------------------


class StorageResponse(BaseModel):
    db_bytes: int
    positions_bytes: int
    positions_total: int
    positions_flushed: int
    jobs_by_status: dict[str, int]
    selfplay_paused: bool


@router.get(
    "/admin/storage",
    response_model=StorageResponse,
    dependencies=[Depends(require_api_key)],
)
def admin_storage(request: Request, detailed: bool = Query(default=False)) -> StorageResponse:
    """DB footprint (PG catalog) + position counts (read-only).

    ``db_bytes`` = ``pg_database_size`` (DB entière) ; ``positions_bytes`` =
    ``pg_total_relation_size('positions')`` (table + index + TOAST). #6 BMAD : par
    défaut (``detailed=false``) l'appel est INSTANTANÉ — tailles disque (catalogue) +
    histogramme jobs, mais ``positions_total`` / ``positions_flushed`` valent ``-1``
    (``COUNT(*)`` lourd sur des dizaines de M de lignes évité). Passer
    ``?detailed=true`` pour les chiffres exacts (ex. avant un VACUUM).
    """
    s = storage_stats(detailed=detailed)
    return StorageResponse(
        db_bytes=s.db_bytes,
        positions_bytes=s.positions_bytes,
        positions_total=s.positions_total,
        positions_flushed=s.positions_flushed,
        jobs_by_status=s.jobs_by_status,
        selfplay_paused=is_selfplay_paused(),
    )


# -----------------------------------------------------------------------------
# Purge
# -----------------------------------------------------------------------------


class PurgeRequest(BaseModel):
    """Drop flushed positions with model_version <= max_model_version.

    Only rows already written to an NPZ shard (flushed_to_npz=1) are deleted —
    nothing un-archived is ever lost. `max_model_version` is required (no
    default) so a purge is always an explicit, intentional bound.
    """

    max_model_version: int = Field(ge=1)
    dry_run: bool = Field(
        default=False,
        description="If true, only report how many rows WOULD be purged.",
    )
    source: str | None = Field(
        default=None,
        description="B4-D1 : restreint la purge à une cohorte ('browser' / 'fleet'). "
        "None = toutes. 'browser' purge la cohorte navigateur flushée sans toucher au fleet.",
    )


class PurgeResponse(BaseModel):
    purgeable: int
    purged: int
    dry_run: bool


@router.post(
    "/admin/purge",
    response_model=PurgeResponse,
    dependencies=[Depends(require_api_key)],
)
def admin_purge(body: PurgeRequest, request: Request) -> PurgeResponse:
    """Purge flushed positions <= a model_version (frees HOT-cache disk).

    The big disk lever : positions already in NPZ shards but kept in the HOT
    cache by the retention window. Run a dry_run first to size the impact. En PG
    le DELETE marque les rows comme dead tuples ; l'autovacuum récupère l'espace
    en continu (ou `POST /admin/vacuum` pour forcer un VACUUM ANALYZE immédiat).
    """
    purgeable = count_purgeable(body.max_model_version, source=body.source)
    if body.dry_run:
        return PurgeResponse(purgeable=purgeable, purged=0, dry_run=True)
    purged = delete_flushed_old(max_model_version=body.max_model_version, source=body.source)
    return PurgeResponse(purgeable=purgeable, purged=purged, dry_run=False)


# -----------------------------------------------------------------------------
# VACUUM ANALYSE (déclenchement manuel post-purge)
# -----------------------------------------------------------------------------


class VacuumResponse(BaseModel):
    status: str
    operation: str


@router.post(
    "/admin/vacuum",
    response_model=VacuumResponse,
    dependencies=[Depends(require_api_key)],
)
def admin_vacuum() -> VacuumResponse:
    """VACUUM ANALYZE positions — récupère les dead tuples + rafraîchit les stats.

    Levier manuel à lancer APRÈS une grosse `/admin/purge` pour récupérer l'espace
    immédiatement. En régime normal l'autovacuum (réglé agressif côté serveur PG)
    suffit. Tourne en autocommit (VACUUM interdit dans une transaction).
    """
    vacuum_analyze()
    return VacuumResponse(status="ok", operation="VACUUM ANALYZE positions")


# -----------------------------------------------------------------------------
# Endpoints SQLite-only supprimés en PostgreSQL → 501 (cf. PG-MIGRATION-plan §8 D3)
#
# Routes conservées (pas supprimées) afin d'informer l'opérateur qui suivrait un
# ancien runbook de maintenance disque, plutôt que de renvoyer un 404 opaque.
# -----------------------------------------------------------------------------

_SQLITE_ONLY_DETAIL = {
    "checkpoint": (
        "Endpoint SQLite-only (wal_checkpoint) supprimé en PG : PostgreSQL gère "
        "son WAL automatiquement. L'espace est récupéré par l'autovacuum ; pour "
        "forcer un nettoyage immédiat, utilisez POST /admin/vacuum (VACUUM ANALYZE)."
    ),
    "vacuum_into": (
        "Endpoint SQLite-only (VACUUM INTO) supprimé en PG : pas de copie compactée "
        "en ligne. Pour une copie offline utilisez `pg_dump` ; l'espace en ligne est "
        "géré par l'autovacuum / POST /admin/vacuum."
    ),
    "compact": (
        "Endpoint SQLite-only (rebuild_compact) supprimé en PG : PostgreSQL n'a pas "
        "la fragmentation B-tree de SQLite sur les BLOBs (TOAST compresse "
        "nativement). L'espace est géré par l'autovacuum / POST /admin/vacuum."
    ),
}


@router.post("/admin/checkpoint", dependencies=[Depends(require_api_key)])
def admin_checkpoint() -> None:
    """SQLite-only (WAL checkpoint) — supprimé en PG, renvoie 501."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=_SQLITE_ONLY_DETAIL["checkpoint"],
    )


@router.post("/admin/vacuum_into", dependencies=[Depends(require_api_key)])
def admin_vacuum_into() -> None:
    """SQLite-only (VACUUM INTO compacted copy) — supprimé en PG, renvoie 501."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=_SQLITE_ONLY_DETAIL["vacuum_into"],
    )


@router.post("/admin/compact", dependencies=[Depends(require_api_key)])
def admin_compact() -> None:
    """SQLite-only (metadata-only rebuild) — supprimé en PG, renvoie 501."""
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail=_SQLITE_ONLY_DETAIL["compact"],
    )
