"""Vue agrégée du cycle AlphaZero — ``GET /cycle/status``.

« Où en est-on ? » en UN appel au lieu de 4-5 (champion, volume fleet vs cible, ETA,
flotte, phase). Lecture pure : assemble des données déjà calculées ailleurs (stats/selfplay,
should_train, control). Zéro nouvelle table.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from nanozero_jobserver.api.stats import _cadence
from nanozero_jobserver.auth import require_api_key
from nanozero_jobserver.storage.batches import batch_timestamps, sum_positions_by_version
from nanozero_jobserver.storage.control import (
    get_autorefill,
    get_selfplay_target,
    is_selfplay_paused,
)
from nanozero_jobserver.storage.jobs import worker_stats
from nanozero_jobserver.storage.models import current_model
from nanozero_jobserver.storage.replay_buffer import (
    count_positions_by_source,
    count_unflushed_by_version,
)

router = APIRouter(prefix="/cycle", tags=["cycle"])


class CycleStatus(BaseModel):
    """État consolidé du cycle self-play → entraînement."""

    phase: str  # no_champion | paused | ready_to_train | collecting
    champion_version: int | None
    champion_name: str | None
    fleet_positions: int  # positions fleet ENTRAÎNABLES de la gen courante (durable + live)
    target_positions: int | None  # cible posée via /selfplay/target (None si aucune)
    missing: int | None  # restant avant la cible (None si pas de cible)
    percent: float | None  # progression vers la cible
    ready_to_train: bool  # fleet_positions >= ready_threshold
    positions_per_hour: float | None
    eta_hours: float | None  # estimation avant d'atteindre la cible
    fleet_workers_active_1h: int
    browser_positions: int  # cohorte browser quarantinée (gen courante)
    last_shard_at: str | None
    paused: bool
    autorefill_enabled: bool


@router.get("/status", response_model=CycleStatus, dependencies=[Depends(require_api_key)])
def cycle_status(
    request: Request,
    ready_threshold: int = Query(default=25000, ge=1, description="Seuil 'prêt à trainer'."),
) -> CycleStatus:
    """« Où en est-on dans le cycle ? » en un appel : champion, volume vs cible, ETA, flotte, phase."""
    paused = is_selfplay_paused()
    ar_enabled, _ = get_autorefill()
    champ = current_model()
    if champ is None:
        return CycleStatus(
            phase="no_champion",
            champion_version=None,
            champion_name=None,
            fleet_positions=0,
            target_positions=None,
            missing=None,
            percent=None,
            ready_to_train=False,
            positions_per_hour=None,
            eta_hours=None,
            fleet_workers_active_1h=0,
            browser_positions=0,
            last_shard_at=None,
            paused=paused,
            autorefill_enabled=ar_enabled,
        )

    mv = champ.version
    durable = sum_positions_by_version(source="fleet").get(mv, 0)
    live = count_unflushed_by_version(source="fleet").get(mv, 0)
    fleet_total = durable + live
    target, _action = get_selfplay_target()
    timestamps = batch_timestamps(mv)
    _sph, pph = _cadence(timestamps, durable)
    browser = count_positions_by_source(model_version=mv).get("browser", 0)
    active_1h = len(worker_stats(since_seconds=3600))

    ready = fleet_total >= ready_threshold
    missing = max(0, target - fleet_total) if target else None
    percent = round(100.0 * fleet_total / target, 1) if target else None
    eta = round(missing / pph, 1) if (missing and pph and pph > 0) else None
    phase = "paused" if paused else ("ready_to_train" if ready else "collecting")

    return CycleStatus(
        phase=phase,
        champion_version=mv,
        champion_name=champ.name,
        fleet_positions=fleet_total,
        target_positions=target,
        missing=missing,
        percent=percent,
        ready_to_train=ready,
        positions_per_hour=pph,
        eta_hours=eta,
        fleet_workers_active_1h=active_1h,
        browser_positions=browser,
        last_shard_at=timestamps[-1] if timestamps else None,
        paused=paused,
        autorefill_enabled=ar_enabled,
    )
