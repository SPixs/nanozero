"""HTTP endpoints for replay buffer sampling + trainer trigger + stats."""

from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel

from nanozero_jobserver.auth import require_api_key
from nanozero_jobserver.storage.batches import sum_positions_by_version
from nanozero_jobserver.storage.jobs import count_jobs_by_status
from nanozero_jobserver.storage.models import current_model, list_models
from nanozero_jobserver.storage.replay_buffer import (
    count_positions,
    count_unflushed_by_version,
    sample_positions,
)

router = APIRouter(tags=["replay"])


class PositionSample(BaseModel):
    game_id: str
    model_version: int
    ply: int
    fen: str
    input_planes_b64: str
    policy_target_b64: str
    outcome: float


class SampleResponse(BaseModel):
    """Trainer batch payload."""

    positions: list[PositionSample]
    requested: int
    returned: int


class ShouldTrainResponse(BaseModel):
    should_train: bool
    new_positions: int
    threshold: int
    missing: int  # positions restantes avant le seuil (0 si atteint) — pas de calcul à la main
    since_version: int  # version depuis laquelle on compte (résolue si non fournie)


class StatsResponse(BaseModel):
    jobs: dict[str, int]
    positions_total: int
    positions_generated_total: int
    models_registered: int
    current_model_version: int | None


def _durable_new_positions(since_version: int) -> int:
    """Positions produced by models at-or-after `since_version`, purge-proof.

    Counts the durable batches registry (flushed → NPZ, never purged) plus the
    live unflushed tail, for every model_version >= since_version.

    This replaces the old `count_new_positions` (positions-table only, filter
    ``model_version > since``), which double-failed in production: it
    under-counted by the HOT-cache purge volume, AND its strict ``>`` missed the
    whole point — self-play positions are tagged with the model that PLAYED them
    (the current promoted version), so "strictly newer than current" was almost
    always zero. ``>=`` counts the in-flight cycle's own output. (cf. memory
    [[feedback_selfplay_status_answer]] : new_positions était buggé.)
    """
    # B4-D2 : exclure la cohorte browser quarantinée du déclencheur d'entraînement,
    # côté durable (shards browser-gen*) ET live (tampon browser non flushé) — sinon du
    # volume browser ferait fire should_train sur de la data hors-corpus.
    durable = sum_positions_by_version(source="fleet")
    live = count_unflushed_by_version(source="fleet")
    versions = set(durable) | set(live)
    return sum(durable.get(v, 0) + live.get(v, 0) for v in versions if v >= since_version)


@router.get(
    "/replay/sample",
    response_model=SampleResponse,
    dependencies=[Depends(require_api_key)],
)
def sample(
    request: Request,
    n: int = Query(default=8192, ge=1, le=100000),
    window: int = Query(default=5, ge=1, le=100),
    current_version: int | None = Query(default=None),
    include_browser: bool = Query(default=False),
) -> SampleResponse:
    """Return up to N random positions from the last `window` model_versions.

    current_version override defaults to the currently-promoted model. Returns
    fewer rows if the buffer is smaller than N within the window.

    include_browser (B.4) : False (défaut) exclut la data ``source='browser'``
    (quarantaine — la route est déjà key-gated, donc opérateur-only). True =
    l'inclut (hook pour le gate Epic D / inspection ; l'admission bornée = D.3).
    """
    cv = current_version
    if cv is None:
        m = current_model()
        cv = m.version if m else 0  # 0 = no promoted model yet, returns []
    rows = sample_positions(
        n=n, current_model_version=cv, window=window, include_browser=include_browser
    )
    return SampleResponse(
        positions=[
            PositionSample(
                game_id=p.game_id,
                model_version=p.model_version,
                ply=p.ply,
                fen=p.fen,
                input_planes_b64=base64.b64encode(p.input_planes).decode("ascii"),
                policy_target_b64=base64.b64encode(p.policy_target).decode("ascii"),
                outcome=p.outcome,
            )
            for p in rows
        ],
        requested=n,
        returned=len(rows),
    )


@router.get(
    "/training/should_train",
    response_model=ShouldTrainResponse,
    dependencies=[Depends(require_api_key)],
)
def should_train(
    request: Request,
    threshold: int = Query(default=25000, ge=1),
    since_version: int | None = Query(default=None),
) -> ShouldTrainResponse:
    """Trainer trigger : returns should_train=True when the current model has
    accumulated ≥ threshold new positions since the given version (default: the
    currently-promoted model's version).
    """
    sv = since_version
    if sv is None:
        m = current_model()
        sv = m.version if m else 0
    new_n = _durable_new_positions(since_version=sv)
    return ShouldTrainResponse(
        should_train=new_n >= threshold,
        new_positions=new_n,
        threshold=threshold,
        missing=max(0, threshold - new_n),
        since_version=sv,
    )


@router.get(
    "/stats",
    response_model=StatsResponse,
    dependencies=[Depends(require_api_key)],
)
def stats(request: Request) -> StatsResponse:
    """Aggregate counters for observability."""
    current = current_model()
    durable = sum_positions_by_version()
    live = count_unflushed_by_version()
    generated_total = sum(durable.values()) + sum(live.values())
    return StatsResponse(
        jobs=count_jobs_by_status(),
        # Rows currently in the HOT cache (under-reports by the purge volume).
        positions_total=count_positions(),
        # Cumulative positions ever produced (durable batches + live tail) —
        # purge-proof, this is the number to trust for "total generated".
        positions_generated_total=generated_total,
        models_registered=len(list_models(limit=10_000)),
        current_model_version=current.version if current else None,
    )
