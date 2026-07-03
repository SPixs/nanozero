"""API SPRT — historique durable des décisions de promotion.

``POST /sprt/record`` archive un test (candidate vs baseline) ; ``GET /sprt/history`` le relit.
Source de vérité du cycle (Elo, accept/reject) au lieu de fichiers locaux + AGENTS.md.
"""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, Field

from nanozero_jobserver.auth import require_api_key
from nanozero_jobserver.storage.sprt import SprtResult, list_sprt, record_sprt

router = APIRouter(prefix="/sprt", tags=["sprt"])


class SprtRecordRequest(BaseModel):
    candidate_version: int = Field(ge=1)
    baseline_version: int = Field(ge=1)
    decision: Literal["accepted", "rejected", "inconclusive"]
    elo_estimate: float | None = None
    games_played: int | None = Field(default=None, ge=0)
    wins: int | None = Field(default=None, ge=0)
    draws: int | None = Field(default=None, ge=0)
    losses: int | None = Field(default=None, ge=0)
    notes: str | None = None


class SprtResultOut(BaseModel):
    id: int
    candidate_version: int
    baseline_version: int
    decision: str
    elo_estimate: float | None
    games_played: int | None
    wins: int | None
    draws: int | None
    losses: int | None
    notes: str | None
    created_at: str

    @classmethod
    def from_record(cls, r: SprtResult) -> SprtResultOut:
        return cls(
            id=r.id,
            candidate_version=r.candidate_version,
            baseline_version=r.baseline_version,
            decision=r.decision,
            elo_estimate=r.elo_estimate,
            games_played=r.games_played,
            wins=r.wins,
            draws=r.draws,
            losses=r.losses,
            notes=r.notes,
            created_at=r.created_at,
        )


@router.post(
    "/record",
    response_model=SprtResultOut,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
def record(body: SprtRecordRequest, request: Request) -> SprtResultOut:
    """Archive un résultat SPRT (candidate vs baseline) — la mémoire durable du cycle.

    Avec un résultat ``rejected``, ``POST /models/{candidate}/promote`` refusera désormais
    de promouvoir ce candidat (garde anti-footgun gen-026) — utiliser ``/set-current`` pour forcer.
    """
    record_sprt(
        body.candidate_version,
        body.baseline_version,
        body.decision,
        elo_estimate=body.elo_estimate,
        games_played=body.games_played,
        wins=body.wins,
        draws=body.draws,
        losses=body.losses,
        notes=body.notes,
    )
    return SprtResultOut.from_record(list_sprt(limit=1)[0])  # la ligne qu'on vient d'insérer


@router.get(
    "/history",
    response_model=list[SprtResultOut],
    dependencies=[Depends(require_api_key)],
)
def history(
    request: Request,
    limit: int = Query(default=50, ge=1, le=500),
    model_version: int | None = Query(
        default=None, description="Filtre les tests où cette gen est candidat OU baseline."
    ),
) -> list[SprtResultOut]:
    """Historique SPRT, plus récent d'abord."""
    return [
        SprtResultOut.from_record(r)
        for r in list_sprt(limit=limit, model_version=model_version)
    ]
