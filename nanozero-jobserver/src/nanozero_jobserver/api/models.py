"""HTTP endpoints for the model registry (current / download / promote)."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from nanozero_jobserver.auth import require_api_key
from nanozero_jobserver.storage.models import (
    ModelRecord,
    current_model,
    get_model,
    list_models,
    promote_model,
    register_model,
    set_current_champion,
)
from nanozero_jobserver.storage.sprt import latest_decision_for_candidate

router = APIRouter(prefix="/models", tags=["models"])


class ModelSummary(BaseModel):
    version: int
    name: str
    sha256_onnx: str
    promoted_at: str | None
    parent_version: int | None
    created_at: str

    @classmethod
    def from_record(cls, r: ModelRecord) -> ModelSummary:
        return cls(
            version=r.version,
            name=r.name,
            sha256_onnx=r.sha256_onnx,
            promoted_at=r.promoted_at,
            parent_version=r.parent_version,
            created_at=r.created_at,
        )


class RegisterRequest(BaseModel):
    version: int
    name: str
    onnx_path: str
    sha256_onnx: str
    parent_version: int | None = None


class PromoteResponse(BaseModel):
    status: str
    version: int


@router.get(
    "/current",
    response_model=ModelSummary,
    responses={404: {"description": "No promoted model yet"}},
    # B.1 — route PUBLIQUE : le navigateur en a besoin pour la version+sha du champion (A.3).
)
def get_current(request: Request) -> ModelSummary:
    """Return the metadata of the currently-promoted champion.

    404 if no model has been promoted yet (pre-bootstrap).
    """
    record = current_model()
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No promoted model yet")
    return ModelSummary.from_record(record)


@router.get(
    "/{version}/download",
    response_class=FileResponse,
    responses={404: {"description": "Model version unknown or file missing"}},
    # B.1 — route PUBLIQUE : téléchargement du .onnx par le navigateur (A.3).
)
def download(version: int, request: Request) -> FileResponse:
    """Stream the .onnx file for the given model version.

    404 if the version isn't registered, or the file on disk is missing.
    """
    record = get_model(version)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model v{version} not found"
        )
    p = Path(record.onnx_path)
    if not p.exists():
        # Route publique (B.1) : ne pas fuiter le path disque absolu à un anonyme.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model v{version} file unavailable",
        )
    return FileResponse(
        path=str(p),
        filename=f"{record.name}.onnx",
        media_type="application/octet-stream",
    )


@router.get(
    "",
    response_model=list[ModelSummary],
    dependencies=[Depends(require_api_key)],
)
def list_all(request: Request, limit: int = 50) -> list[ModelSummary]:
    """List registered models (most recent first)."""
    return [ModelSummary.from_record(r) for r in list_models(limit=limit)]


@router.post(
    "/register",
    response_model=ModelSummary,
    status_code=status.HTTP_201_CREATED,
    responses={409: {"description": "Version or name already registered"}},
    dependencies=[Depends(require_api_key)],
)
def register(body: RegisterRequest, request: Request) -> ModelSummary:
    """Register a new model in the registry (no promotion yet)."""
    try:
        register_model(
            version=body.version,
            name=body.name,
            onnx_path=body.onnx_path,
            sha256_onnx=body.sha256_onnx,
            parent_version=body.parent_version,
        )
    except sqlite3.IntegrityError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model already registered or FK violation: {e}",
        ) from e

    record = get_model(body.version)
    assert record is not None  # just inserted
    return ModelSummary.from_record(record)


@router.post(
    "/{version}/promote",
    response_model=PromoteResponse,
    responses={
        404: {"description": "Model version unknown"},
        409: {"description": "Model already promoted"},
    },
    dependencies=[Depends(require_api_key)],
)
def promote(version: int, request: Request) -> PromoteResponse:
    """Mark a registered model as the new current champion.

    409 if already promoted, 404 if unknown version.
    """
    record = get_model(version)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model v{version} not found"
        )
    # Garde anti-footgun (gen-026) : ne pas promouvoir un candidat dont le dernier SPRT
    # enregistré est 'rejected'. Pour forcer (re)promotion → POST /models/{v}/set-current.
    if latest_decision_for_candidate(version) == "rejected":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model v{version} a un résultat SPRT 'rejected' — utiliser /set-current pour forcer.",
        )
    if not promote_model(version):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Model v{version} already promoted at {record.promoted_at}",
        )
    return PromoteResponse(status="promoted", version=version)


@router.post(
    "/{version}/set-current",
    response_model=ModelSummary,
    responses={404: {"description": "Model version unknown"}},
    dependencies=[Depends(require_api_key)],
)
def set_current(version: int, request: Request) -> ModelSummary:
    """Force cette version à devenir le champion COURANT — rollback / re-promotion.

    Contrairement à ``/promote`` (1ère promotion, 409 si déjà promu), ``set-current``
    re-stampe ``promoted_at`` → la version (re)devient champion, qu'elle ait déjà été
    promue ou non. Usage : **revenir au champion précédent après un SPRT rejeté** (gen-026)
    sans chirurgie SQL en prod. 404 si la version est inconnue.
    """
    if not set_current_champion(version):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Model v{version} not found"
        )
    record = get_model(version)
    assert record is not None  # vient d'être mis à jour
    return ModelSummary.from_record(record)
