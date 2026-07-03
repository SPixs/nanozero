"""HTTP endpoints for the job lifecycle (claim + submit).

Auth: all routes require X-API-Key (or run in dev mode if config.api_key is "").

Workers identify themselves via X-Worker-Id header (free-form string, typically
hostname or UUID). It's stored on the claimed job so we can attribute crashes.

Positions on /jobs/{id}/submit are sent as base64-encoded BLOBs to stay
JSON-friendly (cf. ADR-014 §D2 — JSON over msgpack for debug experience).
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, Field, ValidationError

from nanozero_jobserver import submit_codec
from nanozero_jobserver.auth import require_api_key
from nanozero_jobserver.config import normalize_pseudo
from nanozero_jobserver.storage.contributors import normalize_contributor_token
from nanozero_jobserver.storage.control import get_autorefill, is_selfplay_paused
from nanozero_jobserver.storage.jobs import (
    cancel_pending_jobs,
    claim_job,
    create_and_claim_job,
    create_job,
    get_job,
    get_job_model_version,
    submit_job_with_positions,
)
from nanozero_jobserver.storage.models import current_model
from nanozero_jobserver.storage.replay_buffer import Position
from nanozero_jobserver.validation import (
    BrowserValidationError,
    validate_browser_positions,
)

# Content-Type de la variante binaire de soumission (Story A.2). Absence de ce type
# (= application/json) → chemin JSON base64 existant, inchangé (rétro-compat workers Java).
BINARY_SUBMIT_CONTENT_TYPE = "application/x-nanozero-submit-v1"

# Bornes du submit JSON fleet (durcissement revue BMAD). Un submit = 1 partie ; chaque BLOB
# décodé doit faire la taille EXACTE attendue, sinon corpus corrompu → crash np.frombuffer
# côté trainer. _MAX_POSITIONS borne l'amplification mémoire (sinon 100k micro-positions).
_PLANES_BYTES = submit_codec.PLANES_BYTES  # 7616 × f32 = 30464
_POLICY_BYTES = submit_codec.POLICY_LEN * 4  # 4672 × f32 = 18688
_MAX_POSITIONS = 1024  # 1 partie self-play < 512 plies en pratique (drift max_ply ~400)


async def _raw_body(request: Request) -> bytes:
    """Lit le corps brut de la requête (déjà dé-gzippé par GzipRequestMiddleware).

    Dépendance async pour que le handler `submit` reste sync (threadpool, pas de
    blocage de l'event-loop sur le travail DB/CPU).
    """
    return await request.body()

router = APIRouter(prefix="/jobs", tags=["jobs"])

# X-Worker-Id prefix whose claims are refused (204). NPU workers ship INT8-
# quantized MCTS labels that degraded gen-11 (-174 Elo SPRT vs gen-010). Shared
# with api/stats.py so /stats/workers flags the same set. Remove when the NPU
# worker is fixed (FP16 or CPU EP).
BLACKLISTED_WORKER_PREFIX = "npu-"


# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------


class JobClaimResponse(BaseModel):
    """Payload returned to a worker that successfully claimed a job."""

    job_id: str
    model_version: int
    opening_fen: str | None
    dirichlet_seed: int | None
    num_sims: int


class PositionPayload(BaseModel):
    """One training sample as sent by the worker.

    BLOB fields are base64-encoded. The server decodes and stores them as bytes
    in the SQLite positions table.
    """

    ply: int = Field(ge=0)
    fen: str
    input_planes_b64: str
    policy_target_b64: str
    outcome: float = Field(ge=-1.0, le=1.0)


class JobSubmitRequest(BaseModel):
    """Worker submits a completed game's positions + outcome.

    Attributes:
        game_id: unique identifier (typically worker-generated UUID).
        model_version: echoed from the claim — **IGNORÉ** côté serveur. La version est
            dérivée du job claimé (server-authoritative) ; un payload forgé n'a aucun effet.
        positions: 1..1024 positions, une par ply, ordonnées.
    """

    game_id: str
    model_version: int
    positions: list[PositionPayload] = Field(min_length=1, max_length=_MAX_POSITIONS)


class JobSubmitResponse(BaseModel):
    """Acknowledgement of a successful submit."""

    status: str
    positions_stored: int


class JobDetail(BaseModel):
    """État d'un job pour inspection/debug (GET /jobs/{id})."""

    job_id: str
    status: str
    model_version: int
    num_sims: int
    claimed_by: str | None
    claimed_at: str | None
    submitted_at: str | None
    created_at: str


class EnqueueRequest(BaseModel):
    """Bulk-enqueue N pending jobs (admin / trainer use)."""

    count: int = Field(ge=1, le=10000)
    model_version: int = Field(ge=1)
    num_sims: int = Field(ge=1, le=10000, default=200)
    opening_fen: str | None = None
    dirichlet_seed_base: int | None = Field(
        default=None,
        description=(
            "If set, each enqueued job j (0-indexed) gets dirichlet_seed = "
            "base + j ; otherwise no seed (worker decides)."
        ),
    )


class EnqueueResponse(BaseModel):
    enqueued: int
    job_ids: list[str]


class CancelPendingRequest(BaseModel):
    """Drain the pending queue (clean premature self-play stop)."""

    model_version: int | None = Field(
        default=None,
        description="If set, only cancel pending jobs for this model_version; "
        "otherwise cancel the entire pending queue.",
    )


class CancelPendingResponse(BaseModel):
    cancelled: int


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@router.post(
    "/claim",
    response_model=JobClaimResponse,
    responses={204: {"description": "No pending job (queue empty)"}},
    # B.1 — route PUBLIQUE (anonyme) : pas de require_api_key. Defense-in-depth : les routes de
    # contrôle (admin/selfplay/enqueue/register/promote/stats/replay) restent key-gated.
)
def claim(
    request: Request,
    x_worker_id: str = Header(default="anonymous", alias="X-Worker-Id"),
) -> JobClaimResponse:
    """Atomically claim ONE pending self-play job for this worker.

    Returns 200 with the job descriptor, or 204 if the queue is empty.
    """
    # Blacklist temporaire des workers NPU (2026-05-19) : la quantization INT8
    # de l'inférence NPU produit des labels MCTS de qualité réduite qui ont
    # dégradé gen-11 (-174 Elo SPRT vs gen-010-promoted). À retirer une fois
    # le worker NPU corrigé (passer en FP16 ou rester sur CPU EP).
    if x_worker_id.startswith(BLACKLISTED_WORKER_PREFIX):
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT)

    # Clean stop : when self-play is paused, refuse claims so the fleet idles
    # without draining the pending queue (resume via POST /selfplay/resume).
    if is_selfplay_paused():
        raise HTTPException(status_code=status.HTTP_204_NO_CONTENT)

    job = claim_job(worker_id=x_worker_id)
    if job is None:
        # On-demand autorefill (never-empty queue) : rather than idle the fleet
        # on an empty queue, mint a fresh job for the current champion — but only
        # when autorefill is enabled AND a model has been promoted. Pause is
        # honoured above, so a paused server never auto-generates.
        enabled, num_sims = get_autorefill()
        champion = current_model() if enabled else None
        if champion is None:
            raise HTTPException(status_code=status.HTTP_204_NO_CONTENT)
        job = create_and_claim_job(
            worker_id=x_worker_id,
            model_version=champion.version,
            num_sims=num_sims,
        )
    return JobClaimResponse(
        job_id=job.id,
        model_version=job.model_version,
        opening_fen=job.opening_fen,
        dirichlet_seed=job.dirichlet_seed,
        num_sims=job.num_sims,
    )


@router.post(
    "/{job_id}/submit",
    response_model=JobSubmitResponse,
    responses={
        410: {"description": "Job not claimed, already completed, or unknown"},
    },
    # B.1 — route PUBLIQUE (anonyme) : pas de require_api_key (cf claim).
)
def submit(
    job_id: str,
    request: Request,
    raw_body: bytes = Depends(_raw_body),
    x_pseudo: str | None = Header(default=None, alias="X-Pseudo"),
    x_contributor: str | None = Header(default=None, alias="X-Contributor"),
) -> JobSubmitResponse:
    """Submit positions + outcome for a previously-claimed job.

    Deux formats acceptés (négociés par Content-Type) :
    - **JSON** (`JobSubmitRequest`, base64) — chemin historique des workers Java, inchangé.
    - **Binaire** ``application/x-nanozero-submit-v1`` (Story A.2) — décodé via
      ``submit_codec`` ; les métadonnées (``model_version`` du job, ``ply``=index,
      ``game_id``=``job_id``, ``fen``="") sont dérivées côté serveur (le navigateur
      non-fiable ne les fournit pas).

    STORY-007 — identité « adresse ouverte » : le header optionnel ``X-Pseudo`` porte un
    label PUBLIC cosmétique pour les crédits. Il est normalisé côté serveur (autorité ;
    NFC + lowercase + charset + blocklist) et annoté sur chaque position. **Absent / invalide /
    offensant → ``None`` (anonyme), jamais de rejet de la soumission** (AC-6). RÉTRO-COMPAT
    flotte native : la flotte Java n'envoie PAS ce header → ``None`` → comportement identique.
    Le pseudo n'influence NI l'acceptation NI l'anti-poison (ceux-ci portent sur le soumetteur,
    cf. AC-8) — pure annotation.

    Returns 200 on success with positions_stored count, or 410 if the job
    can't be transitioned to 'completed' (already done, expired, or unknown).
    """
    # STORY-007 — normalisation SERVER-AUTHORITATIVE du pseudo (jamais confiance au client) :
    # invalide/offensant/absent → None. Ne lève jamais → ne fait jamais échouer le submit (AC-6).
    pseudo = normalize_pseudo(x_pseudo)
    # STORY-015 — token contributeur stable (navigateur) : 16 hex, validé server-side.
    # Absent/invalide → None (fleet : non utilisé ; browser vieux client : non compté).
    contributor_token = normalize_contributor_token(x_contributor)
    # Media-type insensible à la casse (RFC 7231), paramètre charset toléré, match EXACT
    # (pas startswith — éviterait une collision de préfixe type ...-v10000).
    media_type = request.headers.get("content-type", "").split(";")[0].strip().lower()

    # B.4 — provenance SERVER-AUTHORITATIVE : décidée par le Content-Type, JAMAIS
    # lue du payload (un browser pourrait sinon se déclarer 'fleet' et échapper à
    # la quarantaine). Binaire = navigateur bénévole → 'browser' (quarantiné) ;
    # JSON = flotte Java de confiance → 'fleet'.
    if media_type == BINARY_SUBMIT_CONTENT_TYPE:
        positions = _positions_from_binary(raw_body, job_id, pseudo)
        source = "browser"
    else:
        positions = _positions_from_json(raw_body, job_id, pseudo)
        source = "fleet"

    # Atomique : complétion du job + insert des positions dans UNE transaction.
    # None = job non 'claimed' → 410 (rien inséré). Sinon n positions stockées.
    # Évite la perte silencieuse de l'ancien ordre (submit_job committé PUIS insert).
    n = submit_job_with_positions(
        job_id,
        positions,
        source=source,
        contributor_token=contributor_token,
        contributor_named=bool(pseudo),
    )
    if n is None:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Job not claimed, already completed, or unknown",
        )
    return JobSubmitResponse(status="ok", positions_stored=n)


@router.get(
    "/{job_id}",
    response_model=JobDetail,
    responses={404: {"description": "Unknown job id"}},
    dependencies=[Depends(require_api_key)],  # debug opérateur (gated)
)
def job_detail(job_id: str, request: Request) -> JobDetail:
    """Inspecte un job (debug fleet) : statut, worker, timestamps. 404 si inconnu.

    Évite le SQL manuel pour diagnostiquer un worker bloqué ou un job stuck in 'claimed'.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found")
    return JobDetail(
        job_id=job.id,
        status=job.status,
        model_version=job.model_version,
        num_sims=job.num_sims,
        claimed_by=job.claimed_by,
        claimed_at=job.claimed_at,
        submitted_at=job.submitted_at,
        created_at=job.created_at,
    )


def _positions_from_json(
    raw_body: bytes, job_id: str, pseudo: str | None = None
) -> list[Position]:
    """Chemin JSON base64 (workers Java de confiance) — DURCI (revue BMAD).

    ``model_version`` est dérivé du JOB claimé (server-authoritative, comme le binaire) ;
    la valeur du payload est IGNORÉE → un client ne peut plus forger la version (fenêtre de
    replay). Bornes : 1..1024 positions (schéma), et chaque BLOB décodé doit faire la taille
    EXACTE (planes 30464 / policy 18688 octets) sinon le trainer crashe sur un corpus mal
    dimensionné → 422.
    """
    model_version = get_job_model_version(job_id)
    if model_version is None:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Job not claimed, already completed, or unknown",
        )
    try:
        body = JobSubmitRequest.model_validate_json(raw_body)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid submit body: {e.error_count()} validation error(s)",
        ) from e
    positions: list[Position] = []
    for p in body.positions:
        try:
            planes = base64.b64decode(p.input_planes_b64)
            policy = base64.b64decode(p.policy_target_b64)
        except (ValueError, base64.binascii.Error) as e:  # type: ignore[attr-defined]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid base64 BLOB in positions: {e}",
            ) from e
        if len(planes) != _PLANES_BYTES or len(policy) != _POLICY_BYTES:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"BLOB de taille invalide (planes={len(planes)}≠{_PLANES_BYTES} "
                    f"ou policy={len(policy)}≠{_POLICY_BYTES})"
                ),
            )
        positions.append(
            Position(
                game_id=body.game_id,
                model_version=model_version,  # ← du JOB, pas du payload (server-authoritative)
                ply=p.ply,
                fen=p.fen,
                input_planes=planes,
                policy_target=policy,
                outcome=p.outcome,
                pseudo=pseudo,  # STORY-007 — normalisé côté serveur (None = anonyme)
            )
        )
    return positions


def _positions_from_binary(
    raw_body: bytes, job_id: str, pseudo: str | None = None
) -> list[Position]:
    """Décode la variante binaire (Story A.2) ; métadonnées dérivées côté serveur.

    ``model_version`` vient du job claimé (server-authoritative, non du client).
    ``game_id``=``job_id``, ``ply``=index 0-based, ``fen``="" (audit-only, non training).
    """
    model_version = get_job_model_version(job_id)
    if model_version is None:
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="Job not claimed, already completed, or unknown",
        )
    try:
        decoded = submit_codec.decode(raw_body)
    except submit_codec.SubmitDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid binary submit: {e}",
        ) from e
    if not decoded:
        # Un submit vide compléterait le job sans stocker → perte de partie. Rejeter.
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty binary submit (num_positions=0)",
        )
    # B.3 — validation structurelle (browser only) : policy = distribution (Σ≈1, ≥0),
    # planes dans la plage encodable. Échec → 422, rien stocké (submit_job non appelé →
    # le job reste 'claimed' → requeue watchdog). Le chemin JSON/fleet n'est PAS validé.
    try:
        validate_browser_positions(decoded)
    except BrowserValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Soumission browser structurellement invalide : {e}",
        ) from e
    return [
        Position(
            game_id=job_id,
            model_version=model_version,
            ply=ply,
            fen="",
            input_planes=planes_bytes,
            policy_target=policy_bytes,
            outcome=outcome,
            pseudo=pseudo,  # STORY-007 — normalisé côté serveur (None = anonyme)
        )
        for ply, (planes_bytes, policy_bytes, outcome) in enumerate(decoded)
    ]


@router.post(
    "/enqueue",
    response_model=EnqueueResponse,
    status_code=status.HTTP_201_CREATED,
    dependencies=[Depends(require_api_key)],
)
def enqueue(body: EnqueueRequest, request: Request) -> EnqueueResponse:
    """Bulk-enqueue {count} pending jobs with given model_version + num_sims.

    Used by the trainer (after each promote) to feed workers with jobs for the
    new model, and by initial-setup scripts to seed the queue.
    """
    ids: list[str] = []
    for j in range(body.count):
        seed: int | None = (
            (body.dirichlet_seed_base + j) if body.dirichlet_seed_base is not None else None
        )
        job_id = create_job(
            model_version=body.model_version,
            num_sims=body.num_sims,
            opening_fen=body.opening_fen,
            dirichlet_seed=seed,
        )
        ids.append(job_id)
    return EnqueueResponse(enqueued=len(ids), job_ids=ids)


@router.post(
    "/cancel_pending",
    response_model=CancelPendingResponse,
    dependencies=[Depends(require_api_key)],
)
def cancel_pending(body: CancelPendingRequest, request: Request) -> CancelPendingResponse:
    """Delete pending (never-claimed) jobs — clean premature self-play stop.

    In-flight (claimed) jobs are untouched and still submit normally; only the
    not-yet-handed-out backlog is dropped. To stop the fleet reversibly without
    losing the queue, use POST /selfplay/pause instead.
    """
    n = cancel_pending_jobs(model_version=body.model_version)
    return CancelPendingResponse(cancelled=n)
