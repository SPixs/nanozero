"""FastAPI application entry point for nanozero-jobserver.

Phase 13.1 scaffolding: server boots, exposes /health endpoint, returns 200.
Endpoints for jobs/models/replay will be wired in 13.2-13.5.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import uvicorn
from fastapi import Depends, FastAPI, Response, status

from nanozero_jobserver import __version__
from nanozero_jobserver.api import admin as admin_api
from nanozero_jobserver.api import cycle as cycle_api
from nanozero_jobserver.api import jobs as jobs_api
from nanozero_jobserver.api import models as models_api
from nanozero_jobserver.api import replay as replay_api
from nanozero_jobserver.api import sprt as sprt_api
from nanozero_jobserver.api import stats as stats_api
from nanozero_jobserver.auth import require_api_key
from nanozero_jobserver.config import ServerConfig, load_config
from nanozero_jobserver.middleware import GzipRequestMiddleware
from nanozero_jobserver.services.flusher import FlusherService
from nanozero_jobserver.storage.db import close_pool, create_pool, init_schema
from nanozero_jobserver.storage.models import current_model
from nanozero_jobserver.ttl_cache import TTLCache

LOG = logging.getLogger(__name__)


def create_app(config: ServerConfig | None = None) -> FastAPI:
    """Build the FastAPI application.

    Args:
        config: optional override (for tests). Default reads env via load_config.

    Returns:
        Configured FastAPI app ready to serve.
    """
    cfg = config if config is not None else load_config()

    # Open the PostgreSQL connection pool + ensure the schema exists (idempotent).
    create_pool(cfg.database_url, cfg.pg_pool_min, cfg.pg_pool_max)
    init_schema()

    # ADR-015 : background FlusherService for write-through cache (HOT SQLite + COLD NPZ).
    flusher = _build_flusher(cfg)

    @asynccontextmanager
    async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
        if flusher is not None:
            flusher.start()
        try:
            yield
        finally:
            if flusher is not None:
                flusher.stop(timeout=10.0)
            close_pool()

    app = FastAPI(
        title="nanozero-jobserver",
        version=__version__,
        description="Distributed job server for AlphaZero self-play (ADR-014)",
        lifespan=_lifespan,
    )

    # Stash config + flusher on app.state so endpoint handlers / tests can access.
    app.state.config = cfg
    app.state.flusher = flusher
    # #6 BMAD : cache TTL du drift report (scan ~34M lignes → minutes) ; le monitoring
    # le poll, on sert un résultat ≤ 5 min sans recalcul. ?fresh=true bypasse.
    app.state.drift_cache = TTLCache[int, object](ttl_seconds=300.0)
    # STORY-001 (gamification) : cache TTL 60 s de /stats/season (jauge collective publique,
    # pollée par la page volontaire). Clé unique (None) car la réponse est globale, pas par-version.
    app.state.season_cache = TTLCache[None, object](ttl_seconds=60.0)
    # STORY-012 (monitoring admin) : cache TTL 60 s de /stats/browser (SUM source non-caché ~20 s sur
    # ~16M lignes → la page admin la pollait sans cache). Même pattern que season ; clé unique (None).
    app.state.browser_cache = TTLCache[None, object](ttl_seconds=60.0)

    # Inflate gzip-encoded request bodies (worker submits) before routing. Backward-compatible:
    # uncompressed requests pass through untouched. Cuts DevSrv WiFi bandwidth on self-play submits.
    app.add_middleware(GzipRequestMiddleware)

    # Wire HTTP routers (auth-protected per route).
    app.include_router(jobs_api.router)
    app.include_router(models_api.router)
    app.include_router(replay_api.router)
    app.include_router(stats_api.router)
    app.include_router(admin_api.router)
    app.include_router(cycle_api.router)
    app.include_router(sprt_api.router)

    @app.get("/health", tags=["meta"])
    def health(response: Response) -> dict[str, object]:
        """Liveness probe + capacity warnings (#5 revue BMAD).

        **Liveness** = la DB répond à un ``SELECT`` — c'est le SEUL critère de 503.
        Avant, des seuils volumétriques (positions > 1M) renvoyaient 503 EN PERMANENCE
        car la prod tourne à ~11M (retention=2) → la sonde criait au loup 24/7 et
        devenait inutilisable pour un vrai monitoring/LB. Désormais la capacité (taille
        DB/WAL, positions) et la santé du flusher ne font QUE remplir ``warnings``
        (status 200) ; seule une DB injoignable dégrade la liveness (503).
        """
        from nanozero_jobserver.storage.db import connect as _connect

        db_size = 0
        try:
            with _connect() as conn:
                positions = conn.execute("SELECT COUNT(*) AS n FROM positions").fetchone()["n"]
                db_size = conn.execute(
                    "SELECT pg_database_size(current_database()) AS b"
                ).fetchone()["b"]
        except Exception:
            positions = -1

        # Seuils recalibrés sur la réalité prod (DB ~25 Go, ~11M positions). WARNINGS
        # informatifs — ils ne font PAS échouer la liveness (≠ ancien comportement).
        warnings: list[str] = []
        if db_size > 60 * 1024**3:
            warnings.append(f"db_size {db_size / 1024**3:.1f} GB > 60")
        # ~34M observé en prod (retention=2 garde 2 gens flushées + tampon live + browser).
        # Seuil au-dessus du normal → ne flague QUE une accumulation anormale (purge/flush bloqué).
        if positions > 50_000_000:
            warnings.append(f"positions {positions} > 50M")
        flusher = app.state.flusher
        if flusher is not None and flusher.consecutive_tick_failures >= 3:
            warnings.append(f"flusher en échec ({flusher.consecutive_tick_failures} ticks consécutifs)")

        alive = positions != -1  # le SEUL critère de liveness → 503
        if not alive:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {
            "status": "down" if not alive else ("warning" if warnings else "ok"),
            "version": __version__,
            "db_size_gb": round(db_size / 1024**3, 2),
            "positions": positions,
            "warnings": warnings,
        }

    @app.get("/whoami", tags=["meta"], dependencies=[Depends(require_api_key)])
    def whoami() -> dict[str, str | bool]:
        """Auth check endpoint — workers can use this to validate their API key.

        Returns 200 with auth status when the X-API-Key header matches config
        (or in dev mode when no API key is configured). Returns 401 otherwise.
        """
        return {
            "status": "authenticated",
            "auth_enabled": bool(app.state.config.api_key),
        }

    return app


def _build_flusher(cfg: ServerConfig) -> FlusherService | None:
    """Construct the FlusherService from config, or None if disabled.

    `get_current_model_version` callable queries storage/models.current_model
    on each call so purge always uses the latest promoted version.
    """
    if not cfg.flusher_enabled:
        LOG.info("FlusherService disabled (cfg.flusher_enabled=False)")
        return None

    def _current_v() -> int | None:
        m = current_model()
        return m.version if m else None

    return FlusherService(
        output_dir=cfg.resolve_npz_output_dir(),
        flush_threshold=cfg.flush_threshold_positions,
        retention_window=cfg.flush_retention_window,
        tick_interval_seconds=cfg.flush_tick_interval_seconds,
        get_current_model_version=_current_v,
        stale_claim_timeout_seconds=cfg.stale_claim_timeout_seconds,
        stale_claim_max_per_tick=cfg.stale_claim_max_per_tick,
        flush_browser=cfg.flush_browser,
        browser_flush_threshold=cfg.browser_flush_threshold,
    )


def cli() -> None:
    """Console entry point invoked by `nanozero-jobserver` script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )
    cfg = load_config()
    LOG.info("Starting nanozero-jobserver v%s on %s:%d", __version__, cfg.host, cfg.port)
    if not cfg.api_key:
        LOG.warning(
            "NANOZERO_JOBSERVER_API_KEY not set — auth DISABLED (dev mode). "
            "Set the env var in any non-local deployment."
        )
    app = create_app(cfg)
    uvicorn.run(app, host=cfg.host, port=cfg.port, log_level="info")


if __name__ == "__main__":
    cli()
