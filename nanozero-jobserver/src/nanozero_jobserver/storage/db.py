"""PostgreSQL connection pool + schema initialization.

Design (migré de SQLite WAL → PostgreSQL, cf. docs/PG-MIGRATION-plan.md) :
- Driver psycopg3 (`psycopg`) + `ConnectionPool` thread-safe. Le pool est un
  singleton de module, créé au démarrage via `create_pool(database_url)`. Les
  handlers FastAPI (sync, threadpool) ET le thread daemon `FlusherService`
  partagent le pool sans verrou applicatif.
- `connect()` est un context manager qui emprunte une connexion au pool : commit
  implicite à la sortie propre du `with`, rollback sur exception, retour au pool.
  Plus de paramètre `db_path` (rupture de signature assumée vs SQLite).
- `row_factory=dict_row` : accès aux colonnes PAR NOM uniquement (`row["status"]`).
  ⚠️ L'accès POSITIONNEL (`row[0]`, `fetchone()[0]`) n'existe PAS avec dict_row —
  toujours nommer les colonnes agrégées (`SELECT COUNT(*) AS n` → `row["n"]`).
- `init_schema()` est idempotent (`CREATE TABLE IF NOT EXISTS`) : sûr à chaque
  démarrage. Base vierge en prod → pas de migration de colonnes (le schéma
  complet inclut déjà flushed_to_npz/batch_id/compressed/source/pseudo).

Tables (cf. ADR-014 §Storage + ADR-015 write-through cache) :
  positions  — replay buffer HOT cache (BYTEA, purgé après flush NPZ)
  jobs       — cycle de vie des jobs self-play (pending → claimed → completed)
  models     — versions de modèles enregistrées
  batches    — registre des shards NPZ (ADR-015)
  server_control — flags de contrôle (pause self-play, ...)
  sprt_results   — historique des décisions SPRT
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

# DDL PostgreSQL complet (cf. PG-MIGRATION-plan.md §2). Statements séparés par ';'
# et exécutés un à un (psycopg3 = un statement par execute en protocole étendu).
_PG_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS batches (
    id            BIGSERIAL PRIMARY KEY,
    model_version INTEGER NOT NULL,
    batch_idx     INTEGER NOT NULL,
    npz_path      TEXT NOT NULL,
    n_positions   INTEGER NOT NULL,
    source        TEXT NOT NULL DEFAULT 'fleet',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (model_version, batch_idx)
);
CREATE INDEX IF NOT EXISTS idx_batches_model_version ON batches(model_version);

CREATE TABLE IF NOT EXISTS positions (
    id             BIGSERIAL PRIMARY KEY,
    game_id        TEXT NOT NULL,
    model_version  INTEGER NOT NULL,
    ply            INTEGER NOT NULL,
    fen            TEXT NOT NULL,
    input_planes   BYTEA NOT NULL,
    policy_target  BYTEA NOT NULL,
    outcome        REAL NOT NULL,
    flushed_to_npz BOOLEAN NOT NULL DEFAULT FALSE,
    batch_id       BIGINT REFERENCES batches(id),
    compressed     SMALLINT NOT NULL DEFAULT 0,
    source         TEXT NOT NULL DEFAULT 'fleet',
    pseudo         TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_positions_model_version ON positions(model_version);
CREATE INDEX IF NOT EXISTS idx_positions_created_at ON positions(created_at);
CREATE INDEX IF NOT EXISTS idx_positions_flushed ON positions(flushed_to_npz, model_version);

CREATE TABLE IF NOT EXISTS jobs (
    id             TEXT PRIMARY KEY,
    status         TEXT NOT NULL CHECK (status IN ('pending', 'claimed', 'completed', 'expired')),
    model_version  INTEGER NOT NULL,
    opening_fen    TEXT,
    dirichlet_seed INTEGER,
    num_sims       INTEGER NOT NULL,
    claimed_by     TEXT,
    claimed_at     TIMESTAMPTZ,
    submitted_at   TIMESTAMPTZ,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_claimed_at ON jobs(claimed_at);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at) WHERE status = 'pending';

CREATE TABLE IF NOT EXISTS models (
    version        INTEGER PRIMARY KEY,
    name           TEXT NOT NULL UNIQUE,
    onnx_path      TEXT NOT NULL,
    sha256_onnx    TEXT NOT NULL,
    promoted_at    TIMESTAMPTZ,
    parent_version INTEGER REFERENCES models(version),
    created_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_models_promoted_at ON models(promoted_at);

CREATE TABLE IF NOT EXISTS server_control (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sprt_results (
    id                BIGSERIAL PRIMARY KEY,
    candidate_version INTEGER NOT NULL,
    baseline_version  INTEGER NOT NULL,
    decision          TEXT NOT NULL CHECK (decision IN ('accepted', 'rejected', 'inconclusive')),
    elo_estimate      REAL,
    games_played      INTEGER,
    wins              INTEGER,
    draws             INTEGER,
    losses            INTEGER,
    notes             TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_sprt_candidate ON sprt_results(candidate_version);
CREATE INDEX IF NOT EXISTS idx_sprt_baseline ON sprt_results(baseline_version);

CREATE TABLE IF NOT EXISTS contributors (
    token      TEXT PRIMARY KEY,
    source     TEXT NOT NULL DEFAULT 'browser',
    first_seen TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_seen  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    named      BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_contributors_last_seen ON contributors(last_seen);
"""

_pool: ConnectionPool | None = None


def create_pool(database_url: str, min_size: int = 2, max_size: int = 10) -> ConnectionPool:
    """Create (idempotently) the global connection pool.

    Args:
        database_url: libpq conninfo / URL, e.g. ``postgresql://u:p@host:5432/db``.
        min_size: connections kept warm.
        max_size: hard cap (handlers HTTP + flusher thread se partagent ce pool ;
            doit rester < ``max_connections`` côté serveur PG).

    Returns:
        The opened ConnectionPool (also stored module-globally).
    """
    global _pool
    if _pool is not None:
        return _pool
    _pool = ConnectionPool(
        conninfo=database_url,
        min_size=min_size,
        max_size=max_size,
        kwargs={"row_factory": dict_row},
        open=True,
    )
    return _pool


def close_pool() -> None:
    """Close the global pool (shutdown). No-op if not created."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


def _statements(script: str) -> list[str]:
    """Split a multi-statement DDL script into individual statements.

    Safe here because the DDL contains no ';' inside string literals or function
    bodies. Comment-only / empty chunks are dropped.
    """
    out: list[str] = []
    for chunk in script.split(";"):
        # Garde le chunk s'il contient au moins une ligne SQL (pas vide / pas que des commentaires).
        if any(line.strip() and not line.strip().startswith("--") for line in chunk.splitlines()):
            out.append(chunk.strip())
    return out


def init_schema() -> None:
    """Create tables/indexes if absent. Idempotent — safe to call on every start.

    Uses the global pool (call :func:`create_pool` first).
    """
    with connect() as conn:
        for stmt in _statements(_PG_SCHEMA_SQL):
            conn.execute(stmt)


@contextmanager
def connect() -> Iterator[psycopg.Connection]:
    """Borrow a pooled connection (dict_row).

    Yields:
        A psycopg3 Connection. psycopg3 commits on clean ``with`` exit, rolls
        back on exception, and returns the connection to the pool. Column access
        is BY NAME only (``row["col"]``) — positional access is unsupported.
    """
    assert _pool is not None, "Connection pool not initialized — call create_pool() first"
    with _pool.connection() as conn:
        yield conn
