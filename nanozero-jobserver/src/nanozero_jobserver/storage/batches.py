"""Batches storage — NPZ shards archive registry (ADR-015).

The `batches` table tracks each NPZ shard produced by the FlusherService.
Used to assign deterministic `batch_idx` per model_version and to query the
archive for audit / replay.

Storage : PostgreSQL via psycopg3 (cf. db.py). Le pool fournit `connect()` — plus de
paramètre `db_path`. `id` vient d'un BIGSERIAL via `INSERT ... RETURNING id` (psycopg3 ne
remplit pas `cursor.lastrowid`). Les agrégats (`COUNT`, `MAX`, ...) sont aliasés pour un
accès PAR NOM (dict_row n'a pas d'accès positionnel). Les colonnes TIMESTAMPTZ
(`created_at`, `earliest`, `latest`) reviennent en `datetime` → reconverties en ISO 8601
pour préserver le contrat `str` (dataclass `Batch`) et JSON.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanozero_jobserver.storage.db import connect


@dataclass(frozen=True)
class Batch:
    """One NPZ shard registered in the `batches` table.

    Attributes:
        id: auto-increment primary key.
        model_version: version of the model whose self-play produced these positions.
        batch_idx: sequential index per model_version (0, 1, 2, ...).
        npz_path: filesystem path to the .npz shard.
        n_positions: number of positions in the shard.
        created_at: ISO 8601 UTC timestamp.
    """

    id: int
    model_version: int
    batch_idx: int
    npz_path: str
    n_positions: int
    created_at: str


def insert_batch(
    model_version: int,
    batch_idx: int,
    npz_path: str,
    n_positions: int,
    source: str = "fleet",
) -> int:
    """Register a freshly-flushed NPZ shard. Returns the inserted batches.id.

    Args:
        model_version: version of the model that generated these positions.
        batch_idx: per-version sequence number (use `next_batch_idx` to compute).
        npz_path: absolute or relative path to the shard.
        n_positions: total positions packed in the shard.
        source: provenance du shard ('fleet' / 'browser', chantier 1 cloisonnement).
            ``batch_idx`` reste GLOBAL par version (séquence partagée entre sources) → la
            contrainte UNIQUE(model_version, batch_idx) tient sans changement ; ``source``
            ne fait que tracer + piloter le préfixe de nommage du shard.

    Returns:
        The newly-created row id (= batches.id).

    Raises:
        psycopg.errors.UniqueViolation: if (model_version, batch_idx) is not unique.
    """
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO batches (model_version, batch_idx, npz_path, n_positions, source)"
            " VALUES (%s, %s, %s, %s, %s)"
            " RETURNING id",
            (model_version, batch_idx, npz_path, n_positions, source),
        )
        return int(cur.fetchone()["id"])


def next_batch_idx(model_version: int) -> int:
    """Return the next batch_idx to use for a given model_version.

    Returns 0 if no batch exists for this version yet, otherwise
    max(batch_idx) + 1.

    Args:
        model_version: the model_version to query.

    Returns:
        Next batch_idx to use (0-based sequence).
    """
    with connect() as conn:
        cur = conn.execute(
            "SELECT COALESCE(MAX(batch_idx), -1) + 1 AS next_idx FROM batches"
            " WHERE model_version = %s",
            (model_version,),
        )
        return int(cur.fetchone()["next_idx"])


def list_batches(
    model_version: int | None = None,
    limit: int = 1000,
) -> list[Batch]:
    """List recorded batches, optionally filtered by model_version.

    Args:
        model_version: if set, only return batches for this version.
            None (default) returns batches across all versions.
        limit: max rows to return. Default 1000.

    Returns:
        List of Batch records, ordered by (model_version DESC, batch_idx ASC).
    """
    with connect() as conn:
        if model_version is None:
            cur = conn.execute(
                "SELECT id, model_version, batch_idx, npz_path, n_positions, created_at"
                " FROM batches"
                " ORDER BY model_version DESC, batch_idx ASC"
                " LIMIT %s",
                (limit,),
            )
        else:
            cur = conn.execute(
                "SELECT id, model_version, batch_idx, npz_path, n_positions, created_at"
                " FROM batches"
                " WHERE model_version = %s"
                " ORDER BY batch_idx ASC"
                " LIMIT %s",
                (model_version, limit),
            )
        return [
            Batch(
                id=row["id"],
                model_version=row["model_version"],
                batch_idx=row["batch_idx"],
                npz_path=row["npz_path"],
                n_positions=row["n_positions"],
                created_at=row["created_at"].isoformat() if row["created_at"] else None,
            )
            for row in cur.fetchall()
        ]


def batches_summary(source: str | None = None) -> list[dict[str, object]]:
    """Rollup des shards NPZ par génération — outillage A/B SPRT (D.3).

    Pour décider si le corpus (browser ou fleet) d'une gen est assez gros pour un A/B :
    n_shards, n_positions, fenêtre temporelle, par version. ``source`` ('browser'/'fleet')
    filtre la cohorte ; None = toutes. Lit le registre durable ``batches`` (survit au purge).
    """
    clause = ""
    if source == "browser":
        clause = " WHERE source = 'browser'"
    elif source == "fleet":
        clause = " WHERE source != 'browser'"
    with connect() as conn:
        cur = conn.execute(
            "SELECT model_version, COUNT(*) AS n_shards, SUM(n_positions) AS n_positions,"
            "  MIN(created_at) AS earliest, MAX(created_at) AS latest"
            f" FROM batches{clause}"
            " GROUP BY model_version ORDER BY model_version DESC"
        )
        return [
            {
                "model_version": int(r["model_version"]),
                "n_shards": int(r["n_shards"]),
                "n_positions": int(r["n_positions"] or 0),
                "earliest": r["earliest"].isoformat() if r["earliest"] else None,
                "latest": r["latest"].isoformat() if r["latest"] else None,
            }
            for r in cur.fetchall()
        ]


def count_batches(model_version: int | None = None) -> int:
    """Count batches, optionally filtered by model_version."""
    with connect() as conn:
        if model_version is None:
            cur = conn.execute("SELECT COUNT(*) AS n FROM batches")
        else:
            cur = conn.execute(
                "SELECT COUNT(*) AS n FROM batches WHERE model_version = %s",
                (model_version,),
            )
        return int(cur.fetchone()["n"])


def count_batches_by_version() -> dict[int, int]:
    """Batch count per model_version en UNE requête (évite le N+1 de /stats/by-version,
    qui appelait ``count_batches`` une fois PAR génération)."""
    with connect() as conn:
        cur = conn.execute("SELECT model_version, COUNT(*) AS n FROM batches GROUP BY model_version")
        return {int(r["model_version"]): int(r["n"]) for r in cur.fetchall()}


def sum_positions_by_version(source: str | None = None) -> dict[int, int]:
    """Durable count of flushed positions per model_version (SUM of n_positions).

    This is the AUTHORITATIVE cumulative count of positions a model generated:
    it survives the HOT-cache purge (`delete_flushed_old`) because the batches
    registry is never purged, only the positions BLOBs are. The `positions`
    table count under-reports by exactly the purged volume, so any "how many
    positions did gen-N produce" question must read here, not `count_positions`.

    Args:
        source: provenance filter sur la colonne ``batches.source`` (B4-D2). None
            (défaut) = toutes sources (observabilité). ``"fleet"`` exclut les shards
            browser ``browser-gen*`` — À UTILISER pour le déclencheur d'entraînement,
            sinon le volume browser flushé compte dans ``should_train``.

    Returns:
        Mapping {model_version: total positions flushed to NPZ}, all versions.
    """
    where = ""
    if source == "fleet":
        where = " WHERE source != 'browser'"
    elif source == "browser":
        where = " WHERE source = 'browser'"
    with connect() as conn:
        cur = conn.execute(
            f"SELECT model_version, SUM(n_positions) AS n FROM batches{where} GROUP BY model_version"
        )
        return {int(row["model_version"]): int(row["n"]) for row in cur.fetchall()}


def batch_timestamps(model_version: int) -> list[str]:
    """Return the created_at timestamps of a version's batches, oldest first.

    Used to compute self-play cadence (shards/hour) without pulling the full
    Batch rows. ISO 8601 UTC strings, ascending by batch_idx.

    Args:
        model_version: target model_version.

    Returns:
        List of ISO timestamp strings (may be empty).
    """
    with connect() as conn:
        cur = conn.execute(
            "SELECT created_at FROM batches WHERE model_version = %s ORDER BY batch_idx ASC",
            (model_version,),
        )
        return [row["created_at"].isoformat() for row in cur.fetchall()]
