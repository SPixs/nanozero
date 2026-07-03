"""Model registry storage : register, list, get current promoted.

Models are immutable once registered. `promoted_at` is set when the model
becomes the current champion (after a successful SPRT H1). At any time exactly
one model has the latest non-null `promoted_at` — that's `current`.

Files (.onnx) are stored on disk at `onnx_path`; this table only holds the
metadata (version, hash, path).

Storage : PostgreSQL via psycopg3 (cf. db.py). Le pool fournit `connect()` —
plus de paramètre `db_path`. Les colonnes TIMESTAMPTZ (`promoted_at`,
`created_at`) reviennent en `datetime` ; on les reconvertit en chaîne ISO 8601
dans `_row_to_model` pour préserver le contrat `str` de la dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanozero_jobserver.storage.db import connect


@dataclass(frozen=True)
class ModelRecord:
    """One entry in the models registry.

    Attributes:
        version: monotonic integer (gen-001 == 1, gen-002 == 2, ...).
        name: human-friendly identifier (e.g., "gen-005-trained").
        onnx_path: filesystem path to the .onnx blob.
        sha256_onnx: hex digest of the .onnx file (integrity check).
        promoted_at: ISO timestamp when this model became `current`, or None.
        parent_version: version of the model this one was trained from.
        created_at: ISO timestamp when this row was inserted.
    """

    version: int
    name: str
    onnx_path: str
    sha256_onnx: str
    promoted_at: str | None
    parent_version: int | None
    created_at: str


def register_model(
    version: int,
    name: str,
    onnx_path: str,
    sha256_onnx: str,
    parent_version: int | None = None,
) -> None:
    """Insert a new (non-promoted) model record. Raises if version already exists.

    Args:
        version: monotonic integer for this model.
        name: unique human-friendly id.
        onnx_path: filesystem path to the .onnx file.
        sha256_onnx: hex digest of the .onnx file.
        parent_version: version this model was trained from (None for init).
    """
    with connect() as conn:
        conn.execute(
            "INSERT INTO models (version, name, onnx_path, sha256_onnx, parent_version)"
            " VALUES (%s, %s, %s, %s, %s)",
            (version, name, onnx_path, sha256_onnx, parent_version),
        )


def promote_model(version: int) -> bool:
    """Mark a model as the new current champion. Returns True if newly promoted.

    Sets promoted_at to NOW for the target version. Does not unset previous
    champions — `current_model()` simply returns the most recently promoted one,
    which preserves audit history.

    Args:
        version: version to promote.

    Returns:
        True if promoted_at was set (model existed AND was previously NULL).
        False if model doesn't exist or was already promoted.
    """
    with connect() as conn:
        cur = conn.execute(
            "UPDATE models SET promoted_at=NOW()"
            " WHERE version=%s AND promoted_at IS NULL",
            (version,),
        )
        return cur.rowcount == 1


def set_current_champion(version: int) -> bool:
    """Force ``version`` à devenir le champion COURANT (rollback / re-promotion).

    Re-stampe ``promoted_at`` à NOW **même si la version était déjà promue** (contrairement
    à ``promote_model`` qui ne touche que ``promoted_at IS NULL``) → ``current_model`` la
    renvoie comme champion (le plus récemment promu). Usage : revenir au champion précédent
    après un SPRT rejeté (cf. gen-026) sans chirurgie SQL en prod.

    Returns:
        True si la version existe (ligne mise à jour), False sinon.
    """
    with connect() as conn:
        cur = conn.execute(
            "UPDATE models SET promoted_at=NOW() WHERE version=%s",
            (version,),
        )
        return cur.rowcount == 1


def current_model() -> ModelRecord | None:
    """Return the currently-promoted champion (most recent promoted_at).

    Returns None if no model has ever been promoted.
    """
    with connect() as conn:
        cur = conn.execute(
            "SELECT version, name, onnx_path, sha256_onnx, promoted_at,"
            "  parent_version, created_at"
            " FROM models"
            " WHERE promoted_at IS NOT NULL"
            " ORDER BY promoted_at DESC"
            " LIMIT 1"
        )
        row = cur.fetchone()
        return _row_to_model(row) if row else None


def get_model(version: int) -> ModelRecord | None:
    """Look up a model by version. Returns None if not found."""
    with connect() as conn:
        cur = conn.execute(
            "SELECT version, name, onnx_path, sha256_onnx, promoted_at,"
            "  parent_version, created_at"
            " FROM models WHERE version=%s",
            (version,),
        )
        row = cur.fetchone()
        return _row_to_model(row) if row else None


def list_models(limit: int = 50) -> list[ModelRecord]:
    """Return models in descending version order (most recent first)."""
    with connect() as conn:
        cur = conn.execute(
            "SELECT version, name, onnx_path, sha256_onnx, promoted_at,"
            "  parent_version, created_at"
            " FROM models ORDER BY version DESC LIMIT %s",
            (limit,),
        )
        return [_row_to_model(row) for row in cur.fetchall()]


def _row_to_model(row: dict[str, Any]) -> ModelRecord:
    promoted_at = row["promoted_at"]
    created_at = row["created_at"]
    return ModelRecord(
        version=row["version"],
        name=row["name"],
        onnx_path=row["onnx_path"],
        sha256_onnx=row["sha256_onnx"],
        promoted_at=promoted_at.isoformat() if promoted_at else None,
        parent_version=row["parent_version"],
        created_at=created_at.isoformat() if created_at else None,
    )
