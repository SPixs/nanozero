"""Database maintenance — storage metrics + manual VACUUM (PostgreSQL).

Migré de SQLite WAL → PostgreSQL (cf. docs/PG-MIGRATION-plan.md §3.8). Ce fichier
était le plus SQLite-spécifique du projet : les leviers disque manuels (WAL
checkpoint, VACUUM INTO, rebuild compact) n'ont plus d'équivalent en PG et sont
supprimés — PostgreSQL gère son WAL automatiquement et l'``autovacuum`` récupère
en continu les dead tuples laissés par les purges (``delete_flushed_old``).

Restent utiles :
- ``storage_stats`` : empreinte disque via le catalogue PG (``pg_database_size`` /
  ``pg_total_relation_size``) + comptes positions.
- ``count_purgeable`` : combien de positions flushées une purge libérerait.
- ``vacuum_analyze`` : déclenchement manuel d'un ``VACUUM ANALYZE positions``
  (l'autovacuum suffit en régime normal ; ce levier sert après une grosse purge).

``VACUUM`` / ``VACUUM ANALYZE`` ne peuvent PAS tourner dans une transaction. En
psycopg3 on bascule la connexion empruntée au pool en ``autocommit = True`` AVANT
le statement (aucune transaction n'est encore ouverte sur une connexion fraîche).
"""

from __future__ import annotations

from dataclasses import dataclass

from nanozero_jobserver.storage.db import connect


@dataclass(frozen=True)
class StorageStats:
    """Snapshot of the on-disk footprint (PostgreSQL catalog view).

    Attributes:
        db_bytes: total size of the database (``pg_database_size``).
        positions_bytes: size of the ``positions`` table incl. indexes + TOAST
            (``pg_total_relation_size``) — the HOT cache footprint.
        positions_total: rows in ``positions`` (-1 if ``detailed=False``).
        positions_flushed: positions already written to an NPZ shard
            (``flushed_to_npz = TRUE``) (-1 if ``detailed=False``).
        jobs_by_status: histogram of jobs by status.
    """

    db_bytes: int
    positions_bytes: int
    positions_total: int
    positions_flushed: int
    jobs_by_status: dict[str, int]


def storage_stats(detailed: bool = True) -> StorageStats:
    """Gather DB footprint + maintenance metrics. Read-only.

    Args:
        detailed: si True (défaut) calcule les ``COUNT(*)`` exacts sur la table
            ``positions`` (scan potentiellement lourd sur des dizaines de M de
            lignes). Si False (#6 BMAD), ``positions_total`` / ``positions_flushed``
            valent ``-1`` et l'appel est quasi instantané — les tailles disque
            (catalogue) et le histogramme jobs restent cheap dans tous les cas.
            Mettre True seulement quand on en a besoin (ex. avant un VACUUM).
    """
    with connect() as conn:
        db_bytes = int(
            conn.execute("SELECT pg_database_size(current_database()) AS n").fetchone()["n"]
        )
        positions_bytes = int(
            conn.execute("SELECT pg_total_relation_size('positions') AS n").fetchone()["n"]
        )
        jobs = {
            row["status"]: int(row["n"])
            for row in conn.execute("SELECT status, COUNT(*) AS n FROM jobs GROUP BY status")
        }
        if detailed:
            positions_total = int(
                conn.execute("SELECT COUNT(*) AS n FROM positions").fetchone()["n"]
            )
            positions_flushed = int(
                conn.execute(
                    "SELECT COUNT(*) AS n FROM positions WHERE flushed_to_npz = TRUE"
                ).fetchone()["n"]
            )
        else:
            positions_total = positions_flushed = -1  # non calculé (cf. detailed)

    return StorageStats(
        db_bytes=db_bytes,
        positions_bytes=positions_bytes,
        positions_total=positions_total,
        positions_flushed=positions_flushed,
        jobs_by_status=jobs,
    )


def count_purgeable(max_model_version: int, source: str | None = None) -> int:
    """Count flushed positions eligible for purge (flushed AND mv <= bound).

    Lets the API report how much a purge would free before committing to it.
    ``source`` ('browser' / 'fleet') restreint le compte à une cohorte (B4-D1) ; None = toutes.
    """
    clause = ""
    if source == "browser":
        clause = " AND source = 'browser'"
    elif source == "fleet":
        clause = " AND source != 'browser'"
    with connect() as conn:
        cur = conn.execute(
            "SELECT COUNT(*) AS n FROM positions"
            f" WHERE flushed_to_npz = TRUE AND model_version <= %s{clause}",
            (max_model_version,),
        )
        return int(cur.fetchone()["n"])


def vacuum_analyze() -> None:
    """Run ``VACUUM ANALYZE positions`` — reclaim dead tuples + refresh planner stats.

    Levier manuel post-purge : en régime normal l'``autovacuum`` (réglé agressif
    côté serveur PG, cf. PG-MIGRATION-plan.md §4) suffit. ``VACUUM`` ne peut pas
    tourner dans une transaction → on bascule la connexion empruntée en
    ``autocommit = True`` (sûr : aucune transaction n'est encore ouverte sur une
    connexion fraîche). On restaure ensuite l'état pour ne pas le fuiter au
    prochain emprunteur du pool.
    """
    with connect() as conn:
        conn.autocommit = True
        try:
            conn.execute("VACUUM ANALYZE positions")
        finally:
            conn.autocommit = False
