"""Job lifecycle storage : create, claim, submit, requeue stale.

State machine :
    pending  -> (claim) ->  claimed  -> (submit) -> completed
                            claimed  -> (timeout) -> expired -> (requeue) -> pending

Workers atomically claim a pending job via "UPDATE ... WHERE status='pending'
LIMIT 1 RETURNING ...". Stale-claim recovery is the server's responsibility :
periodically call `requeue_stale_jobs(timeout=N)` to flip jobs claimed > N
seconds ago back to pending.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from nanozero_jobserver.storage.contributors import (
    fleet_machine_token,
    touch_contributor,
)
from nanozero_jobserver.storage.db import connect
from nanozero_jobserver.storage.replay_buffer import (
    INSERT_POSITIONS_SQL,
    Position,
    build_position_rows,
)


def _iso(value: datetime | None) -> str | None:
    """PostgreSQL ``TIMESTAMPTZ`` (datetime) → ISO 8601 string, or None.

    The :class:`Job` / :class:`WorkerStat` dataclasses type their timestamp fields
    as ``str`` (legacy SQLite stored ISO text). psycopg3 returns ``datetime``
    objects, so convert at the storage boundary to preserve the JSON contract.
    """
    return value.isoformat() if value is not None else None


def _row_to_job(row: dict) -> Job:
    """Build a :class:`Job` from a ``dict_row``, ISO-formatting PG datetimes.

    Centralises the column→field mapping shared by ``claim_job``,
    ``create_and_claim_job`` and ``get_job``. ``created_at`` is NOT NULL
    (``DEFAULT NOW()``) so it is always present.
    """
    return Job(
        id=row["id"],
        status=row["status"],
        model_version=row["model_version"],
        opening_fen=row["opening_fen"],
        dirichlet_seed=row["dirichlet_seed"],
        num_sims=row["num_sims"],
        claimed_by=row["claimed_by"],
        claimed_at=_iso(row["claimed_at"]),
        submitted_at=_iso(row["submitted_at"]),
        created_at=row["created_at"].isoformat(),
    )


@dataclass(frozen=True)
class Job:
    """A self-play job ready to be claimed by a worker.

    Attributes:
        id: server-generated UUID4 string.
        status: pending|claimed|completed|expired.
        model_version: which model the worker should fetch and use.
        opening_fen: starting position FEN (None = standard startpos).
        dirichlet_seed: deterministic seed for the worker's noise.
        num_sims: MCTS simulations per move.
        claimed_by: worker_id if currently claimed, else None.
        claimed_at: ISO timestamp string if claimed, else None.
        submitted_at: ISO timestamp string if completed, else None.
        created_at: ISO timestamp string when the job was enqueued.
    """

    id: str
    status: str
    model_version: int
    opening_fen: str | None
    dirichlet_seed: int | None
    num_sims: int
    claimed_by: str | None
    claimed_at: str | None
    submitted_at: str | None
    created_at: str


def create_job(
    model_version: int,
    num_sims: int,
    opening_fen: str | None = None,
    dirichlet_seed: int | None = None,
) -> str:
    """Enqueue a new pending job. Returns the job id (UUID4).

    Args:
        model_version: target model version the worker must use.
        num_sims: MCTS simulations per move (e.g., 200).
        opening_fen: optional starting position; None = standard startpos.
        dirichlet_seed: optional seed for reproducible noise; None = worker decides.

    Returns:
        Newly-created job id.
    """
    job_id = str(uuid.uuid4())
    with connect() as conn:
        conn.execute(
            "INSERT INTO jobs (id, status, model_version, opening_fen, dirichlet_seed, num_sims)"
            " VALUES (%s, 'pending', %s, %s, %s, %s)",
            (job_id, model_version, opening_fen, dirichlet_seed, num_sims),
        )
    return job_id


def claim_job(worker_id: str) -> Job | None:
    """Atomically claim ONE pending job for `worker_id`. Returns None if queue empty.

    Implementation:
        Single UPDATE with RETURNING (PostgreSQL). Atomic — no race even with
        multiple worker connections polling concurrently (the targeted row is
        row-locked for the duration of the UPDATE).

    Args:
        worker_id: identifier of the claiming worker (free-form string).

    Returns:
        Job with status='claimed', or None if no pending job.
    """
    with connect() as conn:
        cur = conn.execute(
            "UPDATE jobs SET"
            "  status='claimed',"
            "  claimed_by=%s,"
            "  claimed_at=NOW()"
            " WHERE id = (SELECT id FROM jobs WHERE status='pending' ORDER BY created_at LIMIT 1)"
            " RETURNING id, status, model_version, opening_fen, dirichlet_seed, num_sims,"
            " claimed_by, claimed_at, submitted_at, created_at",
            (worker_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return _row_to_job(row)


def create_and_claim_job(
    worker_id: str,
    model_version: int,
    num_sims: int,
    opening_fen: str | None = None,
    dirichlet_seed: int | None = None,
) -> Job:
    """Mint a fresh job already 'claimed' by `worker_id`, and return it.

    Used by /jobs/claim when the pending queue is empty and on-demand autorefill
    is enabled (storage.control.get_autorefill): instead of returning 204, the
    server synthesises a job for the current champion so the fleet never idles
    on an empty queue.

    Single INSERT straight into 'claimed' (no transient 'pending' row): there is
    no race even with many workers minting concurrently — each gets its own UUID
    row. `dirichlet_seed=None` lets the worker pick its own noise, exactly like a
    default-enqueued job.

    Args:
        worker_id: identifier of the claiming worker.
        model_version: champion version the worker must fetch and use.
        num_sims: MCTS simulations per move.
        opening_fen: optional starting position; None = standard startpos.
        dirichlet_seed: optional seed; None = worker decides.

    Returns:
        The freshly-minted Job with status='claimed'.
    """
    job_id = str(uuid.uuid4())
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO jobs"
            "  (id, status, model_version, opening_fen, dirichlet_seed, num_sims,"
            "   claimed_by, claimed_at)"
            " VALUES (%s, 'claimed', %s, %s, %s, %s, %s, NOW())"
            " RETURNING id, status, model_version, opening_fen, dirichlet_seed,"
            "   num_sims, claimed_by, claimed_at, submitted_at, created_at",
            (job_id, model_version, opening_fen, dirichlet_seed, num_sims, worker_id),
        )
        row = cur.fetchone()
        return _row_to_job(row)


# Transition claimed→completed (partagée par submit_job et le submit ATOMIQUE).
# RETURNING claimed_by : le submit ATOMIQUE en a besoin pour attribuer la cohorte fleet
# (worker natif → machine) sans requête supplémentaire. ``submit_job`` ignore le retour.
_COMPLETE_JOB_SQL = (
    "UPDATE jobs SET"
    "  status='completed',"
    "  submitted_at=NOW()"
    " WHERE id = %s AND status = 'claimed'"
    " RETURNING claimed_by"
)


def submit_job(job_id: str) -> bool:
    """Mark a claimed job as completed. Returns True if the transition was valid.

    Args:
        job_id: id of the job to mark completed.

    Returns:
        True if the row was updated (status was 'claimed'). False if not found,
        already completed, or expired.
    """
    with connect() as conn:
        cur = conn.execute(_COMPLETE_JOB_SQL, (job_id,))
        return cur.rowcount == 1


def _record_contributor(
    conn,
    source: str,
    claimed_by: str | None,
    contributor_token: str | None,
    named: bool,
) -> None:
    """Compte le contributeur du submit (STORY-016), dans la transaction ``conn``.

    - ``browser`` : identité = token ``X-Contributor`` stable (STORY-015). Sans token (vieux
      client) → non compté (pas d'identité stable). ``named`` = un pseudo opt-in accompagne le submit.
    - ``fleet`` : identité = machine dérivée du ``claimed_by`` natif (suffixe de slot retiré).
    """
    if source == "browser":
        if contributor_token:
            touch_contributor(conn, contributor_token, "browser", named)
    elif claimed_by and not claimed_by.startswith("browser-"):
        touch_contributor(conn, fleet_machine_token(claimed_by), "fleet", False)


def submit_job_with_positions(
    job_id: str,
    positions: list[Position],
    source: str = "fleet",
    contributor_token: str | None = None,
    contributor_named: bool = False,
) -> int | None:
    """Atomically complete a claimed job AND insert its positions — or neither.

    La transition du job ET l'INSERT des positions tournent dans UNE seule transaction
    (``connect`` commit à la sortie propre du ``with``, rollback sur exception). Garanties :

    - job pas 'claimed' (déjà completed / expired / inconnu) → renvoie ``None`` et
      **rien** n'est inséré (l'appelant mappe None → 410 ; l'idempotence du submit est
      préservée : un double-submit n'insère pas de doublon).
    - **toute** erreur DB pendant l'INSERT → la transaction ENTIÈRE rollback, donc le job
      reste 'claimed' (le worker peut re-soumettre, le watchdog le récupère). Un submit ne
      marque donc JAMAIS un job terminé en perdant silencieusement ses positions — ce que
      l'ancien ordre en deux transactions (``submit_job`` puis ``insert_positions``)
      pouvait faire.

    Returns:
        Nombre de positions insérées (0 si la liste est vide), ou ``None`` si le job
        n'était pas 'claimed'.
    """
    rows = build_position_rows(positions, source)
    with connect() as conn:
        completed = conn.execute(_COMPLETE_JOB_SQL, (job_id,)).fetchone()
        if completed is None:
            return None
        if rows:
            conn.cursor().executemany(INSERT_POSITIONS_SQL, rows)
        # STORY-016 — décompte contributeur dans la MÊME transaction (atomique avec le submit).
        _record_contributor(
            conn, source, completed["claimed_by"], contributor_token, contributor_named
        )
    return len(rows)


def get_job_model_version(job_id: str) -> int | None:
    """Return the ``model_version`` of a job, or None if the job is unknown.

    Used by the binary submit path (Story A.2) pour dériver ``model_version``
    côté serveur au lieu de faire confiance au client (navigateur non-fiable).
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT model_version FROM jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
        return int(row["model_version"]) if row is not None else None


def get_job(job_id: str) -> Job | None:
    """Fetch a job by id for inspection/debug. None if unknown.

    Évite le SQL manuel pour diagnostiquer un worker bloqué / un job stuck in claimed.
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT id, status, model_version, opening_fen, dirichlet_seed, num_sims,"
            "  claimed_by, claimed_at, submitted_at, created_at FROM jobs WHERE id = %s",
            (job_id,),
        ).fetchone()
    if row is None:
        return None
    return _row_to_job(row)


def requeue_stale_jobs(timeout_seconds: int) -> int:
    """Flip jobs claimed > timeout_seconds ago from 'claimed' back to 'pending'.

    Used as a watchdog : a worker that claims a job but dies before submitting
    leaves the job stuck in 'claimed'. Periodically call this to recover.

    Args:
        timeout_seconds: positive duration ; jobs claimed_at older than now-this
            are eligible for requeue.

    Returns:
        Number of jobs requeued.
    """
    with connect() as conn:
        cur = conn.execute(
            "UPDATE jobs SET status='pending', claimed_by=NULL, claimed_at=NULL"
            " WHERE status='claimed'"
            "   AND claimed_at < NOW() - %s * INTERVAL '1 second'",
            (int(timeout_seconds),),
        )
        return cur.rowcount


def delete_stale_claims(
    timeout_seconds: int,
    batch_size: int = 50_000,
    max_total: int | None = None,
) -> int:
    """Delete jobs stuck in 'claimed' for more than timeout_seconds (watchdog).

    A worker that claims a job then dies/restarts before submitting leaves the
    job in 'claimed' forever. Because the pending queue is auto-refilled, these
    NEVER block production — but they accumulate as dead rows (observed : 9.8M
    stale claims over 17 days, because requeue_stale_jobs was designed but never
    wired into a background task).

    We DELETE rather than requeue : the self-play jobs are fungible (autorefill
    regenerates fresh CURRENT-version jobs on demand), whereas requeuing a 17-day
    backlog would flood `pending` with millions of stale, often old-model-version
    tasks. Batched with a FRESH transaction per batch (one ``with connect()`` per
    iteration, committed on clean exit) : PostgreSQL long transactions don't block
    readers (MVCC), but a monolithic multi-million-row DELETE generates heavy WAL +
    dead tuples and stalls autovacuum. Per-batch commits cap the bloat and let a
    periodic caller drain gradually (cf. incident 2026-05-22, when the SQLite
    equivalent held the write lock 10+ min and the DB grew to 128 GB).

    Args:
        timeout_seconds: claims with claimed_at older than now-this are eligible.
        batch_size: rows per DELETE batch (default 50_000).
        max_total: stop after deleting this many (None = until none remain). Lets
            a periodic caller drain a large backlog gradually across ticks rather
            than in one long sweep.

    Returns:
        Number of jobs deleted (cumulative across batches).
    """
    total = 0
    while max_total is None or total < max_total:
        limit = batch_size if max_total is None else min(batch_size, max_total - total)
        with connect() as conn:  # une transaction par batch (commit à la sortie du with)
            cur = conn.execute(
                "DELETE FROM jobs WHERE id IN ("
                "  SELECT id FROM jobs"
                "  WHERE status='claimed'"
                "    AND claimed_at < NOW() - %s * INTERVAL '1 second'"
                f"  LIMIT {int(limit)}"
                ")",
                (int(timeout_seconds),),
            )
            n = int(cur.rowcount)
        if n == 0:
            break
        total += n
    return total


def cancel_pending_jobs(model_version: int | None = None) -> int:
    """Delete pending (never-claimed) jobs. Returns the number removed.

    Clean premature self-play stop : drops jobs that haven't been handed out yet
    so the fleet goes idle, without touching claimed/completed rows (in-flight
    games still submit normally). Optionally restrict to one model_version.

    Args:
        model_version: if set, only cancel pending jobs for this version. None
            cancels the whole pending queue.

    Returns:
        Count of pending jobs deleted.
    """
    with connect() as conn:
        if model_version is None:
            cur = conn.execute("DELETE FROM jobs WHERE status = 'pending'")
        else:
            cur = conn.execute(
                "DELETE FROM jobs WHERE status = 'pending' AND model_version = %s",
                (model_version,),
            )
        return int(cur.rowcount)


def count_jobs_by_status() -> dict[str, int]:
    """Histogram of jobs by status. Useful for /stats endpoint."""
    with connect() as conn:
        cur = conn.execute(
            "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status",
        )
        return {row["status"]: int(row["n"]) for row in cur.fetchall()}


def count_jobs_by_version_status() -> dict[int, dict[str, int]]:
    """Two-level histogram : {model_version: {status: count}}.

    Lets /stats/by-version report queue depth per generation (how many pending
    jobs remain for the in-flight model, how many completed, etc.) in one query.

    Returns:
        Nested mapping keyed by model_version then status.
    """
    out: dict[int, dict[str, int]] = {}
    with connect() as conn:
        cur = conn.execute(
            "SELECT model_version, status, COUNT(*) AS n"
            " FROM jobs GROUP BY model_version, status"
        )
        for row in cur.fetchall():
            out.setdefault(int(row["model_version"]), {})[row["status"]] = int(row["n"])
    return out


@dataclass(frozen=True)
class WorkerStat:
    """Per-worker activity summary for fleet observability.

    Attributes:
        worker_id: the X-Worker-Id the worker self-reported on claim.
        completed: jobs this worker has submitted (status='completed').
        in_flight: jobs currently 'claimed' by this worker (unfinished).
        last_claimed_at: most recent claim timestamp (ISO 8601 UTC), proxy for
            "last seen alive". None only if the worker has no claim on record.
        idle_seconds: secondes écoulées depuis le dernier claim — pour repérer un
            nœud MORT d'un coup d'œil (trier par idle_seconds DESC), sans parser les
            timestamps à la main. None si pas de claim.
    """

    worker_id: str
    completed: int
    in_flight: int
    last_claimed_at: str | None
    idle_seconds: float | None


def worker_stats(
    since_seconds: int | None = None, limit: int | None = None
) -> list[WorkerStat]:
    """Per-worker job attribution, ordered by last activity (most recent first).

    Powers /stats/workers : at ~54-64 workers across DevSrv/W3090/OVH/W1080/...
    this is how we spot a dead or under-performing node. `claimed_by` is the
    free-form X-Worker-Id header; rows with NULL claimed_by (never-claimed jobs)
    are excluded.

    Args:
        since_seconds: if set, only count claims newer than now-this (rolling
            window — e.g. 3600 for "active in the last hour"). None counts all
            history.
        limit: borne le nombre de lignes retournées (les plus récemment actives,
            ORDER BY last_claimed_at DESC). None = toutes. Protège contre l'explosion
            de la réponse si des worker_ids aléatoires (browser buggé) prolifèrent.

    Returns:
        List of WorkerStat, ordered by last_claimed_at DESC.
    """
    where = "claimed_by IS NOT NULL"
    params: list[object] = []
    if since_seconds is not None:
        where += " AND claimed_at >= NOW() - %s * INTERVAL '1 second'"
        params.append(int(since_seconds))
    sql = (
        "SELECT claimed_by AS w,"
        "  SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,"
        "  SUM(CASE WHEN status = 'claimed' THEN 1 ELSE 0 END) AS in_flight,"
        "  MAX(claimed_at) AS last_claimed_at,"
        "  EXTRACT(EPOCH FROM (NOW() - MAX(claimed_at))) AS idle_seconds"
        f" FROM jobs WHERE {where}"
        " GROUP BY claimed_by"
        " ORDER BY last_claimed_at DESC"
    )
    if limit is not None:
        sql += " LIMIT %s"
        params.append(int(limit))
    with connect() as conn:
        cur = conn.execute(sql, tuple(params))
        return [
            WorkerStat(
                worker_id=row["w"],
                completed=int(row["completed"] or 0),
                in_flight=int(row["in_flight"] or 0),
                last_claimed_at=_iso(row["last_claimed_at"]),
                idle_seconds=(
                    round(float(row["idle_seconds"]), 1)
                    if row["idle_seconds"] is not None
                    else None
                ),
            )
            for row in cur.fetchall()
        ]
