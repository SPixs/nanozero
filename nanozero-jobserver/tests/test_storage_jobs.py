"""Unit tests for storage/jobs.py — job lifecycle."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

import pytest
from nanozero_jobserver.storage.db import init_schema
from nanozero_jobserver.storage.jobs import (
    claim_job,
    count_jobs_by_status,
    count_jobs_by_version_status,
    create_job,
    delete_stale_claims,
    get_job,
    requeue_stale_jobs,
    submit_job,
    submit_job_with_positions,
    worker_stats,
)
from nanozero_jobserver.storage.replay_buffer import Position, count_positions


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    p = tmp_path / "jobs.db"
    init_schema(p)
    return p


# -----------------------------------------------------------------------------
# Create + claim
# -----------------------------------------------------------------------------


def test_create_job_returns_uuid(db: Path) -> None:
    job_id = create_job(db, model_version=1, num_sims=200)
    # UUID4 form : 8-4-4-4-12 hex chars with dashes
    assert len(job_id) == 36
    assert job_id.count("-") == 4


def test_create_job_persists_fields(db: Path) -> None:
    job_id = create_job(
        db,
        model_version=7,
        num_sims=400,
        opening_fen="rnbqkbnr/...",
        dirichlet_seed=42,
    )
    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row["status"] == "pending"
    assert row["model_version"] == 7
    assert row["num_sims"] == 400
    assert row["opening_fen"] == "rnbqkbnr/..."
    assert row["dirichlet_seed"] == 42
    assert row["claimed_by"] is None


def test_claim_returns_none_on_empty_queue(db: Path) -> None:
    assert claim_job(db, worker_id="w-1") is None


def test_claim_marks_status_and_worker(db: Path) -> None:
    create_job(db, model_version=1, num_sims=200)
    job = claim_job(db, worker_id="w-1")
    assert job is not None
    assert job.status == "claimed"
    assert job.claimed_by == "w-1"
    assert job.claimed_at is not None


def test_claim_returns_oldest_first(db: Path) -> None:
    """FIFO ordering — workers don't starve."""
    a = create_job(db, model_version=1, num_sims=200)
    time.sleep(0.005)  # ensure distinct timestamps
    b = create_job(db, model_version=1, num_sims=200)
    time.sleep(0.005)
    c = create_job(db, model_version=1, num_sims=200)

    assert claim_job(db, "w").id == a
    assert claim_job(db, "w").id == b
    assert claim_job(db, "w").id == c
    assert claim_job(db, "w") is None


def test_claim_atomicity_multiple_pending_no_double_claim(db: Path) -> None:
    """Each pending job is claimed exactly once even if claim is called multiple times."""
    job_ids = [create_job(db, model_version=1, num_sims=200) for _ in range(5)]
    claimed = []
    while (j := claim_job(db, "w")) is not None:
        claimed.append(j.id)
    assert sorted(claimed) == sorted(job_ids)


def test_claim_atomic_under_real_concurrency(db: Path) -> None:
    """#10 BMAD : 8 workers EN PARALLÈLE → chaque job claimé EXACTEMENT une fois.

    Le test existant ci-dessus est séquentiel (un seul thread en boucle) ; il ne prouve
    PAS l'atomicité du claim sous vraie concurrence. Ici 8 threads démarrent ensemble
    (Barrier) et claiment jusqu'à épuisement ; on vérifie zéro doublon — le verrou
    SQLite (UPDATE ... WHERE status='pending' RETURNING) + busy_timeout doivent tenir.
    """
    n_jobs = 60
    n_workers = 8
    created = {create_job(db, model_version=1, num_sims=200) for _ in range(n_jobs)}
    claimed: list[str] = []
    lock = threading.Lock()
    start = threading.Barrier(n_workers)

    def worker(wid: int) -> None:
        start.wait()  # tous démarrent en même temps → contention maximale
        while True:
            job = claim_job(db, worker_id=f"w{wid}")
            if job is None:
                return
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert len(claimed) == n_jobs  # tous les jobs claimés (rien perdu)
    assert len(set(claimed)) == n_jobs  # AUCUN doublon (pas de double-claim concurrent)
    assert set(claimed) == created  # exactement les jobs créés


# -----------------------------------------------------------------------------
# Submit
# -----------------------------------------------------------------------------


def test_submit_completes_claimed_job(db: Path) -> None:
    job_id = create_job(db, model_version=1, num_sims=200)
    claim_job(db, "w-1")
    assert submit_job(db, job_id) is True

    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT status, submitted_at FROM jobs WHERE id=?", (job_id,)).fetchone()
    assert row["status"] == "completed"
    assert row["submitted_at"] is not None


def test_submit_unclaimed_job_returns_false(db: Path) -> None:
    """Submitting a job that wasn't claimed (still pending) is rejected."""
    job_id = create_job(db, model_version=1, num_sims=200)
    assert submit_job(db, job_id) is False


# -----------------------------------------------------------------------------
# Submit ATOMIQUE : job complété + positions insérées dans UNE transaction
# -----------------------------------------------------------------------------


def _pos(game_id: str = "g", ply: int = 0, mv: int = 1) -> Position:
    return Position(
        game_id=game_id,
        model_version=mv,
        ply=ply,
        fen="startpos",
        input_planes=b"\x00" * 16,
        policy_target=b"\xff" * 8,
        outcome=0.0,
    )


def _job_status(db: Path, job_id: str) -> str | None:
    with sqlite3.connect(str(db)) as conn:
        row = conn.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
    return row[0] if row else None


def test_submit_with_positions_happy(db: Path) -> None:
    job_id = create_job(db, model_version=1, num_sims=200)
    claim_job(db, "w-1")
    n = submit_job_with_positions(db, job_id, [_pos("g", 0), _pos("g", 1)], source="fleet")
    assert n == 2
    assert _job_status(db, job_id) == "completed"
    assert count_positions(db) == 2


def test_submit_with_positions_not_claimed_returns_none_inserts_nothing(db: Path) -> None:
    job_id = create_job(db, model_version=1, num_sims=200)  # pending, jamais claimé
    assert submit_job_with_positions(db, job_id, [_pos()], source="fleet") is None
    assert count_positions(db) == 0  # rien inséré sur un job non-claimé


def test_submit_with_positions_double_submit_no_duplicate(db: Path) -> None:
    job_id = create_job(db, model_version=1, num_sims=200)
    claim_job(db, "w-1")
    assert submit_job_with_positions(db, job_id, [_pos(), _pos("g", 1)]) == 2
    # 2e submit (job déjà completed) → None, AUCUN doublon inséré (idempotence préservée)
    assert submit_job_with_positions(db, job_id, [_pos(), _pos("g", 1)]) is None
    assert count_positions(db) == 2


def test_submit_with_positions_empty_completes_job(db: Path) -> None:
    job_id = create_job(db, model_version=1, num_sims=200)
    claim_job(db, "w-1")
    assert submit_job_with_positions(db, job_id, []) == 0
    assert _job_status(db, job_id) == "completed"
    assert count_positions(db) == 0


def test_submit_with_positions_rolls_back_on_insert_failure(
    db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Le cœur du fix : si l'INSERT échoue, la complétion du job ROLLBACK aussi
    (atomicité) → pas de perte silencieuse (le job reste 'claimed', le worker re-soumet)."""
    job_id = create_job(db, model_version=1, num_sims=200)
    claim_job(db, "w-1")
    # INSERT volontairement cassé (mauvaise arité) → executemany lève → toute la
    # transaction (UPDATE job + INSERT) rollback.
    monkeypatch.setattr(
        "nanozero_jobserver.storage.jobs.INSERT_POSITIONS_SQL",
        "INSERT INTO positions (game_id) VALUES (?, ?)",
    )
    with pytest.raises(sqlite3.Error):
        submit_job_with_positions(db, job_id, [_pos(), _pos("g", 1)], source="fleet")
    # Job PAS marqué completed (rollback) ET aucune position insérée → rien de perdu.
    assert _job_status(db, job_id) == "claimed"
    assert count_positions(db) == 0


def test_submit_unknown_job_returns_false(db: Path) -> None:
    assert submit_job(db, "non-existent-id") is False


def test_submit_already_completed_returns_false(db: Path) -> None:
    """Idempotence — calling submit twice doesn't break, just returns False."""
    job_id = create_job(db, model_version=1, num_sims=200)
    claim_job(db, "w-1")
    assert submit_job(db, job_id) is True
    assert submit_job(db, job_id) is False


# -----------------------------------------------------------------------------
# Requeue stale
# -----------------------------------------------------------------------------


def test_requeue_stale_recovers_claimed_job(db: Path) -> None:
    """Jobs claimed long ago without submit should be requeueable."""
    create_job(db, model_version=1, num_sims=200)
    claim_job(db, "w-dead")

    # Backdate claimed_at to 2 hours ago.
    with sqlite3.connect(str(db)) as conn:
        conn.execute("UPDATE jobs SET claimed_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-2 hours')")

    requeued = requeue_stale_jobs(db, timeout_seconds=3600)
    assert requeued == 1

    # Now it's pending again and a fresh worker can claim it.
    job = claim_job(db, "w-fresh")
    assert job is not None
    assert job.claimed_by == "w-fresh"


def test_requeue_stale_leaves_fresh_jobs_alone(db: Path) -> None:
    """A job claimed seconds ago is not stale yet."""
    create_job(db, model_version=1, num_sims=200)
    claim_job(db, "w-active")

    requeued = requeue_stale_jobs(db, timeout_seconds=3600)
    assert requeued == 0


# -----------------------------------------------------------------------------
# Delete stale claims (watchdog — wired into the flusher)
# -----------------------------------------------------------------------------


def _backdate_claims(db: Path, expr: str = "-2 hours") -> None:
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            f"UPDATE jobs SET claimed_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '{expr}')"
            " WHERE status='claimed'"
        )


def test_delete_stale_claims_removes_old(db: Path) -> None:
    """A job claimed long ago without submit is a dead row → deleted."""
    create_job(db, model_version=1, num_sims=200)
    claim_job(db, "w-dead")
    _backdate_claims(db)

    assert delete_stale_claims(db, timeout_seconds=3600) == 1
    assert count_jobs_by_status(db).get("claimed", 0) == 0


def test_delete_stale_claims_keeps_fresh(db: Path) -> None:
    """A job claimed seconds ago may still be in progress → untouched."""
    create_job(db, model_version=1, num_sims=200)
    claim_job(db, "w-active")

    assert delete_stale_claims(db, timeout_seconds=3600) == 0
    assert count_jobs_by_status(db).get("claimed", 0) == 1


def test_delete_stale_claims_ignores_completed(db: Path) -> None:
    """Only 'claimed' rows are eligible — completed jobs are never deleted."""
    job_id = create_job(db, model_version=1, num_sims=200)
    claim_job(db, "w")
    assert submit_job(db, job_id) is True  # -> completed
    # Even if its (old) claimed_at is ancient, status='completed' protects it.
    with sqlite3.connect(str(db)) as conn:
        conn.execute("UPDATE jobs SET claimed_at=strftime('%Y-%m-%dT%H:%M:%fZ', 'now', '-2 hours')")

    assert delete_stale_claims(db, timeout_seconds=3600) == 0
    assert count_jobs_by_status(db).get("completed", 0) == 1


def test_delete_stale_claims_max_total_caps_and_drains(db: Path) -> None:
    """max_total caps a sweep (gradual drain) ; small batches still cover all."""
    for _ in range(5):
        create_job(db, model_version=1, num_sims=200)
        claim_job(db, "w-dead")
    _backdate_claims(db)

    # First capped sweep deletes exactly 3 (batch_size < max_total exercises batching).
    assert delete_stale_claims(db, timeout_seconds=3600, batch_size=2, max_total=3) == 3
    assert count_jobs_by_status(db).get("claimed", 0) == 2

    # Uncapped follow-up drains the remaining 2.
    assert delete_stale_claims(db, timeout_seconds=3600, batch_size=2) == 2
    assert count_jobs_by_status(db).get("claimed", 0) == 0


def test_count_jobs_by_status(db: Path) -> None:
    """Counters for the /stats endpoint."""
    create_job(db, model_version=1, num_sims=200)
    create_job(db, model_version=1, num_sims=200)
    create_job(db, model_version=1, num_sims=200)
    claim_job(db, "w-1")  # 1 claimed, 2 pending

    counts = count_jobs_by_status(db)
    assert counts["pending"] == 2
    assert counts["claimed"] == 1


def test_count_jobs_by_version_status(db: Path) -> None:
    """Two-level histogram keyed by model_version then status."""
    create_job(db, model_version=16, num_sims=800)
    create_job(db, model_version=16, num_sims=800)
    create_job(db, model_version=17, num_sims=800)
    claim_job(db, "w")  # claims oldest pending (v16)

    hist = count_jobs_by_version_status(db)
    assert hist[16] == {"pending": 1, "claimed": 1}
    assert hist[17] == {"pending": 1}


def test_worker_stats_attribution_and_order(db: Path) -> None:
    create_job(db, model_version=1, num_sims=200)
    create_job(db, model_version=1, num_sims=200)
    j = claim_job(db, "w-old")
    submit_job(db, j.id)  # w-old: 1 completed
    claim_job(db, "w-new")  # w-new: 1 in-flight, claimed later

    stats = worker_stats(db)
    by_id = {w.worker_id: w for w in stats}
    assert by_id["w-old"].completed == 1
    assert by_id["w-old"].in_flight == 0
    assert by_id["w-new"].in_flight == 1
    # Ordered by last_claimed_at DESC → most recently active worker first.
    assert stats[0].worker_id == "w-new"


def test_worker_stats_excludes_unclaimed_jobs(db: Path) -> None:
    create_job(db, model_version=1, num_sims=200)  # never claimed → claimed_by NULL
    assert worker_stats(db) == []


def test_worker_stats_idle_seconds_populated(db: Path) -> None:
    """#focus suivi workers : idle_seconds = secondes depuis le dernier claim (≥ 0)."""
    create_job(db, model_version=1, num_sims=200)
    claim_job(db, "w-x")
    stats = worker_stats(db)
    assert len(stats) == 1
    assert stats[0].idle_seconds is not None
    assert stats[0].idle_seconds >= 0.0


def test_worker_stats_limit_caps_rows(db: Path) -> None:
    """#focus suivi workers : limit borne la réponse (anti-explosion worker_ids)."""
    for i in range(5):
        create_job(db, model_version=1, num_sims=200)
        claim_job(db, f"w-{i}")
    assert len(worker_stats(db)) == 5  # tous sans limit
    assert len(worker_stats(db, limit=2)) == 2  # borné


def test_get_job_returns_detail(db: Path) -> None:
    job_id = create_job(db, model_version=3, num_sims=400)
    claim_job(db, "w-1")
    j = get_job(db, job_id)
    assert j is not None
    assert j.id == job_id
    assert j.status == "claimed"
    assert j.model_version == 3
    assert j.num_sims == 400
    assert j.claimed_by == "w-1"


def test_get_job_unknown_returns_none(db: Path) -> None:
    assert get_job(db, "nope") is None
