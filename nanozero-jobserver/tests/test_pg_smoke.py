"""Smoke test of the SQLite→PostgreSQL migration on a REAL PostgreSQL.

Exercises the code paths that touch the SQL DIALECT (placeholders, RETURNING,
``ANY(%s)`` arrays, ``BYTEA`` bytes, ``ON CONFLICT`` upserts, ``EXTRACT(EPOCH ...)``,
``SUM(CASE ...)``, ``TIMESTAMPTZ`` → ISO strings) end-to-end against
``postgres:16-alpine`` (see ``conftest.py``). Not exhaustive coverage — a tripwire
for migration regressions.
"""

from __future__ import annotations

import base64

from fastapi.testclient import TestClient

from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app
from nanozero_jobserver.storage.batches import insert_batch, list_batches, next_batch_idx
from nanozero_jobserver.storage.control import is_selfplay_paused, set_selfplay_paused
from nanozero_jobserver.storage.jobs import (
    claim_job,
    create_and_claim_job,
    create_job,
    get_job,
    submit_job_with_positions,
    worker_stats,
)
from nanozero_jobserver.storage.models import (
    current_model,
    get_model,
    promote_model,
    register_model,
)
from nanozero_jobserver.storage.replay_buffer import (
    Position,
    count_positions,
    count_unflushed_positions,
    delete_flushed_old,
    insert_positions,
    iter_unflushed_positions,
    mark_positions_flushed,
    sample_positions,
)
from nanozero_jobserver.submit_codec import PLANES_BYTES, POLICY_LEN

_POLICY_BYTES = POLICY_LEN * 4  # 18688 — exact size the JSON submit path enforces


def _pos(
    game_id: str = "g0",
    ply: int = 0,
    model_version: int = 1,
    outcome: float = 0.5,
    planes: bytes = b"\x00" * 64,
    policy: bytes = b"\xff" * 32,
    pseudo: str | None = None,
) -> Position:
    return Position(
        game_id=game_id,
        model_version=model_version,
        ply=ply,
        fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        input_planes=planes,
        policy_target=policy,
        outcome=outcome,
        pseudo=pseudo,
    )


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# -----------------------------------------------------------------------------
# storage/replay_buffer — BYTEA, count, sample, RANDOM()
# -----------------------------------------------------------------------------


def test_replay_buffer_insert_count_sample(pg: str) -> None:
    assert insert_positions([_pos(game_id=f"g{i}", ply=i) for i in range(5)]) == 5
    assert count_positions() == 5
    assert count_positions(min_model_version=1) == 5
    assert count_positions(min_model_version=2) == 0  # all version 1

    sample = sample_positions(n=3, current_model_version=1, window=2)
    assert len(sample) == 3
    assert all(p.model_version == 1 for p in sample)

    # BYTEA round-trip THROUGH the zlib compress(write)/decompress(read) wrapper.
    rt = _pos(game_id="rt", planes=bytes(range(256)), policy=b"\xde\xad\xbe\xef")
    insert_positions([rt])
    got = next(
        p
        for p in sample_positions(n=100, current_model_version=1, window=2)
        if p.game_id == "rt"
    )
    assert got.input_planes == bytes(range(256))
    assert got.policy_target == b"\xde\xad\xbe\xef"
    assert got.outcome == rt.outcome


def test_replay_buffer_flush_cycle(pg: str) -> None:
    """count_unflushed / iter_unflushed / insert_batch RETURNING / mark ANY / purge."""
    insert_positions(
        [_pos(game_id="g", ply=i, model_version=3) for i in range(4)], source="fleet"
    )
    assert count_unflushed_positions() == 4
    assert count_unflushed_positions(model_version=3) == 4
    assert count_unflushed_positions(source="fleet") == 4

    rows = iter_unflushed_positions(model_version=3, limit=10, source="fleet")
    assert len(rows) == 4
    ids = [r.id for r in rows]
    assert ids == sorted(ids)  # FIFO id-order, ids populated from BIGSERIAL

    # next_batch_idx (COALESCE MAX) + insert_batch (RETURNING id).
    assert next_batch_idx(3) == 0
    batch_id = insert_batch(
        model_version=3, batch_idx=0, npz_path="/tmp/shard.npz", n_positions=4
    )
    assert batch_id >= 1
    assert next_batch_idx(3) == 1
    assert [b.id for b in list_batches(model_version=3)] == [batch_id]

    # mark_positions_flushed uses WHERE id = ANY(%s) (psycopg3 list→array).
    assert mark_positions_flushed(ids, batch_id) == 4
    assert count_unflushed_positions() == 0

    # purge flushed rows (batched DELETE loop, fresh tx per batch).
    assert delete_flushed_old(max_model_version=3) == 4
    assert count_positions() == 0


# -----------------------------------------------------------------------------
# storage/jobs — lifecycle + atomic submit + worker_stats aggregates
# -----------------------------------------------------------------------------


def test_jobs_lifecycle_and_worker_stats(pg: str) -> None:
    job_id = create_job(model_version=2, num_sims=200, opening_fen="fen", dirichlet_seed=7)
    pending = get_job(job_id)
    assert pending is not None and pending.status == "pending"
    assert isinstance(pending.created_at, str)  # TIMESTAMPTZ → ISO

    claimed = claim_job(worker_id="w1")
    assert claimed is not None
    assert claimed.id == job_id and claimed.status == "claimed"
    assert claimed.claimed_by == "w1"
    assert isinstance(claimed.claimed_at, str)  # NOW() datetime → ISO string

    # Atomic completion + positions insert in ONE transaction.
    n = submit_job_with_positions(
        job_id, [_pos(game_id="game", ply=i, model_version=2) for i in range(3)]
    )
    assert n == 3
    done = get_job(job_id)
    assert done is not None and done.status == "completed"
    assert isinstance(done.submitted_at, str)
    assert count_positions() == 3

    # Idempotent double-submit: job no longer 'claimed' → None, nothing inserted.
    assert submit_job_with_positions(job_id, [_pos(game_id="dup")]) is None
    assert count_positions() == 3

    # worker_stats: SUM(CASE ...) + EXTRACT(EPOCH FROM (NOW() - MAX(claimed_at))).
    stats = worker_stats()
    assert len(stats) == 1
    ws = stats[0]
    assert ws.worker_id == "w1"
    assert ws.completed == 1 and ws.in_flight == 0
    assert isinstance(ws.last_claimed_at, str)
    assert ws.idle_seconds is not None and ws.idle_seconds >= 0.0

    # windowed variant (claimed_at >= NOW() - N * INTERVAL '1 second').
    assert len(worker_stats(since_seconds=3600, limit=10)) == 1


def test_create_and_claim_job(pg: str) -> None:
    """On-demand autorefill path: single INSERT straight into 'claimed' + RETURNING."""
    job = create_and_claim_job(worker_id="w2", model_version=5, num_sims=400)
    assert job.status == "claimed"
    assert job.model_version == 5 and job.num_sims == 400
    assert job.claimed_by == "w2"
    assert isinstance(job.claimed_at, str)


# -----------------------------------------------------------------------------
# storage/models — register / promote / current (TIMESTAMPTZ → ISO)
# -----------------------------------------------------------------------------


def test_models_register_promote_current(pg: str) -> None:
    register_model(version=1, name="gen-001", onnx_path="/m/1.onnx", sha256_onnx="aaa")
    register_model(
        version=2, name="gen-002", onnx_path="/m/2.onnx", sha256_onnx="bbb", parent_version=1
    )
    assert current_model() is None  # nothing promoted yet

    assert promote_model(1) is True
    assert promote_model(1) is False  # already promoted (promoted_at NOT NULL)

    cur = current_model()
    assert cur is not None and cur.version == 1
    assert isinstance(cur.promoted_at, str)  # TIMESTAMPTZ → ISO string
    assert isinstance(cur.created_at, str)

    assert promote_model(2) is True
    cur2 = current_model()
    assert cur2 is not None and cur2.version == 2  # most-recently promoted wins
    assert get_model(2) is not None and get_model(2).parent_version == 1


# -----------------------------------------------------------------------------
# storage/control — UPSERT (ON CONFLICT DO UPDATE ... NOW())
# -----------------------------------------------------------------------------


def test_control_pause_upsert(pg: str) -> None:
    assert is_selfplay_paused() is False  # unset → None → False
    set_selfplay_paused(True)  # INSERT
    assert is_selfplay_paused() is True
    set_selfplay_paused(False)  # ON CONFLICT DO UPDATE
    assert is_selfplay_paused() is False


# -----------------------------------------------------------------------------
# API — create_app(ServerConfig) end-to-end through FastAPI / TestClient
# -----------------------------------------------------------------------------


def test_api_smoke_enqueue_claim_submit(pg: str) -> None:
    cfg = ServerConfig(
        host="127.0.0.1",
        port=8090,
        api_key="",  # dev mode: auth disabled
        database_url=pg,  # SAME url as the fixture pool → create_pool idempotent
        flusher_enabled=False,
    )
    # Not used as a context manager on purpose: lifespan would close the fixture
    # pool on shutdown. Routes work without lifespan; the flusher is disabled anyway.
    client = TestClient(create_app(cfg))

    # /health: alive (DB answers), zero positions at the start.
    r = client.get("/health")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["positions"] == 0
    assert body["status"] in ("ok", "warning")

    # A promoted champion so /stats/selfplay reports a current cycle.
    register_model(version=1, name="gen-001", onnx_path="/m/1.onnx", sha256_onnx="aaa")
    assert promote_model(1) is True

    # enqueue (key-gated, dev mode passes) → claim (public) → submit (JSON = fleet).
    r = client.post("/jobs/enqueue", json={"count": 1, "model_version": 1, "num_sims": 200})
    assert r.status_code == 201, r.text
    assert r.json()["enqueued"] == 1

    r = client.post("/jobs/claim", headers={"X-Worker-Id": "smoke-worker"})
    assert r.status_code == 200, r.text
    job_id = r.json()["job_id"]
    assert r.json()["model_version"] == 1

    submit_body = {
        "game_id": "game-1",
        "model_version": 1,  # ignored server-side (derived from the job)
        "positions": [
            {
                "ply": i,
                "fen": f"fen-{i}",
                "input_planes_b64": _b64(b"\x00" * PLANES_BYTES),
                "policy_target_b64": _b64(b"\x00" * _POLICY_BYTES),
                "outcome": 0.0,
            }
            for i in range(3)
        ],
    }
    r = client.post(f"/jobs/{job_id}/submit", json=submit_body)
    assert r.status_code == 200, r.text
    assert r.json()["positions_stored"] == 3

    # /health now sees the 3 inserted positions.
    assert client.get("/health").json()["positions"] == 3

    # /stats/selfplay reflects the completed cycle.
    r = client.get("/stats/selfplay")
    assert r.status_code == 200, r.text
    sp = r.json()
    assert sp["model_version"] == 1
    assert sp["positions_live"] == 3  # unflushed tail (flusher disabled)
    assert sp["positions_browser"] == 0
    assert sp["jobs_completed"] == 1
    assert sp["jobs_pending"] == 0
    assert sp["paused"] is False
    assert sp["autorefill_enabled"] is False
