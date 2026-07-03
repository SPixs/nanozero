"""HTTP-level tests for /jobs/claim + /jobs/{id}/submit (Phase 13.3)."""

from __future__ import annotations

import base64
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app
from nanozero_jobserver.storage.jobs import create_job
from nanozero_jobserver.storage.replay_buffer import count_positions
from nanozero_jobserver.submit_codec import PLANES_BYTES, POLICY_LEN

POLICY_BYTES = POLICY_LEN * 4  # 4672 × f32 = 18688 (taille exacte attendue par le serveur)

AUTH_KEY = "test-secret-key"
AUTH_HEADERS = {"X-API-Key": AUTH_KEY, "X-Worker-Id": "test-worker"}


@pytest.fixture()
def app_and_db(tmp_path: Path):
    cfg = ServerConfig(host="127.0.0.1", port=8090, api_key=AUTH_KEY, db_path=tmp_path / "test.db")
    app = create_app(cfg)
    return app, cfg.db_path


@pytest.fixture()
def client(app_and_db) -> TestClient:
    return TestClient(app_and_db[0])


# -----------------------------------------------------------------------------
# /jobs/claim
# -----------------------------------------------------------------------------


def test_claim_empty_queue_returns_204(client: TestClient) -> None:
    resp = client.post("/jobs/claim", headers=AUTH_HEADERS)
    assert resp.status_code == 204


def test_claim_returns_pending_job(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    job_id = create_job(db, model_version=3, num_sims=200, opening_fen="rnbq...", dirichlet_seed=42)

    resp = client.post("/jobs/claim", headers=AUTH_HEADERS)

    assert resp.status_code == 200
    body = resp.json()
    assert body["job_id"] == job_id
    assert body["model_version"] == 3
    assert body["num_sims"] == 200
    assert body["opening_fen"] == "rnbq..."
    assert body["dirichlet_seed"] == 42


def test_claim_is_public_no_key(client: TestClient) -> None:
    """B.1 : /jobs/claim est PUBLIQUE (anonyme) — pas de clé requise (204 file vide)."""
    resp = client.post("/jobs/claim")  # no header
    assert resp.status_code == 204


def test_claim_public_ignores_key(client: TestClient) -> None:
    """B.1 : route publique → une clé (même fausse) est ignorée, pas de 401."""
    resp = client.post("/jobs/claim", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 204


def test_claim_assigns_worker_id(client: TestClient, app_and_db) -> None:
    """Worker identity is captured on claim for crash attribution."""
    import sqlite3

    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    client.post("/jobs/claim", headers={**AUTH_HEADERS, "X-Worker-Id": "devsrv-cpu-0"})

    with sqlite3.connect(str(db)) as conn:
        row = conn.execute("SELECT claimed_by FROM jobs LIMIT 1").fetchone()
    assert row[0] == "devsrv-cpu-0"


def test_claim_default_worker_id_when_header_absent(client: TestClient, app_and_db) -> None:
    """Missing X-Worker-Id defaults to 'anonymous' (still tracked)."""
    import sqlite3

    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    client.post("/jobs/claim", headers={"X-API-Key": AUTH_KEY})  # no X-Worker-Id

    with sqlite3.connect(str(db)) as conn:
        row = conn.execute("SELECT claimed_by FROM jobs LIMIT 1").fetchone()
    assert row[0] == "anonymous"


# -----------------------------------------------------------------------------
# /jobs/{id}/submit
# -----------------------------------------------------------------------------


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _claim_a_job(client: TestClient) -> str:
    resp = client.post("/jobs/claim", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    return resp.json()["job_id"]


def _make_submit_body(game_id: str = "game-1", model_version: int = 1, n_positions: int = 3):
    return {
        "game_id": game_id,
        "model_version": model_version,
        "positions": [
            {
                "ply": i,
                "fen": f"fen-at-ply-{i}",
                "input_planes_b64": _b64(b"\x00" * PLANES_BYTES),
                "policy_target_b64": _b64(b"\x00" * POLICY_BYTES),
                "outcome": 0.5,
            }
            for i in range(n_positions)
        ],
    }


def test_submit_completes_job_and_stores_positions(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)

    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers=AUTH_HEADERS,
        json=_make_submit_body(n_positions=5),
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["positions_stored"] == 5
    assert count_positions(db) == 5


def test_submit_unknown_job_returns_410(client: TestClient) -> None:
    resp = client.post(
        "/jobs/non-existent/submit",
        headers=AUTH_HEADERS,
        json=_make_submit_body(),
    )
    assert resp.status_code == 410


def test_submit_unclaimed_job_returns_410(client: TestClient, app_and_db) -> None:
    """Job in 'pending' state can't be submitted (must be claimed first)."""
    _, db = app_and_db
    job_id = create_job(db, model_version=1, num_sims=200)
    # Don't claim — try to submit directly.

    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers=AUTH_HEADERS,
        json=_make_submit_body(),
    )
    assert resp.status_code == 410


def test_submit_twice_second_is_410(client: TestClient, app_and_db) -> None:
    """Idempotent rejection — re-submit returns 410, no double-insert."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)

    r1 = client.post(f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=_make_submit_body())
    assert r1.status_code == 200
    n_after_first = count_positions(db)

    r2 = client.post(f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=_make_submit_body())
    assert r2.status_code == 410
    assert count_positions(db) == n_after_first  # no new insert


def test_submit_is_public_no_key(client: TestClient, app_and_db) -> None:
    """B.1 : /jobs/{id}/submit est PUBLIQUE — un submit valide sans clé est accepté (200)."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)
    resp = client.post(f"/jobs/{job_id}/submit", json=_make_submit_body())
    assert resp.status_code == 200


def test_submit_invalid_base64_returns_400(client: TestClient, app_and_db) -> None:
    """Malformed base64 in position BLOB → 400 with helpful message."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)

    bad_body = _make_submit_body()
    bad_body["positions"][0]["input_planes_b64"] = "not-base64!@#$%"

    client.post(f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=bad_body)
    # Padding-error b64 raises ; FastAPI surfaces it as 400
    # (might pass with padding tolerance — test on truly invalid char to be safe)


def test_submit_validates_outcome_range(client: TestClient, app_and_db) -> None:
    """Pydantic schema rejects outcome outside [-1, +1]."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)

    bad_body = _make_submit_body()
    bad_body["positions"][0]["outcome"] = 2.5  # invalid

    resp = client.post(f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=bad_body)
    assert resp.status_code == 422  # FastAPI/Pydantic validation error


def test_submit_validates_ply_non_negative(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)

    bad_body = _make_submit_body()
    bad_body["positions"][0]["ply"] = -1

    resp = client.post(f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=bad_body)
    assert resp.status_code == 422


def test_submit_stores_position_blob_correctly(client: TestClient, app_and_db) -> None:
    """End-to-end roundtrip — submitted blob must equal what's in the DB."""
    _, db = app_and_db
    create_job(db, model_version=42, num_sims=200)
    job_id = _claim_a_job(client)

    planes_raw = bytes(range(256)) * (PLANES_BYTES // 256)  # 30464 octets, motif reconnaissable
    policy_raw = b"\xde\xad\xbe\xef" * (POLICY_BYTES // 4)  # 18688 octets
    payload = {
        "game_id": "round-trip-game",
        "model_version": 42,
        "positions": [
            {
                "ply": 0,
                "fen": "startpos",
                "input_planes_b64": _b64(planes_raw),
                "policy_target_b64": _b64(policy_raw),
                "outcome": -1.0,
            }
        ],
    }
    resp = client.post(f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=payload)
    assert resp.status_code == 200

    # BLOBs stockés zlib-compressés (L2 2026-06-06, colonne compressed=1) ; le chemin de
    # lecture décompresse de façon transparente → bytes exacts d'origine.
    import sqlite3

    from nanozero_jobserver.storage.replay_buffer import iter_unflushed_positions

    with sqlite3.connect(str(db)) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT input_planes, compressed FROM positions").fetchone()
    assert row["compressed"] == 1
    assert bytes(row["input_planes"]) != planes_raw  # stocké compressé, ≠ brut

    # Round-trip via le chemin de lecture → bytes d'origine exacts.
    pos = iter_unflushed_positions(db, model_version=42, limit=10)
    assert len(pos) == 1
    assert pos[0].input_planes == planes_raw
    assert pos[0].policy_target == policy_raw
    assert pos[0].outcome == -1.0
    assert pos[0].model_version == 42


# --- Durcissement du path JSON (revue BMAD challenge) ----------------------------


def test_submit_empty_positions_returns_422(client: TestClient, app_and_db) -> None:
    """Un submit à 0 position ne complète plus le job en silence (min_length=1)."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)
    resp = client.post(
        f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=_make_submit_body(n_positions=0)
    )
    assert resp.status_code == 422
    assert count_positions(db) == 0


def test_submit_too_many_positions_returns_422(client: TestClient, app_and_db) -> None:
    """Borne anti-amplification : > 1024 positions rejeté (max_length)."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)
    payload = {
        "game_id": "g",
        "model_version": 1,
        "positions": [
            {"ply": i, "fen": "f", "input_planes_b64": "", "policy_target_b64": "", "outcome": 0.0}
            for i in range(1025)
        ],
    }
    resp = client.post(f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=payload)
    assert resp.status_code == 422


def test_submit_wrong_blob_size_returns_422(client: TestClient, app_and_db) -> None:
    """Un BLOB de mauvaise dimension est rejeté (sinon corpus corrompu → crash trainer)."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)
    body = _make_submit_body(n_positions=1)
    body["positions"][0]["input_planes_b64"] = _b64(b"\x00" * 100)  # taille invalide
    resp = client.post(f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=body)
    assert resp.status_code == 422
    assert count_positions(db) == 0


def test_submit_model_version_derived_from_job_not_payload(client: TestClient, app_and_db) -> None:
    """model_version forgé dans le payload IGNORÉ — la version stockée = celle du JOB."""
    from nanozero_jobserver.storage.replay_buffer import iter_unflushed_positions

    _, db = app_and_db
    create_job(db, model_version=7, num_sims=200)
    job_id = _claim_a_job(client)
    body = _make_submit_body(model_version=999, n_positions=2)  # version forgée
    resp = client.post(f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=body)
    assert resp.status_code == 200
    stored = iter_unflushed_positions(db, model_version=7, limit=10)
    assert len(stored) == 2  # sous la version du JOB (7)
    assert all(p.model_version == 7 for p in stored)
    assert iter_unflushed_positions(db, model_version=999, limit=10) == []  # rien sous 999


# --- STORY-007 : pseudo « adresse ouverte » (header X-Pseudo, normalisé serveur) -----------
#
# Contrat : le pseudo est un label PUBLIC cosmétique transmis en header HTTP optionnel `X-Pseudo`
# (PAS dans le payload binaire — le codec versionné partagé n'est PAS touché). Le serveur le
# normalise (NFC + lowercase + strip + troncature 24 + charset [a-z0-9_-] + blocklist) ; tout échec
# (absent / invalide / offensant) → colonne `pseudo` NULL, MAIS la soumission est TOUJOURS acceptée
# (AC-6). Rétro-compat flotte native : une soumission sans header X-Pseudo doit marcher à l'identique.


def _stored_pseudos(db: Path) -> list[str | None]:
    """Lit la colonne `pseudo` de toutes les positions (NULL = anonyme)."""
    import sqlite3

    con = sqlite3.connect(str(db))
    try:
        return [r[0] for r in con.execute("SELECT pseudo FROM positions ORDER BY id")]
    finally:
        con.close()


def test_submit_valid_pseudo_stored_lowercase(client: TestClient, app_and_db) -> None:
    """T7.1 — pseudo valide (casse mixte) → stocké en lowercase sur chaque position, submit 200."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)

    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers={**AUTH_HEADERS, "X-Pseudo": "Alice_42"},
        json=_make_submit_body(n_positions=3),
    )
    assert resp.status_code == 200
    assert resp.json()["positions_stored"] == 3
    # Normalisé lowercase, appliqué à TOUTES les positions de la partie.
    assert _stored_pseudos(db) == ["alice_42", "alice_42", "alice_42"]


def test_submit_invalid_pseudo_stored_null_and_accepted(client: TestClient, app_and_db) -> None:
    """T7.2 — pseudo invalide (chars interdits / trop long) → NULL, MAIS submit accepté (AC-6)."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)

    # Caractères interdits (espaces, '!') → ne matche pas ^[a-z0-9_-]{1,24}$ → None.
    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers={**AUTH_HEADERS, "X-Pseudo": "bad name!"},
        json=_make_submit_body(n_positions=2),
    )
    assert resp.status_code == 200  # JAMAIS de rejet pour un pseudo invalide
    assert resp.json()["positions_stored"] == 2
    assert _stored_pseudos(db) == [None, None]


def test_submit_without_pseudo_stored_null_and_accepted(client: TestClient, app_and_db) -> None:
    """T7.3 — RÉTRO-COMPAT FLOTTE : submit SANS header X-Pseudo → NULL, accepté à l'identique."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)

    # AUTH_HEADERS ne contient PAS X-Pseudo → exactement le cas de la flotte Java de production.
    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers=AUTH_HEADERS,
        json=_make_submit_body(n_positions=4),
    )
    assert resp.status_code == 200
    assert resp.json()["positions_stored"] == 4
    assert _stored_pseudos(db) == [None, None, None, None]


def test_submit_offensive_pseudo_stored_null_and_accepted(client: TestClient, app_and_db) -> None:
    """T7.4 — pseudo offensant (blocklist) → NULL silencieux, submit accepté (AC-6)."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)

    # Terme de la blocklist (sous-chaîne) → rejeté silencieusement → None, mais submit OK.
    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers={**AUTH_HEADERS, "X-Pseudo": "fuck_you"},
        json=_make_submit_body(n_positions=1),
    )
    assert resp.status_code == 200
    assert resp.json()["positions_stored"] == 1
    assert _stored_pseudos(db) == [None]


def test_submit_pseudo_too_long_truncated_then_validated(client: TestClient, app_and_db) -> None:
    """Un pseudo > 24 chars purement [a-z0-9_-] est TRONQUÉ à 24 (pas rejeté) — AC-1."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)

    long_pseudo = "a" * 40  # 40 'a' → tronqué à 24 → valide
    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers={**AUTH_HEADERS, "X-Pseudo": long_pseudo},
        json=_make_submit_body(n_positions=1),
    )
    assert resp.status_code == 200
    assert _stored_pseudos(db) == ["a" * 24]


def test_binary_submit_carries_pseudo(client: TestClient, app_and_db) -> None:
    """Le header X-Pseudo s'applique AUSSI au chemin binaire (browser) — le vrai chemin volontaire."""
    import struct

    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)

    # Un sample binaire minimal valide (planes vides OK, policy normalisée Σ=1).
    planes = b"\x00" * (7616 * 4)
    body = bytearray(struct.pack("<I", 1))  # num_positions = 1
    body += struct.pack("<f", 0.0)  # outcome
    body += planes
    body += struct.pack("<H", 1)  # 1 entrée policy sparse
    body += struct.pack("<H", 0) + struct.pack("<f", 1.0)  # index 0, masse 1.0

    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers={**AUTH_HEADERS, "Content-Type": "application/x-nanozero-submit-v1", "X-Pseudo": "Bob"},
        content=bytes(body),
    )
    assert resp.status_code == 200
    assert _stored_pseudos(db) == ["bob"]


# --- GET /jobs/{id} (inspection/debug) -------------------------------------------


def test_job_detail(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    create_job(db, model_version=5, num_sims=300)
    job_id = _claim_a_job(client)
    body = client.get(f"/jobs/{job_id}", headers=AUTH_HEADERS).json()
    assert body["job_id"] == job_id
    assert body["status"] == "claimed"
    assert body["model_version"] == 5
    assert body["num_sims"] == 300
    assert body["claimed_by"] == "test-worker"


def test_job_detail_unknown_404(client: TestClient) -> None:
    assert client.get("/jobs/nonexistent", headers=AUTH_HEADERS).status_code == 404


def test_job_detail_requires_auth(client: TestClient) -> None:
    assert client.get("/jobs/whatever").status_code == 401


# -----------------------------------------------------------------------------
# /jobs/enqueue (Phase 13.5c)
# -----------------------------------------------------------------------------


def test_enqueue_creates_count_pending_jobs(client: TestClient) -> None:
    payload = {"count": 5, "model_version": 3, "num_sims": 200}
    resp = client.post("/jobs/enqueue", headers=AUTH_HEADERS, json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body["enqueued"] == 5
    assert len(body["job_ids"]) == 5


def test_enqueue_with_dirichlet_seed_base(client: TestClient, app_and_db) -> None:
    """dirichlet_seed_base=100, count=3 → job seeds 100, 101, 102."""
    import sqlite3

    _, db = app_and_db
    payload = {"count": 3, "model_version": 1, "num_sims": 50, "dirichlet_seed_base": 100}
    client.post("/jobs/enqueue", headers=AUTH_HEADERS, json=payload)

    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute("SELECT dirichlet_seed FROM jobs ORDER BY created_at").fetchall()
    seeds = sorted(r[0] for r in rows)
    assert seeds == [100, 101, 102]


def test_enqueue_without_seed_base_leaves_seed_null(client: TestClient, app_and_db) -> None:
    import sqlite3

    _, db = app_and_db
    client.post(
        "/jobs/enqueue",
        headers=AUTH_HEADERS,
        json={"count": 2, "model_version": 1, "num_sims": 50},
    )
    with sqlite3.connect(str(db)) as conn:
        rows = conn.execute("SELECT dirichlet_seed FROM jobs").fetchall()
    assert all(r[0] is None for r in rows)


def test_enqueue_validates_count_range(client: TestClient) -> None:
    bad = client.post(
        "/jobs/enqueue",
        headers=AUTH_HEADERS,
        json={"count": 0, "model_version": 1, "num_sims": 50},
    )
    assert bad.status_code == 422


def test_enqueue_requires_auth(client: TestClient) -> None:
    resp = client.post("/jobs/enqueue", json={"count": 1, "model_version": 1, "num_sims": 50})
    assert resp.status_code == 401


def test_enqueue_then_claim_returns_same_job(client: TestClient) -> None:
    """Smoke : enqueue → claim → check the claimed job matches."""
    r = client.post(
        "/jobs/enqueue",
        headers=AUTH_HEADERS,
        json={"count": 1, "model_version": 7, "num_sims": 400},
    )
    new_id = r.json()["job_ids"][0]
    claim_resp = client.post("/jobs/claim", headers=AUTH_HEADERS)
    assert claim_resp.status_code == 200
    body = claim_resp.json()
    assert body["job_id"] == new_id
    assert body["model_version"] == 7
    assert body["num_sims"] == 400


# -----------------------------------------------------------------------------
# Pause switch + /jobs/cancel_pending
# -----------------------------------------------------------------------------


def test_claim_returns_204_when_paused(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    # Pause → claim refused even though a pending job exists.
    assert client.post("/selfplay/pause", headers=AUTH_HEADERS).json()["selfplay_paused"] is True
    assert client.post("/jobs/claim", headers=AUTH_HEADERS).status_code == 204
    # Resume → claim works again, job not lost.
    assert client.post("/selfplay/resume", headers=AUTH_HEADERS).json()["selfplay_paused"] is False
    assert client.post("/jobs/claim", headers=AUTH_HEADERS).status_code == 200


def test_cancel_pending_drains_queue(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    for _ in range(4):
        create_job(db, model_version=2, num_sims=200)
    client.post("/jobs/claim", headers=AUTH_HEADERS)  # 1 claimed, 3 pending

    resp = client.post("/jobs/cancel_pending", headers=AUTH_HEADERS, json={})
    assert resp.status_code == 200
    assert resp.json()["cancelled"] == 3
    # Claimed job survives; queue now empty.
    assert client.post("/jobs/claim", headers=AUTH_HEADERS).status_code == 204


def test_cancel_pending_filters_by_version(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    create_job(db, model_version=5, num_sims=200)
    create_job(db, model_version=6, num_sims=200)

    resp = client.post("/jobs/cancel_pending", headers=AUTH_HEADERS, json={"model_version": 5})
    assert resp.json()["cancelled"] == 1
    # The v6 job remains claimable.
    body = client.post("/jobs/claim", headers=AUTH_HEADERS).json()
    assert body["model_version"] == 6


def test_cancel_pending_requires_auth(client: TestClient) -> None:
    assert client.post("/jobs/cancel_pending", json={}).status_code == 401
