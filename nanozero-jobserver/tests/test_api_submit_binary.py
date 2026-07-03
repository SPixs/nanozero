"""HTTP-level tests pour la variante binaire de /jobs/{id}/submit (Story A.2).

Couvre : submit binaire valide (stocké, model_version dérivé du job), binaire
malformé → 400, job non-claimable → 410, et une sanity du chemin JSON (rétro-compat).
"""

from __future__ import annotations

import base64
import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app
from nanozero_jobserver.storage.jobs import create_job
from nanozero_jobserver.storage.replay_buffer import count_positions

AUTH_KEY = "test-secret-key"
AUTH_HEADERS = {"X-API-Key": AUTH_KEY, "X-Worker-Id": "test-worker"}
BINARY_CT = "application/x-nanozero-submit-v1"
PLANES_LEN = 7616
POLICY_LEN = 4672


@pytest.fixture()
def app_and_db(tmp_path: Path):
    cfg = ServerConfig(host="127.0.0.1", port=8090, api_key=AUTH_KEY, db_path=tmp_path / "test.db")
    return create_app(cfg), cfg.db_path


@pytest.fixture()
def client(app_and_db) -> TestClient:
    return TestClient(app_and_db[0])


def _claim_a_job(client: TestClient) -> str:
    resp = client.post("/jobs/claim", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    return resp.json()["job_id"]


def _planes(seed: int = 0) -> np.ndarray:
    p = np.zeros(PLANES_LEN, dtype="<f4")
    p[(np.arange(PLANES_LEN) + seed) % 100 == 0] = 1.0
    return p


def _encode_binary(positions) -> bytes:
    """Encode au layout A.1 (miroir de submit-codec.mjs) pour les tests."""
    out = bytearray(struct.pack("<I", len(positions)))
    for planes, idx, val, outcome in positions:
        out += struct.pack("<f", outcome)
        out += planes.astype("<f4").tobytes()
        out += struct.pack("<H", len(idx))
        for i, v in zip(idx, val, strict=False):
            out += struct.pack("<H", i) + struct.pack("<f", v)
    return bytes(out)


def _stored_model_versions(db: Path) -> list[int]:
    con = sqlite3.connect(str(db))
    try:
        return [r[0] for r in con.execute("SELECT DISTINCT model_version FROM positions")]
    finally:
        con.close()


def test_binary_submit_stores_positions_and_derives_model_version(client, app_and_db) -> None:
    _, db = app_and_db
    create_job(db, model_version=7, num_sims=200)  # le model_version du job
    job_id = _claim_a_job(client)

    body = _encode_binary(
        [(_planes(0), [3, 17], [0.6, 0.4], 1.0), (_planes(1), [5], [1.0], -1.0)]
    )
    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers={**AUTH_HEADERS, "Content-Type": BINARY_CT},
        content=body,
    )

    assert resp.status_code == 200
    assert resp.json()["positions_stored"] == 2
    # model_version dérivé du JOB (7), pas fourni par le client binaire
    assert _stored_model_versions(db) == [7]


def test_binary_submit_malformed_returns_400(client, app_and_db) -> None:
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)

    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers={**AUTH_HEADERS, "Content-Type": BINARY_CT},
        content=b"\xde\xad\xbe\xef garbage hostile",
    )
    assert resp.status_code == 400
    assert count_positions(db) == 0


def test_binary_submit_unclaimed_job_returns_410(client, app_and_db) -> None:
    _, db = app_and_db
    job_id = create_job(db, model_version=1, num_sims=200)  # pas claimé
    body = _encode_binary([(_planes(0), [1], [1.0], 0.0)])

    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers={**AUTH_HEADERS, "Content-Type": BINARY_CT},
        content=body,
    )
    assert resp.status_code == 410


def test_binary_submit_unknown_job_returns_410(client) -> None:
    body = _encode_binary([(_planes(0), [1], [1.0], 0.0)])
    resp = client.post(
        "/jobs/non-existent/submit",
        headers={**AUTH_HEADERS, "Content-Type": BINARY_CT},
        content=body,
    )
    assert resp.status_code == 410


def test_json_submit_still_works(client, app_and_db) -> None:
    """Sanity rétro-compat : un submit JSON base64 reste accepté inchangé."""
    _, db = app_and_db
    create_job(db, model_version=2, num_sims=200)
    job_id = _claim_a_job(client)

    planes_b64 = base64.b64encode(_planes(0).tobytes()).decode()
    policy_b64 = base64.b64encode(np.zeros(POLICY_LEN, dtype="<f4").tobytes()).decode()
    json_body = {
        "game_id": "g1",
        "model_version": 2,
        "positions": [
            {"ply": 0, "fen": "startpos", "input_planes_b64": planes_b64,
             "policy_target_b64": policy_b64, "outcome": 0.0},
        ],
    }
    resp = client.post(f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=json_body)
    assert resp.status_code == 200
    assert resp.json()["positions_stored"] == 1


def test_binary_submit_empty_returns_400(client, app_and_db) -> None:
    """Un submit binaire vide (num_positions=0) ne doit PAS compléter le job (perte)."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)

    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers={**AUTH_HEADERS, "Content-Type": BINARY_CT},
        content=struct.pack("<I", 0),  # num_positions = 0
    )
    assert resp.status_code == 400
    assert count_positions(db) == 0


def test_binary_content_type_with_charset_is_routed_binary(client, app_and_db) -> None:
    """Un paramètre de Content-Type (charset) ne doit pas casser le routage binaire."""
    _, db = app_and_db
    create_job(db, model_version=3, num_sims=200)
    job_id = _claim_a_job(client)
    body = _encode_binary([(_planes(0), [1], [1.0], 0.0)])

    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers={**AUTH_HEADERS, "Content-Type": f"{BINARY_CT}; charset=binary"},
        content=body,
    )
    assert resp.status_code == 200
    assert resp.json()["positions_stored"] == 1


def test_binary_resubmit_completed_job_returns_410(client, app_and_db) -> None:
    """Re-soumettre (binaire) un job déjà complété → 410, pas de double-insert."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)
    body = _encode_binary([(_planes(0), [1], [1.0], 0.0)])
    hdrs = {**AUTH_HEADERS, "Content-Type": BINARY_CT}

    assert client.post(f"/jobs/{job_id}/submit", headers=hdrs, content=body).status_code == 200
    n = count_positions(db)
    assert client.post(f"/jobs/{job_id}/submit", headers=hdrs, content=body).status_code == 410
    assert count_positions(db) == n  # pas de ré-insertion


def test_binary_submit_invalid_policy_returns_422(client, app_and_db) -> None:
    """B.3 : une policy non-normalisée (Σ≠1) → 422, rien stocké, job non complété."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)
    # Σπ = 0.5 → hors [1±0.02]
    body = _encode_binary([(_planes(0), [3, 17], [0.3, 0.2], 0.0)])
    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers={**AUTH_HEADERS, "Content-Type": BINARY_CT},
        content=body,
    )
    assert resp.status_code == 422
    assert count_positions(db) == 0  # rien stocké


def test_binary_submit_plane_out_of_range_returns_422(client, app_and_db) -> None:
    """B.3 : une plane hors plage [0, 10] (finie mais absurde) → 422."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)
    bad_planes = np.full(PLANES_LEN, 1e6, dtype="<f4")  # finie, mais hors plage
    body = _encode_binary([(bad_planes, [1], [1.0], 0.0)])
    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers={**AUTH_HEADERS, "Content-Type": BINARY_CT},
        content=body,
    )
    assert resp.status_code == 422
    assert count_positions(db) == 0


def test_json_fleet_policy_not_validated(client, app_and_db) -> None:
    """B.3 ne touche PAS le chemin fleet : un submit JSON à policy non-normalisée passe (200)."""
    _, db = app_and_db
    create_job(db, model_version=2, num_sims=200)
    job_id = _claim_a_job(client)
    # policy dense arbitraire (non normalisée) — le chemin JSON ne la valide pas
    weird_policy = np.full(POLICY_LEN, 7.0, dtype="<f4")
    json_body = {
        "game_id": "g1",
        "model_version": 2,
        "positions": [
            {"ply": 0, "fen": "startpos",
             "input_planes_b64": base64.b64encode(_planes(0).tobytes()).decode(),
             "policy_target_b64": base64.b64encode(weird_policy.tobytes()).decode(),
             "outcome": 0.0},
        ],
    }
    resp = client.post(f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=json_body)
    assert resp.status_code == 200  # fleet inchangé
    assert resp.json()["positions_stored"] == 1


def test_json_invalid_returns_422_unchanged(client, app_and_db) -> None:
    """Non-régression : un JSON invalide (outcome hors bornes) reste 422 (pas 400)."""
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    job_id = _claim_a_job(client)
    bad = {
        "game_id": "g", "model_version": 1,
        "positions": [{"ply": 0, "fen": "x", "input_planes_b64": "", "policy_target_b64": "",
                       "outcome": 5.0}],  # hors [-1,1]
    }
    resp = client.post(f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=bad)
    assert resp.status_code == 422
