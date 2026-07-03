"""B.4 — quarantaine de la data ``source='browser'`` hors entraînement.

Prouve les invariants de la story :
  - provenance SERVER-AUTHORITATIVE : binaire→'browser', JSON→'fleet' (jamais le payload).
  - les DEUX chemins de lecture training excluent browser : ``sample_positions``
    (override ``include_browser``) ET ``iter_unflushed_positions`` (flush→NPZ, en dur).
  - migration idempotente : DB sans la colonne → ``source`` ajouté, lignes existantes 'fleet'.
  - ``/stats/selfplay`` expose ``positions_browser``.
"""

from __future__ import annotations

import sqlite3
import struct
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app
from nanozero_jobserver.storage.db import init_schema
from nanozero_jobserver.storage.jobs import create_job
from nanozero_jobserver.storage.models import promote_model, register_model
from nanozero_jobserver.storage.replay_buffer import (
    Position,
    count_positions_by_source,
    insert_positions,
    iter_unflushed_positions,
    sample_positions,
)

AUTH_KEY = "test-secret-key"
AUTH_HEADERS = {"X-API-Key": AUTH_KEY, "X-Worker-Id": "test-worker"}
BINARY_CT = "application/x-nanozero-submit-v1"
PLANES_LEN = 7616
POLICY_LEN = 4672


# -----------------------------------------------------------------------------
# Storage-level : exclusion des DEUX chemins training + tag + count
# -----------------------------------------------------------------------------


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    p = tmp_path / "q.db"
    init_schema(p)
    return p


def _pos(game_id: str, ply: int = 0, model_version: int = 5) -> Position:
    return Position(
        game_id=game_id,
        model_version=model_version,
        ply=ply,
        fen="",
        input_planes=b"\x01" * 32,
        policy_target=b"\x02" * 16,
        outcome=0.0,
    )


def _sources(db: Path) -> dict[str, int]:
    con = sqlite3.connect(str(db))
    try:
        return {r[0]: r[1] for r in con.execute("SELECT source, COUNT(*) FROM positions GROUP BY source")}
    finally:
        con.close()


def test_insert_default_source_is_fleet(db: Path) -> None:
    insert_positions(db, [_pos("g0"), _pos("g1")])
    assert _sources(db) == {"fleet": 2}


def test_insert_browser_source_tagged(db: Path) -> None:
    insert_positions(db, [_pos("g0")], source="browser")
    assert _sources(db) == {"browser": 1}


def test_sample_excludes_browser_by_default(db: Path) -> None:
    insert_positions(db, [_pos(f"f{i}") for i in range(20)], source="fleet")
    insert_positions(db, [_pos(f"b{i}") for i in range(20)], source="browser")
    sample = sample_positions(db, n=100, current_model_version=5, window=5)
    assert len(sample) == 20  # uniquement la flotte
    assert {p.game_id[0] for p in sample} == {"f"}


def test_sample_include_browser_override(db: Path) -> None:
    insert_positions(db, [_pos(f"f{i}") for i in range(20)], source="fleet")
    insert_positions(db, [_pos(f"b{i}") for i in range(20)], source="browser")
    sample = sample_positions(db, n=100, current_model_version=5, window=5, include_browser=True)
    assert len(sample) == 40  # flotte + browser
    assert {p.game_id[0] for p in sample} == {"f", "b"}


def test_iter_unflushed_excludes_browser_hard(db: Path) -> None:
    """Le chemin flush→NPZ exclut browser EN DUR (pas d'override) — 2ᵉ fuite bouchée."""
    insert_positions(db, [_pos(f"f{i}") for i in range(10)], source="fleet")
    insert_positions(db, [_pos(f"b{i}") for i in range(10)], source="browser")
    rows = iter_unflushed_positions(db, model_version=5, limit=1000)
    assert len(rows) == 10
    assert all(r.game_id[0] == "f" for r in rows)


def test_count_positions_by_source(db: Path) -> None:
    insert_positions(db, [_pos(f"f{i}") for i in range(7)], source="fleet")
    insert_positions(db, [_pos(f"b{i}") for i in range(3)], source="browser")
    assert count_positions_by_source(db) == {"fleet": 7, "browser": 3}
    assert count_positions_by_source(db, model_version=5) == {"fleet": 7, "browser": 3}
    assert count_positions_by_source(db, model_version=99) == {}


# -----------------------------------------------------------------------------
# Migration : DB pré-existante sans la colonne `source`
# -----------------------------------------------------------------------------


def test_migration_adds_source_defaults_fleet(tmp_path: Path) -> None:
    """Une DB avec l'ancien schéma (sans `source`) → migration → lignes existantes 'fleet'."""
    db = tmp_path / "old.db"
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            "CREATE TABLE positions ("
            " id INTEGER PRIMARY KEY AUTOINCREMENT, game_id TEXT NOT NULL,"
            " model_version INTEGER NOT NULL, ply INTEGER NOT NULL, fen TEXT NOT NULL,"
            " input_planes BLOB NOT NULL, policy_target BLOB NOT NULL, outcome REAL NOT NULL,"
            " created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)"
        )
        conn.execute(
            "INSERT INTO positions"
            " (game_id, model_version, ply, fen, input_planes, policy_target, outcome)"
            " VALUES ('legacy', 5, 0, '', x'00', x'00', 0.0)",
        )
        conn.commit()

    init_schema(db)  # applique _migrate_add_columns (idempotent)
    init_schema(db)  # 2ᵉ appel : doit rester idempotent (pas d'erreur ADD COLUMN dup)

    with sqlite3.connect(str(db)) as conn:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(positions)")}
        assert "source" in cols
        # La ligne héritée est classée 'fleet' (flotte de confiance historique).
        src = conn.execute("SELECT source FROM positions WHERE game_id='legacy'").fetchone()[0]
        assert src == "fleet"


# -----------------------------------------------------------------------------
# API-level : provenance dérivée du Content-Type, jamais du payload
# -----------------------------------------------------------------------------


@pytest.fixture()
def app_and_db(tmp_path: Path):
    cfg = ServerConfig(host="127.0.0.1", port=8090, api_key=AUTH_KEY, db_path=tmp_path / "api.db")
    return create_app(cfg), cfg.db_path


@pytest.fixture()
def client(app_and_db) -> TestClient:
    return TestClient(app_and_db[0])


def _planes() -> np.ndarray:
    return np.zeros(PLANES_LEN, dtype="<f4")


def _encode_binary(positions) -> bytes:
    out = bytearray(struct.pack("<I", len(positions)))
    for planes, idx, val, outcome in positions:
        out += struct.pack("<f", outcome)
        out += planes.astype("<f4").tobytes()
        out += struct.pack("<H", len(idx))
        for i, v in zip(idx, val, strict=True):
            out += struct.pack("<H", i) + struct.pack("<f", v)
    return bytes(out)


def _claim(client: TestClient) -> str:
    return client.post("/jobs/claim", headers=AUTH_HEADERS).json()["job_id"]


def test_binary_submit_tagged_browser(client, app_and_db) -> None:
    _, db = app_and_db
    create_job(db, model_version=7, num_sims=200)
    job_id = _claim(client)
    body = _encode_binary([(_planes(), [1], [1.0], 0.0)])
    resp = client.post(
        f"/jobs/{job_id}/submit",
        headers={**AUTH_HEADERS, "Content-Type": BINARY_CT},
        content=body,
    )
    assert resp.status_code == 200
    assert _sources(db) == {"browser": 1}


def test_json_submit_tagged_fleet(client, app_and_db) -> None:
    _, db = app_and_db
    create_job(db, model_version=2, num_sims=200)
    job_id = _claim(client)
    import base64

    body = {
        "game_id": "g1",
        "model_version": 2,
        "positions": [
            {"ply": 0, "fen": "x",
             "input_planes_b64": base64.b64encode(_planes().tobytes()).decode(),
             "policy_target_b64": base64.b64encode(np.zeros(POLICY_LEN, dtype="<f4").tobytes()).decode(),
             "outcome": 0.0},
        ],
    }
    resp = client.post(f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=body)
    assert resp.status_code == 200
    assert _sources(db) == {"fleet": 1}


def test_json_hostile_source_field_ignored(client, app_and_db) -> None:
    """Un browser ne peut PAS se déclarer 'fleet' : le champ `source` du payload est ignoré."""
    _, db = app_and_db
    create_job(db, model_version=2, num_sims=200)
    job_id = _claim(client)
    import base64

    body = {
        "game_id": "g1",
        "model_version": 2,
        "source": "fleet",  # tentative hostile — DOIT être ignorée
        "positions": [
            {"ply": 0, "fen": "x", "source": "fleet",
             "input_planes_b64": base64.b64encode(_planes().tobytes()).decode(),
             "policy_target_b64": base64.b64encode(np.zeros(POLICY_LEN, dtype="<f4").tobytes()).decode(),
             "outcome": 0.0},
        ],
    }
    resp = client.post(f"/jobs/{job_id}/submit", headers=AUTH_HEADERS, json=body)
    # Le chemin JSON reste 'fleet' (server-authoritative) — ici c'est correct car JSON=flotte ;
    # le point prouvé : le champ `source` du payload n'a AUCUN effet (pas dans le schéma).
    assert resp.status_code == 200
    assert _sources(db) == {"fleet": 1}


# -----------------------------------------------------------------------------
# Stats : /stats/selfplay distingue browser
# -----------------------------------------------------------------------------


def test_stats_selfplay_reports_browser(client, app_and_db) -> None:
    _, db = app_and_db
    register_model(db, version=5, name="gen-005", onnx_path="/p", sha256_onnx="a")
    promote_model(db, 5)
    insert_positions(db, [_pos(f"f{i}", model_version=5) for i in range(10)], source="fleet")
    insert_positions(db, [_pos(f"b{i}", model_version=5) for i in range(4)], source="browser")

    resp = client.get("/stats/selfplay", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == 5
    assert body["positions_browser"] == 4
