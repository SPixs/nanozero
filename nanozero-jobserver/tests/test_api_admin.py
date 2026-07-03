"""HTTP-level tests for admin endpoints — pause, storage, purge, checkpoint, vacuum."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app
from nanozero_jobserver.storage.batches import insert_batch
from nanozero_jobserver.storage.jobs import create_job
from nanozero_jobserver.storage.replay_buffer import (
    Position,
    insert_positions,
    mark_positions_flushed,
)

AUTH = {"X-API-Key": "k"}


@pytest.fixture()
def app_and_db(tmp_path: Path):
    cfg = ServerConfig(host="127.0.0.1", port=8090, api_key="k", db_path=tmp_path / "a.db")
    return create_app(cfg), cfg.db_path


@pytest.fixture()
def client(app_and_db) -> TestClient:
    return TestClient(app_and_db[0])


def _mk_pos(v: int, ply: int = 0) -> Position:
    return Position(
        game_id="g",
        model_version=v,
        ply=ply,
        fen="f",
        input_planes=b"\x00" * 32,
        policy_target=b"\x01" * 32,
        outcome=0.0,
    )


# -----------------------------------------------------------------------------
# pause / resume
# -----------------------------------------------------------------------------


def test_pause_resume_roundtrip(client: TestClient) -> None:
    assert client.post("/selfplay/pause", headers=AUTH).json()["selfplay_paused"] is True
    assert client.post("/selfplay/resume", headers=AUTH).json()["selfplay_paused"] is False


def test_pause_reflected_in_selfplay_stats(client: TestClient) -> None:
    client.post("/selfplay/pause", headers=AUTH)
    assert client.get("/stats/selfplay", headers=AUTH).json()["paused"] is True


def test_pause_requires_auth(client: TestClient) -> None:
    assert client.post("/selfplay/pause").status_code == 401


def test_selfplay_target_set_get_clear(client: TestClient) -> None:
    assert client.get("/selfplay/target", headers=AUTH).json()["target_positions"] is None
    r = client.post(
        "/selfplay/target", headers=AUTH, json={"target_positions": 7_000_000, "action": "notify"}
    )
    assert r.status_code == 200
    assert r.json() == {"target_positions": 7_000_000, "action": "notify"}
    assert client.get("/selfplay/target", headers=AUTH).json()["target_positions"] == 7_000_000
    client.post("/selfplay/target", headers=AUTH, json={"target_positions": 0})  # efface
    assert client.get("/selfplay/target", headers=AUTH).json()["target_positions"] is None


def test_selfplay_target_requires_auth(client: TestClient) -> None:
    assert client.post("/selfplay/target", json={"target_positions": 1}).status_code == 401


# -----------------------------------------------------------------------------
# /admin/storage
# -----------------------------------------------------------------------------


def test_storage_dashboard(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    insert_positions(db, [_mk_pos(v=1, ply=i) for i in range(5)])
    create_job(db, model_version=1, num_sims=200)

    # #6 : ?detailed=true → compteurs lourds exacts.
    body = client.get("/admin/storage?detailed=true", headers=AUTH).json()
    assert body["positions_total"] == 5
    assert body["page_count"] > 0
    assert body["file_bytes"] > 0
    assert body["jobs_by_status"] == {"pending": 1}
    assert body["selfplay_paused"] is False
    # #6 : par défaut (cheap, instantané) les scans lourds ne sont PAS calculés (-1) ;
    # les tailles disque + jobs restent présents.
    cheap = client.get("/admin/storage", headers=AUTH).json()
    assert cheap["positions_total"] == -1
    assert cheap["positions_flushed"] == -1
    assert cheap["freelist_count"] == -1
    assert cheap["file_bytes"] > 0
    assert cheap["jobs_by_status"] == {"pending": 1}


def test_storage_requires_auth(client: TestClient) -> None:
    assert client.get("/admin/storage").status_code == 401


# -----------------------------------------------------------------------------
# /admin/purge
# -----------------------------------------------------------------------------


def test_purge_dry_run_reports_without_deleting(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    insert_positions(db, [_mk_pos(v=1), _mk_pos(v=2)])
    insert_batch(db, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=2)
    mark_positions_flushed(db, [1, 2], batch_id=1)

    resp = client.post("/admin/purge", headers=AUTH, json={"max_model_version": 2, "dry_run": True})
    body = resp.json()
    assert body["purgeable"] == 2
    assert body["purged"] == 0
    assert body["dry_run"] is True
    # Nothing actually deleted.
    assert client.get("/admin/storage?detailed=true", headers=AUTH).json()["positions_total"] == 2


def test_purge_deletes_flushed_within_bound(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    insert_positions(db, [_mk_pos(v=1), _mk_pos(v=2), _mk_pos(v=3)])
    insert_batch(db, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=3)
    mark_positions_flushed(db, [1, 2], batch_id=1)  # v1,v2 flushed; v3 live

    resp = client.post("/admin/purge", headers=AUTH, json={"max_model_version": 2})
    body = resp.json()
    assert body["purged"] == 2
    # v3 (unflushed) survives.
    assert client.get("/admin/storage?detailed=true", headers=AUTH).json()["positions_total"] == 1


def test_purge_source_browser_only(client: TestClient, app_and_db) -> None:
    """B4-D1 : ?source=browser purge UNIQUEMENT la cohorte browser flushée, fleet intact."""
    _, db = app_and_db
    insert_positions(db, [_mk_pos(v=5)], source="fleet")  # id 1
    insert_positions(db, [_mk_pos(v=5), _mk_pos(v=5)], source="browser")  # ids 2,3
    insert_batch(db, model_version=5, batch_idx=0, npz_path="/a.npz", n_positions=3)
    mark_positions_flushed(db, [1, 2, 3], batch_id=1)  # tout flushed

    resp = client.post(
        "/admin/purge", headers=AUTH, json={"max_model_version": 5, "source": "browser"}
    )
    body = resp.json()
    assert body["purgeable"] == 2  # 2 browser flushed
    assert body["purged"] == 2
    # le fleet survit (la version courante n'est PAS purgée pour le fleet)
    assert client.get("/admin/storage?detailed=true", headers=AUTH).json()["positions_total"] == 1


def test_purge_requires_version(client: TestClient) -> None:
    # max_model_version is mandatory (ge=1) — empty body is a validation error.
    assert client.post("/admin/purge", headers=AUTH, json={}).status_code == 422


# -----------------------------------------------------------------------------
# /admin/checkpoint + /admin/vacuum_into
# -----------------------------------------------------------------------------


def test_checkpoint(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    insert_positions(db, [_mk_pos(v=1)])
    body = client.post("/admin/checkpoint", headers=AUTH).json()
    assert "checkpointed" in body
    assert body["busy"] == 0


def test_vacuum_into_writes_copy(client: TestClient, app_and_db, tmp_path: Path) -> None:
    _, db = app_and_db
    insert_positions(db, [_mk_pos(v=1, ply=i) for i in range(8)])
    dest = tmp_path / "copy.db"
    resp = client.post("/admin/vacuum_into", headers=AUTH, json={"dest_path": str(dest)})
    assert resp.status_code == 200
    assert dest.exists()
    assert resp.json()["dest_bytes"] > 0


def test_vacuum_into_refuses_existing_dest(client: TestClient, tmp_path: Path) -> None:
    dest = tmp_path / "exists.db"
    dest.write_text("occupied")
    resp = client.post("/admin/vacuum_into", headers=AUTH, json={"dest_path": str(dest)})
    assert resp.status_code == 409


def test_vacuum_into_requires_auth(client: TestClient, tmp_path: Path) -> None:
    resp = client.post("/admin/vacuum_into", json={"dest_path": str(tmp_path / "x.db")})
    assert resp.status_code == 401


# -----------------------------------------------------------------------------
# /admin/compact
# -----------------------------------------------------------------------------


def test_compact_drops_flushed_keeps_metadata(
    client: TestClient, app_and_db, tmp_path: Path
) -> None:
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    insert_batch(db, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=3)
    insert_positions(db, [_mk_pos(v=1, ply=i) for i in range(3)])
    mark_positions_flushed(db, [1, 2, 3], batch_id=1)  # flushed → dropped
    insert_positions(db, [_mk_pos(v=1, ply=i) for i in range(3, 6)])  # 3 unflushed → kept

    dest = tmp_path / "compact.db"
    resp = client.post(
        "/admin/compact", headers=AUTH, json={"dest_path": str(dest), "keep_unflushed": True}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["rows_copied"]["positions"] == 3  # only unflushed
    assert body["rows_copied"]["batches"] == 1
    assert body["rows_copied"]["jobs"] == 1
    assert dest.exists()


def test_compact_refuses_existing_dest(client: TestClient, tmp_path: Path) -> None:
    dest = tmp_path / "exists.db"
    dest.write_text("occupied")
    resp = client.post("/admin/compact", headers=AUTH, json={"dest_path": str(dest)})
    assert resp.status_code == 409


def test_compact_requires_auth(client: TestClient, tmp_path: Path) -> None:
    resp = client.post("/admin/compact", json={"dest_path": str(tmp_path / "x.db")})
    assert resp.status_code == 401
