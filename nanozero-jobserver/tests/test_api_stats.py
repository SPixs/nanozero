"""HTTP-level tests for /stats/by-version, /stats/selfplay, /stats/workers, /stats/season."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from nanozero_jobserver.config import ServerConfig
from nanozero_jobserver.main import create_app
from nanozero_jobserver.storage.batches import insert_batch
from nanozero_jobserver.storage.jobs import claim_job, create_job, submit_job
from nanozero_jobserver.storage.models import promote_model, register_model
from nanozero_jobserver.storage.replay_buffer import (
    Position,
    insert_positions,
    mark_positions_flushed,
)
from nanozero_jobserver.storage.sprt import record_sprt

AUTH = {"X-API-Key": "k"}


@pytest.fixture()
def app_and_db(tmp_path: Path):
    cfg = ServerConfig(host="127.0.0.1", port=8090, api_key="k", db_path=tmp_path / "s.db")
    return create_app(cfg), cfg.db_path


@pytest.fixture()
def client(app_and_db) -> TestClient:
    return TestClient(app_and_db[0])


def _mk_position(v: int, ply: int = 0) -> Position:
    return Position(
        game_id="g",
        model_version=v,
        ply=ply,
        fen="startpos",
        input_planes=b"\x01" * 8,
        policy_target=b"\x02" * 4,
        outcome=0.0,
    )


# -----------------------------------------------------------------------------
# /stats/by-version
# -----------------------------------------------------------------------------


def test_by_version_combines_durable_and_live(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    # v5 : 300 flushed (batches) + 7 live (positions table, unflushed).
    insert_batch(db, model_version=5, batch_idx=0, npz_path="/a.npz", n_positions=200)
    insert_batch(db, model_version=5, batch_idx=1, npz_path="/b.npz", n_positions=100)
    insert_positions(db, [_mk_position(v=5, ply=i) for i in range(7)])

    resp = client.get("/stats/by-version", headers=AUTH)
    assert resp.status_code == 200
    v5 = next(r for r in resp.json()["versions"] if r["model_version"] == 5)
    assert v5["positions_durable"] == 300
    assert v5["positions_live"] == 7
    assert v5["positions_total"] == 307
    assert v5["batches"] == 2


def test_by_version_durable_survives_purge(client: TestClient, app_and_db) -> None:
    """Marking positions flushed (BLOBs eligible for purge) must not drop durable count."""
    _, db = app_and_db
    insert_batch(db, model_version=3, batch_idx=0, npz_path="/a.npz", n_positions=100)
    insert_positions(db, [_mk_position(v=3, ply=i) for i in range(4)])
    mark_positions_flushed(db, [1, 2, 3, 4], batch_id=1)  # now flushed → live=0

    resp = client.get("/stats/by-version", headers=AUTH)
    v3 = next(r for r in resp.json()["versions"] if r["model_version"] == 3)
    assert v3["positions_durable"] == 100
    assert v3["positions_live"] == 0
    assert v3["positions_total"] == 100


def test_by_version_reports_jobs_and_current_flag(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    register_model(db, version=2, name="gen-002", onnx_path="/p", sha256_onnx="h")
    promote_model(db, 2)
    create_job(db, model_version=2, num_sims=200)
    create_job(db, model_version=2, num_sims=200)
    claim_job(db, "w1")

    resp = client.get("/stats/by-version", headers=AUTH)
    body = resp.json()
    assert body["current_model_version"] == 2
    v2 = next(r for r in body["versions"] if r["model_version"] == 2)
    assert v2["is_current"] is True
    assert v2["promoted"] is True
    assert v2["name"] == "gen-002"
    assert v2["jobs"] == {"pending": 1, "claimed": 1}


def test_by_version_requires_auth(client: TestClient) -> None:
    assert client.get("/stats/by-version").status_code == 401


def test_by_version_respects_limit(client: TestClient, app_and_db) -> None:
    """#focus stats : ?limit borne aux N gens les plus récentes (DESC)."""
    _, db = app_and_db
    for v in range(1, 6):
        insert_batch(db, model_version=v, batch_idx=0, npz_path=f"/{v}.npz", n_positions=10)
    rows = client.get("/stats/by-version?limit=2", headers=AUTH).json()["versions"]
    assert [r["model_version"] for r in rows] == [5, 4]  # 2 plus récentes


def test_workers_exposes_idle_seconds_and_limit(client: TestClient, app_and_db) -> None:
    """#focus suivi workers : idle_seconds dans la réponse + ?limit borne."""
    _, db = app_and_db
    for i in range(3):
        create_job(db, model_version=1, num_sims=200)
        claim_job(db, f"w-{i}")
    body = client.get("/stats/workers", headers=AUTH).json()
    assert body["count"] == 3
    assert all(w["idle_seconds"] is not None for w in body["workers"])
    assert client.get("/stats/workers?limit=1", headers=AUTH).json()["count"] == 1


# -----------------------------------------------------------------------------
# /stats/selfplay
# -----------------------------------------------------------------------------


def test_selfplay_empty_db(client: TestClient) -> None:
    resp = client.get("/stats/selfplay", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] is None
    assert body["shards"] == 0
    assert body["positions_total"] == 0


def test_selfplay_reports_current_cycle(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    register_model(db, version=17, name="gen-022-promoted", onnx_path="/p", sha256_onnx="h")
    promote_model(db, 17)
    insert_batch(db, model_version=17, batch_idx=0, npz_path="/0.npz", n_positions=100_000)
    insert_batch(db, model_version=17, batch_idx=1, npz_path="/1.npz", n_positions=100_000)
    insert_positions(db, [_mk_position(v=17, ply=i) for i in range(4225)])
    create_job(db, model_version=17, num_sims=800)

    resp = client.get("/stats/selfplay", headers=AUTH)
    body = resp.json()
    assert body["model_version"] == 17
    assert body["model_name"] == "gen-022-promoted"
    assert body["shards"] == 2
    assert body["positions_durable"] == 200_000
    assert body["positions_live"] == 4225
    assert body["positions_total"] == 204_225
    assert body["jobs_pending"] == 1
    assert body["next_shard_threshold"] == 100_000


def test_selfplay_cadence_none_with_single_shard(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    register_model(db, version=4, name="gen-x", onnx_path="/p", sha256_onnx="h")
    promote_model(db, 4)
    insert_batch(db, model_version=4, batch_idx=0, npz_path="/0.npz", n_positions=100)

    body = client.get("/stats/selfplay", headers=AUTH).json()
    assert body["shards"] == 1
    assert body["shards_per_hour"] is None
    assert body["positions_per_hour"] is None


# -----------------------------------------------------------------------------
# /stats/workers
# -----------------------------------------------------------------------------


def test_workers_attribution(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    # w1 completes 1 job, holds 1 in-flight; w2 holds 1 in-flight.
    for _ in range(3):
        create_job(db, model_version=1, num_sims=200)
    j = claim_job(db, "w1")
    submit_job(db, j.id)
    claim_job(db, "w1")
    claim_job(db, "w2")

    resp = client.get("/stats/workers", headers=AUTH)
    assert resp.status_code == 200
    body = resp.json()
    workers = {w["worker_id"]: w for w in body["workers"]}
    assert workers["w1"]["completed"] == 1
    assert workers["w1"]["in_flight"] == 1
    assert workers["w2"]["in_flight"] == 1
    assert all(w["blacklisted"] is False for w in body["workers"])


def test_workers_flags_npu_blacklist(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    create_job(db, model_version=1, num_sims=200)
    # An npu- worker can't claim via the API (204), but historical rows may exist.
    # Simulate one by claiming through storage directly (bypasses the API guard).
    claim_job(db, "npu-laptop-01")

    body = client.get("/stats/workers", headers=AUTH).json()
    npu = next(w for w in body["workers"] if w["worker_id"] == "npu-laptop-01")
    assert npu["blacklisted"] is True


def test_workers_requires_auth(client: TestClient) -> None:
    assert client.get("/stats/workers").status_code == 401


# -----------------------------------------------------------------------------
# /stats (enriched) + /training/should_train (durable fix)
# -----------------------------------------------------------------------------


def test_stats_generated_total_is_durable(client: TestClient, app_and_db) -> None:
    """positions_generated_total counts flushed batches even after BLOB purge."""
    _, db = app_and_db
    insert_batch(db, model_version=1, batch_idx=0, npz_path="/a.npz", n_positions=500)
    insert_positions(db, [_mk_position(v=1, ply=i) for i in range(3)])
    mark_positions_flushed(db, [1, 2, 3], batch_id=1)  # live tail now 0

    body = client.get("/stats", headers=AUTH).json()
    # positions table still holds the 3 flushed rows (not yet purged) ...
    assert body["positions_total"] == 3
    # ... but the durable cumulative reflects the full 500-position shard.
    assert body["positions_generated_total"] == 500


def test_should_train_counts_current_version_output(client: TestClient, app_and_db) -> None:
    """Regression: positions are tagged with the playing (current) version, so
    'new since current' must include that version's own output (>=, not >)."""
    _, db = app_and_db
    register_model(db, version=16, name="gen-021", onnx_path="/p", sha256_onnx="h")
    promote_model(db, 16)
    # Self-play for the new cycle is tagged model_version=16 (the model that plays).
    insert_batch(db, model_version=16, batch_idx=0, npz_path="/0.npz", n_positions=30_000)
    insert_positions(db, [_mk_position(v=16, ply=i) for i in range(500)])

    body = client.get("/training/should_train?threshold=25000", headers=AUTH).json()
    assert body["new_positions"] == 30_500  # durable 30k + live 500
    assert body["should_train"] is True


# -----------------------------------------------------------------------------
# /stats/drift  (D.2)
# -----------------------------------------------------------------------------


def _bulk_cohort(db, source: str, n: int, outcome_fn, mv: int = 30) -> None:
    games = max(1, n // 6)
    rows = [
        Position(
            game_id=f"{source}-g{i % games}",
            model_version=mv,
            ply=i % 6,
            fen="f",
            input_planes=b"\x01" * 8,
            policy_target=b"\x02" * 4,
            outcome=outcome_fn(i % games),
        )
        for i in range(n)
    ]
    insert_positions(db, rows, source=source)


def test_drift_requires_auth(client: TestClient) -> None:
    assert client.get("/stats/drift").status_code == 401


def test_drift_flags_browser_all_draws(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    register_model(db, version=30, name="gen-030", onnx_path="/m.onnx", sha256_onnx="x" * 64)
    promote_model(db, version=30)
    # fleet équilibré (+1/0/-1) ; browser 100% nulles → draw_frac drift.
    _bulk_cohort(db, "fleet", 1200, lambda g: [1.0, 0.0, -1.0][g % 3])
    _bulk_cohort(db, "browser", 1200, lambda g: 0.0)

    resp = client.get("/stats/drift", headers=AUTH)  # défaut = champion courant (30)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_version"] == 30
    assert body["verdict"] == "drift"
    assert body["browser"]["n"] == 1200
    assert any("draw_frac" in r for r in body["reasons"])


def test_drift_insufficient_small_browser(client: TestClient, app_and_db) -> None:
    _, db = app_and_db
    register_model(db, version=30, name="gen-030", onnx_path="/m.onnx", sha256_onnx="x" * 64)
    promote_model(db, version=30)
    _bulk_cohort(db, "fleet", 1200, lambda g: [1.0, 0.0, -1.0][g % 3])
    _bulk_cohort(db, "browser", 30, lambda g: 0.0)  # < 1000 (seuil par défaut)

    body = client.get("/stats/drift?model_version=30", headers=AUTH).json()
    assert body["verdict"] == "insufficient"


def test_drift_uses_ttl_cache_and_fresh_bypasses(client: TestClient, app_and_db, monkeypatch) -> None:
    """#6 : 2e appel = cache hit (pas de recalcul du scan ~34M) ; ?fresh=true recalcule."""
    calls = {"n": 0}

    def _counting_report(_db, _mv, *_a, **_k):
        calls["n"] += 1
        return {
            "model_version": 30,
            "verdict": "ok",
            "browser": {},
            "fleet": {},
            "resolution": None,
            "deltas": {},
            "reasons": ["dans les seuils"],
        }

    monkeypatch.setattr("nanozero_jobserver.api.stats.drift_report", _counting_report)

    assert client.get("/stats/drift?model_version=30", headers=AUTH).status_code == 200
    assert client.get("/stats/drift?model_version=30", headers=AUTH).status_code == 200
    assert calls["n"] == 1  # le 2e appel a tapé le cache

    client.get("/stats/drift?model_version=30&fresh=true", headers=AUTH)
    assert calls["n"] == 2  # ?fresh=true force le recalcul


# -----------------------------------------------------------------------------
# /stats/browser
# -----------------------------------------------------------------------------


def test_browser_stats_cohort(client: TestClient, app_and_db) -> None:
    """Volume browser + fleet + ratio (provenance fiable source='browser')."""
    _, db = app_and_db
    register_model(db, version=30, name="g30", onnx_path="/m", sha256_onnx="x" * 64)
    promote_model(db, version=30)
    insert_positions(db, [_mk_position(30, i) for i in range(8)], source="fleet")
    insert_positions(db, [_mk_position(30, i) for i in range(2)], source="browser")
    body = client.get("/stats/browser", headers=AUTH).json()
    assert body["model_version"] == 30
    assert body["positions_browser"] == 2
    assert body["positions_fleet"] == 8
    assert body["browser_fleet_ratio"] == 0.25  # 2/8
    # drift_verdict est None tant que /stats/drift n'a pas peuplé le cache
    assert body["drift_verdict"] is None


def test_browser_stats_requires_auth(client: TestClient) -> None:
    assert client.get("/stats/browser").status_code == 401


def test_batches_summary_endpoint(client: TestClient, app_and_db) -> None:
    """#A/B tooling : /stats/batches?source= rollup par gen depuis le registre batches."""
    _, db = app_and_db
    insert_batch(db, model_version=1, batch_idx=0, npz_path="/f.npz", n_positions=100, source="fleet")
    insert_batch(db, model_version=1, batch_idx=1, npz_path="/b.npz", n_positions=40, source="browser")
    body = client.get("/stats/batches?source=browser", headers=AUTH).json()
    assert body["source"] == "browser"
    assert len(body["versions"]) == 1
    assert body["versions"][0]["model_version"] == 1
    assert body["versions"][0]["n_positions"] == 40


def test_batches_summary_requires_auth(client: TestClient) -> None:
    assert client.get("/stats/batches").status_code == 401


# -----------------------------------------------------------------------------
# /stats/season  (gamification, STORY-001)
# -----------------------------------------------------------------------------


def test_season_empty_db(client: TestClient) -> None:
    """AC-5 (DB vide) : zéros partout + null pour les champs de promotion."""
    resp = client.get("/stats/season")  # AC-1 : PUBLIC, pas d'en-tête d'auth
    assert resp.status_code == 200
    body = resp.json()
    assert body["current_gen"] == 0
    assert body["target_gen"] == 1
    assert body["verified_positions"] == 0
    assert body["active_contributors"] == 0
    assert body["collective_pct"] == 0.0
    assert body["last_promoted_gen"] is None
    assert body["last_promoted_elo_gain"] is None


def test_season_is_public_no_api_key(client: TestClient) -> None:
    """AC-1 / T4.4 : endpoint public — répond 200 SANS X-API-Key (≠ les autres /stats/*)."""
    assert client.get("/stats/season").status_code == 200
    # contraste : un /stats/* protégé renvoie 401 sans clé.
    assert client.get("/stats/selfplay").status_code == 401


def test_season_current_gen_no_promotion(client: TestClient, app_and_db) -> None:
    """AC-2 / AC-4 / T4.2 : gen courante sans aucune promotion archivée → champs promo null."""
    _, db = app_and_db
    register_model(db, version=31, name="gen-031-promoted", onnx_path="/p", sha256_onnx="h")
    promote_model(db, 31)
    # 200k durable (toutes sources) + 5k live → verified=205k.
    insert_batch(db, model_version=31, batch_idx=0, npz_path="/0.npz", n_positions=150_000)
    insert_positions(db, [_mk_position(v=31, ply=i) for i in range(8)], source="fleet")
    insert_positions(db, [_mk_position(v=31, ply=i) for i in range(2)], source="browser")

    body = client.get("/stats/season").json()
    assert body["current_gen"] == 31
    assert body["target_gen"] == 32
    # verified = durable inclut TOUTES sources + tail live (fleet + browser).
    assert body["verified_positions"] == 150_010
    # collective_pct = 150010 / 1_000_000 * 100 = 15.0 (borné à 100).
    assert body["collective_pct"] == 15.0
    # aucun sprt_results 'accepted' → pas de promotion connue.
    assert body["last_promoted_gen"] is None
    assert body["last_promoted_elo_gain"] is None


def test_season_with_promoted_gen_and_elo(client: TestClient, app_and_db) -> None:
    """AC-2 / T4.3 : dernière promo (SPRT accepted) renseigne gen + gain Elo arrondi."""
    _, db = app_and_db
    register_model(db, version=31, name="gen-031", onnx_path="/p", sha256_onnx="h")
    promote_model(db, 31)
    # historique SPRT : 30 promu (+50), 31 promu (+48.3) → on attend le PLUS RÉCENT (31).
    record_sprt(db, candidate_version=30, baseline_version=29, decision="accepted", elo_estimate=50.0)
    record_sprt(db, candidate_version=31, baseline_version=30, decision="accepted", elo_estimate=48.3)
    # un rejet ultérieur ne doit PAS être pris comme dernière promotion.
    record_sprt(db, candidate_version=32, baseline_version=31, decision="rejected", elo_estimate=-25.0)

    body = client.get("/stats/season").json()
    assert body["last_promoted_gen"] == 31
    assert body["last_promoted_elo_gain"] == 48  # round(48.3)


def test_season_counts_browser_contributors(client: TestClient, app_and_db) -> None:
    """AC-2 : active_contributors = sessions distinctes préfixées 'browser-' (24 h)."""
    _, db = app_and_db
    register_model(db, version=31, name="g31", onnx_path="/p", sha256_onnx="h")
    promote_model(db, 31)
    for wid in ("browser-alice", "browser-bob", "devsrv-night-3"):
        create_job(db, model_version=31, num_sims=200)
        claim_job(db, wid)

    body = client.get("/stats/season").json()
    assert body["active_contributors"] == 2  # alice + bob, PAS le worker natif devsrv


def test_season_pct_capped_at_100(client: TestClient, app_and_db) -> None:
    """collective_pct borné à 100 même si le volume dépasse l'objectif de saison."""
    cfg = ServerConfig(host="127.0.0.1", port=8090, api_key="k", db_path=app_and_db[1])
    object.__setattr__(cfg, "season_target", 100)  # frozen dataclass → setattr bas niveau
    client_small = TestClient(create_app(cfg))
    _, db = app_and_db
    register_model(db, version=5, name="g5", onnx_path="/p", sha256_onnx="h")
    promote_model(db, 5)
    insert_batch(db, model_version=5, batch_idx=0, npz_path="/0.npz", n_positions=500)

    body = client_small.get("/stats/season").json()
    assert body["verified_positions"] == 500
    assert body["collective_pct"] == 100.0  # 500/100 → borné


def test_season_cached_60s(client: TestClient, app_and_db, monkeypatch) -> None:
    """AC-3 : 2e appel = cache hit (pas de recalcul) ; le cache TTL 60 s sert la jauge."""
    calls = {"n": 0}
    import nanozero_jobserver.api.stats as stats_mod

    real = stats_mod.sum_positions_by_version

    def _counting(db, *a, **k):
        calls["n"] += 1
        return real(db, *a, **k)

    monkeypatch.setattr(stats_mod, "sum_positions_by_version", _counting)
    _, db = app_and_db
    register_model(db, version=7, name="g7", onnx_path="/p", sha256_onnx="h")
    promote_model(db, 7)
    insert_batch(db, model_version=7, batch_idx=0, npz_path="/0.npz", n_positions=10)

    assert client.get("/stats/season").json()["verified_positions"] == 10
    n_after_first = calls["n"]
    assert n_after_first >= 1
    client.get("/stats/season")  # 2e appel : doit taper le cache
    assert calls["n"] == n_after_first  # pas de recalcul
