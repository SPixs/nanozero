"""Unit tests for jobserver_client.worker (Phase 13.4).

Mocks the JobserverClient HTTP layer and the UciClient subprocess so we can
verify the worker loop logic without spinning up a real server or JVM.
"""

from __future__ import annotations

import base64
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
from nanozero_training.data.sample import Sample
from nanozero_training.jobserver_client.worker import (
    JobserverClient,
    JobserverWorker,
    WorkerConfig,
)
from nanozero_training.selfplay.config import SelfplayConfig

# -----------------------------------------------------------------------------
# Test helpers
# -----------------------------------------------------------------------------


def _make_selfplay_config() -> SelfplayConfig:
    """Default selfplay config — most fields don't matter as we mock play_one_game."""
    return SelfplayConfig(
        target_games_per_gen=1,
        games_per_batch=1,
        mcts_sims=10,
        max_game_plies=100,
        temperature=1.0,
        temperature_switch_ply=30,
        worker_seed=42,
        dirichlet_alpha=300,
        dirichlet_epsilon=250,
        go_timeout_seconds=10.0,
    )


def _make_worker_config(tmp_path: Path) -> WorkerConfig:
    return WorkerConfig(
        server_url="http://test-server:8090",
        api_key="key",
        worker_id="test-worker",
        uci_jar_path=tmp_path / "fake.jar",
        models_dir=tmp_path / "models",
        selfplay_config=_make_selfplay_config(),
        poll_idle_seconds=0.0,  # don't sleep in tests
    )


def _make_fake_sample(ply: int = 0, outcome: float = 0.0) -> Sample:
    return Sample(
        input_planes=np.zeros((119, 8, 8), dtype=np.float32),
        policy_target=np.zeros((4672,), dtype=np.float32),
        value_target=outcome,
        turn=ply % 2,
        ply=ply,
    )


# -----------------------------------------------------------------------------
# JobserverClient — HTTP wrapper
# -----------------------------------------------------------------------------


def test_client_claim_returns_none_on_204(tmp_path: Path) -> None:
    http = MagicMock()
    http.post.return_value = MagicMock(status_code=204)
    client = JobserverClient("http://x", "k", "w", http_client=http)
    assert client.claim() is None
    http.post.assert_called_once_with("/jobs/claim")


def test_client_claim_returns_json_on_200(tmp_path: Path) -> None:
    job = {
        "job_id": "abc",
        "model_version": 3,
        "num_sims": 200,
        "opening_fen": None,
        "dirichlet_seed": None,
    }
    http = MagicMock()
    resp = MagicMock(status_code=200)
    resp.json.return_value = job
    http.post.return_value = resp
    client = JobserverClient("http://x", "k", "w", http_client=http)
    assert client.claim() == job


def test_client_submit_posts_to_correct_path() -> None:
    http = MagicMock()
    http.post.return_value = MagicMock(status_code=200, **{"json.return_value": {"status": "ok"}})
    client = JobserverClient("http://x", "k", "w", http_client=http)
    client.submit("job-42", {"game_id": "g", "positions": []})
    http.post.assert_called_with("/jobs/job-42/submit", json={"game_id": "g", "positions": []})


# -----------------------------------------------------------------------------
# JobserverWorker — loop logic with mocks
# -----------------------------------------------------------------------------


def test_worker_idle_returns_zero_when_no_job(tmp_path, monkeypatch) -> None:
    """If claim() returns None, the loop sleeps + retries ; max_jobs=0 stops immediately."""
    client = MagicMock(spec=JobserverClient)
    client.claim.return_value = None
    uci = MagicMock(spec_set=["start", "quit", "is_alive", "new_game", "go_nodes"])
    uci.is_alive.return_value = False

    cfg = _make_worker_config(tmp_path)
    worker = JobserverWorker(cfg, client=client, uci_client=uci)

    submitted = worker.run_forever(max_jobs=0)
    assert submitted == 0
    client.submit.assert_not_called()


def test_worker_claim_play_submit_happy_path(tmp_path, monkeypatch) -> None:
    """Full single-job loop : claim → ensure model → play → submit."""
    # Mock play_one_game to return 3 fake samples.
    # value_target must be in {-1.0, 0.0, +1.0} (Sample validator) — draw scenario.
    fake_samples = [_make_fake_sample(ply=i, outcome=0.0) for i in range(3)]
    monkeypatch.setattr(
        "nanozero_training.jobserver_client.worker.play_one_game",
        lambda uci, cfg, rng: fake_samples,
    )

    # Mock client : returns one job then nothing.
    client = MagicMock(spec=JobserverClient)
    job = {
        "job_id": "abc-123",
        "model_version": 1,
        "num_sims": 10,
        "opening_fen": None,
        "dirichlet_seed": None,
    }
    client.claim.side_effect = [job, None]  # 2nd iteration: no job → exits at max_jobs=1
    client.submit.return_value = {"status": "ok", "positions_stored": 3}

    # Pre-populate model file so we don't trigger HTTP download.
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "model-v0001.onnx").write_bytes(b"fake-onnx")

    uci = MagicMock(spec_set=["start", "quit", "is_alive", "new_game", "go_nodes"])
    uci.is_alive.return_value = False

    cfg = _make_worker_config(tmp_path)
    worker = JobserverWorker(cfg, client=client, uci_client=uci)

    submitted = worker.run_forever(max_jobs=1)

    assert submitted == 1
    # UCI started exactly once (model fetched & engine launched).
    uci.start.assert_called_once()
    # Submit called with proper payload structure.
    client.submit.assert_called_once()
    call_args = client.submit.call_args
    assert call_args.args[0] == "abc-123"
    body = call_args.args[1]
    assert body["model_version"] == 1
    assert len(body["positions"]) == 3
    # BLOBs are base64.
    assert all(
        base64.b64decode(p["input_planes_b64"]) == fake_samples[i].input_planes.tobytes()
        for i, p in enumerate(body["positions"])
    )


def test_worker_downloads_model_on_first_use(tmp_path, monkeypatch) -> None:
    """Model file absent → ensure_model_loaded calls download_model."""
    monkeypatch.setattr(
        "nanozero_training.jobserver_client.worker.play_one_game",
        lambda uci, cfg, rng: [_make_fake_sample()],
    )

    client = MagicMock(spec=JobserverClient)
    job = {
        "job_id": "j",
        "model_version": 7,
        "num_sims": 10,
        "opening_fen": None,
        "dirichlet_seed": None,
    }
    client.claim.side_effect = [job, None]

    uci = MagicMock(spec_set=["start", "quit", "is_alive", "new_game", "go_nodes"])
    uci.is_alive.return_value = False

    cfg = _make_worker_config(tmp_path)
    # NB: models dir is empty → download must be called.
    worker = JobserverWorker(cfg, client=client, uci_client=uci)

    worker.run_forever(max_jobs=1)

    expected_path = cfg.models_dir / "model-v0007.onnx"
    client.download_model.assert_called_once_with(
        7, expected_path, chunk_size=cfg.download_chunk_size
    )


def test_worker_skips_download_if_model_cached(tmp_path, monkeypatch) -> None:
    """Model file already on disk → no re-download."""
    monkeypatch.setattr(
        "nanozero_training.jobserver_client.worker.play_one_game",
        lambda uci, cfg, rng: [_make_fake_sample()],
    )

    cfg = _make_worker_config(tmp_path)
    cfg.models_dir.mkdir(parents=True)
    (cfg.models_dir / "model-v0007.onnx").write_bytes(b"cached")

    client = MagicMock(spec=JobserverClient)
    client.claim.side_effect = [
        {
            "job_id": "j",
            "model_version": 7,
            "num_sims": 10,
            "opening_fen": None,
            "dirichlet_seed": None,
        },
        None,
    ]

    uci = MagicMock(spec_set=["start", "quit", "is_alive", "new_game", "go_nodes"])
    uci.is_alive.return_value = False

    worker = JobserverWorker(cfg, client=client, uci_client=uci)
    worker.run_forever(max_jobs=1)

    client.download_model.assert_not_called()


def test_worker_restarts_uci_on_model_version_change(tmp_path, monkeypatch) -> None:
    """Different model_version between jobs → UCI quit + restart with new model."""
    monkeypatch.setattr(
        "nanozero_training.jobserver_client.worker.play_one_game",
        lambda uci, cfg, rng: [_make_fake_sample()],
    )

    cfg = _make_worker_config(tmp_path)
    cfg.models_dir.mkdir(parents=True)
    (cfg.models_dir / "model-v0001.onnx").write_bytes(b"v1")
    (cfg.models_dir / "model-v0002.onnx").write_bytes(b"v2")

    client = MagicMock(spec=JobserverClient)
    client.claim.side_effect = [
        {
            "job_id": "j1",
            "model_version": 1,
            "num_sims": 10,
            "opening_fen": None,
            "dirichlet_seed": None,
        },
        {
            "job_id": "j2",
            "model_version": 2,
            "num_sims": 10,
            "opening_fen": None,
            "dirichlet_seed": None,
        },
        None,
    ]
    client.submit.return_value = {"status": "ok"}

    uci = MagicMock(spec_set=["start", "quit", "is_alive", "new_game", "go_nodes"])
    # First call : not alive (initial). Then alive after start.
    uci.is_alive.side_effect = [False, True, True]

    worker = JobserverWorker(cfg, client=client, uci_client=uci)
    worker.run_forever(max_jobs=2)

    # UCI started twice (once per model version), quit() called between.
    assert uci.start.call_count == 2
    assert uci.quit.call_count >= 1  # quit() before restarting for v2


def test_worker_keeps_uci_alive_for_same_model_version(tmp_path, monkeypatch) -> None:
    """Two consecutive jobs with same model_version → UCI reused, no restart."""
    monkeypatch.setattr(
        "nanozero_training.jobserver_client.worker.play_one_game",
        lambda uci, cfg, rng: [_make_fake_sample()],
    )

    cfg = _make_worker_config(tmp_path)
    cfg.models_dir.mkdir(parents=True)
    (cfg.models_dir / "model-v0001.onnx").write_bytes(b"v1")

    client = MagicMock(spec=JobserverClient)
    client.claim.side_effect = [
        {
            "job_id": "j1",
            "model_version": 1,
            "num_sims": 10,
            "opening_fen": None,
            "dirichlet_seed": None,
        },
        {
            "job_id": "j2",
            "model_version": 1,
            "num_sims": 10,
            "opening_fen": None,
            "dirichlet_seed": None,
        },
        None,
    ]
    client.submit.return_value = {"status": "ok"}

    uci = MagicMock(spec_set=["start", "quit", "is_alive", "new_game", "go_nodes"])
    uci.is_alive.side_effect = [False, True, True, True]

    worker = JobserverWorker(cfg, client=client, uci_client=uci)
    worker.run_forever(max_jobs=2)

    # UCI started ONCE — same model, no restart for v1 → v1.
    assert uci.start.call_count == 1


def test_worker_exception_does_not_break_loop(tmp_path, monkeypatch) -> None:
    """If play_one_game raises, worker logs + continues to next job (server requeues)."""
    call_count = {"n": 0}

    def _flaky_play(uci, cfg, rng):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated UCI crash")
        return [_make_fake_sample()]

    monkeypatch.setattr(
        "nanozero_training.jobserver_client.worker.play_one_game",
        _flaky_play,
    )

    cfg = _make_worker_config(tmp_path)
    cfg.models_dir.mkdir(parents=True)
    (cfg.models_dir / "model-v0001.onnx").write_bytes(b"v1")

    client = MagicMock(spec=JobserverClient)
    client.claim.side_effect = [
        {
            "job_id": "j1",
            "model_version": 1,
            "num_sims": 10,
            "opening_fen": None,
            "dirichlet_seed": None,
        },
        {
            "job_id": "j2",
            "model_version": 1,
            "num_sims": 10,
            "opening_fen": None,
            "dirichlet_seed": None,
        },
        None,
    ]
    client.submit.return_value = {"status": "ok"}

    uci = MagicMock(spec_set=["start", "quit", "is_alive", "new_game", "go_nodes"])
    uci.is_alive.return_value = False

    # max_jobs=1 (we only want successful submits to count).
    worker = JobserverWorker(cfg, client=client, uci_client=uci)
    submitted = worker.run_forever(max_jobs=1)

    # Despite the first job's crash, the worker recovered and submitted job 2.
    assert submitted == 1
    assert call_count["n"] == 2  # both jobs attempted


def test_worker_shutdown_quits_uci(tmp_path) -> None:
    """shutdown() must always call uci.quit() (resource cleanup)."""
    cfg = _make_worker_config(tmp_path)
    client = MagicMock(spec=JobserverClient)
    uci = MagicMock(spec_set=["start", "quit", "is_alive"])
    worker = JobserverWorker(cfg, client=client, uci_client=uci)
    worker.shutdown()
    uci.quit.assert_called_once()
    client.close.assert_called_once()


def test_worker_uses_provided_rng_for_determinism(tmp_path, monkeypatch) -> None:
    """Worker can be seeded for reproducible tests / runs."""
    captured = {"rng": None}

    def _capture_rng(uci, cfg, rng):
        captured["rng"] = rng
        return [_make_fake_sample()]

    monkeypatch.setattr("nanozero_training.jobserver_client.worker.play_one_game", _capture_rng)

    cfg = _make_worker_config(tmp_path)
    cfg.models_dir.mkdir(parents=True)
    (cfg.models_dir / "model-v0001.onnx").write_bytes(b"x")

    client = MagicMock(spec=JobserverClient)
    client.claim.side_effect = [
        {
            "job_id": "j",
            "model_version": 1,
            "num_sims": 10,
            "opening_fen": None,
            "dirichlet_seed": None,
        },
        None,
    ]
    client.submit.return_value = {"status": "ok"}

    seeded = np.random.default_rng(12345)
    uci = MagicMock(spec_set=["start", "quit", "is_alive", "new_game", "go_nodes"])
    uci.is_alive.return_value = False

    worker = JobserverWorker(cfg, client=client, uci_client=uci, rng=seeded)
    worker.run_forever(max_jobs=1)

    assert captured["rng"] is seeded
