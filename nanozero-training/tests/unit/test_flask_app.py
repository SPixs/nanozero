"""Unit tests for monitoring/flask_app — REST endpoints + SSE."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest
import yaml
from flask import Flask
from nanozero_training.config.run_config import (
    MonitorConfig,
    PathsConfig,
    RunConfig,
)
from nanozero_training.monitoring.csv_writer import MetricsCSVWriter
from nanozero_training.monitoring.flask_app import create_app


def _make_config(tmp_path: Path) -> RunConfig:
    """Build a minimal RunConfig with monitoring dir in tmp_path."""
    paths = PathsConfig(
        run_root=str(tmp_path),
        datasets_dir="datasets",
        models_dir="models",
        monitoring_dir="monitoring",
        versions_yaml="versions.yaml",
        pgn_path="monitoring/sprt.pgn",
        uci_jar="dummy.jar",
    )
    monitor = MonitorConfig(
        enabled=True,
        port=5000,
        host="127.0.0.1",
        csv_flush_interval_seconds=1.0,
        sse_poll_interval_seconds=0.05,
    )
    (tmp_path / "monitoring").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    return RunConfig(paths=paths, monitor=monitor)


def _write_run_state(tmp_path: Path) -> Path:
    """Write a minimal valid run_state.yaml."""
    state_path = tmp_path / "monitoring" / "run_state.yaml"
    state = {
        "run_id": "test-run-001",
        "started": "2026-05-14T10:00:00+00:00",
        "last_updated": "2026-05-14T10:05:00+00:00",
        "status": "in_progress",
        "config_path": "configs/test.yaml",
        "config_hash": "abc123",
        "max_generations": 5,
        "target_games_per_gen": 100,
        "current_gen": 2,
        "gens_completed": [1],
        "gens_rejected": [],
        "phase": "train",
        "phase_started": None,
        "selfplay": {
            "target_games": 100,
            "completed_batches": 0,
            "completed_games": 0,
            "current_batch_idx": 0,
            "current_batch_games": 0,
            "last_batch_file": None,
            "worker_seed": 42,
        },
        "train": {
            "base_model": "gen-001-init",
            "output_model_target": "gen-002-trained",
            "current_epoch": 3,
            "total_epochs": 10,
            "last_checkpoint": None,
            "optimizer_state_file": None,
        },
        "eval": {
            "challenger": None,
            "baseline": None,
            "pgn_path": None,
            "games_played_at_last_save": 0,
            "last_decision": None,
        },
        "last_error": None,
        "crash_count": 0,
        "last_resume": None,
    }
    state_path.write_text(yaml.safe_dump(state))
    return state_path


def _write_versions(tmp_path: Path, n_entries: int = 2) -> Path:
    """Write a minimal versions.yaml with n_entries."""
    versions_path = tmp_path / "versions.yaml"
    all_entries = []
    for i in range(1, n_entries + 1):
        all_entries.append(
            {
                "name": f"gen-{i:03d}-init" if i == 1 else f"gen-{i:03d}-trained",
                "type": "init" if i == 1 else "trained",
                "promoted": i == 1,
                "created": f"2026-05-14T10:0{i}:00Z",
            }
        )
    versions_path.write_text(
        yaml.safe_dump(
            {
                "current": "gen-001-init",
                "latest_trained": f"gen-{n_entries:03d}-trained" if n_entries > 1 else None,
                "all": all_entries,
            }
        )
    )
    return versions_path


def test_create_app_returns_flask(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    app = create_app(cfg)
    assert isinstance(app, Flask)


def test_get_index_returns_html(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    # Pre-create the template (commit g) -- pour test isolation, skip si missing
    template_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "nanozero_training"
        / "monitoring"
        / "templates"
        / "index.html"
    )
    if not template_path.exists():
        pytest.skip("index.html template not yet created (phase 10 commit g)")
    app = create_app(cfg)
    client = app.test_client()
    resp = client.get("/")
    assert resp.status_code == 200
    assert b"<html" in resp.data.lower() or b"<!doctype" in resp.data.lower()


def test_get_api_state_empty_monitoring_dir(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    app = create_app(cfg)
    client = app.test_client()
    resp = client.get("/api/state")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["run_state"] is None
    assert data["versions_summary"] is None


def test_get_api_state_with_run_state(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    _write_run_state(tmp_path)
    app = create_app(cfg)
    client = app.test_client()
    resp = client.get("/api/state")
    data = resp.get_json()
    assert data["run_state"] is not None
    assert data["run_state"]["run_id"] == "test-run-001"
    assert data["run_state"]["phase"] == "train"
    assert data["run_state"]["train"]["current_epoch"] == 3


def test_get_api_state_with_versions(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    _write_versions(tmp_path, n_entries=2)
    app = create_app(cfg)
    client = app.test_client()
    resp = client.get("/api/state")
    data = resp.get_json()
    assert data["versions_summary"] is not None
    assert data["versions_summary"]["current"] == "gen-001-init"
    assert data["versions_summary"]["total_entries"] == 2


def test_get_api_metrics_empty(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    app = create_app(cfg)
    client = app.test_client()
    resp = client.get("/api/metrics/1")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_get_api_metrics_returns_rows(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    monitoring_dir = cfg.paths.resolve("monitoring_dir")
    writer = MetricsCSVWriter(monitoring_dir=monitoring_dir)
    for epoch in range(3):
        writer.append_train_row(
            gen=1,
            epoch=epoch,
            policy_loss=1.0 - epoch * 0.1,
            value_loss=0.1,
            l2=0.01,
            total_loss=1.11 - epoch * 0.1,
            lr=1e-3,
            grad_norm=2.0,
        )
    app = create_app(cfg)
    client = app.test_client()
    resp = client.get("/api/metrics/1")
    rows = resp.get_json()
    assert len(rows) == 3
    assert int(rows[0]["epoch"]) == 0


def test_get_api_sprt_returns_rows(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    monitoring_dir = cfg.paths.resolve("monitoring_dir")
    writer = MetricsCSVWriter(monitoring_dir=monitoring_dir)
    writer.append_eval_row(
        gen=2,
        games_played=10,
        wins=4,
        losses=4,
        draws=2,
        llr=0.5,
        sprt_status="running",
    )
    app = create_app(cfg)
    client = app.test_client()
    resp = client.get("/api/sprt/2")
    rows = resp.get_json()
    assert len(rows) == 1
    assert rows[0]["sprt_status"] == "running"


def test_get_api_selfplay_returns_rows(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    monitoring_dir = cfg.paths.resolve("monitoring_dir")
    writer = MetricsCSVWriter(monitoring_dir=monitoring_dir)
    writer.append_selfplay_row(gen=1, batch_idx=0, completed_games=50, target_games=250)
    app = create_app(cfg)
    client = app.test_client()
    resp = client.get("/api/selfplay/1")
    rows = resp.get_json()
    assert len(rows) == 1
    assert int(rows[0]["completed_games"]) == 50


def test_get_api_versions_full_list(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    _write_versions(tmp_path, n_entries=3)
    app = create_app(cfg)
    client = app.test_client()
    resp = client.get("/api/versions")
    data = resp.get_json()
    assert data["current"] == "gen-001-init"
    assert len(data["all"]) == 3


def test_get_api_versions_empty_returns_default(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    app = create_app(cfg)
    client = app.test_client()
    resp = client.get("/api/versions")
    data = resp.get_json()
    assert data["current"] is None
    assert data["all"] == []


def test_get_api_stream_returns_event_stream_mime(tmp_path: Path) -> None:
    cfg = _make_config(tmp_path)
    app = create_app(cfg)
    client = app.test_client()
    with client.get("/api/stream") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("Content-Type", "")


def test_sse_yields_state_event_on_first_iter(tmp_path: Path) -> None:
    """Phase 10 Q5 SSE test 1 : first iteration yields a state event."""
    cfg = _make_config(tmp_path)
    app = create_app(cfg)
    client = app.test_client()
    with client.get("/api/stream") as resp:
        gen_iter = iter(resp.response)
        chunk = next(gen_iter)
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        assert "event: state" in text
        assert "data: " in text
        # The payload should parse as JSON
        data_line = next(line for line in text.split("\n") if line.startswith("data: "))
        payload = json.loads(data_line[len("data: ") :])
        assert "run_state" in payload
        assert "versions_summary" in payload


def test_sse_yields_event_on_mtime_change(tmp_path: Path) -> None:
    """Phase 10 Q5 SSE test 2 : modify run_state.yaml mtime -> 2nd event."""
    cfg = _make_config(tmp_path)
    state_path = _write_run_state(tmp_path)
    app = create_app(cfg)
    client = app.test_client()
    with client.get("/api/stream") as resp:
        gen_iter = iter(resp.response)
        # First chunk = initial state event
        first_chunk = next(gen_iter)
        first_text = first_chunk.decode("utf-8") if isinstance(first_chunk, bytes) else first_chunk
        assert "event: state" in first_text

        # Modify run_state.yaml to change mtime
        time.sleep(0.1)
        state_path.write_text(state_path.read_text().replace("train", "selfplay"))

        # Next chunk should fire within poll_interval (~0.05s + overhead)
        time.sleep(0.2)
        second_chunk = next(gen_iter)
        second_text = (
            second_chunk.decode("utf-8") if isinstance(second_chunk, bytes) else second_chunk
        )
        assert "event: state" in second_text
