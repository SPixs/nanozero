"""Integration tests : CSV writers <-> Flask endpoints <-> SSE.

End-to-end : write CSV via MetricsCSVWriter, read via /api/* endpoints.
Validate the monitoring pipeline coherence (file format ↔ reader format).
"""

from __future__ import annotations

import time
from pathlib import Path

from nanozero_training.config.run_config import (
    MonitorConfig,
    PathsConfig,
    RunConfig,
)
from nanozero_training.monitoring.csv_writer import MetricsCSVWriter
from nanozero_training.monitoring.flask_app import create_app


def _make_config(tmp_path: Path) -> RunConfig:
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


def test_train_csv_to_api_metrics_roundtrip(tmp_path: Path) -> None:
    """MetricsCSVWriter.append_train_row -> /api/metrics/<gen> returns same data."""
    cfg = _make_config(tmp_path)
    monitoring_dir = cfg.paths.resolve("monitoring_dir")
    writer = MetricsCSVWriter(monitoring_dir=monitoring_dir)
    writer.append_train_row(
        gen=1,
        epoch=0,
        policy_loss=2.5,
        value_loss=0.3,
        l2=0.01,
        total_loss=2.81,
        lr=1e-3,
        grad_norm=5.2,
    )
    writer.append_train_row(
        gen=1,
        epoch=1,
        policy_loss=2.2,
        value_loss=0.25,
        l2=0.01,
        total_loss=2.46,
        lr=1e-3,
        grad_norm=4.8,
    )

    app = create_app(cfg)
    client = app.test_client()
    resp = client.get("/api/metrics/1")
    rows = resp.get_json()
    assert len(rows) == 2
    assert float(rows[0]["policy_loss"]) == 2.5
    assert float(rows[1]["policy_loss"]) == 2.2
    assert int(rows[1]["epoch"]) == 1


def test_eval_csv_to_api_sprt_roundtrip(tmp_path: Path) -> None:
    """MetricsCSVWriter.append_eval_row -> /api/sprt/<gen> returns same data."""
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
        elo_diff=1.2,
        sprt_status="running",
    )
    writer.append_eval_row(
        gen=2,
        games_played=80,
        wins=40,
        losses=20,
        draws=20,
        llr=2.95,
        elo_diff=11.3,
        sprt_status="h1_accepted",
    )

    app = create_app(cfg)
    client = app.test_client()
    resp = client.get("/api/sprt/2")
    rows = resp.get_json()
    assert len(rows) == 2
    assert rows[1]["sprt_status"] == "h1_accepted"
    assert int(rows[1]["games_played"]) == 80


def test_selfplay_csv_to_api_selfplay_roundtrip(tmp_path: Path) -> None:
    """MetricsCSVWriter.append_selfplay_row -> /api/selfplay/<gen>."""
    cfg = _make_config(tmp_path)
    monitoring_dir = cfg.paths.resolve("monitoring_dir")
    writer = MetricsCSVWriter(monitoring_dir=monitoring_dir)
    for batch_idx in range(3):
        writer.append_selfplay_row(
            gen=1,
            batch_idx=batch_idx,
            completed_games=(batch_idx + 1) * 50,
            target_games=250,
        )

    app = create_app(cfg)
    client = app.test_client()
    resp = client.get("/api/selfplay/1")
    rows = resp.get_json()
    assert len(rows) == 3
    assert int(rows[2]["completed_games"]) == 150


def test_sse_first_iter_yields_state_event(tmp_path: Path) -> None:
    """Phase 10 SSE test 1 : first iter SSE generator -> 'event: state' chunk."""
    cfg = _make_config(tmp_path)
    app = create_app(cfg)
    client = app.test_client()
    with client.get("/api/stream") as resp:
        assert resp.status_code == 200
        gen_iter = iter(resp.response)
        chunk = next(gen_iter)
        text = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        assert "event: state" in text


def test_sse_yields_new_event_on_mtime_change(tmp_path: Path) -> None:
    """Phase 10 SSE test 2 : modify run_state mtime -> 2e event."""
    import yaml

    cfg = _make_config(tmp_path)
    monitoring_dir = cfg.paths.resolve("monitoring_dir")
    state_path = monitoring_dir / "run_state.yaml"
    state = {
        "run_id": "test-001",
        "started": "2026-05-14T10:00:00+00:00",
        "last_updated": "2026-05-14T10:00:00+00:00",
        "status": "in_progress",
        "config_path": "test.yaml",
        "config_hash": "abc",
        "max_generations": 5,
        "target_games_per_gen": 100,
        "current_gen": 1,
        "gens_completed": [],
        "gens_rejected": [],
        "phase": "selfplay",
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
            "base_model": None,
            "output_model_target": None,
            "current_epoch": 0,
            "total_epochs": 0,
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

    app = create_app(cfg)
    client = app.test_client()
    with client.get("/api/stream") as resp:
        gen_iter = iter(resp.response)
        # First event
        first_chunk = next(gen_iter)
        first_text = first_chunk.decode("utf-8") if isinstance(first_chunk, bytes) else first_chunk
        assert "event: state" in first_text

        # Mtime bump
        time.sleep(0.1)
        state["phase"] = "train"
        state_path.write_text(yaml.safe_dump(state))

        # Next chunk should fire within poll_interval (~0.05s) + overhead
        time.sleep(0.2)
        second_chunk = next(gen_iter)
        second_text = (
            second_chunk.decode("utf-8") if isinstance(second_chunk, bytes) else second_chunk
        )
        assert "event: state" in second_text
