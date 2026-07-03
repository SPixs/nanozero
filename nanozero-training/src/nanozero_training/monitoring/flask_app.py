"""Flask app for training dashboard (REST endpoints + Server-Sent Events).

SSE design (Q6 décision actée):
  - GET /api/stream : long-polling SSE endpoint
  - Server checks mtime of run_state.yaml + versions.yaml every
    sse_poll_interval_seconds (default 0.5s)
  - On change : yield SSE event "state" with current snapshot
  - Each client connection has its own generator (no shared state)

Endpoints (SPEC §10.2):
  GET /              : HTML dashboard (templates/index.html)
  GET /api/state     : current snapshot (run_state + versions summary)
  GET /api/metrics/<gen>  : JSON of train-gen-NNN.csv content
  GET /api/sprt/<gen>     : JSON of eval-gen-NNN.csv content
  GET /api/selfplay/<gen> : JSON of selfplay-gen-NNN.csv content
  GET /api/versions  : list of all versions (versions.yaml.all[])
  GET /api/stream    : SSE event stream
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from flask import Flask, Response, jsonify, render_template

from nanozero_training.config.run_config import RunConfig
from nanozero_training.state.manager import RunStateManager
from nanozero_training.version.manager import VersionManager

LOG = logging.getLogger(__name__)

_JOBSERVER_URL = os.environ.get("NANOZERO_JOBSERVER_URL", "http://devsrv:8090")
_JOBSERVER_KEY = os.environ.get("NANOZERO_JOBSERVER_API_KEY", "")


def create_app(config: RunConfig) -> Flask:
    """Build and configure the Flask app."""
    template_dir = Path(__file__).parent / "templates"
    static_dir = Path(__file__).parent / "static"
    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
    )
    app.config["RUN_CONFIG"] = config
    _register_routes(app)
    return app


def _register_routes(app: Flask) -> None:
    @app.route("/")
    def index() -> str:
        return render_template("index.html")

    @app.route("/api/state")
    def api_state() -> Response:
        cfg: RunConfig = app.config["RUN_CONFIG"]
        return jsonify(_collect_state(cfg))

    @app.route("/api/metrics/<int:gen>")
    def api_metrics(gen: int) -> Response:
        cfg: RunConfig = app.config["RUN_CONFIG"]
        return jsonify(_read_csv(cfg, f"train-gen-{gen:03d}.csv"))

    @app.route("/api/sprt/<int:gen>")
    def api_sprt(gen: int) -> Response:
        cfg: RunConfig = app.config["RUN_CONFIG"]
        return jsonify(_read_csv(cfg, f"eval-gen-{gen:03d}.csv"))

    @app.route("/api/selfplay/<int:gen>")
    def api_selfplay(gen: int) -> Response:
        cfg: RunConfig = app.config["RUN_CONFIG"]
        return jsonify(_read_csv(cfg, f"selfplay-gen-{gen:03d}.csv"))

    @app.route("/api/versions")
    def api_versions() -> Response:
        cfg: RunConfig = app.config["RUN_CONFIG"]
        return jsonify(_collect_versions(cfg))

    @app.route("/api/jobserver")
    def api_jobserver() -> Response:
        """Proxy vers le jobserver DevSrv pour stats self-play distribué."""
        try:
            req = urllib.request.Request(
                f"{_JOBSERVER_URL}/stats",
                headers={"X-API-Key": _JOBSERVER_KEY},
            )
            with urllib.request.urlopen(req, timeout=3) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            try:
                req_w = urllib.request.Request(
                    f"{_JOBSERVER_URL}/workers",
                    headers={"X-API-Key": _JOBSERVER_KEY},
                )
                with urllib.request.urlopen(req_w, timeout=3) as resp_w:
                    workers_data = json.loads(resp_w.read().decode("utf-8"))
                if isinstance(workers_data, list):
                    data["workers"] = len(workers_data)
                elif isinstance(workers_data, dict) and "workers" in workers_data:
                    data["workers"] = len(workers_data["workers"])
            except Exception:
                pass
            return jsonify(data)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            return jsonify({"error": str(e)}), 503

    @app.route("/api/stream")
    def api_stream() -> Response:
        cfg: RunConfig = app.config["RUN_CONFIG"]
        return Response(
            _sse_generator(cfg),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )


def _collect_state(cfg: RunConfig) -> dict[str, Any]:
    """Read run_state.yaml + versions.yaml summary."""
    monitoring_dir = cfg.paths.resolve("monitoring_dir")
    versions_yaml = cfg.paths.resolve("versions_yaml")
    models_dir = cfg.paths.resolve("models_dir")

    result: dict[str, Any] = {"run_state": None, "versions_summary": None}

    state_path = monitoring_dir / "run_state.yaml"
    if state_path.exists():
        try:
            sm = RunStateManager(monitoring_dir=monitoring_dir)
            sm.load_existing()
            result["run_state"] = {
                "run_id": sm.state.run_id,
                "status": sm.state.status,
                "phase": sm.state.phase,
                "current_gen": sm.state.current_gen,
                "selfplay": {
                    "completed_games": sm.state.selfplay.completed_games,
                    "completed_batches": sm.state.selfplay.completed_batches,
                    "current_batch_idx": sm.state.selfplay.current_batch_idx,
                    "target_games": sm.state.selfplay.target_games,
                },
                "train": {
                    "current_epoch": sm.state.train.current_epoch,
                    "total_epochs": sm.state.train.total_epochs,
                },
                "eval": {
                    "challenger": sm.state.eval.challenger,
                    "baseline": sm.state.eval.baseline,
                    "games_played": sm.state.eval.games_played_at_last_save,
                    "last_decision": sm.state.eval.last_decision,
                },
            }
        except Exception as e:
            LOG.warning("Failed to load run_state: %s", e)

    if versions_yaml.exists():
        try:
            vm = VersionManager(versions_yaml_path=versions_yaml, models_dir=models_dir)
            vm.load(auto_reconcile=False)
            result["versions_summary"] = {
                "current": vm.versions.current,
                "latest_trained": vm.versions.latest_trained,
                "total_entries": len(vm.versions.all),
            }
        except Exception as e:
            LOG.warning("Failed to load versions: %s", e)

    return result


def _collect_versions(cfg: RunConfig) -> dict[str, Any]:
    """Read full versions.yaml content."""
    versions_yaml = cfg.paths.resolve("versions_yaml")
    models_dir = cfg.paths.resolve("models_dir")
    if not versions_yaml.exists():
        return {"current": None, "latest_trained": None, "all": []}
    try:
        vm = VersionManager(versions_yaml_path=versions_yaml, models_dir=models_dir)
        vm.load(auto_reconcile=False)
        return {
            "current": vm.versions.current,
            "latest_trained": vm.versions.latest_trained,
            "all": [e.to_dict() for e in vm.versions.all],
        }
    except Exception as e:
        LOG.warning("Failed to load versions.yaml: %s", e)
        return {"current": None, "latest_trained": None, "all": []}


def _read_csv(cfg: RunConfig, filename: str) -> list[dict[str, Any]]:
    """Read a CSV file as list of dict rows. Returns [] if absent."""
    monitoring_dir = cfg.paths.resolve("monitoring_dir")
    path = monitoring_dir / filename
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(dict(row))
    except Exception as e:
        LOG.warning("Failed to read CSV %s: %s", path, e)
    return rows


def _sse_generator(cfg: RunConfig) -> Iterator[str]:
    """SSE generator : poll mtime, yield events on change.

    Polls every sse_poll_interval_seconds (default 0.5s). When run_state.yaml or
    versions.yaml mtime changes (or first iteration), yields a 'state' event
    with current snapshot.
    """
    monitoring_dir = cfg.paths.resolve("monitoring_dir")
    versions_path = cfg.paths.resolve("versions_yaml")
    state_path = monitoring_dir / "run_state.yaml"

    last_state_mtime = -1.0
    last_versions_mtime = -1.0
    first_iter = True
    poll_interval = cfg.monitor.sse_poll_interval_seconds

    while True:
        try:
            cur_state_mtime = state_path.stat().st_mtime if state_path.exists() else -1.0
            cur_versions_mtime = versions_path.stat().st_mtime if versions_path.exists() else -1.0

            if (
                first_iter
                or cur_state_mtime != last_state_mtime
                or cur_versions_mtime != last_versions_mtime
            ):
                snapshot = _collect_state(cfg)
                payload = json.dumps(snapshot)
                yield f"event: state\ndata: {payload}\n\n"
                last_state_mtime = cur_state_mtime
                last_versions_mtime = cur_versions_mtime
                first_iter = False

            time.sleep(poll_interval)
        except GeneratorExit:
            LOG.debug("SSE client disconnected")
            return
        except Exception as e:
            LOG.warning("SSE generator error: %s", e)
            try:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
            except Exception:
                return
            time.sleep(poll_interval)
