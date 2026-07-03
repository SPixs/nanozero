"""Worker client : pull jobs from a nanozero-jobserver, play games, submit results.

Architecture:
    [Server] /jobs/claim → JobClaimResponse {job_id, model_version, num_sims, ...}
            ⇣
    [Worker] download_model_if_needed(version)  (cached on disk)
            UciClient.start(jar, model.onnx)    (existing engine subprocess)
            play_one_game(uci, config, rng)     (existing selfplay logic)
            ⇣
    [Server] /jobs/{id}/submit ← b64-encoded positions

JobserverClient is the bare HTTP wrapper (mockable in tests).
JobserverWorker is the loop that ties claim → play → submit.
"""

from __future__ import annotations

import base64
import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np

from nanozero_training.data.sample import Sample
from nanozero_training.selfplay.config import SelfplayConfig
from nanozero_training.selfplay.uci_client import UciClient
from nanozero_training.selfplay.worker import play_one_game

LOG = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config + low-level HTTP client
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkerConfig:
    """Static config for one worker instance.

    Attributes:
        server_url: nanozero-jobserver base URL (e.g., http://devsrv:8090).
        api_key: X-API-Key header value.
        worker_id: identifier the server stores on claim (e.g., hostname).
        uci_jar_path: path to nanozero-uci-X.Y.Z.jar.
        models_dir: local directory where downloaded .onnx files are cached.
        selfplay_config: existing SelfplayConfig — used for game playback.
        poll_idle_seconds: sleep when server returns 204 (empty queue).
        request_timeout_seconds: per-request HTTP timeout.
        download_chunk_size: bytes per chunk when streaming .onnx download.
    """

    server_url: str
    api_key: str
    worker_id: str
    uci_jar_path: Path
    models_dir: Path
    selfplay_config: SelfplayConfig
    poll_idle_seconds: float = 5.0
    request_timeout_seconds: float = 30.0
    download_chunk_size: int = 64 * 1024


class JobserverClient:
    """Thin HTTP layer above the jobserver. Easy to mock in unit tests.

    No business logic — just request/response shaping. The worker drives the
    state machine on top.
    """

    def __init__(
        self,
        server_url: str,
        api_key: str,
        worker_id: str,
        timeout_seconds: float = 30.0,
        http_client: httpx.Client | None = None,
    ):
        self._owns_client = http_client is None
        if http_client is None:
            http_client = httpx.Client(
                base_url=server_url.rstrip("/"),
                headers={"X-API-Key": api_key, "X-Worker-Id": worker_id},
                timeout=timeout_seconds,
            )
        self.http = http_client

    def close(self) -> None:
        if self._owns_client:
            self.http.close()

    def __enter__(self) -> JobserverClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def claim(self) -> dict[str, Any] | None:
        """POST /jobs/claim. Returns the job dict, or None if queue empty (204)."""
        resp = self.http.post("/jobs/claim")
        if resp.status_code == 204:
            return None
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    def submit(self, job_id: str, body: dict[str, Any]) -> dict[str, Any]:
        """POST /jobs/{id}/submit. Returns the JSON response."""
        resp = self.http.post(f"/jobs/{job_id}/submit", json=body)
        resp.raise_for_status()
        return cast(dict[str, Any], resp.json())

    def download_model(self, version: int, target_path: Path, chunk_size: int = 64 * 1024) -> None:
        """GET /models/{version}/download. Stream-writes to target_path.

        Args:
            version: model version requested.
            target_path: local destination file path (parent must exist).
            chunk_size: streaming chunk size in bytes.
        """
        with self.http.stream("GET", f"/models/{version}/download") as resp:
            resp.raise_for_status()
            with target_path.open("wb") as f:
                for chunk in resp.iter_bytes(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)


# -----------------------------------------------------------------------------
# Worker loop
# -----------------------------------------------------------------------------


class JobserverWorker:
    """Top-level worker loop : claim → fetch model → play → submit, forever.

    Single-threaded ; concurrency is achieved by running multiple worker
    PROCESSES (each its own UciClient subprocess). That sidesteps the GIL and
    matches the "1 game at a time per worker" granularity in ADR-014.
    """

    def __init__(
        self,
        config: WorkerConfig,
        client: JobserverClient | None = None,
        uci_client: UciClient | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.config = config
        self.client = client or JobserverClient(
            server_url=config.server_url,
            api_key=config.api_key,
            worker_id=config.worker_id,
            timeout_seconds=config.request_timeout_seconds,
        )
        self.uci = uci_client or UciClient()
        self.rng = rng if rng is not None else np.random.default_rng()
        self._loaded_model_version: int | None = None

    # ------------------------------------------------------------------ public

    def run_forever(self, max_jobs: int | None = None) -> int:
        """Main loop. Returns count of successfully submitted jobs.

        Args:
            max_jobs: if set, stop after this many jobs (useful for tests).
                None = loop forever.
        """
        submitted = 0
        try:
            while max_jobs is None or submitted < max_jobs:
                if not self._run_one_iteration():
                    time.sleep(self.config.poll_idle_seconds)
                    continue
                submitted += 1
        finally:
            self.shutdown()
        return submitted

    def shutdown(self) -> None:
        """Quit the UCI subprocess + close HTTP client. Idempotent."""
        try:
            self.uci.quit()
        except Exception as e:
            LOG.warning("UCI quit failed: %s", e)
        self.client.close()

    # ------------------------------------------------------------------ internals

    def _run_one_iteration(self) -> bool:
        """Try to claim + play + submit ONE job. Returns True if a job was completed."""
        job = self.client.claim()
        if job is None:
            return False

        try:
            self._ensure_model_loaded(job["model_version"])
            samples = play_one_game(self.uci, self.config.selfplay_config, self.rng)
            self._submit_samples(job, samples)
            LOG.info(
                "Job %s completed: %d positions submitted",
                job["job_id"],
                len(samples),
            )
            return True
        except Exception as e:
            LOG.exception("Job %s failed (will be requeued by server): %s", job["job_id"], e)
            return False

    def _ensure_model_loaded(self, version: int) -> None:
        """Download model file if absent locally, (re)start UciClient if version changed."""
        if version == self._loaded_model_version and self.uci.is_alive():
            return

        local = self.config.models_dir / f"model-v{version:04d}.onnx"
        local.parent.mkdir(parents=True, exist_ok=True)
        if not local.exists():
            LOG.info("Downloading model v%d → %s", version, local)
            self.client.download_model(version, local, chunk_size=self.config.download_chunk_size)

        if self.uci.is_alive():
            self.uci.quit()
        self.uci.start(uci_jar=self.config.uci_jar_path, model_path=local)
        self._loaded_model_version = version
        LOG.info("UCI engine started with model v%d", version)

    def _submit_samples(self, job: dict[str, Any], samples: list[Sample]) -> None:
        """Translate Sample list → b64-encoded JSON, POST /jobs/{id}/submit."""
        positions = [
            {
                "ply": int(s.ply),
                "fen": "",  # not stored by play_one_game ; trainer doesn't need it
                "input_planes_b64": base64.b64encode(s.input_planes.tobytes()).decode("ascii"),
                "policy_target_b64": base64.b64encode(s.policy_target.tobytes()).decode("ascii"),
                "outcome": float(s.value_target),
            }
            for s in samples
        ]
        body = {
            "game_id": uuid.uuid4().hex,
            "model_version": int(job["model_version"]),
            "positions": positions,
        }
        self.client.submit(job["job_id"], body)
