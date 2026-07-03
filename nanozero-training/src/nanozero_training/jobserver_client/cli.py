"""CLI entry points for distributed-mode operations (Phase 13.5c).

Provides:
  nanozero-trainer-stream    Run the TrainerStreamingLoop forever.
  nanozero-jobserver-seed    One-shot : register + promote a bootstrap model
                             and enqueue an initial batch of jobs. Useful for
                             starting a nightly run from a clean DB.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click

from nanozero_training.jobserver_client.trainer import TrainerClient
from nanozero_training.jobserver_client.trainer_loop import TrainerStreamingLoop

if TYPE_CHECKING:
    from nanozero_training.train.config import TrainConfig

LOG = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# nanozero-trainer-stream
# ---------------------------------------------------------------------------


@click.command(name="trainer-stream")
@click.option("--server", required=True, help="jobserver base URL, e.g. http://devsrv:8090")
@click.option("--key", default="", help="X-API-Key value (dev mode if empty)")
@click.option(
    "--models-dir",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="shared-fs directory for new .npz/.onnx outputs",
)
@click.option("--threshold", default=25_000, show_default=True, help="positions before training")
@click.option(
    "--window", default=5, show_default=True, help="replay rolling window (model versions)"
)
@click.option(
    "--sample-size", default=None, type=int, help="positions per fetch (default=threshold)"
)
@click.option("--batch-size", default=256, show_default=True)
@click.option("--poll-interval-seconds", default=30, show_default=True)
@click.option(
    "--jobs-per-promote",
    default=250,
    show_default=True,
    help="jobs to enqueue after each promote (0 = manual control)",
)
@click.option("--num-sims-per-job", default=200, show_default=True, help="MCTS sims per move")
@click.option(
    "--max-promotions", default=None, type=int, help="stop after N promotions (default: forever)"
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="optional TrainConfig YAML (uses defaults otherwise)",
)
@click.option("--verbose", "-v", is_flag=True)
def trainer_stream_cmd(
    server: str,
    key: str,
    models_dir: Path,
    threshold: int,
    window: int,
    sample_size: int | None,
    batch_size: int,
    poll_interval_seconds: int,
    jobs_per_promote: int,
    num_sims_per_job: int,
    max_promotions: int | None,
    config: str | None,
    verbose: bool,
) -> None:
    """Start the TrainerStreamingLoop : poll the jobserver, train, register + promote."""
    _setup_logging(verbose)

    # Lazy import torch + Trainer (heavy) so --help is fast.
    from nanozero_training.network.export_npz import (
        export_to_npz,  # noqa: F401 (imported here ensures the trainer file is importable later)
    )
    from nanozero_training.network.resnet import NanoZeroResNet
    from nanozero_training.train.config import TrainConfig
    from nanozero_training.train.trainer import Trainer

    train_config = TrainConfig() if config is None else _load_train_config_from_yaml(config)
    model = NanoZeroResNet()
    trainer = Trainer(model=model, config=train_config, device="cpu")

    LOG.info("Starting trainer-stream → server=%s models-dir=%s", server, models_dir)
    with TrainerClient(server_url=server, api_key=key) as client:
        loop = TrainerStreamingLoop(
            client=client,
            trainer=trainer,
            models_dir=models_dir,
            threshold=threshold,
            window=window,
            sample_size=sample_size,
            batch_size=batch_size,
            poll_interval_seconds=poll_interval_seconds,
            jobs_per_promote=jobs_per_promote,
            num_sims_per_job=num_sims_per_job,
        )
        n = loop.run(max_promotions=max_promotions)
        LOG.info("trainer-stream exited after %d promotions", n)


def _load_train_config_from_yaml(path: str) -> TrainConfig:
    import yaml

    from nanozero_training.train.config import TrainConfig

    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return TrainConfig(**(data.get("train") or data))


# ---------------------------------------------------------------------------
# nanozero-jobserver-seed
# ---------------------------------------------------------------------------


@click.command(name="jobserver-seed")
@click.option("--server", required=True)
@click.option("--key", default="")
@click.option(
    "--init-onnx",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="path to the gen-001-init.onnx (must be reachable by the jobserver too)",
)
@click.option("--init-name", default="gen-001-init", show_default=True)
@click.option("--jobs", default=250, show_default=True, help="how many gen-1 jobs to enqueue")
@click.option("--num-sims", default=200, show_default=True)
@click.option("--verbose", "-v", is_flag=True)
def jobserver_seed_cmd(
    server: str,
    key: str,
    init_onnx: Path,
    init_name: str,
    jobs: int,
    num_sims: int,
    verbose: bool,
) -> None:
    """One-shot bootstrap : register the init model, promote it, enqueue initial jobs."""
    _setup_logging(verbose)
    sha = _sha256(init_onnx)
    LOG.info("Seeding jobserver %s with init model %s (sha=%s)", server, init_name, sha[:12])

    with TrainerClient(server_url=server, api_key=key) as client:
        existing = client.current_model()
        if existing is not None:
            LOG.warning(
                "Jobserver already has a promoted model (v%d, %s). Skipping register.",
                existing["version"],
                existing["name"],
            )
            current_version = existing["version"]
        else:
            client.register_model(
                version=1, name=init_name, onnx_path=str(init_onnx), sha256_onnx=sha
            )
            client.promote_model(version=1)
            LOG.info("Registered + promoted v1 (%s)", init_name)
            current_version = 1

        result = client.enqueue_jobs(count=jobs, model_version=current_version, num_sims=num_sims)
        LOG.info(
            "Enqueued %d jobs for v%d. Workers can start now.",
            result.get("enqueued"),
            current_version,
        )


# ---------------------------------------------------------------------------
# Top-level click group (exposed as console scripts in pyproject.toml)
# ---------------------------------------------------------------------------


def trainer_stream_main() -> None:
    """Entry point for `nanozero-trainer-stream` console script."""
    trainer_stream_cmd()


def jobserver_seed_main() -> None:
    """Entry point for `nanozero-jobserver-seed` console script."""
    jobserver_seed_cmd()
