"""Trainer streaming orchestrator (Phase 13.5b).

Polls the jobserver for the trainer-trigger signal, fetches a batch of replay
positions, runs one training epoch on the existing Trainer, exports the new
model as .npz + .onnx, then registers + promotes the new version on the
jobserver.

This replaces the file-based pipeline (selfplay → train → eval → promote loop
in pipeline/orchestrator.py) for the distributed mode (cf. ADR-014). The
worker side (job claim → play game → submit positions) feeds the server's
replay buffer ; this trainer loop drains it.

NB: no SPRT gate in Phase 13.5. The new model is promoted unconditionally
on each training round (Lc0-style "continuous training"). Phase 13.6 can
re-introduce an SPRT eval step if needed.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch.utils.data import DataLoader, Dataset

from nanozero_training.jobserver_client.trainer import DecodedPosition, TrainerClient

if TYPE_CHECKING:
    from nanozero_training.train.trainer import Trainer

LOG = logging.getLogger(__name__)


class JobserverReplayDataset(Dataset[dict[str, torch.Tensor]]):
    """In-memory PyTorch Dataset wrapping a list of DecodedPosition.

    Returns dicts compatible with Trainer.train_epoch (same keys as
    data.dataset.NpzDataset.__getitem__).
    """

    def __init__(self, positions: list[DecodedPosition]):
        self.positions = positions

    def __len__(self) -> int:
        return len(self.positions)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        p = self.positions[idx]
        return {
            "input_planes": torch.from_numpy(p.input_planes.copy()).float(),
            "policy_target": torch.from_numpy(p.policy_target.copy()).float(),
            "value_target": torch.tensor(p.value_target, dtype=torch.float32),
            "turn": torch.tensor(0, dtype=torch.long),  # not used by alphazero_loss
            "ply": torch.tensor(p.ply, dtype=torch.long),
        }


def compute_sha256(path: Path) -> str:
    """SHA-256 hex digest of a file, streaming-read to avoid loading big .onnx into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(64 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class TrainerStreamingLoop:
    """Streaming trainer : poll → fetch → train → export → register → promote.

    Loop semantics :
        while True:
            status = client.should_train(threshold)
            if not status.should_train: sleep(poll_interval); continue
            positions = client.fetch_sample(threshold, window)
            train_one_epoch(positions)
            export_model(new_version)
            register + promote new_version
    """

    def __init__(
        self,
        client: TrainerClient,
        trainer: Trainer,
        models_dir: Path,
        threshold: int = 25_000,
        window: int = 5,
        sample_size: int | None = None,
        batch_size: int = 256,
        poll_interval_seconds: float = 30.0,
        jobs_per_promote: int = 250,
        num_sims_per_job: int = 200,
    ):
        """
        Args:
            client: TrainerClient already configured against the jobserver.
            trainer: existing Trainer (holds model + optimizer/scheduler factory).
            models_dir: directory writable by both trainer + jobserver (shared FS).
                Each new model writes both {name}.npz + {name}.onnx here.
            threshold: positions threshold for the should_train trigger.
            window: replay buffer rolling window (number of model versions).
            sample_size: number of positions to fetch per epoch. Defaults to
                threshold (= drain rate ≈ creation rate).
            batch_size: PyTorch DataLoader batch size.
            poll_interval_seconds: sleep when should_train is False.
            jobs_per_promote: number of jobs to enqueue after each promote
                (sustain worker queue for the new model version). 0 disables
                auto-enqueue (manual control via /jobs/enqueue).
            num_sims_per_job: MCTS simulations per move enqueued in each job.
        """
        self.client = client
        self.trainer = trainer
        self.models_dir = Path(models_dir)
        self.threshold = threshold
        self.window = window
        self.sample_size = sample_size if sample_size is not None else threshold
        self.batch_size = batch_size
        self.poll_interval_seconds = poll_interval_seconds
        self.jobs_per_promote = jobs_per_promote
        self.num_sims_per_job = num_sims_per_job
        self.models_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------------- public

    def run(self, max_promotions: int | None = None) -> int:
        """Run the loop. Returns the number of successful promotions.

        Args:
            max_promotions: stop after this many ; None = forever.
        """
        promotions = 0
        while max_promotions is None or promotions < max_promotions:
            try:
                if self._run_one_iteration():
                    promotions += 1
            except KeyboardInterrupt:
                LOG.info("KeyboardInterrupt — exiting cleanly after %d promotions", promotions)
                break
            except Exception:
                LOG.exception("Iteration failed (will retry after poll interval).")
                time.sleep(self.poll_interval_seconds)
        return promotions

    # ----------------------------------------------------------------- internals

    def _run_one_iteration(self) -> bool:
        """Try to train + promote once. Returns True if a promotion happened."""
        status = self.client.should_train(threshold=self.threshold)
        if not status.should_train:
            LOG.debug(
                "Not enough new positions yet: %d / %d", status.new_positions, status.threshold
            )
            time.sleep(self.poll_interval_seconds)
            return False

        LOG.info(
            "Trigger fired: %d new positions ≥ %d threshold.",
            status.new_positions,
            status.threshold,
        )

        # Determine current model version (base for the new one's name + parent).
        current = self.client.current_model()
        current_version = int(current["version"]) if current else 0

        # Fetch a batch.
        positions = self.client.fetch_sample(
            n=self.sample_size, window=self.window, current_version=current_version
        )
        if not positions:
            LOG.warning("fetch_sample returned 0 positions despite should_train=True; skip.")
            time.sleep(self.poll_interval_seconds)
            return False
        LOG.info("Fetched %d positions for training.", len(positions))

        # Train 1 epoch.
        dataset = JobserverReplayDataset(positions)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        optimizer = self.trainer.make_optimizer()
        scheduler = self.trainer.make_scheduler(optimizer, total_steps=max(len(loader), 1))
        metrics = self.trainer.train_epoch(loader, optimizer, scheduler)
        LOG.info("Epoch trained: %s", metrics)

        # Export new model (npz + onnx) and register + promote.
        new_version = current_version + 1
        npz_path, onnx_path = self._export_model(new_version)
        sha256 = compute_sha256(onnx_path)
        self.client.register_model(
            version=new_version,
            name=npz_path.stem,
            onnx_path=str(onnx_path),
            sha256_onnx=sha256,
            parent_version=current_version if current_version > 0 else None,
        )
        self.client.promote_model(version=new_version)
        LOG.info(
            "Promoted model v%d (%s, %d positions trained on, parent=v%d)",
            new_version,
            npz_path.stem,
            len(positions),
            current_version,
        )

        # Sustain the worker queue : after each promote, enqueue N jobs for the
        # new model version. Disabled when jobs_per_promote=0 (manual control).
        if self.jobs_per_promote > 0:
            result = self.client.enqueue_jobs(
                count=self.jobs_per_promote,
                model_version=new_version,
                num_sims=self.num_sims_per_job,
            )
            LOG.info(
                "Enqueued %d jobs for model v%d (workers will now produce gen-%d data).",
                result.get("enqueued", 0),
                new_version,
                new_version,
            )
        return True

    def _export_model(self, version: int) -> tuple[Path, Path]:
        """Save the trainer's current model as .npz + .onnx in models_dir.

        Returns (npz_path, onnx_path).
        """
        from nanozero_training.network.export_npz import export_to_npz
        from nanozero_training.network.export_onnx import export_onnx_companion

        name = f"gen-{version:03d}-trained"
        npz_path = self.models_dir / f"{name}.npz"
        export_to_npz(self.trainer.model, npz_path, meta={"version": version})
        onnx_path = export_onnx_companion(npz_path)
        return npz_path, onnx_path
