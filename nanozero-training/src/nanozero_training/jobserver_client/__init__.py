"""HTTP worker that pulls jobs from a nanozero-jobserver and plays games.

Cf. nanozero-jobserver/ADR-014 §Architecture. The worker is a thin HTTP layer
on top of the existing UciClient + play_one_game machinery — no engine changes.
"""

from nanozero_training.jobserver_client.trainer import (
    DecodedPosition,
    ShouldTrainStatus,
    TrainerClient,
    decode_position,
)
from nanozero_training.jobserver_client.trainer_loop import (
    JobserverReplayDataset,
    TrainerStreamingLoop,
    compute_sha256,
)
from nanozero_training.jobserver_client.worker import (
    JobserverClient,
    JobserverWorker,
    WorkerConfig,
)

__all__ = [
    # Worker side (Phase 13.4 — MVP Python reference).
    "JobserverClient",
    "JobserverWorker",
    "WorkerConfig",
    # Trainer side (Phase 13.5).
    "TrainerClient",
    "DecodedPosition",
    "ShouldTrainStatus",
    "decode_position",
    "JobserverReplayDataset",
    "TrainerStreamingLoop",
    "compute_sha256",
]
