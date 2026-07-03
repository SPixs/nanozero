"""RunStateManager — persistent run state with atomic flushes (ADR-009)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from nanozero_training.utils.atomic_io import atomic_rename, atomic_write_yaml

# Type aliases for clarity
PhaseLiteral = str  # "selfplay" | "train" | "eval" | "promote" | "idle"
StatusLiteral = str  # "in_progress" | "completed" | "aborted" | "error"


@dataclass
class SelfplayState:
    target_games: int = 0
    completed_batches: int = 0
    completed_games: int = 0
    current_batch_idx: int = 0
    current_batch_games: int = 0
    last_batch_file: str | None = None
    worker_seed: int = 42


@dataclass
class TrainState:
    base_model: str | None = None
    output_model_target: str | None = None
    current_epoch: int = 0
    total_epochs: int = 0
    last_checkpoint: str | None = None
    optimizer_state_file: str | None = None


@dataclass
class EvalState:
    """Eval phase state (SPRT).

    `last_decision` (phase 1.0.0-7) holds the SPRTStatus enum value as a string once
    fastchess completes : "h1_accepted" | "h0_accepted" | "max_games" | "error".
    Stays None tant que le SPRT n'est pas terminé (running ou not started).
    """

    challenger: str | None = None
    baseline: str | None = None
    pgn_path: str | None = None
    games_played_at_last_save: int = 0
    last_decision: str | None = None


@dataclass
class RunState:
    """Persistent run state. Serialized to monitoring/run_state.yaml."""

    run_id: str
    started: datetime
    last_updated: datetime
    status: StatusLiteral
    config_path: str
    config_hash: str
    max_generations: int
    target_games_per_gen: int
    current_gen: int
    gens_completed: list[int] = field(default_factory=list)
    gens_rejected: list[int] = field(default_factory=list)
    phase: PhaseLiteral = "idle"
    phase_started: datetime | None = None
    selfplay: SelfplayState = field(default_factory=SelfplayState)
    train: TrainState = field(default_factory=TrainState)
    eval: EvalState = field(default_factory=EvalState)
    last_error: str | None = None
    crash_count: int = 0
    last_resume: datetime | None = None


class RunStateManager:
    """Manages the persistent run_state.yaml for resilience (ADR-009).

    Thread safety: single-process v1.0.0. Concurrent access is not supported.
    All flushes use atomic writes (write tmp + rename).
    """

    def __init__(self, monitoring_dir: str | Path) -> None:
        self.monitoring_dir = Path(monitoring_dir)
        self.path = self.monitoring_dir / "run_state.yaml"
        self.log_path = self.monitoring_dir / "run_state.log"
        self._state: RunState | None = None

    @property
    def state(self) -> RunState:
        if self._state is None:
            raise RuntimeError("RunStateManager not initialized — call load_or_create first")
        return self._state

    def detect_existing_run(self) -> bool:
        """Return True if run_state.yaml exists and status='in_progress'."""
        if not self.path.exists():
            return False
        try:
            with self.path.open(encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except (OSError, yaml.YAMLError):
            return False
        return bool(data and data.get("status") == "in_progress")

    def load_existing(self) -> RunState:
        """Load an existing run_state.yaml. Raises if absent or not in_progress."""
        if not self.path.exists():
            raise FileNotFoundError(f"No run_state.yaml at {self.path}")
        with self.path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data.get("status") != "in_progress":
            raise ValueError(f"run_state status is {data.get('status')!r}, not 'in_progress'")
        self._state = self._dict_to_state(data)
        return self._state

    def create_new(
        self,
        run_id: str,
        config_path: str,
        config_hash: str,
        max_generations: int,
        target_games_per_gen: int,
    ) -> RunState:
        """Create a new run state and flush it.

        Refuses if a run is already in progress.
        """
        if self.detect_existing_run():
            raise RuntimeError(
                f"A run is already in progress at {self.path}. "
                "Use 'resume' or archive/delete the state first."
            )
        now = datetime.now(timezone.utc)
        self._state = RunState(
            run_id=run_id,
            started=now,
            last_updated=now,
            status="in_progress",
            config_path=config_path,
            config_hash=config_hash,
            max_generations=max_generations,
            target_games_per_gen=target_games_per_gen,
            current_gen=0,
        )
        self.flush()
        return self._state

    def update(self, **updates: Any) -> None:
        """Update fields on the state, then flush.

        Dot-notation via __ separator allows updating nested dataclasses:
            mgr.update(phase="train")
            mgr.update(train__current_epoch=7, train__last_checkpoint="...")
        """
        if self._state is None:
            raise RuntimeError("State not initialized")
        for key, value in updates.items():
            if "__" in key:
                section, field_name = key.split("__", 1)
                section_obj = getattr(self._state, section)
                setattr(section_obj, field_name, value)
            else:
                setattr(self._state, key, value)
        self._state.last_updated = datetime.now(timezone.utc)
        self.flush()

    def flush(self) -> None:
        """Atomic write run_state.yaml + append run_state.log."""
        if self._state is None:
            raise RuntimeError("State not initialized")
        data = self._state_to_dict(self._state)
        atomic_write_yaml(self.path, data)
        # Best-effort event log
        try:
            self.monitoring_dir.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(
                    f"{self._state.last_updated.isoformat()} "
                    f"phase={self._state.phase} status={self._state.status}\n"
                )
        except OSError:
            pass  # Event log failure is non-fatal

    def archive(self) -> None:
        """Move run_state.yaml to runs/<run_id>/ at completion or abort."""
        if self._state is None:
            raise RuntimeError("State not initialized")
        archive_dir = self.monitoring_dir / "runs" / self._state.run_id
        archive_dir.mkdir(parents=True, exist_ok=True)
        atomic_rename(self.path, archive_dir / "run_state.yaml")
        (archive_dir / "completed.flag").touch()
        if self.log_path.exists():
            atomic_rename(self.log_path, archive_dir / "run_state.log")

    @staticmethod
    def _state_to_dict(state: RunState) -> dict[str, Any]:
        d = asdict(state)
        for key in ("started", "last_updated", "phase_started", "last_resume"):
            val = d.get(key)
            if isinstance(val, datetime):
                d[key] = val.isoformat()
        return d

    @staticmethod
    def _dict_to_state(d: dict[str, Any]) -> RunState:
        for key in ("started", "last_updated", "phase_started", "last_resume"):
            val = d.get(key)
            if isinstance(val, str):
                d[key] = datetime.fromisoformat(val)
        if "selfplay" in d and isinstance(d["selfplay"], dict):
            d["selfplay"] = SelfplayState(**d["selfplay"])
        if "train" in d and isinstance(d["train"], dict):
            d["train"] = TrainState(**d["train"])
        if "eval" in d and isinstance(d["eval"], dict):
            d["eval"] = EvalState(**d["eval"])
        return RunState(**d)
