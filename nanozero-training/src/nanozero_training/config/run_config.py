"""Top-level RunConfig dataclass + sub-configs (ADR-011).

Composes existing configs (SelfplayConfig, TrainConfig, FastchessConfig) and adds
the new ones needed for phase 9 (MonitorConfig, PathsConfig).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from nanozero_training.eval.fastchess_runner import FastchessConfig
from nanozero_training.selfplay.config import SelfplayConfig
from nanozero_training.train.config import TrainConfig


@dataclass(frozen=True)
class MonitorConfig:
    """Monitoring configuration. Defaults match SPEC §13."""

    enabled: bool = True
    port: int = 5000
    host: str = "127.0.0.1"
    csv_flush_interval_seconds: float = 60.0
    sse_poll_interval_seconds: float = 0.5  # phase 10 : SSE server-internal mtime check

    def __post_init__(self) -> None:
        if not (1 <= self.port <= 65535):
            raise ValueError(f"monitor.port must be in [1, 65535], got {self.port}")
        if self.csv_flush_interval_seconds <= 0:
            raise ValueError(
                "monitor.csv_flush_interval_seconds must be > 0, "
                f"got {self.csv_flush_interval_seconds}"
            )
        if self.sse_poll_interval_seconds <= 0:
            raise ValueError(
                "monitor.sse_poll_interval_seconds must be > 0, "
                f"got {self.sse_poll_interval_seconds}"
            )


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem paths configuration.

    All paths are resolved relative to run_root if not absolute.
    Defaults follow SPEC §11.
    """

    run_root: str = "nano-runs/run-001"
    datasets_dir: str = "datasets"
    models_dir: str = "models"
    monitoring_dir: str = "monitoring"
    versions_yaml: str = "versions.yaml"
    pgn_path: str = "monitoring/sprt.pgn"
    uci_jar: str = ""  # absolute or relative; MUST be set in YAML

    def __post_init__(self) -> None:
        if not self.run_root:
            raise ValueError("paths.run_root cannot be empty")

    def resolve(self, name: str) -> Path:
        """Resolve a path field against run_root if relative.

        Args:
            name: field name (datasets_dir, models_dir, etc.).

        Returns:
            Absolute path.

        Raises:
            AttributeError: unknown field name.
            ValueError: field is empty.
        """
        value = getattr(self, name)
        if not value:
            raise ValueError(f"paths.{name} is empty")
        p = Path(value)
        if p.is_absolute():
            return p
        return Path(self.run_root) / p


@dataclass(frozen=True)
class RunConfig:
    """Top-level run configuration: composes all sub-configs.

    Loaded from YAML via `load_config()`. CLI overrides applied via
    `merge_cli_overrides()`. Hashed via `compute_config_hash()` for
    reproducibility audit.
    """

    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval_fastchess: FastchessConfig = field(default_factory=FastchessConfig)
    monitor: MonitorConfig = field(default_factory=MonitorConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    max_generations: int = 100
    run_id: str = ""

    def __post_init__(self) -> None:
        if self.max_generations <= 0:
            raise ValueError(f"max_generations must be > 0, got {self.max_generations}")
