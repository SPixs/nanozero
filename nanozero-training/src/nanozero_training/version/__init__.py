"""Version management: track model generations + promotion logic (ADR-007)."""

from nanozero_training.version.manager import (
    ModelEntry,
    VersionManager,
    Versions,
)
from nanozero_training.version.promotion import (
    PromoteOutcome,
    PromoteResult,
    promote_if_h1,
    reconcile_on_load,
)

__all__ = [
    "ModelEntry",
    "PromoteOutcome",
    "PromoteResult",
    "VersionManager",
    "Versions",
    "promote_if_h1",
    "reconcile_on_load",
]
