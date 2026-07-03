"""Integration: version/__init__ public API exposes expected symbols."""

from __future__ import annotations


def test_version_package_exports() -> None:
    from nanozero_training.version import (
        ModelEntry,
        PromoteOutcome,
        PromoteResult,
        VersionManager,
        Versions,
        promote_if_h1,
        reconcile_on_load,
    )

    # PromoteOutcome string values cohérents (peut être persisté).
    assert PromoteOutcome.PROMOTED.value == "promoted"
    assert PromoteOutcome.REJECTED.value == "rejected"
    assert PromoteOutcome.ALREADY_PROMOTED.value == "already_promoted"

    # Symbols importables.
    assert ModelEntry.__name__ == "ModelEntry"
    assert PromoteResult.__name__ == "PromoteResult"
    assert VersionManager.__name__ == "VersionManager"
    assert Versions.__name__ == "Versions"
    assert callable(promote_if_h1)
    assert callable(reconcile_on_load)
