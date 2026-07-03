"""Promotion logic: H1_ACCEPTED → rename .npz + update versions.yaml.

Workflow promote_if_h1(vm, name, sprt_result):
  Decision Q4 (SPEC §11.5 scenario 6) — 2-phase commit:
    1. If SPRT status != H1_ACCEPTED: register rejection in YAML + save,
       no rename. Return REJECTED.
    2. If gen-N-promoted.npz already exists and registered: idempotent no-op,
       return ALREADY_PROMOTED.
    3. 2-phase commit:
       a. Snapshot in-memory versions for revert.
       b. Update trained entry (promoted=true, sprt_*) + append promoted entry
          + update versions.current.
       c. Save versions.yaml (atomic).
       d. atomic_rename gen-N-trained.npz -> gen-N-promoted.npz.
       e. If rename fails: revert YAML (write previous state back) + raise.
    4. Return PROMOTED.

Guarantees:
  - versions.yaml always reflects intent (current = the model we WANT champion).
  - If crash between (c) and (d), reconcile_on_load detects + completes rename.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum

from nanozero_training.eval.sprt_result import SPRTResult, SPRTStatus
from nanozero_training.utils.atomic_io import atomic_rename
from nanozero_training.version.manager import (
    ModelEntry,
    VersionManager,
    Versions,
)

LOG = logging.getLogger(__name__)


class PromoteOutcome(str, Enum):
    """Outcome of promote_if_h1 call."""

    PROMOTED = "promoted"  # H1 accepted, rename done, YAML updated
    REJECTED = "rejected"  # H0 or MAX_GAMES or ERROR — recorded in YAML, no rename
    ALREADY_PROMOTED = "already_promoted"  # idempotent no-op


@dataclass(frozen=True)
class PromoteResult:
    """Result of promote_if_h1 call."""

    outcome: PromoteOutcome
    promoted_name: str | None  # gen-N-promoted name if PROMOTED, else None
    message: str = ""


def promote_if_h1(
    vm: VersionManager,
    trained_name: str,
    sprt_result: SPRTResult,
    audit_note: str | None = None,
) -> PromoteResult:
    """Promote a trained model if SPRT confirms H1 (challenger > baseline).

    Args:
        vm: VersionManager (must be loaded with current state).
        trained_name: name of the trained model entry, eg. "gen-002-trained".
        sprt_result: SPRTResult from phase 7 eval.
        audit_note: optional override for the `sprt_result` field persisted on
            the entry. Phase 9 use case : CLI manual promote stores
            "manual_override" instead of the SPRT status value to make audit
            distinguishable from automatic SPRT-driven promotions.
            None (default) uses `sprt_result.status.value`.

    Returns:
        PromoteResult with outcome (PROMOTED / REJECTED / ALREADY_PROMOTED).

    Raises:
        ValueError: if trained_name not registered or not type="trained".
        RuntimeError: if 2-phase commit fails (filesystem inconsistency).
    """
    entry = vm.versions.find(trained_name)
    if entry is None:
        raise ValueError(f"Trained model not registered: {trained_name}")
    if entry.type != "trained":
        raise ValueError(f"Cannot promote model of type '{entry.type}': expected 'trained'")

    promoted_name = _derive_promoted_name(trained_name)
    promoted_path = vm.get_path_for(promoted_name)
    trained_path = vm.get_path_for(trained_name)

    # Idempotence: déjà promu (post-rename + entry présente).
    if promoted_path.exists() and vm.versions.find(promoted_name) is not None:
        LOG.info("Model already promoted: %s (idempotent no-op)", promoted_name)
        return PromoteResult(
            outcome=PromoteOutcome.ALREADY_PROMOTED,
            promoted_name=promoted_name,
            message="Already promoted (idempotent)",
        )

    # SPRT non-H1 : record rejection in YAML, no rename.
    if sprt_result.status != SPRTStatus.H1_ACCEPTED:
        _update_entry_with_sprt(
            vm, trained_name, sprt_result, promoted=False, audit_note=audit_note
        )
        vm.save()
        LOG.info(
            "SPRT not H1 (%s) -- model %s rejected",
            sprt_result.status.value,
            trained_name,
        )
        return PromoteResult(
            outcome=PromoteOutcome.REJECTED,
            promoted_name=None,
            message=f"SPRT status={sprt_result.status.value}",
        )

    # H1 accepted : 2-phase commit (Q4 actée).
    if not trained_path.exists():
        raise RuntimeError(f"Cannot promote: trained model file missing: {trained_path}")

    snapshot_versions = _snapshot_versions(vm.versions)

    _update_entry_with_sprt(vm, trained_name, sprt_result, promoted=True, audit_note=audit_note)
    persisted_sprt = audit_note if audit_note is not None else sprt_result.status.value
    promoted_entry = ModelEntry(
        name=promoted_name,
        type="promoted",
        promoted=True,
        parent=trained_name,
        sprt_result=persisted_sprt,
        sprt_games=sprt_result.games_played,
        sprt_elo_diff=sprt_result.elo_diff,
        sprt_llr=sprt_result.llr,
        created=_utc_now_iso(),
    )
    vm.versions.all.append(promoted_entry)
    vm.versions.current = promoted_name

    try:
        vm.save()
    except Exception as e:
        vm._versions = snapshot_versions
        raise RuntimeError(f"versions.yaml write failed: {e}") from e

    try:
        atomic_rename(trained_path, promoted_path)
        # Phase 12+ : si .onnx companion existe, rename aussi pour rester drop-in
        # pour NetworkOnnx Java (uci_client + fastchess_runner pick .onnx auto).
        trained_onnx = trained_path.with_suffix(".onnx")
        if trained_onnx.exists():
            promoted_onnx = promoted_path.with_suffix(".onnx")
            try:
                atomic_rename(trained_onnx, promoted_onnx)
            except Exception as onnx_err:
                LOG.warning("ONNX companion rename failed: %s (selfplay sera Vector API)", onnx_err)
    except Exception as e:
        LOG.error("Rename failed (%s) -- reverting versions.yaml", e)
        vm._versions = snapshot_versions
        try:
            vm.save()
        except Exception as revert_err:
            LOG.error(
                "CRITICAL: revert YAML also failed: %s -- manual intervention required",
                revert_err,
            )
        raise RuntimeError(f"Promote rename failed: {e}") from e

    LOG.info(
        "Promoted %s -> %s (SPRT H1, %d games)",
        trained_name,
        promoted_name,
        sprt_result.games_played,
    )
    return PromoteResult(
        outcome=PromoteOutcome.PROMOTED,
        promoted_name=promoted_name,
        message=f"H1 accepted ({sprt_result.games_played} games)",
    )


def reconcile_on_load(vm: VersionManager) -> bool:
    """Detect and recover from mid-promote crash (SPEC §11.5 scenario 6).

    Scenario: YAML was written with current=gen-N-promoted, but atomic_rename
    didn't complete. Detection: versions.yaml says X.npz should exist but only
    X-trained.npz is on disk. Recovery: complete the rename.

    Args:
        vm: VersionManager (loaded).

    Returns:
        True if reconciliation happened, False if nothing to do.

    Raises:
        RuntimeError: if reconciliation fails (manual intervention needed).
    """
    current = vm.versions.current
    if current is None:
        return False

    current_path = vm.get_path_for(current)
    if current_path.exists():
        return False  # Filesystem already matches

    # Cherche le -trained correspondant (mid-promote crash).
    if current.endswith("-promoted"):
        trained_name = current.replace("-promoted", "-trained")
        trained_path = vm.get_path_for(trained_name)
        if trained_path.exists():
            LOG.warning(
                "Mid-promote crash detected: completing rename %s -> %s",
                trained_path,
                current_path,
            )
            atomic_rename(trained_path, current_path)
            return True

    # Sinon : état corrompu (current fichier absent sans -trained counterpart).
    raise RuntimeError(
        f"Inconsistent state: current={current} but neither {current_path} "
        f"nor a -trained counterpart exists on disk."
    )


def _derive_promoted_name(trained_name: str) -> str:
    """gen-002-trained -> gen-002-promoted."""
    if not trained_name.endswith("-trained"):
        raise ValueError(f"Expected -trained suffix in name: {trained_name}")
    return trained_name.removesuffix("-trained") + "-promoted"


def _update_entry_with_sprt(
    vm: VersionManager,
    name: str,
    sprt_result: SPRTResult,
    promoted: bool,
    audit_note: str | None = None,
) -> None:
    """Update an entry's SPRT metadata in-place via dataclass replace.

    If audit_note is provided, it overrides sprt_result.status.value in the
    persisted entry (phase 9 manual promote support).
    """
    idx = None
    for i, e in enumerate(vm.versions.all):
        if e.name == name:
            idx = i
            break
    if idx is None:
        raise ValueError(f"Entry not found: {name}")
    old = vm.versions.all[idx]
    persisted_sprt = audit_note if audit_note is not None else sprt_result.status.value
    vm.versions.all[idx] = replace(
        old,
        promoted=promoted,
        sprt_result=persisted_sprt,
        sprt_games=sprt_result.games_played,
        sprt_elo_diff=sprt_result.elo_diff,
        sprt_llr=sprt_result.llr,
    )


def _snapshot_versions(v: Versions) -> Versions:
    """Deep-copy Versions for revert on failure (ModelEntry frozen, list copy suffit)."""
    return Versions(
        current=v.current,
        latest_trained=v.latest_trained,
        all=list(v.all),
    )


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
