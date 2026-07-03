"""Unit tests for version/promotion — promote_if_h1 + reconcile_on_load."""

from __future__ import annotations

from pathlib import Path

import pytest
from nanozero_training.eval.sprt_result import SPRTResult, SPRTStatus
from nanozero_training.version.manager import VersionManager
from nanozero_training.version.promotion import (
    PromoteOutcome,
    _derive_promoted_name,
    promote_if_h1,
    reconcile_on_load,
)
from pytest_mock import MockerFixture


def _make_setup(tmp_path: Path) -> VersionManager:
    """Setup a VersionManager with gen-001-init + gen-002-trained + matching .npz files."""
    vm = VersionManager(
        versions_yaml_path=tmp_path / "versions.yaml",
        models_dir=tmp_path / "models",
    )
    vm.create_new()
    vm.add_init("gen-001-init", seed=42)
    vm.add_trained("gen-002-trained", parent="gen-001-init")
    vm.save()
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "models" / "gen-001-init.npz").write_bytes(b"dummy-init")
    (tmp_path / "models" / "gen-002-trained.npz").write_bytes(b"dummy-trained")
    return vm


def _make_sprt(status: SPRTStatus, games: int = 142) -> SPRTResult:
    return SPRTResult(
        status=status,
        llr=2.95 if status == SPRTStatus.H1_ACCEPTED else -2.95,
        games_played=games,
        wins=70,
        losses=52,
        draws=20,
        elo_diff=11.3 if status == SPRTStatus.H1_ACCEPTED else -11.3,
    )


def test_promote_h1_renames_and_updates_yaml(tmp_path: Path) -> None:
    vm = _make_setup(tmp_path)
    sprt = _make_sprt(SPRTStatus.H1_ACCEPTED)
    result = promote_if_h1(vm, "gen-002-trained", sprt)

    assert result.outcome == PromoteOutcome.PROMOTED
    assert result.promoted_name == "gen-002-promoted"
    # File renamed
    assert (tmp_path / "models" / "gen-002-promoted.npz").exists()
    assert not (tmp_path / "models" / "gen-002-trained.npz").exists()
    # YAML current points to promoted
    assert vm.get_current() == "gen-002-promoted"
    # New entry appended
    promoted_entry = vm.versions.find("gen-002-promoted")
    assert promoted_entry is not None
    assert promoted_entry.type == "promoted"
    assert promoted_entry.parent == "gen-002-trained"
    assert promoted_entry.sprt_result == "h1_accepted"
    assert promoted_entry.sprt_games == 142
    # Trained entry updated
    trained_entry = vm.versions.find("gen-002-trained")
    assert trained_entry is not None
    assert trained_entry.promoted is True
    assert trained_entry.sprt_result == "h1_accepted"


def test_promote_h0_records_rejection_no_rename(tmp_path: Path) -> None:
    vm = _make_setup(tmp_path)
    sprt = _make_sprt(SPRTStatus.H0_ACCEPTED)
    result = promote_if_h1(vm, "gen-002-trained", sprt)

    assert result.outcome == PromoteOutcome.REJECTED
    assert result.promoted_name is None
    # Files unchanged
    assert (tmp_path / "models" / "gen-002-trained.npz").exists()
    assert not (tmp_path / "models" / "gen-002-promoted.npz").exists()
    # YAML records SPRT metadata on trained entry
    trained_entry = vm.versions.find("gen-002-trained")
    assert trained_entry is not None
    assert trained_entry.promoted is False
    assert trained_entry.sprt_result == "h0_accepted"
    # Current unchanged (still gen-001-init)
    assert vm.get_current() == "gen-001-init"


def test_promote_max_games_rejects(tmp_path: Path) -> None:
    vm = _make_setup(tmp_path)
    sprt = _make_sprt(SPRTStatus.MAX_GAMES_REACHED)
    result = promote_if_h1(vm, "gen-002-trained", sprt)

    assert result.outcome == PromoteOutcome.REJECTED
    trained_entry = vm.versions.find("gen-002-trained")
    assert trained_entry is not None
    assert trained_entry.sprt_result == "max_games"
    assert trained_entry.promoted is False


def test_promote_already_promoted_idempotent(tmp_path: Path) -> None:
    vm = _make_setup(tmp_path)
    sprt = _make_sprt(SPRTStatus.H1_ACCEPTED)
    # First promote: PROMOTED
    result1 = promote_if_h1(vm, "gen-002-trained", sprt)
    assert result1.outcome == PromoteOutcome.PROMOTED
    # Second promote: no-op ALREADY_PROMOTED
    result2 = promote_if_h1(vm, "gen-002-trained", sprt)
    assert result2.outcome == PromoteOutcome.ALREADY_PROMOTED
    assert result2.promoted_name == "gen-002-promoted"


def test_promote_unknown_model_raises(tmp_path: Path) -> None:
    vm = _make_setup(tmp_path)
    sprt = _make_sprt(SPRTStatus.H1_ACCEPTED)
    with pytest.raises(ValueError, match="not registered"):
        promote_if_h1(vm, "gen-nonexistent-trained", sprt)


def test_promote_wrong_type_raises(tmp_path: Path) -> None:
    vm = _make_setup(tmp_path)
    sprt = _make_sprt(SPRTStatus.H1_ACCEPTED)
    with pytest.raises(ValueError, match="Cannot promote model of type 'init'"):
        promote_if_h1(vm, "gen-001-init", sprt)


def test_promote_missing_trained_file_raises(tmp_path: Path) -> None:
    vm = _make_setup(tmp_path)
    # Remove the trained file to simulate filesystem corruption.
    (tmp_path / "models" / "gen-002-trained.npz").unlink()
    sprt = _make_sprt(SPRTStatus.H1_ACCEPTED)
    with pytest.raises(RuntimeError, match="trained model file missing"):
        promote_if_h1(vm, "gen-002-trained", sprt)


def test_promote_rename_failure_reverts_yaml(tmp_path: Path, mocker: MockerFixture) -> None:
    vm = _make_setup(tmp_path)
    sprt = _make_sprt(SPRTStatus.H1_ACCEPTED)
    # Snapshot pre-state for assertion.
    pre_all_count = len(vm.versions.all)
    pre_current = vm.versions.current

    # Mock atomic_rename to fail.
    mocker.patch(
        "nanozero_training.version.promotion.atomic_rename",
        side_effect=OSError("simulated rename failure"),
    )

    with pytest.raises(RuntimeError, match="Promote rename failed"):
        promote_if_h1(vm, "gen-002-trained", sprt)

    # YAML reverted: same count, same current.
    assert len(vm.versions.all) == pre_all_count
    assert vm.versions.current == pre_current
    # Re-read from disk to confirm save() was called with revert state.
    vm2 = VersionManager(
        versions_yaml_path=tmp_path / "versions.yaml",
        models_dir=tmp_path / "models",
    )
    vm2.load()
    assert vm2.versions.current == pre_current
    assert len(vm2.versions.all) == pre_all_count


def test_reconcile_on_load_completes_rename(tmp_path: Path) -> None:
    """YAML says current=gen-002-promoted but only gen-002-trained.npz on disk."""
    import yaml

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "gen-002-trained.npz").write_bytes(b"dummy-trained")

    versions_path = tmp_path / "versions.yaml"
    with versions_path.open("w") as f:
        yaml.safe_dump(
            {
                "current": "gen-002-promoted",
                "latest_trained": "gen-002-trained",
                "all": [
                    {"name": "gen-001-init", "type": "init", "promoted": True},
                    {
                        "name": "gen-002-trained",
                        "type": "trained",
                        "promoted": True,
                    },
                    {
                        "name": "gen-002-promoted",
                        "type": "promoted",
                        "promoted": True,
                    },
                ],
            },
            f,
        )

    vm = VersionManager(versions_yaml_path=versions_path, models_dir=models_dir)
    # NOTE: load() en commit d ajoutera auto_reconcile=True ; ici on appelle
    # reconcile_on_load explicitement pour tester ce niveau de l'API.
    # Bypass load() pour ne pas dépendre du commit d.
    with versions_path.open() as f:
        data = yaml.safe_load(f)
    from nanozero_training.version.manager import Versions

    vm._versions = Versions.from_dict(data)

    reconciled = reconcile_on_load(vm)
    assert reconciled is True
    assert (models_dir / "gen-002-promoted.npz").exists()
    assert not (models_dir / "gen-002-trained.npz").exists()


def test_reconcile_on_load_no_op_if_consistent(tmp_path: Path) -> None:
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "gen-001-init.npz").write_bytes(b"dummy")

    vm = VersionManager(versions_yaml_path=tmp_path / "versions.yaml", models_dir=models_dir)
    vm.create_new()
    vm.add_init("gen-001-init", seed=42)

    reconciled = reconcile_on_load(vm)
    assert reconciled is False


def test_reconcile_on_load_no_op_if_no_current(tmp_path: Path) -> None:
    vm = VersionManager(
        versions_yaml_path=tmp_path / "versions.yaml", models_dir=tmp_path / "models"
    )
    vm.create_new()
    assert reconcile_on_load(vm) is False


def test_reconcile_on_load_inconsistent_state_raises(tmp_path: Path) -> None:
    """current set but no matching file (neither promoted nor trained) -> raise."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    vm = VersionManager(versions_yaml_path=tmp_path / "versions.yaml", models_dir=models_dir)
    vm.create_new()
    vm.versions.current = "gen-002-promoted"
    # reconcile_on_load checks filesystem only — no need for a matching all[] entry.

    with pytest.raises(RuntimeError, match="Inconsistent state"):
        reconcile_on_load(vm)


def test_derive_promoted_name_valid() -> None:
    assert _derive_promoted_name("gen-002-trained") == "gen-002-promoted"
    assert _derive_promoted_name("gen-042-trained") == "gen-042-promoted"


def test_derive_promoted_name_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Expected -trained suffix"):
        _derive_promoted_name("gen-002-init")
    with pytest.raises(ValueError, match="Expected -trained suffix"):
        _derive_promoted_name("gen-002-promoted")


def test_promote_audit_note_overrides_sprt_result(tmp_path: Path) -> None:
    """Phase 9 : audit_note overrides persisted sprt_result on both entries."""
    vm = _make_setup(tmp_path)
    sprt = _make_sprt(SPRTStatus.H1_ACCEPTED)
    result = promote_if_h1(vm, "gen-002-trained", sprt, audit_note="manual_override")
    assert result.outcome == PromoteOutcome.PROMOTED
    # Trained entry record manual_override
    trained = vm.versions.find("gen-002-trained")
    assert trained is not None
    assert trained.sprt_result == "manual_override"
    # New promoted entry too
    promoted = vm.versions.find("gen-002-promoted")
    assert promoted is not None
    assert promoted.sprt_result == "manual_override"
