"""Test resume scenario 6 SPEC §11.5: mid-promote crash recovery.

Scenario 6: crash entre write YAML (current=gen-N-promoted) et atomic_rename.
Recovery on next load: detect mismatch + complete rename automatiquement via
VersionManager.load(auto_reconcile=True) (default).

Sub-cases tested:
  6a: YAML written, rename NOT done -> reconcile completes rename on load
  6b: idempotence — rerun promote_if_h1 after partial crash -> ALREADY_PROMOTED
  6c: inconsistent state — YAML claims current but neither -promoted nor
      -trained .npz on disk -> RuntimeError "Inconsistent state"
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from nanozero_training.eval.sprt_result import SPRTResult, SPRTStatus
from nanozero_training.version.manager import VersionManager
from nanozero_training.version.promotion import PromoteOutcome, promote_if_h1


def _make_sprt_h1() -> SPRTResult:
    return SPRTResult(
        status=SPRTStatus.H1_ACCEPTED,
        llr=2.95,
        games_played=142,
        wins=70,
        losses=52,
        draws=20,
        elo_diff=11.3,
    )


def test_scenario_6a_yaml_written_rename_lost_then_reconcile(tmp_path: Path) -> None:
    """Simulate post-crash state : YAML says gen-002-promoted, only -trained on disk."""
    versions_path = tmp_path / "versions.yaml"
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    yaml_content = {
        "current": "gen-002-promoted",
        "latest_trained": "gen-002-trained",
        "all": [
            {
                "name": "gen-001-init",
                "type": "init",
                "promoted": True,
                "seed": 42,
            },
            {
                "name": "gen-002-trained",
                "type": "trained",
                "promoted": True,
                "parent": "gen-001-init",
                "sprt_result": "h1_accepted",
                "sprt_games": 142,
            },
            {
                "name": "gen-002-promoted",
                "type": "promoted",
                "promoted": True,
                "parent": "gen-002-trained",
                "sprt_result": "h1_accepted",
                "sprt_games": 142,
            },
        ],
    }
    with versions_path.open("w") as f:
        yaml.safe_dump(yaml_content, f)

    (models_dir / "gen-001-init.npz").write_bytes(b"dummy")
    (models_dir / "gen-002-trained.npz").write_bytes(b"dummy-trained")

    vm = VersionManager(versions_yaml_path=versions_path, models_dir=models_dir)
    vm.load()  # auto_reconcile=True (default)

    # Après load : rename completed, current cohérent.
    assert (models_dir / "gen-002-promoted.npz").exists()
    assert not (models_dir / "gen-002-trained.npz").exists()
    assert vm.get_current() == "gen-002-promoted"


def test_scenario_6b_idempotence_rerun_promote(tmp_path: Path) -> None:
    """Promote fully successful, then rerun -- ALREADY_PROMOTED no-op."""
    versions_path = tmp_path / "versions.yaml"
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    vm = VersionManager(versions_yaml_path=versions_path, models_dir=models_dir)
    vm.create_new()
    vm.add_init("gen-001-init", seed=42)
    vm.add_trained("gen-002-trained", parent="gen-001-init")
    vm.save()
    (models_dir / "gen-001-init.npz").write_bytes(b"dummy")
    (models_dir / "gen-002-trained.npz").write_bytes(b"dummy-trained")

    sprt = _make_sprt_h1()

    # First promote
    result1 = promote_if_h1(vm, "gen-002-trained", sprt)
    assert result1.outcome == PromoteOutcome.PROMOTED
    assert (models_dir / "gen-002-promoted.npz").exists()

    # Second promote attempt -> ALREADY_PROMOTED
    result2 = promote_if_h1(vm, "gen-002-trained", sprt)
    assert result2.outcome == PromoteOutcome.ALREADY_PROMOTED
    assert result2.promoted_name == "gen-002-promoted"


def test_scenario_6c_inconsistent_state_raises_on_load(tmp_path: Path) -> None:
    """YAML claims current=gen-002-promoted but NEITHER promoted nor trained .npz on disk."""
    versions_path = tmp_path / "versions.yaml"
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    yaml_content = {
        "current": "gen-002-promoted",
        "all": [
            {"name": "gen-002-promoted", "type": "promoted", "promoted": True},
        ],
    }
    with versions_path.open("w") as f:
        yaml.safe_dump(yaml_content, f)
    # Aucun .npz sur disque.

    vm = VersionManager(versions_yaml_path=versions_path, models_dir=models_dir)
    with pytest.raises(RuntimeError, match="Inconsistent state"):
        vm.load()
