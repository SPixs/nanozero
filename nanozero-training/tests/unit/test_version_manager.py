"""Unit tests for version/manager — VersionManager + ModelEntry + Versions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import yaml
from nanozero_training.version.manager import (
    ModelEntry,
    VersionManager,
    Versions,
)


def _make_vm(tmp_path: Path) -> VersionManager:
    return VersionManager(
        versions_yaml_path=tmp_path / "versions.yaml",
        models_dir=tmp_path / "models",
    )


def test_create_new_empty_versions(tmp_path: Path) -> None:
    vm = _make_vm(tmp_path)
    vm.create_new()
    assert vm.versions.current is None
    assert vm.versions.latest_trained is None
    assert vm.versions.all == []
    vm.save()
    assert vm.path.exists()
    with vm.path.open() as f:
        loaded = yaml.safe_load(f)
    assert loaded == {"all": []}


def test_load_missing_file_raises(tmp_path: Path) -> None:
    vm = _make_vm(tmp_path)
    with pytest.raises(FileNotFoundError, match="versions.yaml not found"):
        vm.load()


def test_load_malformed_yaml_raises(tmp_path: Path) -> None:
    vm = _make_vm(tmp_path)
    vm.path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed versions.yaml"):
        vm.load()


def test_save_load_roundtrip(tmp_path: Path) -> None:
    vm1 = _make_vm(tmp_path)
    vm1.create_new()
    vm1.add_init("gen-001-init", seed=42)
    vm1.add_trained("gen-002-trained", parent="gen-001-init")
    vm1.save()
    # Auto-reconcile (default true in load) inspects filesystem ; create dummy
    # .npz to match the in-memory state.
    models_dir = tmp_path / "models"
    models_dir.mkdir(exist_ok=True)
    (models_dir / "gen-001-init.npz").write_bytes(b"dummy")
    (models_dir / "gen-002-trained.npz").write_bytes(b"dummy")

    vm2 = _make_vm(tmp_path)
    vm2.load()
    assert len(vm2.versions.all) == 2
    assert vm2.versions.current == "gen-001-init"
    assert vm2.versions.latest_trained == "gen-002-trained"
    init = vm2.versions.find("gen-001-init")
    assert init is not None
    assert init.seed == 42
    trained = vm2.versions.find("gen-002-trained")
    assert trained is not None
    assert trained.parent == "gen-001-init"
    assert trained.promoted is False


def test_add_init_sets_current_and_latest_trained(tmp_path: Path) -> None:
    vm = _make_vm(tmp_path)
    vm.create_new()
    entry = vm.add_init("gen-001-init", seed=42, init_method="kaiming_standard")
    assert entry.type == "init"
    assert entry.promoted is True
    assert vm.versions.current == "gen-001-init"
    assert vm.versions.latest_trained == "gen-001-init"


def test_add_init_duplicate_name_raises(tmp_path: Path) -> None:
    vm = _make_vm(tmp_path)
    vm.create_new()
    vm.add_init("gen-001-init", seed=42)
    with pytest.raises(ValueError, match="already registered"):
        vm.add_init("gen-001-init", seed=43)


def test_add_trained_updates_latest_trained_only(tmp_path: Path) -> None:
    vm = _make_vm(tmp_path)
    vm.create_new()
    vm.add_init("gen-001-init", seed=42)
    vm.add_trained("gen-002-trained", parent="gen-001-init")
    assert vm.versions.current == "gen-001-init"  # unchanged
    assert vm.versions.latest_trained == "gen-002-trained"


def test_add_trained_unknown_parent_raises(tmp_path: Path) -> None:
    vm = _make_vm(tmp_path)
    vm.create_new()
    with pytest.raises(ValueError, match="Parent model not found"):
        vm.add_trained("gen-002-trained", parent="gen-nonexistent")


def test_add_trained_replaces_non_promoted_slot(tmp_path: Path) -> None:
    """Re-train d'un slot REJETÉ/non-évalué : remplace l'entrée, ne lève pas."""
    vm = _make_vm(tmp_path)
    vm.create_new()
    vm.add_init("gen-001-init", seed=42)
    vm.add_trained("gen-002-trained", parent="gen-001-init")
    # Simule un rejet SPRT enregistré sur l'entrée (promoted reste False).
    idx = next(i for i, e in enumerate(vm.versions.all) if e.name == "gen-002-trained")
    vm.versions.all[idx] = replace(vm.versions.all[idx], sprt_result="h0_accepted", sprt_games=400)
    # Ré-entraînement du même slot : doit remplacer, pas lever.
    vm.add_trained("gen-002-trained", parent="gen-001-init")
    matches = [e for e in vm.versions.all if e.name == "gen-002-trained"]
    assert len(matches) == 1  # pas de doublon
    assert matches[0].promoted is False
    assert matches[0].sprt_result is None  # entrée fraîche (rejet précédent effacé)
    assert vm.versions.latest_trained == "gen-002-trained"


def test_add_trained_raises_on_promoted_slot(tmp_path: Path) -> None:
    """Un slot déjà PROMU ne peut jamais être écrasé par un ré-entraînement."""
    vm = _make_vm(tmp_path)
    vm.create_new()
    vm.add_init("gen-001-init", seed=42)
    vm.add_trained("gen-002-trained", parent="gen-001-init")
    idx = next(i for i, e in enumerate(vm.versions.all) if e.name == "gen-002-trained")
    vm.versions.all[idx] = replace(vm.versions.all[idx], promoted=True)
    with pytest.raises(ValueError, match="already-promoted"):
        vm.add_trained("gen-002-trained", parent="gen-001-init")


def test_get_current_raises_if_empty(tmp_path: Path) -> None:
    vm = _make_vm(tmp_path)
    vm.create_new()
    with pytest.raises(RuntimeError, match="No current champion"):
        vm.get_current()


def test_get_path_for_returns_npz_path(tmp_path: Path) -> None:
    vm = _make_vm(tmp_path)
    p = vm.get_path_for("gen-002-promoted")
    assert p == tmp_path / "models" / "gen-002-promoted.npz"


def test_model_entry_to_dict_excludes_none_fields() -> None:
    entry = ModelEntry(name="gen-002-trained", type="trained", parent="gen-001-init")
    d = entry.to_dict()
    assert "name" in d
    assert "type" in d
    assert "parent" in d
    assert "seed" not in d
    assert "sprt_result" not in d
    assert "sprt_llr" not in d


def test_model_entry_from_dict_tolerant_to_missing_optional() -> None:
    minimal = {"name": "gen-001-init", "type": "init"}
    entry = ModelEntry.from_dict(minimal)
    assert entry.name == "gen-001-init"
    assert entry.promoted is False
    assert entry.parent is None
    assert entry.seed is None


def test_versions_find_returns_entry_or_none(tmp_path: Path) -> None:
    vm = _make_vm(tmp_path)
    vm.create_new()
    vm.add_init("gen-001-init", seed=42)
    assert vm.versions.find("gen-001-init") is not None
    assert vm.versions.find("nonexistent") is None


def test_versions_property_raises_before_load() -> None:
    vm = VersionManager(versions_yaml_path="/tmp/dummy.yaml", models_dir="/tmp/models")
    with pytest.raises(RuntimeError, match="not loaded"):
        _ = vm.versions


def test_versions_dict_excludes_none_top_level(tmp_path: Path) -> None:
    """Empty Versions serializes to dict with only 'all' (no current/latest_trained)."""
    v = Versions()
    d = v.to_dict()
    assert d == {"all": []}
    assert "current" not in d
    assert "latest_trained" not in d


def test_load_with_auto_reconcile_repairs_mid_promote_crash(tmp_path: Path) -> None:
    """Phase 8 commit d : load(auto_reconcile=True) completes mid-promote rename.

    Setup : YAML claims current=gen-002-promoted, only gen-002-trained.npz exists.
    Expected : after load(), gen-002-promoted.npz exists, gen-002-trained.npz absent,
    YAML re-saved (effectively same content but flushed post-reconcile).
    """
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "gen-002-trained.npz").write_bytes(b"dummy")

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
    vm.load()  # auto_reconcile=True (default)

    assert (models_dir / "gen-002-promoted.npz").exists()
    assert not (models_dir / "gen-002-trained.npz").exists()
    assert vm.get_current() == "gen-002-promoted"


def test_load_without_auto_reconcile_does_not_touch_filesystem(tmp_path: Path) -> None:
    """auto_reconcile=False preserves mid-promote state for inspection."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    (models_dir / "gen-002-trained.npz").write_bytes(b"dummy")

    versions_path = tmp_path / "versions.yaml"
    with versions_path.open("w") as f:
        yaml.safe_dump(
            {
                "current": "gen-002-promoted",
                "all": [
                    {"name": "gen-002-promoted", "type": "promoted", "promoted": True},
                ],
            },
            f,
        )

    vm = VersionManager(versions_yaml_path=versions_path, models_dir=models_dir)
    vm.load(auto_reconcile=False)

    # Filesystem unchanged
    assert (models_dir / "gen-002-trained.npz").exists()
    assert not (models_dir / "gen-002-promoted.npz").exists()
    # But in-memory state loaded
    assert vm.versions.current == "gen-002-promoted"
