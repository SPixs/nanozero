"""Tests for state/manager.py — RunState + RunStateManager (ADR-009)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from nanozero_training.state.manager import (
    EvalState,
    RunState,
    RunStateManager,
    SelfplayState,
    TrainState,
)


def _make_mgr(tmp_path: Path) -> RunStateManager:
    return RunStateManager(tmp_path / "monitoring")


def _create_default_run(mgr: RunStateManager, run_id: str = "20260513-100000") -> RunState:
    return mgr.create_new(
        run_id=run_id,
        config_path="configs/default.yaml",
        config_hash="abc123",
        max_generations=5,
        target_games_per_gen=500,
    )


def test_create_new_writes_yaml(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    state = _create_default_run(mgr)
    assert mgr.path.exists()
    assert state.run_id == "20260513-100000"
    assert state.status == "in_progress"
    with mgr.path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded["run_id"] == "20260513-100000"
    assert loaded["status"] == "in_progress"
    assert loaded["current_gen"] == 0


def test_create_new_refuses_if_run_in_progress(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    _create_default_run(mgr)
    with pytest.raises(RuntimeError, match="already in progress"):
        _create_default_run(mgr)


def test_detect_existing_run_true_when_in_progress(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    _create_default_run(mgr)
    mgr2 = _make_mgr(tmp_path)  # Fresh manager on same path
    assert mgr2.detect_existing_run() is True


def test_detect_existing_run_false_when_no_file(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    assert mgr.detect_existing_run() is False


def test_detect_existing_run_false_when_status_completed(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    _create_default_run(mgr)
    mgr.update(status="completed")
    mgr2 = _make_mgr(tmp_path)
    assert mgr2.detect_existing_run() is False


def test_detect_existing_run_false_on_corrupt_yaml(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    mgr.monitoring_dir.mkdir(parents=True, exist_ok=True)
    mgr.path.write_text("not: valid: yaml: : :", encoding="utf-8")
    assert mgr.detect_existing_run() is False


def test_load_existing_roundtrip(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    state_created = _create_default_run(mgr)
    mgr.update(train__current_epoch=5, train__base_model="gen-002-promoted.npz")

    mgr2 = _make_mgr(tmp_path)
    state_loaded = mgr2.load_existing()
    assert state_loaded.run_id == state_created.run_id
    assert state_loaded.train.current_epoch == 5
    assert state_loaded.train.base_model == "gen-002-promoted.npz"


def test_load_existing_raises_when_no_file(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    with pytest.raises(FileNotFoundError):
        mgr.load_existing()


def test_load_existing_raises_when_not_in_progress(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    _create_default_run(mgr)
    mgr.update(status="completed")
    mgr2 = _make_mgr(tmp_path)
    with pytest.raises(ValueError, match="not 'in_progress'"):
        mgr2.load_existing()


def test_update_simple_field(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    _create_default_run(mgr)
    mgr.update(phase="train")
    assert mgr.state.phase == "train"
    with mgr.path.open(encoding="utf-8") as f:
        assert yaml.safe_load(f)["phase"] == "train"


def test_update_nested_field(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    _create_default_run(mgr)
    mgr.update(train__current_epoch=7, selfplay__completed_games=350)
    assert mgr.state.train.current_epoch == 7
    assert mgr.state.selfplay.completed_games == 350


def test_state_property_raises_before_init(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    with pytest.raises(RuntimeError, match="not initialized"):
        _ = mgr.state


def test_update_raises_before_init(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    with pytest.raises(RuntimeError, match="not initialized"):
        mgr.update(phase="train")


def test_flush_raises_before_init(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    with pytest.raises(RuntimeError, match="not initialized"):
        mgr.flush()


def test_archive_moves_state_to_runs_dir(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    _create_default_run(mgr, run_id="20260513-120000")
    mgr.archive()
    assert not mgr.path.exists()
    archive_path = mgr.monitoring_dir / "runs" / "20260513-120000" / "run_state.yaml"
    assert archive_path.exists()
    flag_path = mgr.monitoring_dir / "runs" / "20260513-120000" / "completed.flag"
    assert flag_path.exists()


def test_archive_raises_before_init(tmp_path: Path) -> None:
    mgr = _make_mgr(tmp_path)
    with pytest.raises(RuntimeError, match="not initialized"):
        mgr.archive()


def test_flush_is_atomic_no_partial_yaml_on_failure(tmp_path: Path) -> None:
    """If atomic_write_yaml fails mid-flush, no run_state.yaml should exist."""
    mgr = _make_mgr(tmp_path)
    mgr.monitoring_dir.mkdir(parents=True, exist_ok=True)
    # Inject a state then mock yaml.safe_dump to crash
    _create_default_run(mgr)
    mgr.update(phase="selfplay")  # First flush succeeds
    assert mgr.path.exists()
    # Now mock crash during yaml.safe_dump and try another update
    with (
        patch(
            "nanozero_training.utils.atomic_io.yaml.safe_dump",
            side_effect=RuntimeError("yaml crash"),
        ),
        pytest.raises(RuntimeError, match="yaml crash"),
    ):
        mgr.update(phase="train")
    # Previous state still readable (last successful flush)
    with mgr.path.open(encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    assert loaded["phase"] == "selfplay"
    # No leftover tmp
    assert list(mgr.monitoring_dir.glob(".*.yaml.tmp")) == []


def test_state_dataclass_defaults() -> None:
    """SelfplayState / TrainState defaults are sensible."""
    sp = SelfplayState()
    assert sp.target_games == 0
    assert sp.last_batch_file is None
    tr = TrainState()
    assert tr.current_epoch == 0
    assert tr.base_model is None


def test_eval_state_last_decision_default_none() -> None:
    """Phase 1.0.0-7 : EvalState.last_decision defaults to None."""
    ev = EvalState()
    assert ev.last_decision is None
    assert ev.pgn_path is None
    assert ev.games_played_at_last_save == 0


def test_eval_state_last_decision_roundtrip(tmp_path: Path) -> None:
    """Phase 1.0.0-7 : last_decision survives flush + load."""
    mgr = _make_mgr(tmp_path)
    _create_default_run(mgr)
    mgr.update(eval__last_decision="h1_accepted", eval__games_played_at_last_save=142)

    mgr2 = _make_mgr(tmp_path)
    state_loaded = mgr2.load_existing()
    assert state_loaded.eval.last_decision == "h1_accepted"
    assert state_loaded.eval.games_played_at_last_save == 142


def test_load_old_yaml_without_last_decision_field(tmp_path: Path) -> None:
    """Phase 1.0.0-7 : backwards-compat — vieux YAML sans last_decision defaultise à None.

    Garantit qu'un run en cours pré-phase 7 reste resumable post-extension EvalState.
    """
    mgr = _make_mgr(tmp_path)
    mgr.monitoring_dir.mkdir(parents=True, exist_ok=True)
    # YAML manuel sans le champ last_decision (simulant un run pré-phase 7).
    old_yaml = {
        "run_id": "20260512-100000",
        "started": "2026-05-12T10:00:00+00:00",
        "last_updated": "2026-05-12T10:00:00+00:00",
        "status": "in_progress",
        "config_path": "configs/default.yaml",
        "config_hash": "abc123",
        "max_generations": 5,
        "target_games_per_gen": 500,
        "current_gen": 2,
        "gens_completed": [1],
        "gens_rejected": [],
        "phase": "eval",
        "phase_started": None,
        "selfplay": {
            "target_games": 0,
            "completed_batches": 0,
            "completed_games": 0,
            "current_batch_idx": 0,
            "current_batch_games": 0,
            "last_batch_file": None,
            "worker_seed": 42,
        },
        "train": {
            "base_model": None,
            "output_model_target": None,
            "current_epoch": 0,
            "total_epochs": 0,
            "last_checkpoint": None,
            "optimizer_state_file": None,
        },
        "eval": {
            "challenger": "gen-002-trained.npz",
            "baseline": "gen-001-init.npz",
            "pgn_path": "monitoring/sprt-gen-002.pgn",
            "games_played_at_last_save": 87,
            # NO last_decision field
        },
        "last_error": None,
        "crash_count": 0,
        "last_resume": None,
    }
    with mgr.path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(old_yaml, f)

    state = mgr.load_existing()
    assert state.eval.last_decision is None  # default kicked in
    assert state.eval.challenger == "gen-002-trained.npz"
    assert state.eval.games_played_at_last_save == 87
