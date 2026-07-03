"""Unit tests for orchestrator RunStateManager integration."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from nanozero_training.data.sample import Sample, make_valid_sample_arrays
from nanozero_training.selfplay.config import SelfplayConfig
from nanozero_training.selfplay.orchestrator import SelfplayOrchestrator
from nanozero_training.selfplay.uci_client import UciCrashError


def _make_valid_sample() -> Sample:
    return Sample(**make_valid_sample_arrays())


def _make_mock_state(phase: str = "idle", current_gen: int = 0) -> MagicMock:
    mgr = MagicMock()
    state = MagicMock()
    state.phase = phase
    state.current_gen = current_gen
    state.status = "in_progress"
    selfplay = MagicMock()
    selfplay.completed_games = 0
    selfplay.completed_batches = 0
    selfplay.current_batch_idx = 0
    selfplay.current_batch_games = 0
    state.selfplay = selfplay
    mgr.state = state

    def fake_update(**kwargs):
        for k, v in kwargs.items():
            if "__" in k:
                section, field = k.split("__", 1)
                setattr(getattr(state, section), field, v)
            else:
                setattr(state, k, v)

    mgr.update.side_effect = fake_update
    return mgr


def _make_orch(tmp_path, state_mgr, **cfg_kwargs):
    config_args = {
        "games_per_batch": 100,
        "target_games_per_gen": 5,
        "max_consecutive_crashes": 5,
    }
    config_args.update(cfg_kwargs)
    return SelfplayOrchestrator(
        uci_client=MagicMock(),
        config=SelfplayConfig(**config_args),
        state_manager=state_mgr,
        abort_flag={"requested": False},
        datasets_dir=tmp_path / "datasets",
    )


def test_run_generation_sets_phase_selfplay(tmp_path, mocker) -> None:
    """phase=='idle' initial -> phase=='selfplay' après run_generation."""
    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    state_mgr = _make_mock_state(phase="idle")
    orch = _make_orch(tmp_path, state_mgr)
    orch.run_generation(gen=1, target_games=3)
    assert state_mgr.state.phase == "selfplay"


def test_run_generation_updates_current_gen(tmp_path, mocker) -> None:
    """state.current_gen=2, run_generation(gen=3) -> state.current_gen=3."""
    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    state_mgr = _make_mock_state(phase="idle", current_gen=2)
    orch = _make_orch(tmp_path, state_mgr)
    orch.run_generation(gen=3, target_games=2)
    assert state_mgr.state.current_gen == 3


def test_run_generation_updates_completed_games_per_game(tmp_path, mocker) -> None:
    """mock 5 games -> state.completed_games == 5 à la fin."""
    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    state_mgr = _make_mock_state()
    orch = _make_orch(tmp_path, state_mgr, target_games_per_gen=5)
    orch.run_generation(gen=1, target_games=5)
    assert state_mgr.state.selfplay.completed_games == 5


def test_run_generation_updates_last_batch_file_on_flush(tmp_path, mocker) -> None:
    """1 flush -> state.last_batch_file == 'selfplay-genXXX-batch-YYY.npz'."""
    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    state_mgr = _make_mock_state()
    orch = _make_orch(tmp_path, state_mgr, games_per_batch=3)
    orch.run_generation(gen=1, target_games=3)
    # Vérifier que last_batch_file a été mis à jour.
    assert state_mgr.state.selfplay.last_batch_file is not None
    assert "selfplay-gen001-batch" in state_mgr.state.selfplay.last_batch_file


def test_run_generation_updates_completed_batches_on_flush(tmp_path, mocker) -> None:
    """2 flushes -> state.completed_batches == 2."""
    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    state_mgr = _make_mock_state()
    orch = _make_orch(tmp_path, state_mgr, games_per_batch=2, target_games_per_gen=4)
    orch.run_generation(gen=1, target_games=4)
    assert state_mgr.state.selfplay.completed_batches == 2


def test_run_generation_marks_status_aborted_on_signal(tmp_path, mocker) -> None:
    """abort_flag set -> state.status=='aborted' à la fin."""
    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    state_mgr = _make_mock_state()
    abort = {"requested": True}
    orch = SelfplayOrchestrator(
        uci_client=MagicMock(),
        config=SelfplayConfig(games_per_batch=100, target_games_per_gen=5),
        state_manager=state_mgr,
        abort_flag=abort,
        datasets_dir=tmp_path / "datasets",
    )
    orch.run_generation(gen=1, target_games=5)
    assert state_mgr.state.status == "aborted"


def test_run_generation_marks_status_error_on_max_crashes(tmp_path, mocker) -> None:
    """5 crashes consécutifs -> state.status='error' + last_error."""
    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        side_effect=UciCrashError("boom"),
    )
    state_mgr = _make_mock_state()
    orch = _make_orch(tmp_path, state_mgr, max_consecutive_crashes=3)
    with pytest.raises(RuntimeError, match="UCI crashed"):
        orch.run_generation(gen=1, target_games=5)
    assert state_mgr.state.status == "error"
    assert state_mgr.state.last_error is not None


def test_run_generation_resume_does_not_reset_phase(tmp_path, mocker) -> None:
    """state.phase déjà 'selfplay' (resume) -> n'écrase pas phase_started."""
    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    state_mgr = _make_mock_state(phase="selfplay")
    orch = _make_orch(tmp_path, state_mgr)
    initial_update_count = state_mgr.update.call_count
    orch.run_generation(gen=1, target_games=2)
    # Pas d'update phase au démarrage (déjà selfplay).
    phase_updates = [
        c
        for c in state_mgr.update.call_args_list[initial_update_count:]
        if "phase" in c.kwargs and c.kwargs.get("phase") == "selfplay"
    ]
    assert len(phase_updates) == 0
