"""Unit tests for orchestrator _run_one_game_with_recovery."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from nanozero_training.data.sample import Sample, make_valid_sample_arrays
from nanozero_training.selfplay.config import SelfplayConfig
from nanozero_training.selfplay.orchestrator import SelfplayOrchestrator
from nanozero_training.selfplay.uci_client import UciCrashError, UciTimeoutError


def _make_valid_sample() -> Sample:
    return Sample(**make_valid_sample_arrays())


def _make_orchestrator(
    tmp_path: Path,
    max_consecutive_crashes: int = 5,
) -> tuple[SelfplayOrchestrator, MagicMock, MagicMock]:
    """Construct orchestrator with mocked client + state."""
    mock_client = MagicMock()
    mock_state = MagicMock()
    state = MagicMock()
    state.phase = "idle"
    state.status = "in_progress"
    selfplay = MagicMock()
    selfplay.completed_games = 0
    selfplay.completed_batches = 0
    selfplay.current_batch_idx = 0
    selfplay.current_batch_games = 0
    state.selfplay = selfplay
    mock_state.state = state
    mock_state.update = MagicMock()
    abort_flag = {"requested": False}
    config = SelfplayConfig(mcts_sims=10, max_consecutive_crashes=max_consecutive_crashes)
    orch = SelfplayOrchestrator(
        uci_client=mock_client,
        config=config,
        state_manager=mock_state,
        abort_flag=abort_flag,
        datasets_dir=tmp_path / "datasets",
    )
    return orch, mock_client, mock_state


def test_recovery_succeeds_first_attempt(tmp_path, mocker) -> None:
    """play_one_game succeeds first call -> samples returned, no restart."""
    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    orch, client, _state = _make_orchestrator(tmp_path)
    rng = np.random.default_rng(42)
    samples = orch._run_one_game_with_recovery(rng, game_idx=0)
    assert samples is not None
    assert len(samples) == 1
    client.restart.assert_not_called()
    assert orch._consecutive_crashes == 0


def test_recovery_crash_then_succeeds(tmp_path, mocker) -> None:
    """1st call raises UciCrashError, 2nd succeeds -> restart + samples returned."""
    side_effect = [UciCrashError("boom"), [_make_valid_sample()]]
    mocker.patch("nanozero_training.selfplay.orchestrator.play_one_game", side_effect=side_effect)
    orch, client, _state = _make_orchestrator(tmp_path)
    rng = np.random.default_rng(42)
    samples = orch._run_one_game_with_recovery(rng, game_idx=0)
    assert samples is not None
    assert len(samples) == 1
    client.restart.assert_called_once()
    # Counter reset after success.
    assert orch._consecutive_crashes == 0


def test_recovery_timeout_treated_as_crash(tmp_path, mocker) -> None:
    """UciTimeoutError treated identically to UciCrashError."""
    side_effect = [UciTimeoutError("slow"), [_make_valid_sample()]]
    mocker.patch("nanozero_training.selfplay.orchestrator.play_one_game", side_effect=side_effect)
    orch, client, _state = _make_orchestrator(tmp_path)
    rng = np.random.default_rng(42)
    samples = orch._run_one_game_with_recovery(rng, game_idx=0)
    assert samples is not None
    client.restart.assert_called_once()


def test_recovery_max_crashes_aborts(tmp_path, mocker) -> None:
    """5 crashes consécutifs -> RuntimeError + state.status='error'."""
    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        side_effect=UciCrashError("boom"),
    )
    orch, client, mock_state = _make_orchestrator(tmp_path, max_consecutive_crashes=3)
    rng = np.random.default_rng(42)
    with pytest.raises(RuntimeError, match="UCI crashed"):
        orch._run_one_game_with_recovery(rng, game_idx=0)
    # State updated with status=error.
    error_updates = [
        c for c in mock_state.update.call_args_list if c.kwargs.get("status") == "error"
    ]
    assert len(error_updates) >= 1


def test_recovery_consecutive_reset_after_success(tmp_path, mocker) -> None:
    """2 crashes -> success -> counter reset, 3 nouveaux crashes ne raise pas."""
    # 1ère partie : 2 crashes puis success.
    seq = [
        UciCrashError("boom1"),
        UciCrashError("boom2"),
        [_make_valid_sample()],
    ]
    mocker.patch("nanozero_training.selfplay.orchestrator.play_one_game", side_effect=seq)
    orch, _client, _state = _make_orchestrator(tmp_path, max_consecutive_crashes=5)
    rng = np.random.default_rng(42)
    samples = orch._run_one_game_with_recovery(rng, game_idx=0)
    assert samples is not None
    # Counter reset après success.
    assert orch._consecutive_crashes == 0


def test_recovery_restart_failure_propagates(tmp_path, mocker) -> None:
    """client.restart() raise -> state.status='error' + raise."""
    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        side_effect=UciCrashError("boom"),
    )
    orch, client, mock_state = _make_orchestrator(tmp_path)
    client.restart.side_effect = RuntimeError("subprocess re-spawn failed")
    rng = np.random.default_rng(42)
    with pytest.raises(RuntimeError, match="subprocess re-spawn failed"):
        orch._run_one_game_with_recovery(rng, game_idx=0)
    # state.status='error' enregistré.
    error_updates = [
        c for c in mock_state.update.call_args_list if c.kwargs.get("status") == "error"
    ]
    assert len(error_updates) >= 1
