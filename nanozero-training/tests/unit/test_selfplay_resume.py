"""Resume scenarios SPEC §11.5 pour self-play.

Scenario 1 (abort mid-batch) : abort signal arrive mid-batch, partial batch
flushé, status='aborted'. Au resume, on continue depuis state.completed_games.

Scenario 2 (entre batches) : state pre-filled comme si N batches déjà faits,
resume continue à completed_games. Nouveau batch flushé.

Ces 2 scénarios sont les canoniques que phase 1.0.0-11 (pipeline assembly)
exploitera.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from nanozero_training.data.sample import Sample, make_valid_sample_arrays
from nanozero_training.selfplay.config import SelfplayConfig
from nanozero_training.selfplay.orchestrator import SelfplayOrchestrator


def _make_valid_sample() -> Sample:
    return Sample(**make_valid_sample_arrays())


@pytest.fixture()
def mock_state_manager():
    """RunStateManager mock with state field réflectant updates."""
    mgr = MagicMock()
    state = MagicMock()
    state.phase = "idle"
    state.current_gen = 0
    state.status = "in_progress"
    state.last_error = None
    state.crash_count = 0
    selfplay_state = MagicMock()
    selfplay_state.completed_games = 0
    selfplay_state.completed_batches = 0
    selfplay_state.current_batch_idx = 0
    selfplay_state.current_batch_games = 0
    selfplay_state.last_batch_file = None
    state.selfplay = selfplay_state
    mgr.state = state

    updates_log = []

    def fake_update(**kwargs):
        updates_log.append(kwargs)
        for k, v in kwargs.items():
            if "__" in k:
                section, field = k.split("__", 1)
                setattr(getattr(state, section), field, v)
            else:
                setattr(state, k, v)

    mgr.update.side_effect = fake_update
    mgr._updates_log = updates_log
    return mgr


def test_resume_scenario_2_between_batches(mock_state_manager, tmp_path, mocker):
    """SPEC §11.5 scenario 2 : state pre-filled (2 batches faits), resume
    continue depuis completed_games. Nouveau batch flushé.

    Setup :
    - state.completed_games=20, completed_batches=2.
    - games_per_batch=10, target=30.

    Expected :
    - 10 parties jouées (game_idx 20..29).
    - 1 nouveau flush -> completed_batches=3, completed_games=30.
    """
    # Pre-fill state comme si 20 parties (2 batches x 10) déjà faites.
    mock_state_manager.state.selfplay.completed_games = 20
    mock_state_manager.state.selfplay.completed_batches = 2

    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )

    config = SelfplayConfig(
        games_per_batch=10,
        target_games_per_gen=30,
        worker_restart_every=1000,
        max_consecutive_crashes=5,
    )

    orch = SelfplayOrchestrator(
        uci_client=MagicMock(),
        config=config,
        state_manager=mock_state_manager,
        abort_flag={"requested": False},
        datasets_dir=tmp_path,
    )

    orch.run_generation(gen=1, target_games=30)

    # 10 parties jouées (de game_idx 20 à 29).
    from nanozero_training.selfplay.orchestrator import play_one_game  # mocked

    assert play_one_game.call_count == 10

    # State final.
    assert mock_state_manager.state.selfplay.completed_games == 30
    assert mock_state_manager.state.selfplay.completed_batches == 3

    # 1 nouveau batch file écrit (le 3e, batch_idx=2).
    batch_files = list(tmp_path.glob("selfplay-gen001-batch-*.npz"))
    assert len(batch_files) == 1
    # Naming = batch_idx=2 (after 2 batches déjà existants).
    assert "batch-002" in batch_files[0].name


def test_resume_scenario_1_mid_batch_abort(mock_state_manager, tmp_path, mocker):
    """SPEC §11.5 scenario 1 : abort mid-batch flush partial + status='aborted'.

    Setup :
    - state.completed_games=0, completed_batches=0.
    - games_per_batch=10, target=20.
    - Abort signal fired après 3 parties terminées (sera vu au début de la 4e iter).

    Expected :
    - 3 parties jouées avant abort détecté.
    - Partial batch flushé (3 games).
    - state.status='aborted'.
    """
    mock_state_manager.state.selfplay.completed_games = 0
    mock_state_manager.state.selfplay.completed_batches = 0

    abort_flag = {"requested": False}
    call_count = {"n": 0}

    def play_with_abort(*args, **kwargs):
        call_count["n"] += 1
        # Set abort flag après 3 parties terminées.
        if call_count["n"] == 3:
            abort_flag["requested"] = True
        return [_make_valid_sample()]

    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        side_effect=play_with_abort,
    )

    config = SelfplayConfig(
        games_per_batch=10,
        target_games_per_gen=20,
        worker_restart_every=1000,
        max_consecutive_crashes=5,
    )

    orch = SelfplayOrchestrator(
        uci_client=MagicMock(),
        config=config,
        state_manager=mock_state_manager,
        abort_flag=abort_flag,
        datasets_dir=tmp_path,
    )

    orch.run_generation(gen=1, target_games=20)

    # 3 parties jouées (4e iter check abort détecte le flag).
    from nanozero_training.selfplay.orchestrator import play_one_game  # mocked

    assert play_one_game.call_count == 3
    # State status aborted.
    assert mock_state_manager.state.status == "aborted"
    # Partial flush effectué (3 games).
    batch_files = list(tmp_path.glob("selfplay-gen001-batch-*.npz"))
    assert len(batch_files) == 1
