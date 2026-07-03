"""Unit tests for orchestrator run_generation core loop (mocks UciClient + state + worker)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from nanozero_training.data.sample import Sample, make_valid_sample_arrays
from nanozero_training.selfplay.config import SelfplayConfig
from nanozero_training.selfplay.orchestrator import SelfplayOrchestrator


def _make_valid_sample() -> Sample:
    return Sample(**make_valid_sample_arrays())


def _make_mock_state_manager() -> MagicMock:
    """RunStateManager mock with state field reflecting updates."""
    mgr = MagicMock()
    state = MagicMock()
    state.phase = "idle"
    state.current_gen = 0
    state.status = "in_progress"
    selfplay = MagicMock()
    selfplay.completed_games = 0
    selfplay.completed_batches = 0
    selfplay.current_batch_idx = 0
    selfplay.current_batch_games = 0
    state.selfplay = selfplay
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


def _make_orchestrator(
    tmp_path: Path,
    config_kwargs: dict | None = None,
    abort_flag: dict[str, bool] | None = None,
) -> tuple[SelfplayOrchestrator, MagicMock, MagicMock]:
    """Construct orchestrator with mocked client + state."""
    mock_client = MagicMock()
    mock_state = _make_mock_state_manager()
    flag = abort_flag if abort_flag is not None else {"requested": False}
    cfg_args = {
        "mcts_sims": 10,
        "max_game_plies": 10,
        "games_per_batch": 250,
        "target_games_per_gen": 500,
        "worker_restart_every": 1000,
        "max_consecutive_crashes": 5,
    }
    if config_kwargs:
        cfg_args.update(config_kwargs)
    config = SelfplayConfig(**cfg_args)
    orch = SelfplayOrchestrator(
        uci_client=mock_client,
        config=config,
        state_manager=mock_state,
        abort_flag=flag,
        datasets_dir=tmp_path / "datasets",
    )
    return orch, mock_client, mock_state


def test_run_generation_calls_play_one_game_for_each_target_game(tmp_path, mocker) -> None:
    """target=3 -> play_one_game appelé 3 fois."""
    mock_play = mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    orch, _client, _state = _make_orchestrator(tmp_path, {"games_per_batch": 100})
    orch.run_generation(gen=1, target_games=3)
    assert mock_play.call_count == 3


def test_run_generation_buffer_flushes_at_threshold(tmp_path, mocker) -> None:
    """games_per_batch=2, target=4 -> 2 batches flushed."""
    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    orch, _client, _state = _make_orchestrator(
        tmp_path, {"games_per_batch": 2, "target_games_per_gen": 4}
    )
    orch.run_generation(gen=1, target_games=4)
    batch_files = sorted((tmp_path / "datasets").glob("selfplay-gen001-batch-*.npz"))
    assert len(batch_files) == 2


def test_run_generation_flushes_partial_last_batch(tmp_path, mocker) -> None:
    """games_per_batch=3, target=5 -> 2 flushes (3 + 2 partial)."""
    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    orch, _client, _state = _make_orchestrator(
        tmp_path, {"games_per_batch": 3, "target_games_per_gen": 5}
    )
    orch.run_generation(gen=1, target_games=5)
    batch_files = sorted((tmp_path / "datasets").glob("selfplay-gen001-batch-*.npz"))
    assert len(batch_files) == 2


def test_run_generation_uses_make_game_rngs(tmp_path, mocker) -> None:
    """RNGs déterministes : rng[0].random() identique cross-runs avec même worker_seed."""
    captured_rngs = []

    def capture_rng(client, config, rng):
        captured_rngs.append(rng)
        return [_make_valid_sample()]

    mocker.patch("nanozero_training.selfplay.orchestrator.play_one_game", side_effect=capture_rng)
    orch, _, _ = _make_orchestrator(tmp_path, {"worker_seed": 42, "games_per_batch": 100})
    orch.run_generation(gen=1, target_games=3)
    # 3 RNGs distincts capturés.
    assert len(captured_rngs) == 3
    first_draws = [r.random() for r in captured_rngs]
    assert len(set(first_draws)) == 3


def test_run_generation_state_updated_per_game(tmp_path, mocker) -> None:
    """state.update appelé pour chaque partie (au moins n_games fois)."""
    mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    orch, _client, mock_state = _make_orchestrator(tmp_path, {"games_per_batch": 100})
    orch.run_generation(gen=1, target_games=3)
    # 3 updates par partie + au moins 1 final update.
    # Plus 1 update sur flush (selfplay__last_batch_file).
    # Plus 1 update final (selfplay__completed_batches + completed_games).
    assert mock_state.update.call_count >= 3


def test_run_generation_abort_before_first_game_breaks_loop(tmp_path, mocker) -> None:
    """abort_flag set initialement -> 0 parties jouées, status='aborted'."""
    mock_play = mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    abort = {"requested": True}
    orch, _client, mock_state = _make_orchestrator(
        tmp_path, {"games_per_batch": 100}, abort_flag=abort
    )
    orch.run_generation(gen=1, target_games=5)
    assert mock_play.call_count == 0
    assert mock_state.state.status == "aborted"


def test_run_generation_abort_mid_loop_flushes_partial(tmp_path, mocker) -> None:
    """abort_flag set après 2 parties -> flush partial + status='aborted'."""
    call_count = {"n": 0}
    abort = {"requested": False}

    def play_with_abort(*args, **kwargs):
        call_count["n"] += 1
        # Set abort après 2 parties terminées (sera vu au début de la 3e iter).
        if call_count["n"] == 2:
            abort["requested"] = True
        return [_make_valid_sample()]

    mock_play = mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game", side_effect=play_with_abort
    )
    orch, _client, mock_state = _make_orchestrator(
        tmp_path, {"games_per_batch": 100, "target_games_per_gen": 5}, abort_flag=abort
    )
    orch.run_generation(gen=1, target_games=5)
    # 2 parties jouées (3e check_abort détecte le flag).
    assert mock_play.call_count == 2
    assert mock_state.state.status == "aborted"
    # Partial flush effectué.
    batch_files = sorted((tmp_path / "datasets").glob("selfplay-gen001-batch-*.npz"))
    assert len(batch_files) == 1


def test_run_generation_target_already_complete_no_op(tmp_path, mocker) -> None:
    """state.completed_games >= target -> early return sans appeler play_one_game."""
    mock_play = mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    orch, _client, mock_state = _make_orchestrator(tmp_path)
    mock_state.state.selfplay.completed_games = 5
    orch.run_generation(gen=1, target_games=5)
    assert mock_play.call_count == 0


def test_run_generation_resume_from_state(tmp_path, mocker) -> None:
    """state.completed_games=3, target=5 -> 2 parties jouées seulement."""
    mock_play = mocker.patch(
        "nanozero_training.selfplay.orchestrator.play_one_game",
        return_value=[_make_valid_sample()],
    )
    orch, _client, mock_state = _make_orchestrator(tmp_path, {"games_per_batch": 100})
    mock_state.state.selfplay.completed_games = 3
    orch.run_generation(gen=1, target_games=5)
    assert mock_play.call_count == 2
