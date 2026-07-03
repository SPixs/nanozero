"""Unit tests for selfplay/worker.py — play_one_game with mocked UciClient."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from nanozero_training.selfplay.config import SelfplayConfig
from nanozero_training.selfplay.uci_client import UciCrashError, UciResult, UciTimeoutError
from nanozero_training.selfplay.worker import play_one_game


def _rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_mock_client(visit_sequence: list[UciResult]) -> MagicMock:
    """MagicMock UciClient yielding visit_sequence on successive go_nodes calls."""
    client = MagicMock()
    client.go_nodes.side_effect = visit_sequence
    return client


def _foolsmate_results() -> list[UciResult]:
    """Build mock UciResult sequence reproducing Fool's mate (4 plies, black wins).

    Sequence: 1. f2f3 e7e5 2. g2g4 Qd8h4#
    Plies : 0 white f2f3, 1 black e7e5, 2 white g2g4, 3 black d8h4#.
    """
    return [
        UciResult(visits={"f2f3": 10, "e2e4": 5}, bestmove="f2f3"),
        UciResult(visits={"e7e5": 10, "d7d5": 5}, bestmove="e7e5"),
        UciResult(visits={"g2g4": 10, "d2d4": 5}, bestmove="g2g4"),
        UciResult(visits={"d8h4": 10}, bestmove="d8h4"),
    ]


# ----- core play_one_game tests -----


def test_play_one_game_calls_new_game_first() -> None:
    """new_game() est appelé AVANT le 1er go_nodes."""
    client = _make_mock_client(_foolsmate_results())
    cfg = SelfplayConfig(mcts_sims=10, max_game_plies=10, temperature_switch_ply=0)
    play_one_game(client, cfg, _rng())
    # new_game appelé une fois.
    client.new_game.assert_called_once()
    # Et avant tout go_nodes (ordre d'appels).
    call_names = [c[0] for c in client.method_calls if c[0] in {"new_game", "go_nodes"}]
    assert call_names[0] == "new_game"


def test_play_one_game_returns_samples_for_each_ply() -> None:
    """Fool's mate : 4 plies joués -> 4 samples retournés."""
    client = _make_mock_client(_foolsmate_results())
    cfg = SelfplayConfig(mcts_sims=10, max_game_plies=10, temperature_switch_ply=0)
    samples = play_one_game(client, cfg, _rng())
    assert len(samples) == 4
    # Ply monotone 0, 1, 2, 3.
    assert [s.ply for s in samples] == [0, 1, 2, 3]
    # Turn alternés 0, 1, 0, 1.
    assert [s.turn for s in samples] == [0, 1, 0, 1]


def test_play_one_game_foolsmate_value_targets_black_wins() -> None:
    """Fool's mate : noir gagne -> outcome_white=-1.
    turn=0 samples value=-1, turn=1 samples value=+1.
    """
    client = _make_mock_client(_foolsmate_results())
    cfg = SelfplayConfig(mcts_sims=10, max_game_plies=10, temperature_switch_ply=0)
    samples = play_one_game(client, cfg, _rng())
    assert samples[0].turn == 0
    assert samples[0].value_target == -1.0
    assert samples[1].turn == 1
    assert samples[1].value_target == 1.0
    assert samples[2].turn == 0
    assert samples[2].value_target == -1.0
    assert samples[3].turn == 1
    assert samples[3].value_target == 1.0


def test_play_one_game_position_cmd_no_moves_first_ply() -> None:
    """1er go_nodes -> position_cmd == 'position startpos' exact."""
    client = _make_mock_client(_foolsmate_results())
    cfg = SelfplayConfig(mcts_sims=10, max_game_plies=10, temperature_switch_ply=0)
    play_one_game(client, cfg, _rng())
    # 1er appel go_nodes : args = (position_cmd, nodes=..., timeout=...).
    first_call = client.go_nodes.call_args_list[0]
    position_cmd = first_call.args[0]
    assert position_cmd == "position startpos"


def test_play_one_game_position_cmd_includes_moves_after_ply_1() -> None:
    """2e go_nodes -> position_cmd contient les coups joués."""
    client = _make_mock_client(_foolsmate_results())
    cfg = SelfplayConfig(mcts_sims=10, max_game_plies=10, temperature_switch_ply=0)
    play_one_game(client, cfg, _rng())
    # 2e appel go_nodes.
    second_call = client.go_nodes.call_args_list[1]
    position_cmd = second_call.args[0]
    assert position_cmd.startswith("position startpos moves ")
    assert "f2f3" in position_cmd  # 1er coup joué


def test_play_one_game_terminal_bestmove_none_stops() -> None:
    """UciResult(visits={}, bestmove=None) -> boucle s'arrête immédiatement."""
    client = _make_mock_client(
        [
            UciResult(visits={"e2e4": 10}, bestmove="e2e4"),
            UciResult(visits={"e7e5": 10}, bestmove="e7e5"),
            UciResult(visits={}, bestmove=None),  # Engine indique terminal
        ]
    )
    cfg = SelfplayConfig(mcts_sims=10, max_game_plies=100, temperature_switch_ply=0)
    samples = play_one_game(client, cfg, _rng())
    # Seulement 2 samples avant le bestmove=None.
    assert len(samples) == 2
    # bestmove=None ne déclenche pas d'is_game_over naturel -> outcome=draw.
    assert all(s.value_target == 0.0 for s in samples)


def test_play_one_game_max_plies_cutoff_draw() -> None:
    """max_plies atteint sans terminaison naturelle -> draw."""
    # Mock visits infini (e2e4 puis répétition impossible, simplifié).
    # On crée une longue liste de UciResult retournant des coups légaux.
    # En pratique, après quelques coups le board atteindra l'état naturel,
    # mais on configure max_plies très bas pour cutoff explicite.
    visits_inf = [
        UciResult(visits={"e2e4": 10}, bestmove="e2e4"),
        UciResult(visits={"e7e5": 10}, bestmove="e7e5"),
        UciResult(visits={"g1f3": 10}, bestmove="g1f3"),
        UciResult(visits={"b8c6": 10}, bestmove="b8c6"),
    ]
    client = _make_mock_client(visits_inf)
    cfg = SelfplayConfig(mcts_sims=10, max_game_plies=4, temperature_switch_ply=0)  # cutoff @ ply 4
    samples = play_one_game(client, cfg, _rng())
    assert len(samples) == 4
    assert all(s.value_target == 0.0 for s in samples)


def test_play_one_game_propagates_uci_timeout() -> None:
    """UciTimeoutError raised in go_nodes -> propagate."""
    client = MagicMock()
    client.go_nodes.side_effect = UciTimeoutError("timeout")
    cfg = SelfplayConfig(mcts_sims=10, max_game_plies=10)
    with pytest.raises(UciTimeoutError):
        play_one_game(client, cfg, _rng())


def test_play_one_game_propagates_uci_crash() -> None:
    """UciCrashError raised in go_nodes -> propagate."""
    client = MagicMock()
    client.go_nodes.side_effect = UciCrashError("crash")
    cfg = SelfplayConfig(mcts_sims=10, max_game_plies=10)
    with pytest.raises(UciCrashError):
        play_one_game(client, cfg, _rng())


def test_play_one_game_fresh_board_per_call() -> None:
    """2 invocations consécutives -> samples du 2e appel n'ont PAS l'historique du 1er."""
    # 1er appel : Fool's mate (4 plies).
    client1 = _make_mock_client(_foolsmate_results())
    cfg = SelfplayConfig(mcts_sims=10, max_game_plies=10, temperature_switch_ply=0)
    samples_a = play_one_game(client1, cfg, _rng())
    assert samples_a[0].ply == 0

    # 2e appel sur un nouveau mock client.
    client2 = _make_mock_client(_foolsmate_results())
    samples_b = play_one_game(client2, cfg, _rng())
    # 2e appel commence aussi au ply 0 (fresh board).
    assert samples_b[0].ply == 0
    # Et samples_b[0].turn == 0 (white commence à nouveau).
    assert samples_b[0].turn == 0


def test_play_one_game_uses_temperature_sampling_early() -> None:
    """ply < switch -> select_move consomme le RNG (temperature_sample)."""
    client = _make_mock_client(
        [
            UciResult(visits={"e2e4": 80, "d2d4": 20}, bestmove="e2e4"),
            UciResult(visits={"e7e5": 80, "d7d5": 20}, bestmove="e7e5"),
        ]
    )
    cfg = SelfplayConfig(mcts_sims=10, max_game_plies=2, temperature_switch_ply=10, temperature=1.0)
    rng = _rng(42)
    state_before = rng.bit_generator.state
    play_one_game(client, cfg, rng)
    state_after = rng.bit_generator.state
    assert state_before != state_after, "RNG should be consumed by temperature_sample"


def test_play_one_game_uses_argmax_late() -> None:
    """temperature_switch_ply=0 -> argmax dès ply 0, RNG inchangé."""
    client = _make_mock_client(
        [
            UciResult(visits={"e2e4": 10, "d2d4": 20}, bestmove="d2d4"),  # argmax = d2d4
            UciResult(visits={"e7e5": 10, "d7d5": 20}, bestmove="d7d5"),
        ]
    )
    cfg = SelfplayConfig(
        mcts_sims=10, max_game_plies=2, temperature_switch_ply=0
    )  # tau-zero direct
    rng = _rng(42)
    state_before = rng.bit_generator.state
    play_one_game(client, cfg, rng)
    state_after = rng.bit_generator.state
    assert state_before == state_after, "RNG should NOT be consumed in argmax mode"


def test_play_one_game_samples_validated() -> None:
    """Tous les samples retournés passent Sample.__post_init__ (shape/dtype/range)."""
    client = _make_mock_client(_foolsmate_results())
    cfg = SelfplayConfig(mcts_sims=10, max_game_plies=10, temperature_switch_ply=0)
    samples = play_one_game(client, cfg, _rng())
    for s in samples:
        assert s.input_planes.shape == (119, 8, 8)
        assert s.input_planes.dtype == np.float32
        assert s.policy_target.shape == (4672,)
        assert s.policy_target.dtype == np.float32
        assert s.value_target in (-1.0, 0.0, 1.0)
        assert s.turn in (0, 1)
        assert s.ply >= 0
