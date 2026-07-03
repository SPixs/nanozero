"""Self-play worker : play one complete game and return collected samples.

Pattern figé SPEC-training §6.2 :

  1. Fresh chess.Board() (pas de pollution historique inter-parties).
  2. uci_client.new_game().
  3. Loop :
     a. Check terminal : board.is_game_over(claim_draw=True) OR ply >= max_plies.
     b. encode_position(board) -> input_planes (NO MUTATION garanti phase 4-a).
     c. uci_client.go_nodes(position_cmd, mcts_sims) -> UciResult(visits, bestmove).
     d. visits_to_policy_target(visits, board) -> policy_target.
     e. Store (input_planes, policy_target, turn, ply) (value pending).
     f. select_move(visits, board, ply, ...) -> chess.Move.
     g. board.push(move).
  4. Determine outcome from board.result(claim_draw=True) OR draw si max_plies cutoff.
  5. backfill_value_targets -> list[Sample] validés.

Error handling : UciTimeoutError / UciCrashError propagate au caller
(l'orchestrator phase 1.0.0-5 décide : restart worker / discard game / abort).
"""

from __future__ import annotations

import chess
import numpy as np

from nanozero_training.data.sample import Sample
from nanozero_training.network.move_encoding import visits_to_policy_target
from nanozero_training.network.position_encoding import encode_position
from nanozero_training.selfplay.config import SelfplayConfig
from nanozero_training.selfplay.temperature import select_move
from nanozero_training.selfplay.uci_client import UciClient
from nanozero_training.selfplay.value_backfill import (
    PartialSample,
    backfill_value_targets,
    outcome_white_from_result,
)


def play_one_game(
    uci_client: UciClient,
    config: SelfplayConfig,
    rng: np.random.Generator,
) -> list[Sample]:
    """Play one complete self-play game and return collected samples.

    Args:
        uci_client: started UciClient (handshake done, Dirichlet options set).
        config: SelfplayConfig (mcts_sims, max_game_plies, temperature_switch_ply, ...).
        rng: numpy RNG pour temperature sampling.

    Returns:
        list[Sample] validés avec value_targets backfilled.
        len = nombre de plies joués (peut être 0 si startpos terminal, en
        pratique ≥ 1 pour une partie normale ; 0 reste valide retour).

    Raises:
        UciTimeoutError: subprocess UCI n'a pas répondu dans go_timeout_seconds.
        UciCrashError: subprocess UCI exited unexpectedly.
    """
    board = chess.Board()
    uci_client.new_game()

    samples_partial: list[PartialSample] = []
    moves_uci: list[str] = []

    while not _is_terminal(board, config.max_game_plies):
        input_planes = encode_position(board)
        turn = 0 if board.turn == chess.WHITE else 1
        ply = len(board.move_stack)

        position_cmd = (
            "position startpos moves " + " ".join(moves_uci) if moves_uci else "position startpos"
        )

        result = uci_client.go_nodes(
            position_cmd,
            nodes=config.mcts_sims,
            timeout=config.go_timeout_seconds,
        )

        # If UCI returned no bestmove (terminal from its POV), stop.
        if result.bestmove is None:
            break

        policy_target = visits_to_policy_target(result.visits, board)
        samples_partial.append((input_planes, policy_target, turn, ply))

        chosen_move = select_move(
            visits=result.visits,
            board=board,
            ply=ply,
            temperature_switch_ply=config.temperature_switch_ply,
            tau=config.temperature,
            rng=rng,
        )

        board.push(chosen_move)
        moves_uci.append(chosen_move.uci())

    # Determine outcome.
    if board.is_game_over(claim_draw=True):
        outcome_white = outcome_white_from_result(board.result(claim_draw=True))
    else:
        # max_plies cutoff sans terminaison naturelle -> draw (SPEC §6 pas de biais).
        outcome_white = 0.0

    return backfill_value_targets(samples_partial, outcome_white)


def _is_terminal(board: chess.Board, max_plies: int) -> bool:
    """True si game over naturel OU max_plies atteint."""
    return board.is_game_over(claim_draw=True) or len(board.move_stack) >= max_plies
