"""Self-play module: UCI client + visits parser + temperature + worker + orchestrator."""

from nanozero_training.selfplay.config import SelfplayConfig
from nanozero_training.selfplay.orchestrator import (
    SelfplayOrchestrator,
    derive_dirichlet_seed,
    make_game_rngs,
)
from nanozero_training.selfplay.temperature import (
    argmax_move,
    select_move,
    temperature_sample,
)
from nanozero_training.selfplay.uci_client import (
    UciClient,
    UciCrashError,
    UciResult,
    UciTimeoutError,
)
from nanozero_training.selfplay.value_backfill import (
    backfill_value_targets,
    outcome_white_from_result,
)
from nanozero_training.selfplay.visits_parser import parse_visits_line
from nanozero_training.selfplay.worker import play_one_game

__all__ = [
    "SelfplayConfig",
    "SelfplayOrchestrator",
    "UciClient",
    "UciCrashError",
    "UciResult",
    "UciTimeoutError",
    "argmax_move",
    "backfill_value_targets",
    "derive_dirichlet_seed",
    "make_game_rngs",
    "outcome_white_from_result",
    "parse_visits_line",
    "play_one_game",
    "select_move",
    "temperature_sample",
]
