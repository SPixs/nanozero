"""Selfplay configuration (mini-version phase 1.0.0-4-b).

Full configuration system (YAML + CLI overrides + RunStateManager integration)
en phase 1.0.0-9. Ce dataclass minimal est le strict nécessaire pour
typer/passer les hyperparams à `temperature.select_move` et au futur worker
phase 1.0.0-4-c.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SelfplayConfig:
    """Self-play hyperparameters (defaults alignés SPEC-training §4).

    Attributes:
        mcts_sims: nombre de simulations MCTS par coup (`go nodes N`).
        temperature: tau pour les plies [0, temperature_switch_ply).
        temperature_switch_ply: ply à partir duquel on bascule en argmax.
        dirichlet_alpha: alpha x 1000 (eg. 300 ↔ alpha=0.3) — hidden option UCI.
        dirichlet_epsilon: epsilon x 1000 (eg. 250 ↔ epsilon=0.25) — hidden option.
        go_timeout_seconds: max wall-clock pour une recherche `go nodes`.
        worker_seed: RNG seed pour Dirichlet noise + temperature sampling.
        max_game_plies: cutoff de sécurité empêchant parties infinies. Typique
                        60-200 plies en partie naturelle ; 400 laisse une grande
                        marge. Si atteint sans terminaison naturelle, la partie
                        est traitée comme draw (cohérent SPEC §6, pas de biais).
        games_per_batch: parties par fichier .npz batché (SPEC-training §4, §5.2).
        target_games_per_gen: cible de parties par génération (SPEC §4, §6).
        worker_restart_every: redémarrer le subprocess UCI tous les N parties
                              pour robustesse mémoire/VRAM (SPEC §6.1).
        max_consecutive_crashes: nombre max de crashes UCI consécutifs avant
                                 d'avorter la génération (SPEC §6.4, ADR-009).
    """

    # Worker single-game (phase 4)
    mcts_sims: int = 400
    temperature: float = 1.0
    temperature_switch_ply: int = 30
    dirichlet_alpha: int = 300  # alpha x 1000
    dirichlet_epsilon: int = 250  # epsilon x 1000
    go_timeout_seconds: float = 60.0
    worker_seed: int = 42
    max_game_plies: int = 400

    # Orchestrator multi-parties (phase 5)
    games_per_batch: int = 250
    target_games_per_gen: int = 500
    worker_restart_every: int = 1000
    max_consecutive_crashes: int = 5
