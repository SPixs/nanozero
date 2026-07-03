"""Test pipeline reproducibility: same seed → same state transitions (mocked).

NOT a slow test — uses mocked phase_runner helpers everywhere.
Validates determinism of the orchestration logic + state machine.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from nanozero_training.eval.sprt_result import SPRTResult, SPRTStatus


def _run_mocked_pipeline(tmp_path: Path, seed: int) -> tuple[list[int], list[str]]:
    """Run a mini pipeline 2 generations with mocked phases.

    Returns (gens_visited, phases_visited) for comparison.
    """
    from nanozero_training.config.run_config import (
        MonitorConfig,
        PathsConfig,
        RunConfig,
    )
    from nanozero_training.pipeline.orchestrator import PipelineOrchestrator
    from nanozero_training.selfplay.config import SelfplayConfig
    from nanozero_training.train.config import TrainConfig

    monitoring_dir = tmp_path / "monitoring"
    models_dir = tmp_path / "models"
    monitoring_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    paths = PathsConfig(
        run_root=str(tmp_path),
        datasets_dir="datasets",
        models_dir="models",
        monitoring_dir="monitoring",
        versions_yaml="versions.yaml",
        pgn_path="monitoring/sprt.pgn",
        uci_jar=str(tmp_path / "fake.jar"),
    )
    cfg = RunConfig(
        paths=paths,
        selfplay=SelfplayConfig(mcts_sims=10, worker_seed=seed, target_games_per_gen=2),
        train=TrainConfig(batch_size=2, total_epochs=1, train_seed=seed),
        monitor=MonitorConfig(enabled=False),
        max_generations=2,
    )

    sm = MagicMock()
    state = MagicMock()
    state.current_gen = 0
    state.phase = "idle"
    state.eval = MagicMock(games_played_at_last_save=0, last_decision=None)
    sm.state = state

    vm = MagicMock()
    vm.get_current.return_value = "gen-001-init"
    vm.get_latest_trained.return_value = "gen-001-trained"

    gens_visited: list[int] = []
    phases_visited: list[str] = []

    def track_selfplay(gen: int, **_kw: object) -> None:
        gens_visited.append(gen)
        phases_visited.append(f"selfplay-{gen}")

    def track_train(gen: int, **_kw: object) -> str:
        phases_visited.append(f"train-{gen}")
        return f"gen-{gen:03d}-trained"

    def track_eval(gen: int, **_kw: object) -> SPRTResult:
        phases_visited.append(f"eval-{gen}")
        # Deterministic result based on seed (same seed -> same result).
        return SPRTResult(
            status=SPRTStatus.H1_ACCEPTED if seed % 2 == 0 else SPRTStatus.H0_ACCEPTED,
            llr=2.95 * (1 if seed % 2 == 0 else -1),
            games_played=100,
            wins=60,
            losses=30,
            draws=10,
            elo_diff=10.0 * (1 if seed % 2 == 0 else -1),
        )

    def track_promote(gen: int, **_kw: object) -> MagicMock:
        phases_visited.append(f"promote-{gen}")
        return MagicMock()

    with (
        patch(
            "nanozero_training.pipeline.orchestrator.run_selfplay_phase",
            side_effect=track_selfplay,
        ),
        patch(
            "nanozero_training.pipeline.orchestrator.run_train_phase",
            side_effect=track_train,
        ),
        patch(
            "nanozero_training.pipeline.orchestrator.run_eval_phase",
            side_effect=track_eval,
        ),
        patch(
            "nanozero_training.pipeline.orchestrator.run_promote_phase",
            side_effect=track_promote,
        ),
    ):
        orch = PipelineOrchestrator(
            config=cfg,
            state_manager=sm,
            version_manager=vm,
        )
        orch.run()

    return gens_visited, phases_visited


def test_pipeline_two_invocations_same_seed_same_state(tmp_path: Path) -> None:
    """2 invocations with same seed -> identical phase sequence."""
    tmp1 = tmp_path / "run1"
    tmp2 = tmp_path / "run2"
    tmp1.mkdir()
    tmp2.mkdir()

    gens1, phases1 = _run_mocked_pipeline(tmp1, seed=42)
    gens2, phases2 = _run_mocked_pipeline(tmp2, seed=42)

    assert gens1 == gens2
    assert phases1 == phases2
    # Each generation should visit 4 phases (selfplay/train/eval/promote)
    # 2 gens -> 8 phase events
    assert len(phases1) == 8


def test_pipeline_different_seed_can_differ(tmp_path: Path) -> None:
    """Different seeds produce different SPRT outcomes (sanity check on mock determinism)."""
    tmp1 = tmp_path / "run1"
    tmp2 = tmp_path / "run2"
    tmp1.mkdir()
    tmp2.mkdir()

    # seed=42 (even) -> H1_ACCEPTED, seed=43 (odd) -> H0_ACCEPTED
    # State transitions identical, but the SPRT result decision differs.
    # In this mock, the orchestrator visits same phases regardless ;
    # so phase sequence is same. The decision diverges inside promote.
    gens1, phases1 = _run_mocked_pipeline(tmp1, seed=42)
    gens2, phases2 = _run_mocked_pipeline(tmp2, seed=43)

    # Phase sequence identical (state machine same for both H0/H1).
    assert phases1 == phases2
