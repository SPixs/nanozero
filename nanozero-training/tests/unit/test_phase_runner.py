"""Unit tests for pipeline/phase_runner — 4 phase helpers with mocks."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from nanozero_training.config.run_config import (
    MonitorConfig,
    PathsConfig,
    RunConfig,
)
from nanozero_training.eval.sprt_result import SPRTResult, SPRTStatus
from nanozero_training.pipeline.phase_runner import (
    run_eval_phase,
    run_promote_phase,
    run_selfplay_phase,
    run_train_phase,
)
from nanozero_training.selfplay.config import SelfplayConfig
from nanozero_training.train.config import TrainConfig
from nanozero_training.version.promotion import PromoteOutcome


def _make_cfg(tmp_path: Path) -> RunConfig:
    paths = PathsConfig(
        run_root=str(tmp_path),
        datasets_dir="datasets",
        models_dir="models",
        monitoring_dir="monitoring",
        versions_yaml="versions.yaml",
        pgn_path="monitoring/sprt.pgn",
        uci_jar=str(tmp_path / "fake-uci.jar"),
    )
    (tmp_path / "monitoring").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "datasets").mkdir(parents=True, exist_ok=True)
    return RunConfig(
        paths=paths,
        selfplay=SelfplayConfig(mcts_sims=10, target_games_per_gen=2),
        train=TrainConfig(batch_size=2, total_epochs=1),
        monitor=MonitorConfig(enabled=False),
    )


def _make_sm() -> MagicMock:
    mgr = MagicMock()
    state = MagicMock()
    state.phase = "idle"
    state.current_gen = 0
    state.selfplay = MagicMock(completed_games=0, completed_batches=0)
    state.train = MagicMock(current_epoch=0)
    state.eval = MagicMock(games_played_at_last_save=0, last_decision=None)
    mgr.state = state
    return mgr


def _make_vm(tmp_path: Path) -> MagicMock:
    mgr = MagicMock()
    mgr.get_current.return_value = "gen-001-init"
    mgr.get_path_for.side_effect = lambda name: tmp_path / "models" / f"{name}.npz"
    mgr.get_latest_trained.return_value = "gen-002-trained"
    return mgr


def test_run_selfplay_phase_raises_if_jar_missing(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    sm = _make_sm()
    vm = _make_vm(tmp_path)
    # JAR doesn't exist
    (tmp_path / "models" / "gen-001-init.npz").write_bytes(b"dummy")
    with pytest.raises(FileNotFoundError, match="UCI JAR"):
        run_selfplay_phase(gen=1, sm=sm, vm=vm, cfg=cfg, abort_flag={"requested": False})


def test_run_selfplay_phase_raises_if_model_missing(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    sm = _make_sm()
    vm = _make_vm(tmp_path)
    # No init model on disk
    with pytest.raises(FileNotFoundError, match="Current model not found"):
        run_selfplay_phase(gen=1, sm=sm, vm=vm, cfg=cfg, abort_flag={"requested": False})


def test_run_selfplay_phase_calls_orchestrator(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    sm = _make_sm()
    vm = _make_vm(tmp_path)
    (tmp_path / "models" / "gen-001-init.npz").write_bytes(b"dummy")
    (tmp_path / "fake-uci.jar").write_bytes(b"jar")

    with (
        patch("nanozero_training.pipeline.phase_runner.UciClient") as mock_client,
        patch("nanozero_training.pipeline.phase_runner.SelfplayOrchestrator") as mock_orch,
    ):
        run_selfplay_phase(gen=1, sm=sm, vm=vm, cfg=cfg, abort_flag={"requested": False})

    mock_client.return_value.start.assert_called_once()
    mock_orch.return_value.run_generation.assert_called_once()
    mock_client.return_value.quit.assert_called_once()


def test_run_train_phase_calls_trainer(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    sm = _make_sm()
    vm = _make_vm(tmp_path)

    fake_npz = tmp_path / "models" / "gen-002-trained.npz"
    fake_npz.write_bytes(b"dummy-trained")

    with (
        patch("nanozero_training.pipeline.phase_runner.Trainer") as mock_trainer,
        patch("nanozero_training.pipeline.phase_runner.make_train_dataloader_for_gen") as mock_dl,
    ):
        mock_trainer.return_value.train_generation.return_value = fake_npz
        mock_dl.return_value = MagicMock()
        trained_name = run_train_phase(
            gen=2, sm=sm, vm=vm, cfg=cfg, abort_flag={"requested": False}
        )

    assert trained_name == "gen-002-trained"
    mock_trainer.return_value.train_generation.assert_called_once()


def test_run_train_phase_registers_in_versions(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    sm = _make_sm()
    vm = _make_vm(tmp_path)
    fake_npz = tmp_path / "models" / "gen-002-trained.npz"
    fake_npz.write_bytes(b"dummy")

    with (
        patch("nanozero_training.pipeline.phase_runner.Trainer") as mock_trainer,
        patch("nanozero_training.pipeline.phase_runner.make_train_dataloader_for_gen"),
    ):
        mock_trainer.return_value.train_generation.return_value = fake_npz
        run_train_phase(gen=2, sm=sm, vm=vm, cfg=cfg, abort_flag={"requested": False})

    vm.add_trained.assert_called_once_with(name="gen-002-trained", parent="gen-001-init")
    vm.save.assert_called_once()


def test_run_train_phase_a3_uses_champion_checkpoint(tmp_path: Path) -> None:
    """Phase A3 : l'init continued-training repart du checkpoint du CHAMPION."""
    base_cfg = _make_cfg(tmp_path)
    cfg = replace(base_cfg, train=replace(base_cfg.train, continued_training=True))
    sm = _make_sm()
    vm = _make_vm(tmp_path)
    vm.get_current.return_value = "gen-025-promoted"

    models = tmp_path / "models"
    champion_pt = models / "train-state-gen-025-epoch-000.pt"  # total_epochs=1 -> epoch-000
    champion_pt.write_bytes(b"champion-weights")
    (models / "gen-026-trained.npz").write_bytes(b"dummy")

    with (
        patch("nanozero_training.pipeline.phase_runner.Trainer") as mock_trainer,
        patch("nanozero_training.pipeline.phase_runner.make_train_dataloader_for_gen"),
    ):
        mock_trainer.return_value.train_generation.return_value = models / "gen-026-trained.npz"
        run_train_phase(gen=26, sm=sm, vm=vm, cfg=cfg, abort_flag={"requested": False})

    kwargs = mock_trainer.return_value.train_generation.call_args.kwargs
    assert kwargs["base_model_path"] == champion_pt


def test_run_train_phase_a3_prefers_champion_over_rejected_genminus1(tmp_path: Path) -> None:
    """Régression : gen-(N-1) REJETÉE -> A3 repart du champion, PAS de gen-(N-1).

    gen-026 rejetée -> champion reste gen-025-promoted. Pour gen-027, le code ne
    doit PAS charger train-state-gen-026 (poids rejetés) même s'il existe sur disque.
    """
    base_cfg = _make_cfg(tmp_path)
    cfg = replace(base_cfg, train=replace(base_cfg.train, continued_training=True))
    sm = _make_sm()
    vm = _make_vm(tmp_path)
    vm.get_current.return_value = "gen-025-promoted"  # gen-026 rejetée, champion inchangé

    models = tmp_path / "models"
    champion_pt = models / "train-state-gen-025-epoch-000.pt"
    champion_pt.write_bytes(b"champion")
    rejected_pt = models / "train-state-gen-026-epoch-000.pt"
    rejected_pt.write_bytes(b"rejected")  # piège : le .pt de gen-(N-1) existe aussi
    (models / "gen-027-trained.npz").write_bytes(b"dummy")

    with (
        patch("nanozero_training.pipeline.phase_runner.Trainer") as mock_trainer,
        patch("nanozero_training.pipeline.phase_runner.make_train_dataloader_for_gen"),
    ):
        mock_trainer.return_value.train_generation.return_value = models / "gen-027-trained.npz"
        run_train_phase(gen=27, sm=sm, vm=vm, cfg=cfg, abort_flag={"requested": False})

    kwargs = mock_trainer.return_value.train_generation.call_args.kwargs
    assert kwargs["base_model_path"] == champion_pt
    assert kwargs["base_model_path"] != rejected_pt


def test_run_train_phase_a3_missing_champion_checkpoint_falls_back(tmp_path: Path) -> None:
    """Pas de .pt champion -> base_model_path None (random init, warmup off)."""
    base_cfg = _make_cfg(tmp_path)
    cfg = replace(base_cfg, train=replace(base_cfg.train, continued_training=True))
    sm = _make_sm()
    vm = _make_vm(tmp_path)
    vm.get_current.return_value = "gen-025-promoted"
    (tmp_path / "models" / "gen-026-trained.npz").write_bytes(b"dummy")

    with (
        patch("nanozero_training.pipeline.phase_runner.Trainer") as mock_trainer,
        patch("nanozero_training.pipeline.phase_runner.make_train_dataloader_for_gen"),
    ):
        mock_trainer.return_value.train_generation.return_value = (
            tmp_path / "models" / "gen-026-trained.npz"
        )
        run_train_phase(gen=26, sm=sm, vm=vm, cfg=cfg, abort_flag={"requested": False})

    kwargs = mock_trainer.return_value.train_generation.call_args.kwargs
    assert kwargs["base_model_path"] is None


def test_run_eval_phase_calls_fastchess_runner(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    sm = _make_sm()
    vm = _make_vm(tmp_path)

    expected_result = SPRTResult(
        status=SPRTStatus.H1_ACCEPTED,
        llr=2.95,
        games_played=100,
        wins=60,
        losses=30,
        draws=10,
        elo_diff=10.0,
    )
    with patch("nanozero_training.pipeline.phase_runner.FastchessRunner") as mock_runner:
        mock_runner.return_value.run_sprt.return_value = expected_result
        result = run_eval_phase(
            gen=2,
            trained_name="gen-002-trained",
            sm=sm,
            vm=vm,
            cfg=cfg,
            abort_flag={"requested": False},
        )

    assert result == expected_result
    mock_runner.return_value.run_sprt.assert_called_once()


def test_run_eval_phase_updates_state_initial(tmp_path: Path) -> None:
    """state.eval.challenger/baseline/last_decision set before SPRT launch."""
    cfg = _make_cfg(tmp_path)
    sm = _make_sm()
    vm = _make_vm(tmp_path)

    with patch("nanozero_training.pipeline.phase_runner.FastchessRunner") as mock_runner:
        mock_runner.return_value.run_sprt.return_value = SPRTResult(
            status=SPRTStatus.H0_ACCEPTED,
            llr=-2.95,
            games_played=100,
            wins=30,
            losses=60,
            draws=10,
            elo_diff=-10.0,
        )
        run_eval_phase(
            gen=2,
            trained_name="gen-002-trained",
            sm=sm,
            vm=vm,
            cfg=cfg,
            abort_flag={"requested": False},
        )

    # Initial state.update call must have set challenger/baseline/last_decision=None.
    call_args = [c.kwargs for c in sm.update.call_args_list]
    found_initial = any(
        "eval__challenger" in kw and kw.get("eval__last_decision") is None for kw in call_args
    )
    assert found_initial


def test_run_eval_phase_returns_sprt_result(tmp_path: Path) -> None:
    cfg = _make_cfg(tmp_path)
    sm = _make_sm()
    vm = _make_vm(tmp_path)
    expected = SPRTResult(
        status=SPRTStatus.MAX_GAMES_REACHED,
        llr=0.5,
        games_played=2000,
        wins=900,
        losses=900,
        draws=200,
        elo_diff=0.0,
    )
    with patch("nanozero_training.pipeline.phase_runner.FastchessRunner") as mock_runner:
        mock_runner.return_value.run_sprt.return_value = expected
        result = run_eval_phase(
            gen=2,
            trained_name="gen-002-trained",
            sm=sm,
            vm=vm,
            cfg=cfg,
            abort_flag={"requested": False},
        )
    assert result.status == SPRTStatus.MAX_GAMES_REACHED


def test_run_promote_phase_h1_promotes() -> None:
    vm = MagicMock()
    with patch("nanozero_training.pipeline.phase_runner.promote_if_h1") as mock_promote:
        from nanozero_training.version.promotion import PromoteResult

        mock_promote.return_value = PromoteResult(
            outcome=PromoteOutcome.PROMOTED,
            promoted_name="gen-002-promoted",
            message="ok",
        )
        sprt = SPRTResult(
            status=SPRTStatus.H1_ACCEPTED,
            llr=2.95,
            games_played=100,
            wins=60,
            losses=30,
            draws=10,
            elo_diff=10.0,
        )
        result = run_promote_phase(gen=2, trained_name="gen-002-trained", sprt_result=sprt, vm=vm)
    assert result.outcome == PromoteOutcome.PROMOTED


def test_run_promote_phase_h0_rejects() -> None:
    vm = MagicMock()
    with patch("nanozero_training.pipeline.phase_runner.promote_if_h1") as mock_promote:
        from nanozero_training.version.promotion import PromoteResult

        mock_promote.return_value = PromoteResult(
            outcome=PromoteOutcome.REJECTED,
            promoted_name=None,
            message="h0",
        )
        sprt = SPRTResult(
            status=SPRTStatus.H0_ACCEPTED,
            llr=-2.95,
            games_played=100,
            wins=30,
            losses=60,
            draws=10,
            elo_diff=-10.0,
        )
        result = run_promote_phase(gen=2, trained_name="gen-002-trained", sprt_result=sprt, vm=vm)
    assert result.outcome == PromoteOutcome.REJECTED


def test_run_promote_phase_max_games_rejects() -> None:
    vm = MagicMock()
    with patch("nanozero_training.pipeline.phase_runner.promote_if_h1") as mock_promote:
        from nanozero_training.version.promotion import PromoteResult

        mock_promote.return_value = PromoteResult(
            outcome=PromoteOutcome.REJECTED,
            promoted_name=None,
            message="max_games",
        )
        sprt = SPRTResult(
            status=SPRTStatus.MAX_GAMES_REACHED,
            llr=0.1,
            games_played=2000,
            wins=900,
            losses=900,
            draws=200,
            elo_diff=0.0,
        )
        result = run_promote_phase(gen=2, trained_name="gen-002-trained", sprt_result=sprt, vm=vm)
    assert result.outcome == PromoteOutcome.REJECTED


def test_run_promote_phase_already_promoted_idempotent() -> None:
    vm = MagicMock()
    with patch("nanozero_training.pipeline.phase_runner.promote_if_h1") as mock_promote:
        from nanozero_training.version.promotion import PromoteResult

        mock_promote.return_value = PromoteResult(
            outcome=PromoteOutcome.ALREADY_PROMOTED,
            promoted_name="gen-002-promoted",
            message="already promoted",
        )
        sprt = SPRTResult(
            status=SPRTStatus.H1_ACCEPTED,
            llr=2.95,
            games_played=100,
            wins=60,
            losses=30,
            draws=10,
            elo_diff=10.0,
        )
        result = run_promote_phase(gen=2, trained_name="gen-002-trained", sprt_result=sprt, vm=vm)
    assert result.outcome == PromoteOutcome.ALREADY_PROMOTED
