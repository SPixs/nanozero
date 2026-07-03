"""Unit tests for pipeline/orchestrator — PipelineOrchestrator state machine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from nanozero_training.eval.sprt_result import SPRTResult, SPRTStatus
from nanozero_training.pipeline.orchestrator import PipelineOrchestrator


def _make_sprt(status: SPRTStatus = SPRTStatus.H1_ACCEPTED) -> SPRTResult:
    return SPRTResult(
        status=status,
        llr=2.95 if status == SPRTStatus.H1_ACCEPTED else -2.95,
        games_played=100,
        wins=60,
        losses=30,
        draws=10,
        elo_diff=10.0,
    )


def _make_sm(current_gen: int = 0, phase: str = "idle") -> MagicMock:
    mgr = MagicMock()
    state = MagicMock()
    state.current_gen = current_gen
    state.phase = phase
    state.eval = MagicMock(games_played_at_last_save=0, last_decision=None)
    mgr.state = state
    return mgr


def _make_vm() -> MagicMock:
    vm = MagicMock()
    vm.get_current.return_value = "gen-001-init"
    vm.get_latest_trained.return_value = "gen-002-trained"
    return vm


def _make_cfg(max_gens: int = 3) -> MagicMock:
    cfg = MagicMock()
    cfg.max_generations = max_gens
    return cfg


def _patch_phases() -> dict[str, MagicMock]:
    """Patch the 4 phase runner helpers used by orchestrator."""
    patches = {
        "selfplay": patch("nanozero_training.pipeline.orchestrator.run_selfplay_phase"),
        "train": patch("nanozero_training.pipeline.orchestrator.run_train_phase"),
        "eval": patch("nanozero_training.pipeline.orchestrator.run_eval_phase"),
        "promote": patch("nanozero_training.pipeline.orchestrator.run_promote_phase"),
    }
    return {name: p.start() for name, p in patches.items()}


def test_run_single_generation() -> None:
    cfg = _make_cfg(max_gens=1)
    sm = _make_sm()
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        mocks["train"].return_value = "gen-001-trained"
        mocks["eval"].return_value = _make_sprt()
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        orch.run()

        # 4 phases called for gen 1
        mocks["selfplay"].assert_called_once()
        mocks["train"].assert_called_once()
        mocks["eval"].assert_called_once()
        mocks["promote"].assert_called_once()
    finally:
        patch.stopall()


def test_run_skips_completed_gens() -> None:
    """state.current_gen=2 -> run() targets remaining gens (3..max)."""
    cfg = _make_cfg(max_gens=3)
    sm = _make_sm(current_gen=2)
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        mocks["train"].return_value = "gen-003-trained"
        mocks["eval"].return_value = _make_sprt()
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        orch.run()
        # Only gen 3 ran -> exactly 1 call to each phase
        assert mocks["selfplay"].call_count == 1
        assert mocks["train"].call_count == 1
    finally:
        patch.stopall()


def test_run_marks_completed_when_done() -> None:
    cfg = _make_cfg(max_gens=1)
    sm = _make_sm()
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        mocks["train"].return_value = "gen-001-trained"
        mocks["eval"].return_value = _make_sprt()
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        orch.run()

        # Final update should set status="completed"
        final_calls = [c for c in sm.update.call_args_list if c.kwargs.get("status") == "completed"]
        assert len(final_calls) == 1
    finally:
        patch.stopall()


def test_run_abort_before_gens() -> None:
    cfg = _make_cfg(max_gens=3)
    sm = _make_sm()
    vm = _make_vm()
    abort = {"requested": True}
    mocks = _patch_phases()
    try:
        orch = PipelineOrchestrator(
            config=cfg, state_manager=sm, version_manager=vm, abort_flag=abort
        )
        orch.run()
        # No phases called
        mocks["selfplay"].assert_not_called()
        # Status aborted
        assert any(c.kwargs.get("status") == "aborted" for c in sm.update.call_args_list)
    finally:
        patch.stopall()


def test_run_abort_between_phases() -> None:
    cfg = _make_cfg(max_gens=1)
    sm = _make_sm()
    vm = _make_vm()
    abort = {"requested": False}
    mocks = _patch_phases()
    try:

        def trigger_abort(*_args: object, **_kwargs: object) -> None:
            abort["requested"] = True

        mocks["selfplay"].side_effect = trigger_abort
        orch = PipelineOrchestrator(
            config=cfg, state_manager=sm, version_manager=vm, abort_flag=abort
        )
        orch.run()
        # selfplay ran, but train should not have
        mocks["selfplay"].assert_called_once()
        mocks["train"].assert_not_called()
        assert any(c.kwargs.get("status") == "aborted" for c in sm.update.call_args_list)
    finally:
        patch.stopall()


def test_resume_selfplay_continues() -> None:
    cfg = _make_cfg(max_gens=2)
    sm = _make_sm(current_gen=2, phase="selfplay")
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        mocks["train"].return_value = "gen-002-trained"
        mocks["eval"].return_value = _make_sprt()
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        orch.resume()

        # Selfplay called for gen 2 (resume), then full pipeline for next gens
        mocks["selfplay"].assert_called()
    finally:
        patch.stopall()


def test_resume_train_skips_selfplay() -> None:
    cfg = _make_cfg(max_gens=2)
    sm = _make_sm(current_gen=2, phase="train")
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        mocks["train"].return_value = "gen-002-trained"
        mocks["eval"].return_value = _make_sprt()
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        orch.resume()
        # selfplay NOT called for gen 2 (we resumed past it)
        # but train IS called for gen 2
        assert mocks["train"].call_count >= 1
        # selfplay calls only for next gens, if any
        # For max_gens=2 + current_gen=2, no further gens -> selfplay never called
        mocks["selfplay"].assert_not_called()
    finally:
        patch.stopall()


def test_resume_eval_skips_selfplay_and_train() -> None:
    cfg = _make_cfg(max_gens=2)
    sm = _make_sm(current_gen=2, phase="eval")
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        mocks["eval"].return_value = _make_sprt()
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        orch.resume()
        mocks["selfplay"].assert_not_called()
        mocks["train"].assert_not_called()
        mocks["eval"].assert_called_once()
        mocks["promote"].assert_called_once()
    finally:
        patch.stopall()


def test_resume_promote_only_promotes() -> None:
    cfg = _make_cfg(max_gens=2)
    sm = _make_sm(current_gen=2, phase="promote")
    # Provide last_decision so _resume_sprt_result doesn't raise
    sm.state.eval.last_decision = "h0_accepted"
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        orch.resume()
        mocks["selfplay"].assert_not_called()
        mocks["train"].assert_not_called()
        mocks["eval"].assert_not_called()
        mocks["promote"].assert_called_once()
    finally:
        patch.stopall()


def test_resume_then_continues_next_gens() -> None:
    """Resume gen 2 mid-train, then continues gens 3..5."""
    cfg = _make_cfg(max_gens=5)
    sm = _make_sm(current_gen=2, phase="train")
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        mocks["train"].return_value = "gen-X-trained"
        mocks["eval"].return_value = _make_sprt()
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        orch.resume()
        # Train called for gen 2 (resume) + gens 3, 4, 5 -> total 4 calls
        assert mocks["train"].call_count == 4
        # Selfplay called only for gens 3, 4, 5 (not gen 2 which was past)
        assert mocks["selfplay"].call_count == 3
    finally:
        patch.stopall()


def test_run_promote_h0_continues_to_next_gen() -> None:
    """SPRT H0 -> promote rejects, but pipeline continues to next gen."""
    cfg = _make_cfg(max_gens=2)
    sm = _make_sm()
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        mocks["train"].return_value = "gen-X-trained"
        mocks["eval"].return_value = _make_sprt(status=SPRTStatus.H0_ACCEPTED)
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        orch.run()
        # 2 generations of full pipeline
        assert mocks["selfplay"].call_count == 2
        assert mocks["promote"].call_count == 2
    finally:
        patch.stopall()


def _train_current_epoch_resets(sm: MagicMock) -> list[int]:
    """Return list of values train__current_epoch was set to via sm.update()."""
    return [
        c.kwargs["train__current_epoch"]
        for c in sm.update.call_args_list
        if "train__current_epoch" in c.kwargs
    ]


def _selfplay_resets(sm: MagicMock) -> list[int]:
    """Return list of values selfplay__completed_games was set to via sm.update()."""
    return [
        c.kwargs["selfplay__completed_games"]
        for c in sm.update.call_args_list
        if "selfplay__completed_games" in c.kwargs
    ]


def test_new_gen_resets_selfplay_state() -> None:
    """Regression Bug #6 : starting a new gen MUST reset state.selfplay.completed_games=0.

    Without this, completed_games carries over from previous gen (=target) and the selfplay
    orchestrator sees completed >= target → SKIPS the selfplay phase silently. The new gen
    is then trained on previous gen's data only. Silent corruption (no crash, no error log).

    Observed in Phase 12 prod gen 2 (2026-05-15 23:50): gen-002-trained.npz created without
    any 'Phase selfplay (gen 2)' log line because state.selfplay.completed_games=250 was
    inherited from gen 1.
    """
    cfg = _make_cfg(max_gens=2)
    sm = _make_sm()
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        mocks["train"].return_value = "gen-001-trained"
        mocks["eval"].return_value = _make_sprt()
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        orch.run()

        resets = _selfplay_resets(sm)
        assert (
            0 in resets
        ), f"Expected sm.update(selfplay__completed_games=0) at gen entry, got {resets}"
    finally:
        patch.stopall()


def test_resume_mid_selfplay_does_not_reset_selfplay_state() -> None:
    """Regression Bug #6 : resuming mid-selfplay MUST preserve completed_games to allow
    correct cursor-based resume of partial batch."""
    cfg = _make_cfg(max_gens=1)
    sm = _make_sm(current_gen=1, phase="selfplay")
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        mocks["train"].return_value = "gen-001-trained"
        mocks["eval"].return_value = _make_sprt()
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        orch.resume()

        resets = _selfplay_resets(sm)
        assert (
            0 not in resets
        ), f"Resume mid-selfplay must NOT reset completed_games, but found resets: {resets}"
    finally:
        patch.stopall()


def test_new_gen_resets_train_current_epoch() -> None:
    """Regression Bug #3 : starting a new gen MUST reset state.train.current_epoch=0.

    Without this, current_epoch carries over from previous gen's last epoch
    (= total_epochs) and Trainer.train_generation tries to load a non-existent
    train-state-gen-N-epoch-(total-1).pt checkpoint → crash.
    """
    cfg = _make_cfg(max_gens=1)
    sm = _make_sm()
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        mocks["train"].return_value = "gen-001-trained"
        mocks["eval"].return_value = _make_sprt()
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        orch.run()

        resets = _train_current_epoch_resets(sm)
        assert (
            0 in resets
        ), f"Expected at least one sm.update(train__current_epoch=0) call, got {resets}"
    finally:
        patch.stopall()


def test_resume_mid_train_does_not_reset_current_epoch() -> None:
    """Regression Bug #3 : resuming mid-train MUST preserve state.train.current_epoch.

    The reset only happens for fresh phases, not when resuming a crashed train phase.
    """
    cfg = _make_cfg(max_gens=1)
    sm = _make_sm(current_gen=1, phase="train")
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        mocks["train"].return_value = "gen-001-trained"
        mocks["eval"].return_value = _make_sprt()
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        orch.resume()

        resets = _train_current_epoch_resets(sm)
        assert (
            0 not in resets
        ), f"Resume mid-train must NOT reset current_epoch, but found resets: {resets}"
    finally:
        patch.stopall()


def test_resume_from_eval_resets_current_epoch() -> None:
    """Resume from eval/promote (post-train) should reset for the next gen if any."""
    cfg = _make_cfg(max_gens=2)
    sm = _make_sm(current_gen=1, phase="eval")
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        mocks["train"].return_value = "gen-002-trained"
        mocks["eval"].return_value = _make_sprt()
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        orch.resume()

        resets = _train_current_epoch_resets(sm)
        # Resume from eval gen 1 = no reset for gen 1, but next gen 2 must reset
        assert 0 in resets, f"Expected reset for gen 2 fresh train phase, got {resets}"
    finally:
        patch.stopall()


def test_orchestrator_propagates_phase_runner_exceptions() -> None:
    """If a phase runner raises, orchestrator does NOT catch — propagates up."""
    cfg = _make_cfg(max_gens=1)
    sm = _make_sm()
    vm = _make_vm()
    mocks = _patch_phases()
    try:
        mocks["selfplay"].side_effect = RuntimeError("selfplay crash")
        orch = PipelineOrchestrator(config=cfg, state_manager=sm, version_manager=vm)
        with pytest.raises(RuntimeError, match="selfplay crash"):
            orch.run()
    finally:
        patch.stopall()
