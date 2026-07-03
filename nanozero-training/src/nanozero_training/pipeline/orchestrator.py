"""Pipeline orchestrator : multi-generation loop (ADR-012).

State machine implicite via run_state.yaml :
  - state.phase = "idle"     -> start selfplay for current_gen+1
  - state.phase = "selfplay" -> resume selfplay (or restart from completed_games)
  - state.phase = "train"    -> resume train (skip selfplay)
  - state.phase = "eval"     -> resume eval (restart SPRT from scratch v1.0.0)
  - state.phase = "promote"  -> idempotent promote retry

Resume logic:
  - state.current_gen < max_generations -> continue.
  - For current_gen, dispatch on state.phase.

Workflow run() for new run:
  for gen in 1..max_generations:
    state.update(phase="selfplay", current_gen=gen)
    run_selfplay_phase(gen, ...)
    state.update(phase="train")
    trained_name = run_train_phase(gen, ...)
    state.update(phase="eval")
    sprt_result = run_eval_phase(gen, trained_name, ...)
    state.update(phase="promote")
    promote_result = run_promote_phase(...)
    state.update(phase="idle")  # gen complete
  state.update(status="completed")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from nanozero_training.eval.sprt_result import SPRTResult, SPRTStatus
from nanozero_training.pipeline.phase_runner import (
    run_eval_phase,
    run_promote_phase,
    run_selfplay_phase,
    run_train_phase,
)

if TYPE_CHECKING:
    from nanozero_training.config.run_config import RunConfig
    from nanozero_training.monitoring.csv_writer import MetricsCSVWriter
    from nanozero_training.state.manager import RunStateManager
    from nanozero_training.version.manager import VersionManager

LOG = logging.getLogger(__name__)

_PHASES: tuple[str, ...] = ("selfplay", "train", "eval", "promote")


class PipelineOrchestrator:
    """Orchestrate the full training pipeline across multiple generations (ADR-012).

    Wires together (without modifying):
      - SelfplayOrchestrator (phase 5)
      - Trainer (phase 6)
      - FastchessRunner (phase 7)
      - VersionManager + promote_if_h1 (phase 8)
      - MetricsCSVWriter (phase 10)

    State persistence : RunStateManager (phase 1). Resume support via state.phase
    + state.current_gen.

    Usage:
        orch = PipelineOrchestrator(cfg, sm, vm, metrics_writer, abort_flag)
        orch.run(max_generations=10)
        # OR
        orch.resume()  # reads state.current_gen + state.phase, continues
    """

    def __init__(
        self,
        config: RunConfig,
        state_manager: RunStateManager,
        version_manager: VersionManager,
        metrics_writer: MetricsCSVWriter | None = None,
        abort_flag: dict[str, bool] | None = None,
    ) -> None:
        self.config = config
        self.sm = state_manager
        self.vm = version_manager
        self.metrics_writer = metrics_writer
        self.abort_flag = abort_flag if abort_flag is not None else {"requested": False}

    def run(self, max_generations: int | None = None) -> None:
        """Run from current state to max_generations.

        Args:
            max_generations: target total generations. If None, uses config.max_generations.
        """
        target = max_generations if max_generations is not None else self.config.max_generations

        start_gen = self.sm.state.current_gen + 1 if self.sm.state.current_gen > 0 else 1

        LOG.info("Pipeline run: start_gen=%d target=%d", start_gen, target)

        for gen in range(start_gen, target + 1):
            if self._check_abort():
                LOG.info("Abort requested before gen %d -- stopping", gen)
                self.sm.update(status="aborted")
                return

            self._run_one_generation(gen)

        if not self._check_abort():
            self.sm.update(status="completed")
            LOG.info("Pipeline complete: %d generations done", target)

    def resume(self) -> None:
        """Resume from current state.

        Reads state.phase + state.current_gen and continues at that point.
        After current gen completes (or skipped if already complete), continues
        through remaining generations up to config.max_generations.
        """
        gen = self.sm.state.current_gen
        phase = self.sm.state.phase
        LOG.info("Resume: gen=%d phase=%s", gen, phase)

        if phase in _PHASES and gen > 0:
            # Resume current generation from where it stopped.
            self._run_one_generation(gen, resume_phase=phase)

        # Continue remaining generations from the next one.
        self.run(max_generations=self.config.max_generations)

    def _run_one_generation(
        self,
        gen: int,
        resume_phase: str | None = None,
    ) -> None:
        """Execute all 4 phases for one generation.

        Args:
            gen: generation number.
            resume_phase: if set, skip ahead to this phase (resume case).
        """
        start_idx = 0
        if resume_phase is not None:
            try:
                start_idx = _PHASES.index(resume_phase)
            except ValueError:
                LOG.warning("Unknown resume_phase %r, starting from selfplay", resume_phase)
                start_idx = 0

        # Reset per-phase state unless resuming mid-phase. Without this, state carries over
        # from previous gen and corrupts the new gen's pipeline:
        #
        # - train.current_epoch carries over → Trainer tries to load non-existent
        #   train-state-gen-N-epoch-(total-1).pt checkpoint for new gen N → crash (Bug #3).
        # - selfplay.completed_games carries over → Selfplay orchestrator sees completed >= target
        #   and SKIPS the whole selfplay phase silently → new gen trained on stale data (Bug #6,
        #   silent corruption — observed in Phase 12 prod gen 2).
        # - eval state is reset by FastchessRunner.start() itself so we don't touch it here.
        if resume_phase != "selfplay":
            self.sm.update(
                selfplay__completed_games=0,
                selfplay__completed_batches=0,
                selfplay__current_batch_idx=0,
                selfplay__current_batch_games=0,
            )
        if resume_phase != "train":
            self.sm.update(train__current_epoch=0)

        trained_name: str | None = None
        sprt_result: SPRTResult | None = None

        for phase_name in _PHASES[start_idx:]:
            if self._check_abort():
                LOG.info("Abort requested at phase %s", phase_name)
                self.sm.update(status="aborted")
                return

            if phase_name == "selfplay":
                self._enter_phase("selfplay", gen)
                run_selfplay_phase(
                    gen=gen,
                    sm=self.sm,
                    vm=self.vm,
                    cfg=self.config,
                    abort_flag=self.abort_flag,
                    metrics_writer=self.metrics_writer,
                )
            elif phase_name == "train":
                self._enter_phase("train", gen)
                trained_name = run_train_phase(
                    gen=gen,
                    sm=self.sm,
                    vm=self.vm,
                    cfg=self.config,
                    abort_flag=self.abort_flag,
                    metrics_writer=self.metrics_writer,
                )
            elif phase_name == "eval":
                self._enter_phase("eval", gen)
                trained_name = trained_name or self._resume_trained_name()
                sprt_result = run_eval_phase(
                    gen=gen,
                    trained_name=trained_name,
                    sm=self.sm,
                    vm=self.vm,
                    cfg=self.config,
                    abort_flag=self.abort_flag,
                    metrics_writer=self.metrics_writer,
                )
            elif phase_name == "promote":
                self._enter_phase("promote", gen)
                trained_name = trained_name or self._resume_trained_name()
                sprt_result = sprt_result or self._resume_sprt_result()
                run_promote_phase(
                    gen=gen,
                    trained_name=trained_name,
                    sprt_result=sprt_result,
                    vm=self.vm,
                )

        # Generation complete -- reset phase to idle.
        self.sm.update(phase="idle")

    def _resume_trained_name(self) -> str:
        """Recover trained_name from versions.yaml on resume."""
        trained = self.vm.get_latest_trained()
        if trained is None:
            raise RuntimeError("Cannot resume eval/promote : no latest_trained in versions.yaml")
        return trained

    def _resume_sprt_result(self) -> SPRTResult:
        """Recover minimal SPRTResult from state on resume.

        Limited info : we only have last_decision string + games_played_at_last_save.
        Other fields (llr/wins/losses) are not persisted in run_state.
        """
        last_dec = self.sm.state.eval.last_decision
        if last_dec is None:
            raise RuntimeError("Cannot resume promote : state.eval.last_decision is None")
        return SPRTResult(
            status=SPRTStatus(last_dec),
            llr=None,
            games_played=self.sm.state.eval.games_played_at_last_save,
            wins=0,
            losses=0,
            draws=0,
            elo_diff=None,
        )

    def _enter_phase(self, phase: str, gen: int) -> None:
        """Set state.phase + phase_started + current_gen atomically."""
        self.sm.update(
            phase=phase,
            phase_started=datetime.now(timezone.utc),
            current_gen=gen,
        )

    def _check_abort(self) -> bool:
        return self.abort_flag.get("requested", False)
