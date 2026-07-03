"""Phase runner helpers: encapsulate each phase invocation with uniform wiring (ADR-012).

Each helper takes (gen, sm, vm, cfg, abort_flag, metrics_writer) and runs ONE
phase end-to-end, updating state appropriately.

Backwards-compat : phase_runner.py does NOT modify the existing modules
(SelfplayOrchestrator, Trainer, FastchessRunner, VersionManager). It only
COMPOSES them.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace
from typing import TYPE_CHECKING

import torch

from nanozero_training.eval.fastchess_runner import FastchessRunner
from nanozero_training.eval.sprt_result import SPRTResult
from nanozero_training.network.resnet import NanoZeroResNet
from nanozero_training.selfplay.orchestrator import SelfplayOrchestrator
from nanozero_training.selfplay.uci_client import UciClient
from nanozero_training.train.dataloader import make_train_dataloader_for_gen
from nanozero_training.train.trainer import Trainer
from nanozero_training.version.manager import VersionManager
from nanozero_training.version.promotion import (
    PromoteOutcome,
    PromoteResult,
    promote_if_h1,
)

if TYPE_CHECKING:
    from nanozero_training.config.run_config import RunConfig
    from nanozero_training.monitoring.csv_writer import MetricsCSVWriter
    from nanozero_training.state.manager import RunStateManager

LOG = logging.getLogger(__name__)

_GEN_NUMBER_RE = re.compile(r"gen-(\d+)")


def _gen_number_from_name(name: str) -> int | None:
    """Extract the generation number from a model name.

    ``gen-025-promoted`` -> 25, ``gen-001-init`` -> 1, ``gen-026-trained`` -> 26.
    Returns None if the name does not match the ``gen-NNN-*`` convention.
    """
    m = _GEN_NUMBER_RE.search(name)
    return int(m.group(1)) if m else None


def run_selfplay_phase(
    gen: int,
    sm: RunStateManager,
    vm: VersionManager,
    cfg: RunConfig,
    abort_flag: dict[str, bool],
    metrics_writer: MetricsCSVWriter | None = None,
) -> None:
    """Run self-play phase for one generation.

    Uses vm.get_current() as the model for self-play.
    Writes batches into cfg.paths.datasets_dir.
    Updates sm.state.selfplay.* through SelfplayOrchestrator.
    """
    current_model = vm.get_current()
    model_path = vm.get_path_for(current_model)
    if not model_path.exists():
        raise FileNotFoundError(f"Current model not found: {model_path}")

    uci_jar = cfg.paths.resolve("uci_jar")
    if not uci_jar.exists():
        raise FileNotFoundError(f"UCI JAR not found: {uci_jar}")

    LOG.info("Phase selfplay (gen %d): model=%s", gen, current_model)

    client = UciClient()
    client.start(
        uci_jar=uci_jar,
        model_path=model_path,
        dirichlet_alpha=cfg.selfplay.dirichlet_alpha,
        dirichlet_epsilon=cfg.selfplay.dirichlet_epsilon,
        dirichlet_seed=cfg.selfplay.worker_seed,
    )

    try:
        orchestrator = SelfplayOrchestrator(
            uci_client=client,
            config=cfg.selfplay,
            state_manager=sm,
            abort_flag=abort_flag,
            datasets_dir=cfg.paths.resolve("datasets_dir"),
        )
        orchestrator.run_generation(gen=gen, target_games=cfg.selfplay.target_games_per_gen)
    finally:
        client.quit()


def run_train_phase(
    gen: int,
    sm: RunStateManager,
    vm: VersionManager,
    cfg: RunConfig,
    abort_flag: dict[str, bool],
    metrics_writer: MetricsCSVWriter | None = None,
) -> str:
    """Run training phase for one generation. Returns name of trained model.

    Phase A3 (2026-05-23) : si ``cfg.train.continued_training=True`` et gen > 1,
    cherche le ``.pt`` du checkpoint final de gen-(N-1) (
    ``train-state-gen-(N-1)-epoch-(total_epochs-1).pt``) dans ``models_dir``.
    S'il existe, le passe à ``trainer.train_generation`` qui chargera les
    poids comme init (continued training, pattern Lc0/KataGo/AlphaZero).
    Sinon (gen 1, fichier absent, ou flag désactivé) : init random (comportement
    pré-A3 strict).
    """
    current_model = vm.get_current()
    LOG.info(
        "Phase train (gen %d): base_model=%s, archi=%dx%d",
        gen,
        current_model,
        cfg.train.body_blocks,
        cfg.train.body_channels,
    )

    model = NanoZeroResNet(n_blocks=cfg.train.body_blocks, channels=cfg.train.body_channels)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(model=model, config=cfg.train, device=device)

    datasets_dir = cfg.paths.resolve("datasets_dir")
    dataloader = make_train_dataloader_for_gen(
        datasets_dir=datasets_dir,
        current_gen=gen,
        config=cfg.train,
    )

    models_dir = cfg.paths.resolve("models_dir")

    # Phase A3 — résolution du base_model_path pour continued training.
    # IMPORTANT : on repart du checkpoint final du CHAMPION (vm.get_current()),
    # PAS de gen-(N-1). Sur un cycle normal c'est identique (champion == gen
    # précédente). MAIS si la gen précédente a été REJETÉE au SPRT, le champion
    # reste l'ancien : repartir de gen-(N-1) chargerait les poids REJETÉS et
    # propagerait la régression (cf. mémoire rejected-gen-continuation-gap).
    # Le checkpoint du champion = train-state-gen-{champion_gen}-epoch-{E-1}.pt.
    base_model_path = None
    if cfg.train.continued_training and gen > 1:
        champion_gen = _gen_number_from_name(current_model)
        if champion_gen is None:
            LOG.warning(
                "Phase A3 : n° de gen introuvable dans le champion '%s' — "
                "fallback random init (warmup désactivé)",
                current_model,
            )
        else:
            prev_pt = (
                models_dir
                / f"train-state-gen-{champion_gen:03d}-epoch-{cfg.train.total_epochs - 1:03d}.pt"
            )
            if prev_pt.is_file():
                base_model_path = prev_pt
                LOG.info(
                    "Phase A3 continued training : checkpoint du champion %s -> %s "
                    "(init pour gen %d)",
                    current_model,
                    prev_pt.name,
                    gen,
                )
            else:
                LOG.warning(
                    "Phase A3 continued training activé mais checkpoint du champion "
                    "ABSENT à %s — fallback random init (warmup désactivé)",
                    prev_pt,
                )

    final_npz = trainer.train_generation(
        gen=gen,
        dataloader=dataloader,
        models_dir=models_dir,
        state_manager=sm,
        abort_flag=abort_flag,
        metrics_writer=metrics_writer,
        base_model_path=base_model_path,
    )

    trained_name = final_npz.stem
    vm.add_trained(name=trained_name, parent=current_model)
    vm.save()

    return trained_name


def run_eval_phase(
    gen: int,
    trained_name: str,
    sm: RunStateManager,
    vm: VersionManager,
    cfg: RunConfig,
    abort_flag: dict[str, bool],
    metrics_writer: MetricsCSVWriter | None = None,
) -> SPRTResult:
    """Run SPRT eval phase. Returns the SPRTResult for promotion decision."""
    baseline = vm.get_current()
    LOG.info(
        "Phase eval (gen %d): challenger=%s vs baseline=%s",
        gen,
        trained_name,
        baseline,
    )

    challenger_npz = vm.get_path_for(trained_name)
    baseline_npz = vm.get_path_for(baseline)

    pgn_path = cfg.paths.resolve("pgn_path")
    pgn_path.parent.mkdir(parents=True, exist_ok=True)

    # FastchessConfig is frozen ; rebuild with paths populated from cfg.paths.
    eval_cfg = replace(
        cfg.eval_fastchess,
        pgn_output=str(pgn_path),
        uci_jar=str(cfg.paths.resolve("uci_jar")),
    )

    sm.update(
        eval__challenger=trained_name,
        eval__baseline=baseline,
        eval__pgn_path=str(pgn_path),
        eval__last_decision=None,
    )

    runner = FastchessRunner(eval_cfg)
    return runner.run_sprt(
        challenger_npz=challenger_npz,
        baseline_npz=baseline_npz,
        state_manager=sm,
        metrics_writer=metrics_writer,
        gen=gen,
    )


def run_promote_phase(
    gen: int,
    trained_name: str,
    sprt_result: SPRTResult,
    vm: VersionManager,
) -> PromoteResult:
    """Run promotion logic based on SPRT result. Returns PromoteResult.

    Decision:
      - SPRT H1_ACCEPTED -> promote (PROMOTED outcome)
      - SPRT H0/MAX_GAMES/ERROR -> reject (REJECTED outcome, no promote)

    Logs but does not raise on rejection (training continues with same champion).
    """
    LOG.info(
        "Phase promote (gen %d): trained=%s, sprt_status=%s",
        gen,
        trained_name,
        sprt_result.status.value,
    )
    result = promote_if_h1(vm=vm, trained_name=trained_name, sprt_result=sprt_result)
    if result.outcome == PromoteOutcome.PROMOTED:
        LOG.info("Promoted: %s", result.promoted_name)
    else:
        LOG.info("Not promoted (%s): %s", result.outcome.value, result.message)
    return result
