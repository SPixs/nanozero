"""CLI entry point for nanozero-training (SPEC §12, ADR-011)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import click

from nanozero_training.config.loader import (
    compute_config_hash,
    load_config,
    merge_cli_overrides,
)
from nanozero_training.config.run_config import RunConfig

LOG = logging.getLogger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """nanozero-training: AlphaZero-style chess engine training pipeline."""
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@cli.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True),
    required=True,
    help="Path to config YAML file.",
)
@click.option(
    "--model",
    required=True,
    help="Model name to use for self-play (eg. gen-001-init).",
)
@click.option(
    "--n-games",
    type=int,
    default=None,
    help="Number of games to play (overrides target_games_per_gen).",
)
@click.option(
    "--mcts-sims",
    type=int,
    default=None,
    help="MCTS simulations per move (overrides config).",
)
@click.option(
    "--resume-from-batch",
    type=int,
    default=None,
    help="Resume from a specific batch index (advanced).",
)
def selfplay(
    config: str,
    model: str,
    n_games: int | None,
    mcts_sims: int | None,
    resume_from_batch: int | None,
) -> None:
    """Run one self-play generation (standalone)."""
    from nanozero_training.pipeline.phase_runner import run_selfplay_phase

    cfg = _load_and_merge(config, mcts_sims=mcts_sims)
    if n_games is not None:
        cfg = merge_cli_overrides(cfg, target_games_per_gen=n_games)

    sm, vm, abort_flag, writer = _bootstrap_managers(cfg, config, allow_existing=True)
    gen = (sm.state.current_gen or 0) + 1

    click.echo(f"selfplay: model={model}, config={config}, gen={gen}")
    click.echo(f"  config_hash={_compute_hash(config)}")
    if resume_from_batch is not None:
        click.echo(f"  resume_from_batch={resume_from_batch}")
    try:
        run_selfplay_phase(
            gen=gen,
            sm=sm,
            vm=vm,
            cfg=cfg,
            abort_flag=abort_flag,
            metrics_writer=writer,
        )
        click.echo("Done.")
    except Exception as e:
        sm.update(status="error", last_error=str(e))
        raise click.ClickException(str(e)) from e


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.option("--base-model", required=True, help="Baseline model for replay buffer.")
@click.option("--replay-window", type=int, default=None)
@click.option("--batch-size", type=int, default=None)
@click.option("--total-epochs", type=int, default=None)
@click.option("--resume-from-epoch", type=int, default=None)
def train(
    config: str,
    base_model: str,
    replay_window: int | None,
    batch_size: int | None,
    total_epochs: int | None,
    resume_from_epoch: int | None,
) -> None:
    """Run one training generation (standalone)."""
    from nanozero_training.pipeline.phase_runner import run_train_phase

    cfg = _load_and_merge(
        config,
        replay_window=replay_window,
        batch_size=batch_size,
        total_epochs=total_epochs,
    )

    sm, vm, abort_flag, writer = _bootstrap_managers(cfg, config, allow_existing=True)
    gen = (sm.state.current_gen or 0) + 1

    click.echo(f"train: base_model={base_model}, config={config}, gen={gen}")
    click.echo(f"  batch_size={cfg.train.batch_size}, total_epochs={cfg.train.total_epochs}")
    if resume_from_epoch is not None:
        click.echo(f"  resume_from_epoch={resume_from_epoch}")
    try:
        trained_name = run_train_phase(
            gen=gen,
            sm=sm,
            vm=vm,
            cfg=cfg,
            abort_flag=abort_flag,
            metrics_writer=writer,
        )
        click.echo(f"Trained: {trained_name}")
    except Exception as e:
        sm.update(status="error", last_error=str(e))
        raise click.ClickException(str(e)) from e


@cli.command(name="eval")
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.option("--challenger", required=True)
@click.option("--baseline", required=True)
@click.option("--time-control", "tc", default=None)
@click.option("--max-games", type=int, default=None)
def eval_cmd(
    config: str,
    challenger: str,
    baseline: str,
    tc: str | None,
    max_games: int | None,
) -> None:
    """Run SPRT eval between two models (standalone)."""
    from nanozero_training.pipeline.phase_runner import (
        run_eval_phase,
        run_promote_phase,
    )

    cfg = _load_and_merge(
        config,
        time_control=tc,
        max_games=max_games,
    )

    sm, vm, abort_flag, writer = _bootstrap_managers(cfg, config, allow_existing=True)
    gen = (sm.state.current_gen or 0) + 1

    click.echo(f"eval: challenger={challenger}, baseline={baseline}")
    click.echo(
        f"  time_control={cfg.eval_fastchess.time_control}, "
        f"max_games={cfg.eval_fastchess.max_games}"
    )
    try:
        sprt_result = run_eval_phase(
            gen=gen,
            trained_name=challenger,
            sm=sm,
            vm=vm,
            cfg=cfg,
            abort_flag=abort_flag,
            metrics_writer=writer,
        )
        click.echo(
            f"SPRT: status={sprt_result.status.value}, "
            f"games={sprt_result.games_played}, llr={sprt_result.llr}"
        )

        if click.confirm("Promote challenger if H1?", default=False):
            promote_result = run_promote_phase(
                gen=gen,
                trained_name=challenger,
                sprt_result=sprt_result,
                vm=vm,
            )
            click.echo(f"Promote: {promote_result.outcome.value}")
    except Exception as e:
        sm.update(status="error", last_error=str(e))
        raise click.ClickException(str(e)) from e


@cli.group()
def versions() -> None:
    """Manage model versions (versions.yaml)."""


@versions.command("list")
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
def versions_list(config: str) -> None:
    """List all model versions in versions.yaml."""
    from nanozero_training.version.manager import VersionManager

    cfg = load_config(config)
    versions_yaml = cfg.paths.resolve("versions_yaml")
    models_dir = cfg.paths.resolve("models_dir")

    if not versions_yaml.exists():
        click.echo(f"No versions.yaml at {versions_yaml}.")
        return

    vm = VersionManager(versions_yaml_path=versions_yaml, models_dir=models_dir)
    vm.load(auto_reconcile=False)  # inspection mode

    rows: list[list[str]] = []
    for e in vm.versions.all:
        rows.append(
            [
                e.name,
                e.type,
                "yes" if e.promoted else "no",
                e.sprt_result or "-",
                (e.created or "-")[:19],
            ]
        )

    headers = ["NAME", "TYPE", "PROMOTED", "SPRT", "CREATED"]
    if rows:
        widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    else:
        widths = [len(h) for h in headers]

    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    click.echo(fmt.format(*headers))
    click.echo(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        click.echo(fmt.format(*row))

    click.echo()
    click.echo(f"current: {vm.versions.current or '(none)'}")
    click.echo(f"latest_trained: {vm.versions.latest_trained or '(none)'}")


@versions.command("promote")
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.argument("model_name")
@click.option(
    "--force",
    is_flag=True,
    help="Required to bypass SPRT decision (manual promotion).",
)
@click.confirmation_option(
    prompt="Manual promote bypasses SPRT validation. Continue?",
)
def versions_promote(config: str, model_name: str, force: bool) -> None:
    """Manually promote a trained model (Q4 actée : fake SPRTResult + audit_note)."""
    from nanozero_training.eval.sprt_result import SPRTResult, SPRTStatus
    from nanozero_training.version.manager import VersionManager
    from nanozero_training.version.promotion import promote_if_h1

    if not force:
        raise click.ClickException("Manual promotion requires --force flag (SPRT bypass).")

    cfg = load_config(config)
    versions_yaml = cfg.paths.resolve("versions_yaml")
    models_dir = cfg.paths.resolve("models_dir")

    vm = VersionManager(versions_yaml_path=versions_yaml, models_dir=models_dir)
    vm.load()

    fake_sprt = SPRTResult(
        status=SPRTStatus.H1_ACCEPTED,
        llr=None,
        games_played=0,
        wins=0,
        losses=0,
        draws=0,
        elo_diff=None,
        raw_output="manual_override",
    )

    result = promote_if_h1(vm, model_name, fake_sprt, audit_note="manual_override")
    click.echo(f"Promotion: {result.outcome.value}")
    if result.promoted_name:
        click.echo(f"Promoted: {result.promoted_name}")
    click.echo(f"Message: {result.message}")


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
def status(config: str) -> None:
    """Show run state + versions summary (read-only)."""
    from nanozero_training.state.manager import RunStateManager
    from nanozero_training.version.manager import VersionManager

    cfg = load_config(config)
    monitoring_dir = cfg.paths.resolve("monitoring_dir")
    versions_yaml = cfg.paths.resolve("versions_yaml")
    models_dir = cfg.paths.resolve("models_dir")

    run_state_path = monitoring_dir / "run_state.yaml"
    if run_state_path.exists():
        sm = RunStateManager(monitoring_dir=monitoring_dir)
        sm.load_existing()
        click.echo("=== Run State ===")
        click.echo(f"  run_id: {sm.state.run_id}")
        click.echo(f"  status: {sm.state.status}")
        click.echo(f"  phase: {sm.state.phase}")
        click.echo(f"  current_gen: {sm.state.current_gen}")
    else:
        click.echo("(No active run -- run_state.yaml not found)")

    click.echo()

    if versions_yaml.exists():
        vm = VersionManager(versions_yaml_path=versions_yaml, models_dir=models_dir)
        vm.load(auto_reconcile=False)
        click.echo("=== Versions ===")
        click.echo(f"  current: {vm.versions.current or '(none)'}")
        click.echo(f"  latest_trained: {vm.versions.latest_trained or '(none)'}")
        click.echo(f"  total entries: {len(vm.versions.all)}")
    else:
        click.echo("(No versions.yaml)")


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.option(
    "--run-id",
    default=None,
    help="Resume a specific run_id (if multiple exist).",
)
def resume(config: str, run_id: str | None) -> None:
    """Resume an interrupted run from run_state.yaml."""
    from nanozero_training.pipeline.orchestrator import PipelineOrchestrator

    cfg = _load_and_merge(config)

    sm, vm, abort_flag, writer = _bootstrap_managers(cfg, config, allow_existing=True)

    if not sm.detect_existing_run():
        if sm.state.status == "completed":
            click.echo("Run already completed -- nothing to resume.")
            return
        if sm.state.status in ("aborted", "error"):
            if not click.confirm(f"Run status={sm.state.status!r}. Resume anyway?", default=False):
                return
            sm.update(status="in_progress")

    click.echo(
        f"Resuming run: run_id={sm.state.run_id}, "
        f"current_gen={sm.state.current_gen}, phase={sm.state.phase}"
    )
    if run_id is not None:
        click.echo(f"  --run-id={run_id} (informational ; single-run v1.0.0)")

    orch = PipelineOrchestrator(
        config=cfg,
        state_manager=sm,
        version_manager=vm,
        metrics_writer=writer,
        abort_flag=abort_flag,
    )

    try:
        orch.resume()
        click.echo("Resume completed.")
    except Exception as e:
        sm.update(status="error", last_error=str(e))
        raise click.ClickException(str(e)) from e


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.option("--max-generations", type=int, default=None)
def run(config: str, max_generations: int | None) -> None:
    """Run the full pipeline (self-play -> train -> eval -> promote, repeat)."""
    from nanozero_training.pipeline.orchestrator import PipelineOrchestrator

    cfg = _load_and_merge(config, max_generations=max_generations)

    # Q2 actée : refuse existing run (allow_existing=False).
    sm, vm, abort_flag, writer = _bootstrap_managers(cfg, config, allow_existing=False)

    click.echo(f"Starting pipeline run: max_generations={cfg.max_generations}")
    click.echo(f"Run ID: {sm.state.run_id}")
    click.echo("Press Ctrl+C to abort gracefully.")

    orch = PipelineOrchestrator(
        config=cfg,
        state_manager=sm,
        version_manager=vm,
        metrics_writer=writer,
        abort_flag=abort_flag,
    )

    try:
        orch.run(max_generations=cfg.max_generations)
        click.echo("Pipeline completed.")
    except Exception as e:
        sm.update(status="error", last_error=str(e))
        raise click.ClickException(str(e)) from e


@cli.command()
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.option("--port", type=int, default=None)
@click.option("--host", default=None)
def monitor(config: str, port: int | None, host: str | None) -> None:
    """Launch the monitoring dashboard (Flask + SSE, ADR-006)."""
    from nanozero_training.monitoring.flask_app import create_app

    cfg = _load_and_merge(config, port=port, host=host)

    if not cfg.monitor.enabled:
        raise click.ClickException("Monitoring disabled in config (monitor.enabled=false).")

    click.echo(f"Starting dashboard at http://{cfg.monitor.host}:{cfg.monitor.port}/")
    click.echo(f"Config: {config}")
    click.echo("Press Ctrl+C to stop.")

    app = create_app(cfg)
    # threaded=True required for SSE (each long-polling connection needs a thread).
    app.run(
        host=cfg.monitor.host,
        port=cfg.monitor.port,
        threaded=True,
        debug=False,
    )


@cli.command("generate-gen0")
@click.option("--config", "-c", type=click.Path(exists=True), required=True)
@click.option("--seed", type=int, default=42)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Override output path (default: paths.models_dir/gen-001-init.npz).",
)
def generate_gen0(config: str, seed: int, output: str | None) -> None:
    """Generate the initial (gen-0) model and register it in versions.yaml."""
    import subprocess
    import sys

    from nanozero_training.version.manager import VersionManager

    cfg = load_config(config)
    models_dir = cfg.paths.resolve("models_dir")
    models_dir.mkdir(parents=True, exist_ok=True)

    init_name = "gen-001-init"
    output_path = Path(output) if output else models_dir / f"{init_name}.npz"

    script = Path(__file__).resolve().parents[2] / "scripts" / "generate_gen0_model.py"
    if not script.exists():
        raise click.ClickException(f"generate_gen0_model.py not found at {script}")

    cmd = [
        sys.executable,
        str(script),
        "--seed",
        str(seed),
        "--output",
        str(output_path),
    ]
    click.echo(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise click.ClickException(
            f"generate_gen0_model.py failed: {result.stderr or result.stdout}"
        )

    versions_yaml = cfg.paths.resolve("versions_yaml")
    vm = VersionManager(versions_yaml_path=versions_yaml, models_dir=models_dir)
    if versions_yaml.exists():
        vm.load(auto_reconcile=False)
    else:
        vm.create_new()
    vm.add_init(name=init_name, seed=seed, init_method="kaiming_standard")
    vm.save()

    click.echo(f"Generated {output_path}")
    click.echo(f"Registered {init_name} in {versions_yaml}")


def _load_and_merge(config_path: str, **overrides: Any) -> RunConfig:
    cfg = load_config(config_path)
    return merge_cli_overrides(cfg, **overrides)


def _bootstrap_managers(
    cfg: RunConfig,
    config_path: str,
    allow_existing: bool = False,
) -> tuple[Any, Any, dict[str, bool], Any]:
    """Initialize RunStateManager + VersionManager + abort_flag + optional writer.

    Args:
        cfg: loaded RunConfig.
        config_path: original YAML path (for config_hash + persist in run_state).
        allow_existing: if False, refuse to start if existing run in progress
            (run command). If True, allow continuation (selfplay/train/eval
            standalone, resume).

    Returns:
        (sm, vm, abort_flag, metrics_writer | None) tuple.

    Raises:
        click.ClickException: if allow_existing=False and a run is already in progress.
    """
    import uuid

    from nanozero_training.config.loader import compute_config_hash
    from nanozero_training.monitoring.csv_writer import MetricsCSVWriter
    from nanozero_training.state.manager import RunStateManager
    from nanozero_training.state.signals import install_signal_handlers
    from nanozero_training.version.manager import VersionManager

    monitoring_dir = cfg.paths.resolve("monitoring_dir")
    versions_yaml = cfg.paths.resolve("versions_yaml")
    models_dir = cfg.paths.resolve("models_dir")

    sm = RunStateManager(monitoring_dir=monitoring_dir)
    if sm.detect_existing_run():
        if not allow_existing:
            raise click.ClickException(
                "An existing run is in progress. Use 'resume' to continue, "
                "or archive/delete the state first."
            )
        sm.load_existing()
    else:
        run_id = cfg.run_id or f"run-{uuid.uuid4().hex[:8]}"
        with Path(config_path).open("r", encoding="utf-8") as f:
            yaml_content = f.read()
        config_hash = compute_config_hash(yaml_content)
        sm.create_new(
            run_id=run_id,
            config_path=config_path,
            config_hash=config_hash,
            max_generations=cfg.max_generations,
            target_games_per_gen=cfg.selfplay.target_games_per_gen,
        )

    vm = VersionManager(versions_yaml_path=versions_yaml, models_dir=models_dir)
    if versions_yaml.exists():
        vm.load()
    else:
        vm.create_new()

    abort_flag = install_signal_handlers()

    writer = MetricsCSVWriter(monitoring_dir=monitoring_dir) if cfg.monitor.enabled else None

    return sm, vm, abort_flag, writer


def _compute_hash(config_path: str) -> str:
    with Path(config_path).open("r", encoding="utf-8") as f:
        return compute_config_hash(f.read())[:16]


def main() -> None:
    """Entry point invoked by pyproject scripts (nanozero-training)."""
    cli()


if __name__ == "__main__":
    main()
