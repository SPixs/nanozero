"""Training orchestrator : Trainer manages epochs, batches, optimization."""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR
from torch.utils.data import DataLoader, IterableDataset

from nanozero_training.network.export_npz import export_to_npz
from nanozero_training.train.checkpoint import (
    load_model_for_continued_training,
    load_model_for_resume,
    load_training_state,
    save_training_state,
)
from nanozero_training.train.config import TrainConfig
from nanozero_training.train.loss import LossComponents, alphazero_loss
from nanozero_training.utils.atomic_io import atomic_rename

if TYPE_CHECKING:
    from nanozero_training.monitoring.csv_writer import MetricsCSVWriter
    from nanozero_training.state.manager import RunStateManager

LOG = logging.getLogger(__name__)


def _estimate_batches(dataloader: DataLoader[dict[str, torch.Tensor]]) -> int:
    """Return number of batches per epoch, handling IterableDataset gracefully.

    For map-style datasets ``len(dataloader)`` works ; for IterableDataset
    (``LazyNpzDataset``) it raises, so we fall back to a count derived from
    ``dataset.total_samples()`` and ``dataloader.batch_size``.
    """
    dataset = dataloader.dataset
    if isinstance(dataset, IterableDataset):
        if hasattr(dataset, "total_samples"):
            n = int(dataset.total_samples())
            bs = dataloader.batch_size or 1
            return max(math.ceil(n / bs), 1)
        return 1  # unknown — scheduler degrades to constant LR for 1 step
    return max(len(dataloader), 1)


class Trainer:
    """Training orchestrator AlphaZero.

    Pattern (cf. SPEC §8) :
    - Adam optimizer + CosineAnnealingLR schedule.
    - Per-epoch checkpointing atomique (npz model + pt training state).
    - Gradient clipping Lc0-style (defensive safety net).
    - Resume support from state.train.current_epoch (phase 1.0.0-6 g).

    Workflow train_generation(gen, dataloader, models_dir, state_manager,
    abort_flag) ajouté en commit g (phase 6).
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        device: str | torch.device = "cpu",
    ) -> None:
        """Initialize Trainer.

        Args:
            model: NanoZeroResNet (ou compatible) à entraîner.
            config: TrainConfig avec hyperparams.
            device: 'cpu' / 'cuda' / torch.device. Le model est déplacé sur ce device.
        """
        self.model = model
        self.config = config
        self.device = torch.device(device)
        self.model.to(self.device)

    def make_optimizer(self) -> Optimizer:
        """Build AdamW optimizer with decoupled weight decay (Loshchilov 2017).

        AdamW remplace Adam-vanilla + L2-in-loss qui souffrait du bug "invisible
        weight blowup" documenté par Lc0 (le L2 dans la loss est divisé par le
        second-moment Adam → régularisation non uniforme). AdamW applique le
        weight_decay decoupled : ``p = p × (1 - lr × wd)`` directement, ce qui
        est mathématiquement correct.

        BN gammas/betas et biases sont exclus du weight_decay (convention
        standard ResNet : ces paramètres ne devraient pas être pénalisés).
        """
        decay_params = []
        no_decay_params = []
        for name, p in self.model.named_parameters():
            if "bn" in name.lower() or name.endswith(".bias"):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
        param_groups = [
            {"params": decay_params, "weight_decay": self.config.l2_reg},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return AdamW(param_groups, lr=self.config.learning_rate)

    def make_scheduler(
        self,
        optimizer: Optimizer,
        total_steps: int,
        warmup_steps: int = 0,
    ) -> LRScheduler:
        """Build CosineAnnealingLR scheduler avec eta_min floor.

        Le floor ``eta_min = lr × lr_min_ratio`` (default 5 %) empêche le LR
        de descendre à 0 en fin de cycle. Sans ce floor, le weight_decay
        continue à pousser les poids vers 0 sans gradient pour rattraper —
        cause documentée des 17 % denormal weights observés gen-13 (cf.
        diagnostic 2026-05-21).

        Phase A3 (2026-05-23) : si ``warmup_steps > 0``, retourne un
        ``SequentialLR(LinearLR, CosineAnnealingLR)`` qui fait un warmup
        linéaire de ``start_factor=0.01 → 1.0`` sur ``warmup_steps`` steps
        avant d'enchaîner le cosine sur ``total_steps - warmup_steps``.
        Pattern Gupta 2023 (Continual Pre-Training of LLMs) : indispensable
        quand on continue depuis un checkpoint avec un optimizer reset, sinon
        loss spike "stability gap" au démarrage.
        """
        eta_min = self.config.learning_rate * self.config.lr_min_ratio
        total = max(total_steps, 1)
        if warmup_steps <= 0:
            return CosineAnnealingLR(optimizer, T_max=total, eta_min=eta_min)

        warmup = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=max(total - warmup_steps, 1),
            eta_min=eta_min,
        )
        return SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_steps],
        )

    def train_epoch(
        self,
        dataloader: DataLoader[dict[str, torch.Tensor]],
        optimizer: Optimizer,
        scheduler: LRScheduler,
        metrics_writer: MetricsCSVWriter | None = None,
        gen: int | None = None,
        epoch: int | None = None,
    ) -> dict[str, float]:
        """Run one training epoch over dataloader.

        Args:
            dataloader: PyTorch DataLoader (consume make_train_dataloader_for_gen).
            optimizer: Adam optimizer (state_dict updated in-place).
            scheduler: LR scheduler (state_dict updated in-place).
            metrics_writer: optional MetricsCSVWriter for live monitoring CSV.
                If provided AND gen + epoch are also provided, appends 1 CSV
                row at end of epoch with mean loss components + lr + grad_norm.
                Backwards-compatible : None default = no CSV side effect.
            gen: generation number for CSV filename (required if metrics_writer).
            epoch: epoch index for CSV row (required if metrics_writer).

        Returns:
            Dict des metrics moyennes sur l'epoch :
            'policy_loss', 'value_loss', 'l2', 'total_loss', 'grad_norm'.
        """
        self.model.train()

        sum_metrics: dict[str, float] = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "l2": 0.0,
            "total_loss": 0.0,
            "grad_norm": 0.0,
        }
        n_batches = 0

        for batch in dataloader:
            # non_blocking=True : overlap du transfert H2D avec le compute (effectif
            # uniquement si pin_memory=True ET num_workers>0, sinon no-op inoffensif).
            # Évite que la boucle principale bloque sur chaque copie CPU->GPU.
            input_planes = batch["input_planes"].to(self.device, non_blocking=True)
            policy_target = batch["policy_target"].to(self.device, non_blocking=True)
            value_target = batch["value_target"].to(self.device, non_blocking=True)

            # Forward.
            policy_logits, value_pred = self.model(input_planes)

            # Loss : policy + value seulement. Le weight_decay est appliqué
            # decoupled par AdamW (cf. make_optimizer). Le terme L2-in-loss
            # historique a été retiré pour éliminer le bug Adam+L2 (Lc0).
            components: LossComponents = alphazero_loss(
                policy_logits=policy_logits,
                policy_target=policy_target,
                value_pred=value_pred,
                value_target=value_target,
                model=self.model,
                value_target_smoothing=self.config.value_target_smoothing,
            )

            # Backward.
            optimizer.zero_grad()
            components.total.backward()  # type: ignore[no-untyped-call]

            # Gradient clipping (Lc0 style — safety net).
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            # Step.
            optimizer.step()
            scheduler.step()

            # Accumulate metrics.
            sum_metrics["policy_loss"] += components.policy.item()
            sum_metrics["value_loss"] += components.value.item()
            sum_metrics["l2"] += components.l2.item()
            sum_metrics["total_loss"] += components.total.item()
            sum_metrics["grad_norm"] += float(grad_norm)
            n_batches += 1

        metrics = {k: v / max(n_batches, 1) for k, v in sum_metrics.items()}

        # Phase 10 : optional CSV append (graceful — never break training).
        if metrics_writer is not None and gen is not None and epoch is not None:
            try:
                metrics_writer.append_train_row(
                    gen=gen,
                    epoch=epoch,
                    policy_loss=metrics["policy_loss"],
                    value_loss=metrics["value_loss"],
                    l2=metrics["l2"],
                    total_loss=metrics["total_loss"],
                    lr=optimizer.param_groups[0]["lr"],
                    grad_norm=metrics["grad_norm"],
                )
            except Exception as e:
                LOG.warning("CSV append failed: %s (training continues)", e)

        return metrics

    def train_generation(  # noqa: PLR0915 — orchestration séquentielle (resume + A3 + boucle epochs)
        self,
        gen: int,
        dataloader: DataLoader[dict[str, torch.Tensor]],
        models_dir: Path,
        state_manager: RunStateManager,
        abort_flag: dict[str, bool] | None = None,
        metrics_writer: MetricsCSVWriter | None = None,
        base_model_path: Path | None = None,
    ) -> Path:
        """Train one generation pour total_epochs, avec resume + checkpointing atomique.

        Args:
            gen: index de génération (1-based).
            dataloader: DataLoader sur le replay buffer.
            models_dir: répertoire pour les checkpoints (npz + pt).
            state_manager: RunStateManager pour resilience.
            abort_flag: optional dict {requested: bool} pour graceful abort.
            base_model_path: Phase A3 (2026-05-23) — chemin vers le ``.pt`` de
                la génération précédente (typiquement
                ``train-state-gen-(N-1)-epoch-(total_epochs-1).pt``). Si fourni
                ET ``start_epoch == 0``, les poids sont chargés dans
                ``self.model`` AVANT le training loop (continued training).
                L'optimizer est volontairement reset (moments à zéro) ; combiné
                au warmup linéaire du scheduler, c'est le pattern Gupta 2023
                pour continued pretraining.

        Returns:
            Path vers le checkpoint final `gen-NNN-trained.npz`.

        Resume behavior :
            Lit state.train.current_epoch. Si > 0 :
            - Load model + optimizer + scheduler + rng depuis
              train-state-gen-NNN-epoch-(current_epoch-1).pt.
            - Reprend depuis epoch=current_epoch.
            - ``base_model_path`` est IGNORÉ (resume prime sur continued init).
            Si 0 :
            - Si ``base_model_path`` fourni : load weights (continued training).
            - Sinon : init random (from scratch).

        Notes :
            - Abort check ENTRE epochs (jamais mid-epoch v1.0.0).
            - Cleanup intermediates .npz à la fin (seul gen-NNN-trained.npz reste).
            - .pt files preserved (utiles pour ré-analyse future).
        """
        models_dir = Path(models_dir)
        models_dir.mkdir(parents=True, exist_ok=True)
        abort = abort_flag if abort_flag is not None else {"requested": False}

        # Set phase=train + current_gen si pas déjà.
        if state_manager.state.phase != "train":
            state_manager.update(
                phase="train",
                phase_started=datetime.now(timezone.utc),
                current_gen=gen,
                train__total_epochs=self.config.total_epochs,
            )

        # Initialize optimizer + scheduler.
        # Phase A3 : warmup_steps actif uniquement si continued_training=True
        # ET pas un resume (start_epoch == 0). Permet de garder les anciens
        # tests bit-pour-bit identiques (warmup=0 par défaut).
        optimizer = self.make_optimizer()
        steps_per_epoch = max(_estimate_batches(dataloader), 1)
        total_steps = self.config.total_epochs * steps_per_epoch
        start_epoch_pre = state_manager.state.train.current_epoch
        warmup_steps = 0
        if self.config.continued_training and start_epoch_pre == 0 and base_model_path is not None:
            warmup_steps = self.config.warmup_epochs * steps_per_epoch
        scheduler = self.make_scheduler(optimizer, total_steps, warmup_steps=warmup_steps)

        # Resume support (prend priorité sur continued training).
        start_epoch = state_manager.state.train.current_epoch
        if start_epoch > 0:
            prev_state_path = (
                models_dir / f"train-state-gen-{gen:03d}-epoch-{start_epoch - 1:03d}.pt"
            )
            LOG.info("Resuming training from %s (epoch %d)", prev_state_path, start_epoch)
            load_training_state(prev_state_path, optimizer, scheduler, restore_rng=True)
            load_model_for_resume(prev_state_path, self.model)
        elif base_model_path is not None:
            # Phase A3 — continued training : load weights from previous generation.
            # Optimizer reste reset (moments à zéro) — combiné au warmup linéaire,
            # c'est le pattern Gupta 2023 (Continual Pre-Training of LLMs).
            LOG.info(
                "Continued training (gen %d): loading base weights from %s "
                "(optimizer reset, %d steps linear warmup, %d steps cosine T_max)",
                gen,
                base_model_path,
                warmup_steps,
                max(total_steps - warmup_steps, 1),
            )
            # Load tolérant aux mismatches de shape : réutilise body + policy +
            # value_conv/fc1, ré-initialise le value head WDL (value_fc2 64->3).
            # Sur une gen sans changement d'archi, rien n'est skippé.
            load_model_for_continued_training(base_model_path, self.model)
        else:
            LOG.info("Starting fresh training for generation %d", gen)

        # Training loop.
        for epoch in range(start_epoch, self.config.total_epochs):
            # Check abort AVANT epoch (jamais mid-epoch v1.0.0).
            if abort.get("requested", False):
                LOG.info("Abort requested at epoch %d", epoch)
                state_manager.update(status="aborted")
                # Return last completed epoch checkpoint (or .npz suffix if none).
                if epoch > 0:
                    last_npz = models_dir / f"gen-{gen:03d}-trained-epoch-{epoch - 1:03d}.npz"
                    if last_npz.exists():
                        return last_npz
                return models_dir / f"gen-{gen:03d}-trained.npz"

            LOG.info("Epoch %d/%d", epoch + 1, self.config.total_epochs)
            metrics = self.train_epoch(
                dataloader,
                optimizer,
                scheduler,
                metrics_writer=metrics_writer,
                gen=gen,
                epoch=epoch,
            )
            LOG.info("Epoch %d metrics: %s", epoch + 1, metrics)

            # Save model .npz (engine-consumable).
            npz_path = models_dir / f"gen-{gen:03d}-trained-epoch-{epoch:03d}.npz"
            export_to_npz(self.model, npz_path, meta={"epoch": epoch, "gen": gen})

            # Save training state .pt (resume).
            state_path = models_dir / f"train-state-gen-{gen:03d}-epoch-{epoch:03d}.pt"
            save_training_state(
                path=state_path,
                optimizer=optimizer,
                scheduler=scheduler,
                torch_rng_state=torch.get_rng_state(),
                extra={
                    "model_state": self.model.state_dict(),
                    "epoch": epoch,
                    "metrics": metrics,
                },
            )

            # Update state.
            state_manager.update(
                train__current_epoch=epoch + 1,
                train__last_checkpoint=npz_path.name,
                train__optimizer_state_file=state_path.name,
            )

        # Promote final epoch as gen-NNN-trained.npz.
        last_npz = (
            models_dir / f"gen-{gen:03d}-trained-epoch-{self.config.total_epochs - 1:03d}.npz"
        )
        final_npz = models_dir / f"gen-{gen:03d}-trained.npz"
        atomic_rename(last_npz, final_npz)
        LOG.info("Final checkpoint: %s", final_npz)

        # Phase 12+ : export ONNX companion pour NetworkOnnx Java (~17x speedup SPRT
        # vs Vector API). Le fastchess_runner et uci_client picknt automatiquement
        # le .onnx si présent next to .npz.
        try:
            from nanozero_training.network.export_onnx import export_onnx_companion

            export_onnx_companion(final_npz)
        except Exception as e:
            LOG.warning("ONNX companion export failed (SPRT will use slower Vector API): %s", e)

        # Cleanup intermediate .npz (keep .pt).
        for epoch in range(self.config.total_epochs - 1):
            intermediate = models_dir / f"gen-{gen:03d}-trained-epoch-{epoch:03d}.npz"
            if intermediate.exists():
                intermediate.unlink()

        state_manager.update(train__output_model_target=final_npz.name)
        return final_npz
