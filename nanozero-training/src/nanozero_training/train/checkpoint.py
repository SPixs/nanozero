"""Checkpoint save/load pour resume support (ADR-009).

Deux formats de fichiers par checkpoint :
1. Model export .npz : produit via export_to_npz (ADR-002 nn naming, consumable
   par engine Java). Pour engine inference + checkpoints intermédiaires per-epoch.
2. Training state .pt : optimizer state_dict + scheduler state_dict + RNG state
   + full model state_dict (via `extra={"model_state": ...}`).
   Pour resume training (PyTorch-internal, pas consommé par Java).

Au resume à epoch K :
- Load optimizer + scheduler + RNG state depuis train-state-gen-NNN-epoch-(K-1).pt.
- Load model.state_dict() depuis le même .pt via extra["model_state"]
  (PAS depuis .npz qui a folded BN, lossy pour training).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from nanozero_training.utils.atomic_io import atomic_write_torch

LOG = logging.getLogger(__name__)


def save_training_state(
    path: str | Path,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    torch_rng_state: torch.Tensor | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Atomically save training state pour resume.

    Args:
        path: target .pt file.
        optimizer: PyTorch optimizer (state_dict sauvé).
        scheduler: optional LR scheduler.
        torch_rng_state: optional RNG state (typically torch.get_rng_state()).
        extra: optional dict de champs supplémentaires (typiquement
               {"model_state": model.state_dict(), "epoch": K, "metrics": {...}}).
    """
    state: dict[str, Any] = {
        "optimizer": optimizer.state_dict(),
    }
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if torch_rng_state is not None:
        state["torch_rng_state"] = torch_rng_state
    if extra is not None:
        state["extra"] = extra
    atomic_write_torch(state, path)


def load_training_state(
    path: str | Path,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """Load training state dans optimizer + scheduler + RNG.

    Args:
        path: source .pt file.
        optimizer: PyTorch optimizer (state restauré in-place).
        scheduler: optional LR scheduler (restauré si présent).
        restore_rng: si True, restore torch RNG state.

    Returns:
        dict avec au moins 'optimizer' key, optionnellement 'scheduler',
        'torch_rng_state', 'extra'.

    Raises:
        FileNotFoundError: si path absent.
        KeyError: si 'optimizer' key manquante.
    """
    state: dict[str, Any] = torch.load(path, weights_only=False)

    if "optimizer" not in state:
        raise KeyError(f"Missing 'optimizer' key in {path}")
    optimizer.load_state_dict(state["optimizer"])

    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])

    if restore_rng and "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])

    return state


def load_model_for_resume(
    train_state_path: str | Path,
    model: nn.Module,
    map_location: str | torch.device = "cpu",
) -> None:
    """Load model weights pour resume depuis un train-state-*.pt file.

    Note : NOT depuis un .npz file (qui a folded BN, lossy pour training resume).
    Le .pt file contient le full PyTorch state_dict dans extra['model_state'].

    Args:
        train_state_path: source .pt file.
        model: model PyTorch (state_dict chargé in-place).
        map_location: device pour tensor placement.

    Raises:
        FileNotFoundError: si path absent.
        KeyError: si extra.model_state manquant.
    """
    state = torch.load(train_state_path, weights_only=False, map_location=map_location)
    if "extra" not in state or "model_state" not in state.get("extra", {}):
        raise KeyError(f"Missing extra.model_state in {train_state_path}")
    model.load_state_dict(state["extra"]["model_state"])


def load_model_for_continued_training(
    train_state_path: str | Path,
    model: nn.Module,
    map_location: str | torch.device = "cpu",
) -> tuple[list[str], list[str]]:
    """Load PARTIEL des poids d'une génération précédente (Phase A3 tolérante).

    Comme ``load_model_for_resume`` MAIS tolère les mismatches de shape : charge
    uniquement les tenseurs dont le nom ET la shape correspondent, et laisse les
    autres à leur initialisation courante (random). C'est le pattern de transfert
    nécessaire au switch WDL (v1.5.0) : body ResNet + policy head + value_conv +
    value_fc1 sont réutilisés (shapes identiques), seul ``value_fc2`` (64->1 dans
    l'ancien réseau scalaire, 64->3 dans le WDL) est ré-initialisé.

    Sur une génération SANS changement d'archi, tous les tenseurs matchent →
    comportement identique à ``load_model_for_resume`` (rien de skippé).

    Args:
        train_state_path: source .pt (extra['model_state']).
        model: model PyTorch (state_dict chargé in-place, partiellement).
        map_location: device.

    Returns:
        (loaded, skipped) : noms des tenseurs chargés et ré-initialisés.

    Raises:
        FileNotFoundError: si path absent.
        KeyError: si extra.model_state manquant.
    """
    state = torch.load(train_state_path, weights_only=False, map_location=map_location)
    if "extra" not in state or "model_state" not in state.get("extra", {}):
        raise KeyError(f"Missing extra.model_state in {train_state_path}")
    base_sd = state["extra"]["model_state"]
    model_sd = model.state_dict()
    filtered = {k: v for k, v in base_sd.items() if k in model_sd and model_sd[k].shape == v.shape}
    loaded = sorted(filtered.keys())
    skipped = sorted(set(model_sd.keys()) - set(filtered.keys()))
    model.load_state_dict(filtered, strict=False)
    if skipped:
        LOG.warning(
            "Continued training (load partiel) : %d/%d tenseurs chargés, "
            "%d ré-initialisés (mismatch shape — ex. value head WDL) : %s",
            len(loaded),
            len(model_sd),
            len(skipped),
            skipped,
        )
    return loaded, skipped
