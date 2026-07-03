"""Export PyTorch model to .npz format compliant ADR-002 nn (model export).

Phase 1.0.0-3 — implementation complète :
- Fold BN dans conv précédente (ADR-008 nn) pour les 17 conv qui en ont une.
- 42 tenseurs poids folded conformes SPEC-nn §6.3.
- 5 metadata obligatoires conformes SPEC-nn §6.5 :
  _meta_architecture_version, _meta_input_plane_format, _meta_model_hash,
  _meta_training_step, _meta_export_date.
- Hash SHA-256 reproduit bit-pour-bit la logique nn (sorted tensor names,
  bytes float32 little-endian via numpy.tobytes()).

Note format selfplay-data vs model-export (cf. ADR-002 training) : ce module
gère le format **model-export** (poids du réseau, cross-language Java).
Le format selfplay-data (samples) vit dans `data/npz_writer.py`.

Source de vérité : `nanozero-nn/scripts/python/nanozero_resnet.py::fold_conv_bn`
et `generate_parity_fixtures.py::export_model` + `compute_hash`.
Duplication contrôlée ADR-003.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from nanozero_training.utils.atomic_io import atomic_write_npz_uncompressed

ARCHITECTURE_VERSION = "resnet8x96-v1"
INPUT_PLANE_FORMAT = "alphazero-119"
WRITER_NAME = "nanozero-training v1.0.0"
EXPECTED_WEIGHT_TENSOR_COUNT = 42


def _tensor_to_float32_numpy(tensor: torch.Tensor) -> npt.NDArray[np.float32]:
    """Detach + CPU + float32 numpy conversion. Centralise pour mypy strict."""
    return tensor.detach().cpu().numpy().astype(np.float32)


def _fold_conv_bn(
    conv: nn.Conv2d, bn: nn.BatchNorm2d
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """Fold BN dans la conv précédente, retour (W', b') float32 (ADR-008 nn).

    Reproduit bit-pour-bit `nanozero_resnet.fold_conv_bn` (nn reference).

    En mode eval, BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta.
    Soit conv(x) = W * x + b. Alors BN(conv(x)) = W' * x + b' avec :
        scale = gamma / sqrt(var + eps)
        W'    = W * scale.view(-1, 1, 1, 1)
        b'    = (b - mean) * scale + beta
    """
    eps = bn.eps
    running_mean = bn.running_mean
    running_var = bn.running_var
    assert running_mean is not None
    assert running_var is not None

    mean = running_mean.detach()
    var = running_var.detach()
    gamma = bn.weight.detach()
    beta = bn.bias.detach()

    scale = gamma / torch.sqrt(var + eps)

    w_folded = conv.weight.detach() * scale.view(-1, 1, 1, 1)
    b_conv = conv.bias.detach() if conv.bias is not None else torch.zeros_like(beta)
    b_folded = (b_conv - mean) * scale + beta

    return (
        w_folded.cpu().numpy().astype(np.float32),
        b_folded.cpu().numpy().astype(np.float32),
    )


def _collect_export_tensors(model: nn.Module) -> dict[str, npt.NDArray[Any]]:
    """Produce les 42 tenseurs poids folded (SPEC-nn §6.3).

    Naming aligné parity-model.npz (cf. nn generate_parity_fixtures.py::export_model) :
      - input_conv.{weight,bias}        : folded avec input_bn
      - block_{i}.conv{1,2}.{weight,bias} : folded avec bn{1,2} (i in 0..7)
      - policy_head.conv.{weight,bias}  : direct (pas de BN)
      - value_head.conv.{weight,bias}   : direct
      - value_head.fc{1,2}.{weight,bias} : direct

    Total : 2 (input) + 8*4 (blocks) + 2 + 2 + 2 + 2 (heads) = 42.

    Args:
        model: NanoZeroResNet (ou compatible — accès aux attributs input_conv,
               input_bn, blocks (ModuleList de ResidualBlock avec
               conv1/bn1/conv2/bn2), policy_conv, value_conv,
               value_fc1, value_fc2).

    Raises:
        AssertionError: si on n'obtient pas exactement 42 tenseurs.
    """
    tensors: dict[str, npt.NDArray[Any]] = {}

    # Input conv : fold input_bn dans input_conv.
    w, b = _fold_conv_bn(model.input_conv, model.input_bn)  # type: ignore[arg-type]
    tensors["input_conv.weight"] = w
    tensors["input_conv.bias"] = b

    # 8 blocs résiduels.
    blocks: nn.ModuleList = model.blocks  # type: ignore[assignment]
    for i, block in enumerate(blocks):
        w1, b1 = _fold_conv_bn(block.conv1, block.bn1)  # type: ignore[arg-type]
        w2, b2 = _fold_conv_bn(block.conv2, block.bn2)  # type: ignore[arg-type]
        tensors[f"block_{i}.conv1.weight"] = w1
        tensors[f"block_{i}.conv1.bias"] = b1
        tensors[f"block_{i}.conv2.weight"] = w2
        tensors[f"block_{i}.conv2.bias"] = b2

    # Policy head (pas de BN, bias direct).
    policy_conv: nn.Conv2d = model.policy_conv  # type: ignore[assignment]
    assert policy_conv.bias is not None
    tensors["policy_head.conv.weight"] = _tensor_to_float32_numpy(policy_conv.weight)
    tensors["policy_head.conv.bias"] = _tensor_to_float32_numpy(policy_conv.bias)

    # Value head (pas de BN, bias direct).
    value_conv: nn.Conv2d = model.value_conv  # type: ignore[assignment]
    value_fc1: nn.Linear = model.value_fc1  # type: ignore[assignment]
    value_fc2: nn.Linear = model.value_fc2  # type: ignore[assignment]
    assert value_conv.bias is not None
    assert value_fc1.bias is not None
    assert value_fc2.bias is not None
    tensors["value_head.conv.weight"] = _tensor_to_float32_numpy(value_conv.weight)
    tensors["value_head.conv.bias"] = _tensor_to_float32_numpy(value_conv.bias)
    tensors["value_head.fc1.weight"] = _tensor_to_float32_numpy(value_fc1.weight)
    tensors["value_head.fc1.bias"] = _tensor_to_float32_numpy(value_fc1.bias)
    tensors["value_head.fc2.weight"] = _tensor_to_float32_numpy(value_fc2.weight)
    tensors["value_head.fc2.bias"] = _tensor_to_float32_numpy(value_fc2.bias)

    # Compte attendu = fonction du nombre de blocs (arch paramétrique) :
    # 2 (input_conv) + 4 × n_blocks (conv1/conv2 × w/b) + 2 (policy) + 6 (value).
    # 8 blocs -> 42 (archi historique) ; 10 blocs -> 50 ; etc.
    n_blocks = len(model.blocks)  # type: ignore[arg-type]
    expected = 2 + 4 * n_blocks + 2 + 6
    if len(tensors) != expected:
        raise AssertionError(
            f"Expected {expected} weight tensors for {n_blocks} blocks, got {len(tensors)}"
        )
    return tensors


def _compute_weights_hash(tensors: dict[str, npt.NDArray[Any]]) -> str:
    """SHA-256 hex sur les tenseurs triés alphabétiquement (bytes LE).

    Reproduit bit-pour-bit `nanozero_resnet/generate_parity_fixtures.compute_hash` :
      - Itère sur les clés triées alphabétiquement.
      - Skip les clés commençant par '_meta_'.
      - Update SHA-256 avec `tensor.tobytes()` (pas de clé incluse dans le hash).

    Garantit cohérence cross-language : NetworkLoader Java doit produire le
    même hash via le même algorithme (cf. SPEC-nn §6.5).
    """
    hasher = hashlib.sha256()
    for name in sorted(tensors.keys()):
        if name.startswith("_meta_"):
            continue
        hasher.update(tensors[name].tobytes())
    return hasher.hexdigest()


def export_to_npz(
    model: nn.Module,
    path: str | Path,
    meta: dict[str, Any] | None = None,
) -> None:
    """Export PyTorch model to .npz format (ADR-002 nn model-export, ADR-003 training).

    Args:
        model: NanoZeroResNet PyTorch initialisé (typically via init_kaiming_standard
               ou init_fixup_gamma_zero pour parity tests).
        path: target .npz file path.
        meta: optional extra metadata dict. Sera prefixé `_meta_<key>=<value>`,
              SAUF `training_step` qui remplace la default value
              `_meta_training_step` (int64 conforme nn).

    Format produit (cf. SPEC-nn §6) :
      - 42 tenseurs poids folded (BN absorbed dans conv).
      - 5 metadata obligatoires :
        * `_meta_architecture_version` = "resnet8x96-v1"
        * `_meta_input_plane_format`   = "alphazero-119"
        * `_meta_model_hash`           = SHA-256 hex des 42 tenseurs (triés)
        * `_meta_training_step`        = int64, default 0
        * `_meta_export_date`          = ISO 8601 UTC
      - Métadonnées custom additionnelles via `meta` paramètre.

    Atomic write via `atomic_write_npz_uncompressed` (uncompressed pour load
    speed côté Java NetworkLoader, cf. ADR-002 training distinction vs
    selfplay-data compressed).
    """
    tensors = _collect_export_tensors(model)

    # Hash sur les 42 tenseurs avant ajout des _meta_*.
    model_hash = _compute_weights_hash(tensors)

    # Training step override (default 0) — convert en int64 pour matcher nn.
    training_step = 0
    if meta is not None and "training_step" in meta:
        training_step = int(meta["training_step"])

    metadata: dict[str, npt.NDArray[Any]] = {
        "_meta_architecture_version": np.array(ARCHITECTURE_VERSION),
        "_meta_input_plane_format": np.array(INPUT_PLANE_FORMAT),
        "_meta_model_hash": np.array(model_hash),
        "_meta_training_step": np.array(training_step, dtype=np.int64),
        "_meta_export_date": np.array(datetime.now(timezone.utc).isoformat()),
    }

    # Custom meta supplémentaire (hors training_step déjà géré).
    #
    # Cross-platform dtype : np.array(int) sans dtype produit int32 sur Windows
    # et int64 sur Linux. NpzReader.java (nanozero-uci) accepte uniquement <i8
    # pour int, <f4 pour float, <U/|S pour string. Tout autre dtype fait
    # crasher le subprocess UCI au load. Détecté phase 12-deploy 2026-05-14
    # (validation W3090) — tests slow Python→Java skipés sur DevSrv JDK 21.
    #
    # Ordre des isinstance : bool AVANT int (bool est subclass de int en Python,
    # isinstance(True, int) est True). bool casté en int64 car |b1 refusé par
    # NpzReader (sémantiquement 0/1 préserve l'info).
    if meta is not None:
        for k, v in meta.items():
            if k == "training_step":
                continue  # déjà inclus avec dtype int64 explicite ci-dessus
            if isinstance(v, bool):
                metadata[f"_meta_{k}"] = np.array(int(v), dtype=np.int64)
            elif isinstance(v, int):
                metadata[f"_meta_{k}"] = np.array(v, dtype=np.int64)
            elif isinstance(v, float):
                metadata[f"_meta_{k}"] = np.array(v, dtype=np.float32)
            else:
                metadata[f"_meta_{k}"] = np.array(v)  # str → <U, bytes → |S

    # Writer name informatif (différent du hash, ignoré dans la comparaison parity).
    metadata.setdefault("_meta_writer", np.array(WRITER_NAME))

    arrays: dict[str, npt.NDArray[Any]] = {**tensors, **metadata}
    atomic_write_npz_uncompressed(path, **arrays)
