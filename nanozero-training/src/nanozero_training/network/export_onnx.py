"""Export ONNX companion à côté de .npz BN-folded (Phase 12 v1.1.0+).

Le .npz produit par export_to_npz contient des weights BN-folded (cf. _fold_conv_bn).
Pour permettre le drop-in NetworkOnnx Java (~17x speedup CPU vs Vector API SIMD),
on génère un .onnx équivalent next to chaque .npz.

Architecture : NanoZeroResNetDeployment (no BN, mêmes shapes que NanoZeroResNet
post-fold) — chargé depuis le .npz puis exporté via torch.onnx.export.

Used by :
  - generate_gen0_model.py : post init
  - Trainer.train_generation : post atomic_rename gen-NNN-trained.npz
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

LOG = logging.getLogger(__name__)

# float32 minimum positive normal value : 2^-126
# Anything smaller (but non-zero) is denormal/subnormal.
_FLOAT32_MIN_NORMAL = 1.175494e-38


class _DeploymentResBlock(nn.Module):
    """ResBlock sans BN (= BN-folded dans conv weights/bias)."""

    # Class-level annotations so mypy resolves `block.conv1` as nn.Conv2d
    # (without these, nn.Module.__getattr__ returns Tensor | Module).
    conv1: nn.Conv2d
    conv2: nn.Conv2d

    def __init__(self, channels: int = 96) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + skip)


class _DeploymentResNet(nn.Module):
    """Same architecture que NanoZeroResNet mais sans BN (weights BN-folded)."""

    INPUT_CHANNELS = 119
    BODY_CHANNELS = 96
    NUM_BLOCKS = 8
    POLICY_PLANES = 73
    VALUE_HIDDEN = 64
    VALUE_WDL_CLASSES = 3  # Win / Draw / Loss (v1.5.0)
    POLICY_LOGITS = POLICY_PLANES * 8 * 8

    # Class-level annotations for mypy (cf. _DeploymentResBlock comment).
    input_conv: nn.Conv2d
    blocks: nn.ModuleList
    policy_conv: nn.Conv2d
    value_conv: nn.Conv2d
    value_fc1: nn.Linear
    value_fc2: nn.Linear

    def __init__(self, n_blocks: int = NUM_BLOCKS, channels: int = BODY_CHANNELS) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(self.INPUT_CHANNELS, channels, 3, padding=1, bias=True)
        self.blocks = nn.ModuleList([_DeploymentResBlock(channels) for _ in range(n_blocks)])
        self.policy_conv = nn.Conv2d(channels, self.POLICY_PLANES, kernel_size=1, bias=True)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.value_fc1 = nn.Linear(self.VALUE_HIDDEN, self.VALUE_HIDDEN, bias=True)
        self.value_fc2 = nn.Linear(self.VALUE_HIDDEN, self.VALUE_WDL_CLASSES, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.input_conv(x))
        for block in self.blocks:
            out = block(out)
        batch_size = out.shape[0]
        policy_nchw = self.policy_conv(out)
        policy_3d = policy_nchw.reshape(batch_size, self.POLICY_PLANES, 64)
        policy_transposed = policy_3d.transpose(1, 2)
        policy_logits = policy_transposed.reshape(batch_size, self.POLICY_LOGITS)
        value_conv_out = self.value_conv(out)
        value_flat = value_conv_out.reshape(batch_size, self.VALUE_HIDDEN)
        value_hidden = F.relu(self.value_fc1(value_flat))
        value_logits = self.value_fc2(value_hidden)  # [N, 3] WDL logits (v1.5.0)
        return policy_logits, value_logits


def _copy_layer(layer: nn.Conv2d | nn.Linear, prefix: str, d: Any) -> None:
    """Copy `prefix.weight` and `prefix.bias` from a loaded .npz into a layer.

    `layer` is built with bias=True everywhere in this file, so both weight and
    bias are non-None Parameters — assertions are tight, not defensive.
    """
    assert layer.weight is not None
    assert layer.bias is not None
    layer.weight.data.copy_(torch.from_numpy(d[f"{prefix}.weight"]))
    layer.bias.data.copy_(torch.from_numpy(d[f"{prefix}.bias"]))


def _load_npz_into_deployment(npz_path: Path, model: _DeploymentResNet) -> None:
    """Populate weights from .npz BN-folded (format export_to_npz)."""
    d = np.load(npz_path, allow_pickle=False)
    with torch.no_grad():
        _copy_layer(model.input_conv, "input_conv", d)
        for i, raw_blk in enumerate(model.blocks):
            assert isinstance(raw_blk, _DeploymentResBlock)
            _copy_layer(raw_blk.conv1, f"block_{i}.conv1", d)
            _copy_layer(raw_blk.conv2, f"block_{i}.conv2", d)
        _copy_layer(model.policy_conv, "policy_head.conv", d)
        _copy_layer(model.value_conv, "value_head.conv", d)
        _copy_layer(model.value_fc1, "value_head.fc1", d)
        _copy_layer(model.value_fc2, "value_head.fc2", d)


def _zero_denormals_in_module(model: nn.Module) -> int:
    """Zero in-place all denormal float32 weights/biases of a PyTorch module.

    Denormal float32 (0 < |x| < 1.175e-38) trigger an Intel AVX2 microcode
    handler that's ~100x slower than normal floats. A single denormal-poisoned
    weight tensor can drop ONNX Runtime inference throughput by 50x on CPU
    (observed end-to-end in nanozero self-play : 92k positions/h → 1.7k
    positions/h after 30 training epochs caused weights to drift into the
    denormal range without reaching zero).

    Mathematically harmless : denormals contribute < 1.175e-38 to any sum,
    well below float32 rounding noise of normal-magnitude weights (~1e-3).
    Equivalent to the FTZ/DAZ MXCSR bits ONNX Runtime can set via
    `session.set_denormal_as_zero`, but applied permanently to the weights.

    Called BEFORE `torch.onnx.export` so the exported file is clean. Constant
    folding inside the exporter operates on Reshape/Concat/etc. of existing
    constants — it does not synthesize new denormals from normal weights —
    so a single pre-export pass suffices.

    Args:
        model: nn.Module to clean in place (operates on `.parameters()`).

    Returns:
        Number of denormal float32 weights zeroed (across all parameters).
    """
    total = 0
    with torch.no_grad():
        for p in model.parameters():
            if p.dtype != torch.float32:
                continue
            abs_p = p.abs()
            mask = (abs_p > 0) & (abs_p < _FLOAT32_MIN_NORMAL)
            n = int(mask.sum().item())
            if n == 0:
                continue
            p[mask] = 0.0
            total += n
    return total


def _infer_arch_from_npz(npz_path: Path) -> tuple[int, int]:
    """Infère (n_blocks, channels) depuis les shapes des poids du npz.

    channels = out-channels de input_conv ([channels, 119, 3, 3]).
    n_blocks = nombre de préfixes ``block_{i}`` distincts présents.
    """
    import numpy as np

    d = np.load(npz_path)
    channels = int(d["input_conv.weight"].shape[0])
    block_ids = {
        int(k.split(".")[0].removeprefix("block_")) for k in d.files if k.startswith("block_")
    }
    n_blocks = max(block_ids) + 1 if block_ids else 0
    return n_blocks, channels


def export_onnx_companion(npz_path: Path, onnx_path: Path | None = None) -> Path:
    """Export .onnx BN-folded companion à côté du .npz.

    Args:
        npz_path: path vers .npz BN-folded (output de export_to_npz)
        onnx_path: si None, écrit à npz_path.with_suffix('.onnx')

    Returns:
        Path du .onnx exporté.

    Raises:
        FileNotFoundError: si .npz absent
        IOError: si export échoue
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f".npz absent: {npz_path}")
    onnx_path = npz_path.with_suffix(".onnx") if onnx_path is None else Path(onnx_path)

    # Infère l'architecture depuis les shapes du npz (supporte tout body, pas
    # seulement 8×96) : channels = out-dim de input_conv, n_blocks = nb de
    # préfixes block_N distincts. Permet d'exporter des réseaux rebumpés (ex.
    # 10×120) sans hardcoder les dims.
    n_blocks, channels = _infer_arch_from_npz(npz_path)
    model = _DeploymentResNet(n_blocks=n_blocks, channels=channels)
    _load_npz_into_deployment(npz_path, model)
    model.eval()

    # Pre-process : zero denormal float32 weights to avoid AVX2 microcode
    # slowdown at inference time. Done unconditionally — a clean model has no
    # denormals and the scan is cheap (~few ms on a ResNet 8x96).
    n_denormals = _zero_denormals_in_module(model)
    if n_denormals > 0:
        LOG.warning(
            "Zeroed %d denormal float32 weights before ONNX export "
            "(if ≥1%% of params, investigate training : weight decay / epochs / lr).",
            n_denormals,
        )

    dummy = torch.zeros(1, 119, 8, 8, dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["board"],
        output_names=["policy_logits", "value"],
        opset_version=17,
        dynamic_axes={
            "board": {0: "batch"},
            "policy_logits": {0: "batch"},
            "value": {0: "batch"},
        },
        do_constant_folding=True,
        # Legacy TorchScript exporter — stable, no onnxscript dependency.
        # torch >= 2.6 defaults dynamo=True which routes through the dynamo
        # exporter (requires onnxscript + onnx_ir + ...). Stay on the path
        # this code was designed for.
        dynamo=False,
    )
    LOG.info("Exported ONNX companion: %s (%d bytes)", onnx_path, onnx_path.stat().st_size)
    return onnx_path
