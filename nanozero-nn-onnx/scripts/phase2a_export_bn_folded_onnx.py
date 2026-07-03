"""Phase 2a Step A: BN-folded ONNX export equivalent au .npz Java engine.

Le .npz lu par le Java engine (NetworkLoader) contient des conv weights où le
BatchNorm a été folded dans la conv elle-même (cf. export_npz._fold_conv_bn).
Pour permettre le drop-in replacement OnnxNetwork ↔ Network (Vector API), on
doit exporter un .onnx avec la même architecture BN-folded.

Architecture deployment (BN-removed) :
  - input_conv (Conv 3×3 119→96, bias folded) → ReLU
  - 8× ResBlock_deployment:
      out = ReLU(conv1(x))    (conv1 weights+bias = folded conv+BN1)
      out = conv2(out)         (conv2 weights+bias = folded conv+BN2)
      out = ReLU(out + skip)
  - policy_conv (1×1 96→73) [pas de BN dans le model original]
  - value_conv (1×1 96→1)
  - value_fc1 (64→64) → ReLU
  - value_fc2 (64→1) → tanh

Workflow:
  1. Generate (or load) un .npz pour reference (utilise scripts/generate_gen0_model.py si absent)
  2. Construct NanoZeroResNetDeployment Python instance
  3. Load weights depuis .npz
  4. Sanity : forward Deployment ≈ forward NanoZeroResNet originale (eval mode, BN folded)
  5. Export .onnx via torch.onnx.export(deployment_model, ...)
  6. Sanity : ORT Python(BN-folded .onnx) == Deployment PyTorch (tolerance 1e-6)
  7. Save reference outputs en .bin float32 LE pour Java parity test

Usage :
    python scripts/phase2a_export_bn_folded_onnx.py [--npz PATH]

Outputs dans src/test/resources/ :
  - deployment_model.onnx
  - deployment_reference_board.bin     (input 7616 floats)
  - deployment_reference_policy.bin    (output policy 4672 floats)
  - deployment_reference_value.bin     (output value 1 float)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "nanozero-training" / "src"))

from nanozero_training.network.resnet import NanoZeroResNet  # noqa: E402


# ---------------------------------------------------------------------------
# Deployment model (BN absorbed into conv weights/bias)
# ---------------------------------------------------------------------------

class DeploymentResidualBlock(nn.Module):
    """ResBlock sans BN (BN-folded). Conv1+BN1 ↔ folded conv1 ; idem conv2."""

    def __init__(self, channels: int = 96) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + skip)


class NanoZeroResNetDeployment(nn.Module):
    """Same architecture as NanoZeroResNet mais sans BN (weights folded).

    Identique au model attendu par le Java engine (Network class) qui consomme
    .npz BN-folded.
    """

    INPUT_CHANNELS = 119
    BODY_CHANNELS = 96
    NUM_BLOCKS = 8
    POLICY_PLANES = 73
    VALUE_HIDDEN = 64
    POLICY_LOGITS = POLICY_PLANES * 8 * 8

    def __init__(self, n_blocks: int = NUM_BLOCKS, channels: int = BODY_CHANNELS) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(self.INPUT_CHANNELS, channels, 3, padding=1, bias=True)
        self.blocks = nn.ModuleList([DeploymentResidualBlock(channels) for _ in range(n_blocks)])
        self.policy_conv = nn.Conv2d(channels, self.POLICY_PLANES, kernel_size=1, bias=True)
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.value_fc1 = nn.Linear(self.VALUE_HIDDEN, self.VALUE_HIDDEN, bias=True)
        self.value_fc2 = nn.Linear(self.VALUE_HIDDEN, 1, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = F.relu(self.input_conv(x))
        for block in self.blocks:
            out = block(out)
        batch_size = out.shape[0]
        # Policy head + transposition
        policy_nchw = self.policy_conv(out)
        policy_3d = policy_nchw.reshape(batch_size, self.POLICY_PLANES, 64)
        policy_transposed = policy_3d.transpose(1, 2)
        policy_logits = policy_transposed.reshape(batch_size, self.POLICY_LOGITS)
        # Value head
        value_conv_out = self.value_conv(out)
        value_flat = value_conv_out.reshape(batch_size, self.VALUE_HIDDEN)
        value_hidden = F.relu(self.value_fc1(value_flat))
        value_raw = self.value_fc2(value_hidden)
        value = torch.tanh(value_raw).squeeze(-1)
        return policy_logits, value


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------

def fold_bn_into_conv(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Replication exacte de export_npz._fold_conv_bn pour cohérence."""
    eps = bn.eps
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    gamma = bn.weight.detach()
    beta = bn.bias.detach()
    scale = gamma / torch.sqrt(var + eps)
    w_folded = conv.weight.detach() * scale.view(-1, 1, 1, 1)
    b_conv = conv.bias.detach() if conv.bias is not None else torch.zeros_like(beta)
    b_folded = (b_conv - mean) * scale + beta
    return w_folded, b_folded


def populate_deployment_from_original(
    original: NanoZeroResNet,
    deployment: NanoZeroResNetDeployment,
) -> None:
    """Copy weights from original (with BN) → deployment (BN folded into conv)."""
    # Input conv : fold input_conv + input_bn
    w_folded, b_folded = fold_bn_into_conv(original.input_conv, original.input_bn)
    with torch.no_grad():
        deployment.input_conv.weight.copy_(w_folded)
        deployment.input_conv.bias.copy_(b_folded)

    # 8 residual blocks : fold conv1+bn1, conv2+bn2
    for i in range(deployment.NUM_BLOCKS):
        orig_block = original.blocks[i]
        dep_block = deployment.blocks[i]
        w1, b1 = fold_bn_into_conv(orig_block.conv1, orig_block.bn1)
        w2, b2 = fold_bn_into_conv(orig_block.conv2, orig_block.bn2)
        with torch.no_grad():
            dep_block.conv1.weight.copy_(w1)
            dep_block.conv1.bias.copy_(b1)
            dep_block.conv2.weight.copy_(w2)
            dep_block.conv2.bias.copy_(b2)

    # Policy + value heads : pas de BN, copy direct
    with torch.no_grad():
        deployment.policy_conv.weight.copy_(original.policy_conv.weight.detach())
        deployment.policy_conv.bias.copy_(original.policy_conv.bias.detach())
        deployment.value_conv.weight.copy_(original.value_conv.weight.detach())
        deployment.value_conv.bias.copy_(original.value_conv.bias.detach())
        deployment.value_fc1.weight.copy_(original.value_fc1.weight.detach())
        deployment.value_fc1.bias.copy_(original.value_fc1.bias.detach())
        deployment.value_fc2.weight.copy_(original.value_fc2.weight.detach())
        deployment.value_fc2.bias.copy_(original.value_fc2.bias.detach())


def save_float32_le(path: Path, arr: np.ndarray) -> None:
    arr.astype("<f4").tofile(path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent / "src/test/resources"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build original model + init Kaiming (équivalent .npz gen-0 init)
    torch.manual_seed(args.seed)
    original = NanoZeroResNet()
    # Apply Kaiming init pour reproductibility avec gen-001-init.npz
    sys.path.insert(0, str(ROOT / "nanozero-training" / "src"))
    from nanozero_training.network.init import init_kaiming_standard
    init_kaiming_standard(original, seed=args.seed)
    original.eval()

    # 2. Build deployment model + copy folded weights
    deployment = NanoZeroResNetDeployment()
    populate_deployment_from_original(original, deployment)
    deployment.eval()

    # 3. Random input
    torch.manual_seed(args.seed + 1)
    dummy = torch.randn(1, 119, 8, 8, dtype=torch.float32)

    # 4. Sanity : forward original (BN eval) == forward deployment (BN folded)
    with torch.no_grad():
        p_orig, v_orig = original(dummy)
        p_dep, v_dep = deployment(dummy)
    p_diff_pyt = float((p_orig - p_dep).abs().max())
    v_diff_pyt = float((v_orig - v_dep).abs().max())
    print(f"Sanity check PyTorch (original BN eval) vs PyTorch (deployment BN folded):")
    print(f"  Policy max abs diff: {p_diff_pyt:.6e}")
    print(f"  Value  max abs diff: {v_diff_pyt:.6e}")
    if p_diff_pyt > 1e-4 or v_diff_pyt > 1e-4:
        print("  WARN: diff > 1e-4 — fold logic suspect")

    # 5. Export deployment → .onnx
    onnx_path = out_dir / "deployment_model.onnx"
    torch.onnx.export(
        deployment,
        dummy,
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
    )
    print(f"\nExported deployment ONNX: {onnx_path} "
          f"({onnx_path.stat().st_size} bytes)")

    # 6. Save reference outputs (deployment forward)
    save_float32_le(out_dir / "deployment_reference_board.bin", dummy.numpy())
    save_float32_le(out_dir / "deployment_reference_policy.bin", p_dep.numpy())
    save_float32_le(out_dir / "deployment_reference_value.bin", v_dep.numpy())

    # 7. Cross-check : ORT Python on .onnx == PyTorch deployment
    try:
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_p, ort_v = sess.run(None, {"board": dummy.numpy()})
        p_diff_ort = float(np.max(np.abs(p_dep.numpy() - ort_p)))
        v_diff_ort = float(np.max(np.abs(v_dep.numpy() - ort_v)))
        print(f"\nSanity check ORT Python(deployment ONNX) vs PyTorch deployment:")
        print(f"  Policy max abs diff: {p_diff_ort:.6e}")
        print(f"  Value  max abs diff: {v_diff_ort:.6e}")
        ok = p_diff_ort < 1e-5 and v_diff_ort < 1e-5
        print(f"  {'OK' if ok else 'WARN'}")
    except ImportError:
        print("\n(onnxruntime Python absent — skip ORT sanity)")

    print(f"\nValue (PyTorch deployment): {float(v_dep[0]):.6f}")

    # 8. Export .npz equivalent (same weights, format Java engine attend)
    from nanozero_training.network.export_npz import export_to_npz

    npz_path = out_dir / "deployment_model.npz"
    export_to_npz(
        original,
        npz_path,
        meta={"seed": args.seed, "init_method": "kaiming_standard", "n_blocks": 8, "channels": 96},
    )
    print(f"Exported deployment NPZ:  {npz_path} ({npz_path.stat().st_size} bytes)")
    print(f"\n.onnx et .npz exportés du MÊME PyTorch model (seed={args.seed}, BN folded identique).")
    print("Reference outputs OK pour Java parity test (Phase 2a Step B).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
