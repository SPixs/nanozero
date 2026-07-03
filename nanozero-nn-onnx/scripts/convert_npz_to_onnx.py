"""Convert un .npz BN-folded existant → .onnx BN-folded (drop-in pour NetworkOnnx Java).

Le .npz produit par `nanozero-training.export_to_npz` contient déjà les weights BN-folded
(via `_fold_conv_bn`). On les charge dans un PyTorch DeploymentResNet (no BN) puis exporte
en ONNX.

Usage :
    python scripts/convert_npz_to_onnx.py --npz path/to/model.npz [--onnx path/to/model.onnx]

Si --onnx omis, écrit next to .npz avec extension .onnx.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

# Add scripts/ dir au sys.path pour importer NanoZeroResNetDeployment
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase2a_export_bn_folded_onnx import NanoZeroResNetDeployment


def load_npz_into_deployment(npz_path: Path, model: NanoZeroResNetDeployment) -> None:
    """Populate DeploymentResNet weights depuis un .npz BN-folded existant."""
    d = np.load(npz_path, allow_pickle=False)
    with torch.no_grad():
        model.input_conv.weight.copy_(torch.from_numpy(d["input_conv.weight"]))
        model.input_conv.bias.copy_(torch.from_numpy(d["input_conv.bias"]))
        for i in range(model.NUM_BLOCKS):
            blk = model.blocks[i]
            blk.conv1.weight.copy_(torch.from_numpy(d[f"block_{i}.conv1.weight"]))
            blk.conv1.bias.copy_(torch.from_numpy(d[f"block_{i}.conv1.bias"]))
            blk.conv2.weight.copy_(torch.from_numpy(d[f"block_{i}.conv2.weight"]))
            blk.conv2.bias.copy_(torch.from_numpy(d[f"block_{i}.conv2.bias"]))
        model.policy_conv.weight.copy_(torch.from_numpy(d["policy_head.conv.weight"]))
        model.policy_conv.bias.copy_(torch.from_numpy(d["policy_head.conv.bias"]))
        model.value_conv.weight.copy_(torch.from_numpy(d["value_head.conv.weight"]))
        model.value_conv.bias.copy_(torch.from_numpy(d["value_head.conv.bias"]))
        model.value_fc1.weight.copy_(torch.from_numpy(d["value_head.fc1.weight"]))
        model.value_fc1.bias.copy_(torch.from_numpy(d["value_head.fc1.bias"]))
        model.value_fc2.weight.copy_(torch.from_numpy(d["value_head.fc2.weight"]))
        model.value_fc2.bias.copy_(torch.from_numpy(d["value_head.fc2.bias"]))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--npz", type=Path, required=True, help="Input .npz file (BN-folded)")
    parser.add_argument("--onnx", type=Path, default=None, help="Output .onnx (default: <npz>.onnx)")
    args = parser.parse_args()

    if not args.npz.exists():
        print(f"ERROR: .npz not found: {args.npz}")
        return 1
    onnx_out = args.onnx if args.onnx else args.npz.with_suffix(".onnx")

    print(f"Loading {args.npz} into DeploymentResNet ...")
    model = NanoZeroResNetDeployment()
    load_npz_into_deployment(args.npz, model)
    model.eval()

    dummy = torch.zeros(1, 119, 8, 8, dtype=torch.float32)
    print(f"Exporting → {onnx_out}")
    torch.onnx.export(
        model,
        dummy,
        onnx_out,
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
    size_mb = onnx_out.stat().st_size / (1024 * 1024)

    # Verify externalized weights file (.onnx.data) si présent
    data_path = onnx_out.with_name(onnx_out.name + ".data")
    if data_path.exists():
        data_mb = data_path.stat().st_size / (1024 * 1024)
        print(f"Done. .onnx={size_mb:.2f} MB + .onnx.data={data_mb:.2f} MB (external weights)")
    else:
        print(f"Done. .onnx={size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
