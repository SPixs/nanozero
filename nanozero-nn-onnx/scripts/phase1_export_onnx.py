"""Phase 1 PoC: export PyTorch NanoZeroResNet → ONNX + save reference outputs.

Usage (from nanozero-nn-onnx/ root):
    python scripts/phase1_export_onnx.py

Produit dans src/test/resources/ :
  - poc_model.onnx               : model ONNX (avec BN inclus, opset 17)
  - poc_reference_board.bin      : input float32 LE shape [1, 119, 8, 8] = 7616 floats
  - poc_reference_policy.bin     : output policy_logits float32 LE shape [1, 4672]
  - poc_reference_value.bin      : output value float32 LE shape [1]

Le format binaire LE permet une lecture trivial via ByteBuffer côté Java
(évite la dep numpy/jnumpy lourde pour parser .npz côté JVM).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

# Append nanozero-training/src to sys.path pour importer NanoZeroResNet
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "nanozero-training" / "src"))

from nanozero_training.network.resnet import NanoZeroResNet


def save_float32_le(path: Path, arr: np.ndarray) -> None:
    """Save numpy array as flat float32 little-endian binary."""
    arr.astype("<f4").tofile(path)


def main() -> int:
    out_dir = Path(__file__).resolve().parent.parent / "src/test/resources"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic seed pour reproducibility
    torch.manual_seed(42)
    model = NanoZeroResNet()
    model.eval()

    # Dummy input - random tensor pour testing
    dummy = torch.randn(1, 119, 8, 8, dtype=torch.float32)

    # Export ONNX (opset 17, dynamic batch dim)
    onnx_path = out_dir / "poc_model.onnx"
    torch.onnx.export(
        model,
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
    print(f"Exported ONNX: {onnx_path} ({onnx_path.stat().st_size} bytes)")

    # Run reference inference + save outputs as plain binary
    with torch.no_grad():
        policy, value = model(dummy)

    board_bin = out_dir / "poc_reference_board.bin"
    policy_bin = out_dir / "poc_reference_policy.bin"
    value_bin = out_dir / "poc_reference_value.bin"
    save_float32_le(board_bin, dummy.numpy())
    save_float32_le(policy_bin, policy.numpy())
    save_float32_le(value_bin, value.numpy())

    print(f"Reference input:  {board_bin} (shape {tuple(dummy.shape)}, "
          f"{dummy.numel()} floats, {board_bin.stat().st_size} bytes)")
    print(f"Reference policy: {policy_bin} (shape {tuple(policy.shape)}, "
          f"{policy.numel()} floats)")
    print(f"Reference value:  {value_bin} (shape {tuple(value.shape)}, "
          f"value={float(value[0]):.6f})")

    # Sanity verify in Python via onnxruntime
    try:
        import onnxruntime as ort  # type: ignore[import-not-found]

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_outputs = sess.run(None, {"board": dummy.numpy()})
        ort_policy, ort_value = ort_outputs[0], ort_outputs[1]
        policy_diff = float(np.max(np.abs(policy.numpy() - ort_policy)))
        value_diff = float(np.max(np.abs(value.numpy() - ort_value)))
        print(f"\nSanity check ORT Python vs PyTorch:")
        print(f"  Policy max abs diff: {policy_diff:.6e}")
        print(f"  Value  max abs diff: {value_diff:.6e}")
        tol = 1e-5
        if policy_diff < tol and value_diff < tol:
            print(f"  OK (both < {tol})")
        else:
            print(f"  WARN diff exceeds tol {tol}")
    except ImportError:
        print("\n(onnxruntime Python pas installé, sanity check skipped — install via")
        print(" `pip install onnxruntime` pour valider PyTorch == ORT Python output)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
