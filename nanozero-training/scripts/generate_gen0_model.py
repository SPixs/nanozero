"""Generate gen-001-init.npz — random init for nanozero-training pipeline.

Phase 1.0.0-3 : production-ready CLI standalone.

Run:
    poetry run python scripts/generate_gen0_model.py \\
        --seed 42 --output S:/nano2/models/gen-001-init.npz

Le .npz produit est conforme ADR-002 nn (model-export) : 42 tenseurs poids
folded + 5 metadata obligatoires + custom meta seed/init_method/generation.
Chargeable par Java NetworkLoader.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from nanozero_training.network.export_npz import export_to_npz
from nanozero_training.network.init import init_kaiming_standard
from nanozero_training.network.resnet import NanoZeroResNet


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate gen-001-init.npz for nanozero-training pipeline."
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default: 42)")
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output .npz path (eg. S:/nano2/models/gen-001-init.npz)",
    )
    parser.add_argument(
        "--n-blocks", type=int, default=8, help="Number of residual blocks (default: 8)"
    )
    parser.add_argument("--channels", type=int, default=96, help="Conv channels (default: 96)")
    args = parser.parse_args()

    print(
        f"Generating gen0 model with seed={args.seed}, n_blocks={args.n_blocks}, "
        f"channels={args.channels}...",
        file=sys.stderr,
    )
    model = NanoZeroResNet(n_blocks=args.n_blocks, channels=args.channels)
    init_kaiming_standard(model, seed=args.seed)

    print(f"Exporting to {args.output}...", file=sys.stderr)
    export_to_npz(
        model,
        args.output,
        meta={
            "training_step": 0,
            "seed": args.seed,
            "init_method": "kaiming_standard",
            "n_blocks": args.n_blocks,
            "channels": args.channels,
            "generation": 1,
        },
    )
    # Vérification post-export — sanity check sur le .npz produit.
    print("Verifying export...", file=sys.stderr)
    with np.load(args.output, allow_pickle=True) as data:
        weight_count = sum(1 for k in data.files if not k.startswith("_meta_"))
        print(f"  Weight tensors:  {weight_count}", file=sys.stderr)
        print(
            f"  Architecture:    {data['_meta_architecture_version'].item()}",
            file=sys.stderr,
        )
        print(
            f"  Input format:    {data['_meta_input_plane_format'].item()}",
            file=sys.stderr,
        )
        print(
            f"  Weights SHA-256: {data['_meta_model_hash'].item()}",
            file=sys.stderr,
        )

    print(f"Done. Path: {args.output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
