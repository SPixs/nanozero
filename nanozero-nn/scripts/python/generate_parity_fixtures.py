"""Génère les fixtures phase 9 pour le test de parité Java vs PyTorch.

Produit deux fichiers consommés par le test Java de phase 9b :
- src/test/resources/parity-model.npz   : modèle conforme §6 (42 tenseurs
  poids BN-folded + 5 _meta_*), chargeable par NetworkLoader.
- src/test/resources/parity-fixtures.npz : 100 inputs random + outputs PyTorch
  de référence (logits, values).

Usage :
    cd nanozero-nn
    python3 scripts/python/generate_parity_fixtures.py

Reproductibilité totale : seed 42 pour init modèle, seed 42 pour inputs random,
export_date figé. Le hash _meta_model_hash écrit dans parity-model.npz est
recalculable bit-pour-bit côté Java par NetworkLoader.computeWeightsHash
(testé phase 8).

Convention de hash §6.5 : SHA-256 sur les 42 tenseurs de poids
EXCLUSIVEMENT (les noms commençant par "_meta_" sont exclus), triés
alphabétiquement, bytes float32 little-endian via numpy.tobytes().
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from nanozero_resnet import NanoZeroResNet, fold_conv_bn, init_with_seed

# ---------------------------------------------------------------------------------
# Constantes (toutes figées pour reproductibilité)
# ---------------------------------------------------------------------------------

SEED_MODEL = 42
SEED_INPUTS = 42
NUM_FIXTURES = 100
EXPORT_DATE = "2026-05-08T12:00:00Z"
TRAINING_STEP = 0  # pas d'entraînement réel, fixture utilitaire

OUT_MODEL = SCRIPT_DIR.parents[1] / "src" / "test" / "resources" / "npz" / "parity-model.npz"
OUT_FIXTURES = SCRIPT_DIR.parents[1] / "src" / "test" / "resources" / "npz" / "parity-fixtures.npz"


def _wdl_to_value(value_logits: "torch.Tensor") -> "torch.Tensor":
    """Convertit les logits WDL [N, 3] (Win/Draw/Loss) en value scalaire [N].

    value = P(Win) - P(Loss) ∈ [-1, 1] via softmax. C'est EXACTEMENT ce que
    calcule l'inference Java (NetworkVectorApi / NetworkOnnx) : la fixture de
    référence doit donc stocker ce scalaire, pas les 3 logits (v1.5.0).
    """
    p = torch.softmax(value_logits, dim=1)  # [N, 3]
    return p[:, 0] - p[:, 2]


# ---------------------------------------------------------------------------------
# Étape 1-3 : modèle, fold, export parity-model.npz
# ---------------------------------------------------------------------------------


def export_model(model: NanoZeroResNet) -> dict[str, np.ndarray]:
    """Exporte les 42 tenseurs §6.3 + 5 _meta_* (cf. §6.5).

    Fold BN dans les 17 conv qui en ont une (input_conv + 8 × conv1 + 8 × conv2).
    Conv heads (policy, value) et FC : weight/bias directs (pas de BN).
    """
    tensors: dict[str, np.ndarray] = {}

    # Input conv : fold input_bn dans input_conv.
    w, b = fold_conv_bn(model.input_conv, model.input_bn)
    tensors["input_conv.weight"] = w
    tensors["input_conv.bias"] = b

    # 8 blocs résiduels.
    for i, block in enumerate(model.blocks):
        w1, b1 = fold_conv_bn(block.conv1, block.bn1)
        w2, b2 = fold_conv_bn(block.conv2, block.bn2)
        tensors[f"block_{i}.conv1.weight"] = w1
        tensors[f"block_{i}.conv1.bias"] = b1
        tensors[f"block_{i}.conv2.weight"] = w2
        tensors[f"block_{i}.conv2.bias"] = b2

    # Policy head (pas de BN, biais direct).
    tensors["policy_head.conv.weight"] = (
        model.policy_conv.weight.detach().cpu().numpy().astype(np.float32)
    )
    tensors["policy_head.conv.bias"] = (
        model.policy_conv.bias.detach().cpu().numpy().astype(np.float32)
    )

    # Value head (pas de BN, biais direct).
    tensors["value_head.conv.weight"] = (
        model.value_conv.weight.detach().cpu().numpy().astype(np.float32)
    )
    tensors["value_head.conv.bias"] = (
        model.value_conv.bias.detach().cpu().numpy().astype(np.float32)
    )
    tensors["value_head.fc1.weight"] = (
        model.value_fc1.weight.detach().cpu().numpy().astype(np.float32)
    )
    tensors["value_head.fc1.bias"] = (
        model.value_fc1.bias.detach().cpu().numpy().astype(np.float32)
    )
    tensors["value_head.fc2.weight"] = (
        model.value_fc2.weight.detach().cpu().numpy().astype(np.float32)
    )
    tensors["value_head.fc2.bias"] = (
        model.value_fc2.bias.detach().cpu().numpy().astype(np.float32)
    )

    if len(tensors) != 42:
        raise AssertionError(f"Expected 42 weight tensors, got {len(tensors)}")
    return tensors


def compute_hash(tensors: dict[str, np.ndarray]) -> str:
    """SHA-256 hex sur les 42 tenseurs triés alphabétiquement (bytes LE)."""
    hasher = hashlib.sha256()
    for name in sorted(tensors.keys()):
        if name.startswith("_meta_"):
            continue
        hasher.update(tensors[name].tobytes())
    return hasher.hexdigest()


# ---------------------------------------------------------------------------------
# Étape 4-7 : inputs random + forward Python + sanity + export parity-fixtures.npz
# ---------------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("Phase 9a — generate_parity_fixtures.py")
    print("=" * 70)

    # Étape 1 : construire le modèle.
    model = NanoZeroResNet()
    init_with_seed(model, seed=SEED_MODEL)
    model.eval()
    print(f"\n[1/8] Modèle NanoZeroResNet construit, init seed={SEED_MODEL}, eval mode.")

    # Étape 2 : sanity activations par couche (Fixup init doit borner ~ const).
    print("\n[2/8] Sanity activations Fixup (gamma_bn2=0) :")
    rng_sanity = np.random.default_rng(0)
    sanity_inputs = rng_sanity.standard_normal(size=(8, 119, 8, 8)).astype(np.float32)
    with torch.no_grad():
        h = F.relu(model.input_bn(model.input_conv(torch.from_numpy(sanity_inputs))))
        std_after_input = h.std().item()
        print(f"  apres input  : std={std_after_input:.3f}")
        max_block_std = std_after_input
        for i, block in enumerate(model.blocks):
            h = block(h)
            s = h.std().item()
            max_block_std = max(max_block_std, s)
            print(f"  apres block {i}: std={s:.3f}")
    if max_block_std > 3.0:
        raise AssertionError(
            f"std activations dépasse 3.0 (max={max_block_std:.3f}) → "
            "init Fixup probablement cassée (bn2.weight mal référencée ?)"
        )
    print(f"  max std observé = {max_block_std:.3f} (seuil sanity = 3.0)")

    # Étape 3 : fold + export parity-model.npz.
    print("\n[3/8] Fold BN + export parity-model.npz :")
    tensors = export_model(model)
    model_hash = compute_hash(tensors)
    print(f"  42 tenseurs de poids générés.")
    print(f"  Hash SHA-256 (sorted, hors _meta_) : {model_hash}")

    # Sanity sur les magnitudes des poids folded (§5.2.2 : |w| < 100).
    for name, arr in tensors.items():
        max_abs = float(np.max(np.abs(arr)))
        if max_abs >= 100.0:
            raise AssertionError(
                f"Poids {name} dépasse |w| >= 100 (max_abs={max_abs:.2f}). "
                "NetworkLoader.validateValues rejettera le fixture."
            )

    tensors["_meta_architecture_version"] = np.array("resnet8x96-v1")
    tensors["_meta_input_plane_format"] = np.array("alphazero-119")
    tensors["_meta_model_hash"] = np.array(model_hash)
    tensors["_meta_training_step"] = np.array(TRAINING_STEP, dtype=np.int64)
    tensors["_meta_export_date"] = np.array(EXPORT_DATE)

    OUT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_MODEL, **tensors)
    size_kb = os.path.getsize(OUT_MODEL) / 1024
    print(f"  parity-model.npz : {size_kb:.1f} KiB ({len(tensors)} entrées)")

    # Étape 4 : inputs random.
    print(f"\n[4/8] Génération de {NUM_FIXTURES} inputs random (seed={SEED_INPUTS}) :")
    rng = np.random.default_rng(SEED_INPUTS)
    inputs = rng.standard_normal(size=(NUM_FIXTURES, 119, 8, 8)).astype(np.float32)
    print(f"  inputs shape={inputs.shape} dtype={inputs.dtype}")

    # Étape 5 : forward Python sur le modèle NON-folded.
    print("\n[5/8] Forward PyTorch (eval, no_grad) :")
    with torch.no_grad():
        x = torch.from_numpy(inputs)
        policy_t, value_logits_t = model(x)
        values_t = _wdl_to_value(value_logits_t)  # [N] = P(W) - P(L), WDL v1.5.0
    logits = policy_t.numpy().astype(np.float32)  # [100, 4672]
    values = values_t.numpy().astype(np.float32)  # [100] = P(W) - P(L)
    print(f"  logits shape={logits.shape} dtype={logits.dtype}")
    print(f"  values shape={values.shape} dtype={values.dtype}")

    # Étape 6 : sanity Python-side.
    print("\n[6/8] Sanity Python-side :")

    # 6a. Déterminisme : re-forward → égalité bit-pour-bit.
    with torch.no_grad():
        policy_t2, value_logits_t2 = model(torch.from_numpy(inputs))
        values_t2 = _wdl_to_value(value_logits_t2)
    if not np.array_equal(logits, policy_t2.numpy().astype(np.float32)):
        raise AssertionError("Re-forward logits diffère bit-pour-bit (non déterminisme).")
    if not np.array_equal(values, values_t2.numpy().astype(np.float32)):
        raise AssertionError("Re-forward values diffère bit-pour-bit (non déterminisme).")
    print("  6a. Déterminisme bit-pour-bit confirmé sur 2 forwards.")

    # 6b. Aucun NaN / Inf.
    if not np.all(np.isfinite(logits)):
        raise AssertionError("NaN/Inf détecté dans logits.")
    if not np.all(np.isfinite(values)):
        raise AssertionError("NaN/Inf détecté dans values.")
    print("  6b. Aucun NaN/Inf dans logits ou values.")

    # 6c. values strictement dans [-1, 1] (post-tanh).
    if values.min() < -1.0 or values.max() > 1.0:
        raise AssertionError(
            f"values hors [-1, 1] : min={values.min():.4f} max={values.max():.4f}"
        )
    print(f"  6c. values ∈ [{values.min():.4f}, {values.max():.4f}] ⊂ [-1, 1].")

    # 6d. logits non-dégénérés (std > 0.01 sur les 100 × 4672 valeurs).
    logits_std = float(logits.std())
    if logits_std < 0.01:
        raise AssertionError(
            f"logits std={logits_std:.4f} < 0.01 → modèle trivialement constant ?"
        )
    print(
        f"  6d. logits non-dégénérés : min={logits.min():.3f} max={logits.max():.3f} "
        f"mean={logits.mean():.3f} std={logits_std:.3f}"
    )
    print(
        f"  values stats : min={values.min():.4f} max={values.max():.4f} "
        f"mean={values.mean():.4f} std={values.std():.4f}"
    )

    # Étape 7 : export parity-fixtures.npz.
    print("\n[7/8] Export parity-fixtures.npz :")
    np.savez_compressed(
        OUT_FIXTURES,
        inputs=inputs,
        logits=logits,
        values=values,
    )
    size_mb = os.path.getsize(OUT_FIXTURES) / (1024 * 1024)
    print(f"  parity-fixtures.npz : {size_mb:.2f} MiB")

    # Étape 8 : récap.
    print("\n[8/8] Récap :")
    print(f"  Hash modèle (à matcher côté Java) : {model_hash}")
    print(f"  parity-model.npz    : {os.path.getsize(OUT_MODEL) / 1024:.1f} KiB")
    print(f"  parity-fixtures.npz : {size_mb:.2f} MiB")
    if size_mb > 20.0:
        print(
            f"\n  /!\\ Taille parity-fixtures.npz ({size_mb:.1f} MiB) > 20 MiB. "
            "Considérer génération à la demande au lieu d'un commit."
        )
    print("\n[OK] Phase 9a fixtures générées.")


if __name__ == "__main__":
    main()
