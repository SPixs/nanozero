"""Génère le fixture phase 8 : test-realistic.npz conforme §6 SPEC-nn.

À exécuter une seule fois pour produire src/test/resources/npz/test-realistic.npz.
Le fixture est figé (PRNG seed=42, sigma=0.05, biais=0, export_date figée) pour
reproductibilité bit-pour-bit du hash _meta_model_hash entre Python et Java.

Usage :
    cd nanozero-nn
    python3 scripts/python/generate_test_realistic_npz.py

Sortie : nanozero-nn/src/test/resources/npz/test-realistic.npz

Conventions critiques (à miroirer côté Java NetworkLoader.computeWeightsHash) :
- Le hash SHA-256 est calculé sur la concaténation des bytes des 42 tenseurs
  de poids EXCLUSIVEMENT (les noms commençant par "_meta_" sont exclus du
  hash), triés alphabétiquement par nom.
- Chaque tenseur est sérialisé en float32 little-endian via numpy.tobytes()
  (équivalent à un ByteBuffer LE Java).
- Le hash est calculé AVANT d'ajouter les _meta_* au dict — cohérent avec le
  script de référence §6.5.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

# Reproductibilité totale.
SEED = 42
SIGMA_CONV = 0.05
SIGMA_FC = 0.05
TRAINING_STEP = 1_000_000
EXPORT_DATE = "2026-05-08T12:00:00Z"

OUT_PATH = Path(__file__).resolve().parents[2] / "src" / "test" / "resources" / "npz" / "test-realistic.npz"


def main() -> None:
    rng = np.random.default_rng(SEED)
    tensors: dict[str, np.ndarray] = {}

    # Input conv (96, 119, 3, 3) + bias (96,) — biais à 0 par convention fixture.
    tensors["input_conv.weight"] = (rng.standard_normal((96, 119, 3, 3)) * SIGMA_CONV).astype(np.float32)
    tensors["input_conv.bias"] = np.zeros((96,), dtype=np.float32)

    # 8 blocs résiduels.
    for i in range(8):
        tensors[f"block_{i}.conv1.weight"] = (rng.standard_normal((96, 96, 3, 3)) * SIGMA_CONV).astype(np.float32)
        tensors[f"block_{i}.conv1.bias"] = np.zeros((96,), dtype=np.float32)
        tensors[f"block_{i}.conv2.weight"] = (rng.standard_normal((96, 96, 3, 3)) * SIGMA_CONV).astype(np.float32)
        tensors[f"block_{i}.conv2.bias"] = np.zeros((96,), dtype=np.float32)

    # Policy head conv 1x1 (73, 96, 1, 1) + bias (73,).
    tensors["policy_head.conv.weight"] = (rng.standard_normal((73, 96, 1, 1)) * SIGMA_CONV).astype(np.float32)
    tensors["policy_head.conv.bias"] = np.zeros((73,), dtype=np.float32)

    # Value head WDL v1.5.0 : conv 1x1 (1, 96, 1, 1) + bias (1,) puis FC 64x64 et FC 3x64 (W/D/L).
    tensors["value_head.conv.weight"] = (rng.standard_normal((1, 96, 1, 1)) * SIGMA_CONV).astype(np.float32)
    tensors["value_head.conv.bias"] = np.zeros((1,), dtype=np.float32)
    tensors["value_head.fc1.weight"] = (rng.standard_normal((64, 64)) * SIGMA_FC).astype(np.float32)
    tensors["value_head.fc1.bias"] = np.zeros((64,), dtype=np.float32)
    tensors["value_head.fc2.weight"] = (rng.standard_normal((3, 64)) * SIGMA_FC).astype(np.float32)
    tensors["value_head.fc2.bias"] = np.zeros((3,), dtype=np.float32)

    # Vérification : 42 tenseurs de poids exactement.
    assert len(tensors) == 42, f"Expected 42 weight tensors, got {len(tensors)}"

    # Hash SHA-256 sur les 42 tenseurs sortés alphabétiquement, en bytes
    # natifs little-endian via tobytes() (équivalent à ByteBuffer LE côté Java).
    hasher = hashlib.sha256()
    for name in sorted(tensors.keys()):
        # tobytes() retourne les bytes dans l'ordre C row-major en little-endian
        # pour les arrays float32 sur x86 (descripteur '<f4' dans le header .npy).
        hasher.update(tensors[name].tobytes())
    model_hash = hasher.hexdigest()
    print(f"Computed hash on {len(tensors)} weight tensors (sorted): {model_hash}")

    # Ajout des 5 _meta_* APRÈS le calcul du hash (les méta NE sont PAS hashés).
    tensors["_meta_architecture_version"] = np.array("resnet8x96-v1")
    tensors["_meta_input_plane_format"] = np.array("alphazero-119")
    tensors["_meta_model_hash"] = np.array(model_hash)
    tensors["_meta_training_step"] = np.array(TRAINING_STEP, dtype=np.int64)
    tensors["_meta_export_date"] = np.array(EXPORT_DATE)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_PATH, **tensors)

    size_kb = os.path.getsize(OUT_PATH) / 1024
    print(f"Wrote {OUT_PATH} ({size_kb:.1f} KiB, {len(tensors)} entries = 42 weights + 5 _meta_*)")


if __name__ == "__main__":
    main()
