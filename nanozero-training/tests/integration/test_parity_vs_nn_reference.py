"""Parity tests: training.network ↔ nn Python reference (transitive proof for Java).

Strategy v1.0.0 (cf. prompt phase 1.0.0-3) — pas de subprocess Java :

By transitivity :
- nn Python ≡ nn Java (validé par tests nn v1.0.0 phase 9b sur parity-fixtures.npz)
- IF training.NanoZeroResNet ≡ nn Python reference (ce test)
- THEN training.NanoZeroResNet ≡ nn Java (la propriété recherchée).

Test direct subprocess Java différé v1.1.0+ si bullet-proof confidence requise.

Schéma fixtures nn (capturé depuis nanozero-nn/scripts/python/generate_parity_fixtures.py) :
- parity-fixtures.npz : keys = 'inputs' (100, 119, 8, 8), 'logits' (100, 4672),
  'values' (100,) — tous float32.
- parity-model.npz : 42 tenseurs poids folded + 5 _meta_* (architecture_version,
  input_plane_format, model_hash, training_step, export_date).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from nanozero_training.network.export_npz import export_to_npz
from nanozero_training.network.init import init_fixup_gamma_zero
from nanozero_training.network.resnet import NanoZeroResNet

# Path to nn fixtures (relative to repo root).
NN_FIXTURES_DIR = (
    Path(__file__).resolve().parents[3] / "nanozero-nn" / "src" / "test" / "resources" / "npz"
)
PARITY_FIXTURES = NN_FIXTURES_DIR / "parity-fixtures.npz"
PARITY_MODEL = NN_FIXTURES_DIR / "parity-model.npz"

PARITY_SEED = 42  # Doit matcher SEED_MODEL dans nn generate_parity_fixtures.py
FORWARD_TOLERANCE = 1e-6


@pytest.mark.slow()
def test_parity_forward_vs_nn_reference() -> None:
    """Forward pass on parity-fixtures.npz inputs must match expected outputs.

    Setup:
      - NanoZeroResNet training, init_fixup_gamma_zero(seed=42) == nn reference init.
      - Load parity-fixtures.npz (inputs + expected logits/values from nn Python).
      - Forward on inputs, compare with expected outputs.

    Tolerance: 1e-6 (drift numérique float32 acceptable). Si échec marginal,
    monter à 1e-5 cohérent ADR-008 nn. Échec grossier (>1e-4) = bug logique,
    pas drift.
    """
    if not PARITY_FIXTURES.exists():
        pytest.skip(f"Missing nn fixture: {PARITY_FIXTURES}")

    model = NanoZeroResNet()
    init_fixup_gamma_zero(model, seed=PARITY_SEED)
    model.eval()

    with np.load(PARITY_FIXTURES) as data:
        inputs = data["inputs"]  # (100, 119, 8, 8) float32
        expected_logits = data["logits"]  # (100, 4672) float32
        expected_values = data["values"]  # (100,) float32

    inputs_t = torch.from_numpy(inputs)

    with torch.no_grad():
        policy_logits, value = model(inputs_t)

    actual_logits = policy_logits.numpy()
    actual_values = value.numpy()

    np.testing.assert_allclose(
        actual_logits,
        expected_logits,
        rtol=0,
        atol=FORWARD_TOLERANCE,
        err_msg=(
            f"Policy logits diverge from nn Python reference (>{FORWARD_TOLERANCE} absolute).\n"
            "Probable causes: init_fixup_gamma_zero RNG order mismatch, ou divergence "
            "architecturale resnet (test_resnet_parity_state_dict_keys aurait dû échouer)."
        ),
    )
    np.testing.assert_allclose(
        actual_values,
        expected_values,
        rtol=0,
        atol=FORWARD_TOLERANCE,
        err_msg=(
            f"Value diverges from nn Python reference (>{FORWARD_TOLERANCE} absolute).\n"
            "Probable causes: même que ci-dessus, ou squeeze final value différent."
        ),
    )


@pytest.mark.slow()
def test_parity_export_vs_nn_parity_model(tmp_path: Path) -> None:
    """Export to .npz of Fixup-initialized model must match parity-model.npz bit-perfect.

    Setup:
      - NanoZeroResNet training, init_fixup_gamma_zero(seed=42).
      - export_to_npz to temp file.
      - Load both exported file et parity-model.npz, comparer :
        * All non-meta tensors identical (bit-perfect float32 — pas de tolérance).
        * _meta_model_hash identical (SHA-256 stable).
        * _meta_architecture_version identical ("resnet8x96-v1").
        * _meta_input_plane_format identical ("alphazero-119").
        * _meta_training_step identical (0).
      - Ignorer meta variables : _meta_export_date (timestamp distinct).
    """
    if not PARITY_MODEL.exists():
        pytest.skip(f"Missing nn fixture: {PARITY_MODEL}")

    model = NanoZeroResNet()
    init_fixup_gamma_zero(model, seed=PARITY_SEED)

    exported_path = tmp_path / "training-export.npz"
    export_to_npz(model, exported_path, meta={"training_step": 0})

    with np.load(exported_path) as ours, np.load(PARITY_MODEL) as ref:
        ours_keys = set(ours.files)
        ref_keys = set(ref.files)

        # Tous les tenseurs poids doivent être présents des deux côtés.
        weight_keys = {k for k in ref_keys if not k.startswith("_meta_")}
        missing_in_ours = weight_keys - ours_keys
        assert not missing_in_ours, f"Missing weight keys in our export: {missing_in_ours}"

        # Compare chaque tenseur poids bit-pour-bit.
        for key in sorted(weight_keys):
            ours_arr = ours[key]
            ref_arr = ref[key]
            assert (
                ours_arr.shape == ref_arr.shape
            ), f"Shape mismatch for {key}: ours {ours_arr.shape} vs ref {ref_arr.shape}"
            np.testing.assert_array_equal(
                ours_arr,
                ref_arr,
                err_msg=(
                    f"Tensor {key} differs from nn reference (not bit-perfect).\n"
                    "Probable causes: init_fixup_gamma_zero RNG order mismatch, "
                    "fold_conv_bn divergence (eps?), ou bug arithmétique."
                ),
            )

        # Compare metadata stables.
        stable_meta_keys = [
            "_meta_architecture_version",
            "_meta_input_plane_format",
            "_meta_training_step",
            "_meta_model_hash",
        ]
        for key in stable_meta_keys:
            assert key in ours.files, f"Missing meta key in our export: {key}"
            assert key in ref.files, f"Missing meta key in nn reference: {key}"
            ours_val = ours[key].item()
            ref_val = ref[key].item()
            assert (
                ours_val == ref_val
            ), f"Meta key {key} differs: ours {ours_val!r} vs ref {ref_val!r}"
