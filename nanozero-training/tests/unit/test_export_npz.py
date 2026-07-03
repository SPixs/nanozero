"""Tests for network/export_npz.py — fold + 42 tensors + 5 meta + SHA-256.

Phase 1.0.0-3 — implementation complète conforme ADR-002 nn (model-export).
Le test de parité bit-pour-bit vs parity-model.npz vit dans
tests/integration/test_parity_vs_nn_reference.py (slow opt-in).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from nanozero_training.network.export_npz import (
    ARCHITECTURE_VERSION,
    EXPECTED_WEIGHT_TENSOR_COUNT,
    INPUT_PLANE_FORMAT,
    _collect_export_tensors,
    _compute_weights_hash,
    _fold_conv_bn,
    export_to_npz,
)
from nanozero_training.network.init import init_fixup_gamma_zero, init_kaiming_standard
from nanozero_training.network.resnet import NanoZeroResNet

# ----- _fold_conv_bn -----


def test_fold_conv_bn_identity_bn_preserves_conv() -> None:
    """Fold(conv, identity_BN) should leave conv weights essentially unchanged.

    With BN gamma=1, beta=0, running_mean=0, running_var=1 (identity), the folded
    conv weights equal the originals (modulo small eps numerical drift).
    """
    conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True)
    bn = torch.nn.BatchNorm2d(8)
    # Force identity BN.
    torch.nn.init.ones_(bn.weight)
    torch.nn.init.zeros_(bn.bias)
    bn.running_mean.zero_()
    bn.running_var.fill_(1.0)

    w_folded, b_folded = _fold_conv_bn(conv, bn)
    expected_w = conv.weight.detach().cpu().numpy().astype(np.float32)
    expected_b = conv.bias.detach().cpu().numpy().astype(np.float32)

    # With eps != 0, scale = 1/sqrt(1+eps) ≈ 1 (tiny drift). Tolerance 1e-3.
    np.testing.assert_allclose(w_folded, expected_w, atol=1e-3)
    np.testing.assert_allclose(b_folded, expected_b, atol=1e-3)


def test_fold_conv_bn_output_dtype_float32() -> None:
    conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=True)
    bn = torch.nn.BatchNorm2d(8)
    w, b = _fold_conv_bn(conv, bn)
    assert w.dtype == np.float32
    assert b.dtype == np.float32


# ----- _collect_export_tensors -----


def test_collect_export_tensors_count_42() -> None:
    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)
    tensors = _collect_export_tensors(model)
    assert len(tensors) == EXPECTED_WEIGHT_TENSOR_COUNT == 42


def test_collect_export_tensors_keys_match_nn_naming() -> None:
    """Naming aligné parity-model.npz : input_conv.*, block_i.conv*.*, heads."""
    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)
    keys = set(_collect_export_tensors(model).keys())

    expected = {
        "input_conv.weight",
        "input_conv.bias",
        *(f"block_{i}.conv{j}.{k}" for i in range(8) for j in (1, 2) for k in ("weight", "bias")),
        "policy_head.conv.weight",
        "policy_head.conv.bias",
        "value_head.conv.weight",
        "value_head.conv.bias",
        "value_head.fc1.weight",
        "value_head.fc1.bias",
        "value_head.fc2.weight",
        "value_head.fc2.bias",
    }
    assert keys == expected, f"Missing: {expected - keys}, Extra: {keys - expected}"


def test_collect_export_tensors_all_float32() -> None:
    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)
    tensors = _collect_export_tensors(model)
    for name, arr in tensors.items():
        assert arr.dtype == np.float32, f"{name} dtype {arr.dtype} != float32"


# ----- _compute_weights_hash -----


def test_compute_weights_hash_deterministic() -> None:
    """Same tensors → same hash."""
    rng = np.random.default_rng(0)
    a = {f"t_{i}": rng.standard_normal((4, 4)).astype(np.float32) for i in range(5)}
    h1 = _compute_weights_hash(a)
    h2 = _compute_weights_hash(a)
    assert h1 == h2


def test_compute_weights_hash_independent_of_dict_order() -> None:
    """Dict insertion order should not affect hash (sorted internally)."""
    rng = np.random.default_rng(0)
    arrs = [rng.standard_normal((3,)).astype(np.float32) for _ in range(4)]
    a = {"key_a": arrs[0], "key_b": arrs[1], "key_c": arrs[2], "key_d": arrs[3]}
    b = {"key_d": arrs[3], "key_c": arrs[2], "key_b": arrs[1], "key_a": arrs[0]}
    assert _compute_weights_hash(a) == _compute_weights_hash(b)


def test_compute_weights_hash_changes_on_modification() -> None:
    rng = np.random.default_rng(0)
    a = {"t": rng.standard_normal((4,)).astype(np.float32)}
    h1 = _compute_weights_hash(a)
    a["t"] = a["t"] + np.float32(1e-3)  # modify
    h2 = _compute_weights_hash(a)
    assert h1 != h2


def test_compute_weights_hash_ignores_meta_keys() -> None:
    """Keys starting with _meta_ should be excluded from the hash."""
    rng = np.random.default_rng(0)
    base = {"t": rng.standard_normal((4,)).astype(np.float32)}
    with_meta = {**base, "_meta_extra": np.array("foo")}
    assert _compute_weights_hash(base) == _compute_weights_hash(with_meta)


# ----- export_to_npz -----


def test_export_to_npz_creates_file(tmp_path: Path) -> None:
    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)
    target = tmp_path / "model.npz"
    export_to_npz(model, target)
    assert target.exists()


def test_export_to_npz_includes_all_42_weight_tensors(tmp_path: Path) -> None:
    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)
    target = tmp_path / "model.npz"
    export_to_npz(model, target)
    with np.load(target, allow_pickle=True) as data:
        weight_keys = [k for k in data.files if not k.startswith("_meta_")]
        assert len(weight_keys) == 42


def test_export_to_npz_includes_5_required_meta(tmp_path: Path) -> None:
    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)
    target = tmp_path / "model.npz"
    export_to_npz(model, target)
    required = {
        "_meta_architecture_version",
        "_meta_input_plane_format",
        "_meta_model_hash",
        "_meta_training_step",
        "_meta_export_date",
    }
    with np.load(target, allow_pickle=True) as data:
        assert required.issubset(set(data.files))


def test_export_to_npz_meta_values(tmp_path: Path) -> None:
    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)
    target = tmp_path / "model.npz"
    export_to_npz(model, target, meta={"training_step": 0})
    with np.load(target, allow_pickle=True) as data:
        assert str(data["_meta_architecture_version"].item()) == ARCHITECTURE_VERSION
        assert str(data["_meta_input_plane_format"].item()) == INPUT_PLANE_FORMAT
        assert int(data["_meta_training_step"].item()) == 0
        export_date = str(data["_meta_export_date"].item())
        assert export_date.startswith("20")
        assert "T" in export_date
        # _meta_model_hash is a 64-char hex string (SHA-256).
        hash_str = str(data["_meta_model_hash"].item())
        assert len(hash_str) == 64
        assert all(c in "0123456789abcdef" for c in hash_str)


def test_export_to_npz_training_step_passthrough(tmp_path: Path) -> None:
    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)
    target = tmp_path / "model.npz"
    export_to_npz(model, target, meta={"training_step": 42})
    with np.load(target, allow_pickle=True) as data:
        assert int(data["_meta_training_step"].item()) == 42
        # int64 (matches nn format).
        assert data["_meta_training_step"].dtype == np.int64


def test_export_to_npz_custom_meta_passthrough(tmp_path: Path) -> None:
    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)
    target = tmp_path / "model.npz"
    export_to_npz(model, target, meta={"seed": 7, "generation": 3})
    with np.load(target, allow_pickle=True) as data:
        assert int(data["_meta_seed"].item()) == 7
        assert int(data["_meta_generation"].item()) == 3


def test_export_to_npz_hash_matches_collect(tmp_path: Path) -> None:
    """The _meta_model_hash in the .npz must match _compute_weights_hash(collected)."""
    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)
    target = tmp_path / "model.npz"
    export_to_npz(model, target)
    tensors = _collect_export_tensors(model)
    expected_hash = _compute_weights_hash(tensors)
    with np.load(target, allow_pickle=True) as data:
        assert str(data["_meta_model_hash"].item()) == expected_hash


def test_export_to_npz_fixup_init_deterministic_hash(tmp_path: Path) -> None:
    """Init Fixup avec seed=42 doit produire un hash reproductible cross-runs.

    Sanity essentielle pour la parity vs parity-model.npz : si le hash diffère
    entre 2 runs même seed, la parity bit-pour-bit ne peut pas marcher.
    """
    hashes = []
    for _ in range(2):
        model = NanoZeroResNet()
        init_fixup_gamma_zero(model, seed=42)
        target = tmp_path / "model.npz"
        export_to_npz(model, target)
        with np.load(target, allow_pickle=True) as data:
            hashes.append(str(data["_meta_model_hash"].item()))
    assert hashes[0] == hashes[1], f"Hash drift cross-runs: {hashes[0]} != {hashes[1]}"


def test_export_to_npz_export_includes_model_hash_consistent_with_meta(
    tmp_path: Path,
) -> None:
    """Le hash _meta_model_hash doit être consistent avec re-calcul des tenseurs exportés."""
    model = NanoZeroResNet()
    init_fixup_gamma_zero(model, seed=42)
    target = tmp_path / "model.npz"
    export_to_npz(model, target)
    with np.load(target, allow_pickle=True) as data:
        # Reconstituer les tenseurs depuis le .npz.
        reloaded = {k: data[k] for k in data.files if not k.startswith("_meta_")}
        reloaded_hash = _compute_weights_hash(reloaded)
        stored_hash = str(data["_meta_model_hash"].item())
        assert stored_hash == reloaded_hash


@pytest.fixture()
def fresh_model() -> NanoZeroResNet:
    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)
    return model


def test_collect_export_tensors_no_nan_inf(fresh_model: NanoZeroResNet) -> None:
    tensors = _collect_export_tensors(fresh_model)
    for name, arr in tensors.items():
        assert np.all(np.isfinite(arr)), f"NaN/Inf detected in {name}"
