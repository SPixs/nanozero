"""Tests skeleton phase 1.0.0-1 — network/{resnet, init, export_npz}.

Tests de parité Java/Python complets phase 1.0.0-3.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from nanozero_training.network.export_npz import export_to_npz
from nanozero_training.network.init import init_kaiming_standard
from nanozero_training.network.resnet import NanoZeroResNet


def test_resnet_forward_shape() -> None:
    """Forward produit (policy [N, 4672], value_logits [N, 3]) WDL (v1.5.0)."""
    model = NanoZeroResNet()
    model.eval()  # BN en mode eval (sinon batch=2 erre les stats)
    x = torch.randn(2, NanoZeroResNet.INPUT_CHANNELS, 8, 8)
    policy, value = model(x)
    assert policy.shape == (2, NanoZeroResNet.POLICY_LOGITS)
    # Value head WDL : 3 logits Win/Draw/Loss, NON bornés (pas de tanh).
    assert value.shape == (2, NanoZeroResNet.VALUE_WDL_CLASSES)
    assert policy.dtype == torch.float32
    assert value.dtype == torch.float32


def test_resnet_forward_batch_size_1() -> None:
    """Forward fonctionne avec batch_size=1 (cas inférence MCTS)."""
    model = NanoZeroResNet()
    model.eval()
    x = torch.randn(1, NanoZeroResNet.INPUT_CHANNELS, 8, 8)
    policy, value = model(x)
    assert policy.shape == (1, NanoZeroResNet.POLICY_LOGITS)
    assert value.shape == (1, NanoZeroResNet.VALUE_WDL_CLASSES)


def test_resnet_param_count_8x96() -> None:
    """Sanity check : architecture conforme SPEC (8 blocs x 96 channels)."""
    model = NanoZeroResNet()
    assert len(model.blocks) == NanoZeroResNet.NUM_BLOCKS == 8
    assert model.input_conv.out_channels == NanoZeroResNet.BODY_CHANNELS == 96
    assert model.policy_conv.out_channels == NanoZeroResNet.POLICY_PLANES == 73


def test_init_kaiming_reproducible() -> None:
    """Deux modèles init avec même seed produisent params identiques."""
    model_a = NanoZeroResNet()
    model_b = NanoZeroResNet()
    init_kaiming_standard(model_a, seed=42)
    init_kaiming_standard(model_b, seed=42)
    for (n_a, p_a), (n_b, p_b) in zip(
        model_a.named_parameters(), model_b.named_parameters(), strict=True
    ):
        assert n_a == n_b
        assert torch.equal(p_a, p_b), f"Mismatch on {n_a}"


def test_init_kaiming_different_seeds() -> None:
    """Deux modèles init avec seeds différents → params différents."""
    model_a = NanoZeroResNet()
    model_b = NanoZeroResNet()
    init_kaiming_standard(model_a, seed=42)
    init_kaiming_standard(model_b, seed=43)
    diffs = [
        not torch.equal(p_a, p_b)
        for p_a, p_b in zip(model_a.parameters(), model_b.parameters(), strict=True)
    ]
    # Au moins quelques params doivent différer (probabilité 0 d'avoir l'identité).
    assert any(diffs)


def test_init_kaiming_bn_default() -> None:
    """Init BN standard : gamma=1, beta=0 (vs Fixup gamma=0 dans nn)."""
    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            assert torch.equal(
                m.weight, torch.ones_like(m.weight)
            ), "BN gamma should be 1 (standard init), not 0 (Fixup)"
            assert torch.equal(m.bias, torch.zeros_like(m.bias))


def test_export_to_npz_creates_file(tmp_path: Path) -> None:
    """Export produit un .npz loadable contenant au moins les metadata."""
    model = NanoZeroResNet()
    init_kaiming_standard(model, seed=42)
    target = tmp_path / "test-model.npz"
    export_to_npz(model, target, meta={"training_step": 0, "seed": 42})
    assert target.exists()
    data = np.load(target, allow_pickle=True)
    assert len(data.files) > 0
    assert "_meta_export_date" in data.files
    assert "_meta_seed" in data.files
    assert "_meta_training_step" in data.files
    # Sanity sur le contenu des metadata custom (item() pour 0-d arrays).
    assert int(data["_meta_seed"].item()) == 42
    assert int(data["_meta_training_step"].item()) == 0


# Tests obsolètes phase 1 supprimés en phase 1.0.0-3 :
#  - test_export_to_npz_contains_state_dict_keys (stub naming dots->underscores
#    remplacé par fold BN + naming aligné parity-model.npz, ce test ne s'applique
#    plus). Voir tests/unit/test_export_npz.py pour les vérifications complètes
#    sur les 42 tenseurs et 5 metadata.
#  - test_pytorch_to_nn_naming_skeleton (la fonction _pytorch_to_nn_naming a
#    été supprimée — le mapping est désormais stateful via fold).
