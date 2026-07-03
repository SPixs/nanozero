"""Unit tests for train/loss.py — value head WDL (v1.5.0)."""

from __future__ import annotations

import pytest
import torch
from nanozero_training.train.loss import (
    LossComponents,
    _z_to_wdl_target,
    alphazero_loss,
)
from torch import nn


class _MiniModel(nn.Module):
    """Tiny model for fast loss tests (named modules to expose 'bn' in param names).

    NanoZeroResNet a des noms 'input_bn.weight', 'blocks.0.bn1.weight' etc.
    On reproduit le pattern de naming via attributs explicites.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(119, 4, kernel_size=1, bias=True)
        self.bn = nn.BatchNorm2d(4)
        self.fc = nn.Linear(4 * 8 * 8, 4672 + 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover (not used)
        x = self.conv(x)
        x = self.bn(x)
        return self.fc(x.flatten(1))


def _make_mini_model() -> nn.Module:
    """Tiny model for fast loss tests (named modules)."""
    return _MiniModel()


def _make_inputs(batch_size: int = 2):
    """Build random valid inputs for loss tests (value head WDL)."""
    policy_logits = torch.randn(batch_size, 4672, requires_grad=True)
    policy_target = torch.zeros(batch_size, 4672)
    policy_target[:, 0] = 1.0  # one-hot at index 0
    value_pred = torch.randn(batch_size, 3, requires_grad=True)  # WDL logits [N, 3]
    value_target = torch.zeros(batch_size)  # cible scalaire z (draws)
    return policy_logits, policy_target, value_pred, value_target


# ---------------------------------------------------------------------------
# Helper z -> WDL target
# ---------------------------------------------------------------------------


def test_z_to_wdl_target_mapping() -> None:
    """z ∈ {+1, 0, -1} -> one-hot Win/Draw/Loss (classes 0/1/2)."""
    z = torch.tensor([1.0, 0.0, -1.0])
    wdl = _z_to_wdl_target(z)
    expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert torch.allclose(wdl, expected)
    # somme 1.0 par ligne
    assert torch.allclose(wdl.sum(dim=1), torch.ones(3))


def test_z_to_wdl_target_smoothing() -> None:
    """smoothing=0.3 -> (1-eps)*one_hot + eps/3 = [0.8, 0.1, 0.1] pour un Win."""
    z = torch.tensor([1.0])
    wdl = _z_to_wdl_target(z, smoothing=0.3)
    assert torch.allclose(wdl, torch.tensor([[0.8, 0.1, 0.1]]), atol=1e-6)
    assert wdl.sum().item() == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# alphazero_loss structure
# ---------------------------------------------------------------------------


def test_alphazero_loss_returns_loss_components() -> None:
    policy_logits, policy_target, value_pred, value_target = _make_inputs()
    model = _make_mini_model()
    components = alphazero_loss(policy_logits, policy_target, value_pred, value_target, model)
    assert isinstance(components, LossComponents)
    assert components.total.dim() == 0  # scalar
    assert components.policy.dim() == 0
    assert components.value.dim() == 0
    assert components.l2.dim() == 0


def test_alphazero_loss_total_has_grad() -> None:
    policy_logits, policy_target, value_pred, value_target = _make_inputs()
    model = _make_mini_model()
    components = alphazero_loss(policy_logits, policy_target, value_pred, value_target, model)
    assert components.total.requires_grad is True


def test_alphazero_loss_policy_perfect_low() -> None:
    """policy_target one-hot aligné avec argmax(logits) -> policy_loss faible."""
    batch_size = 4
    policy_logits = torch.full((batch_size, 4672), -10.0)
    policy_logits[:, 0] = 10.0  # Strong preference for index 0
    policy_target = torch.zeros(batch_size, 4672)
    policy_target[:, 0] = 1.0
    value_pred = torch.zeros(batch_size, 3)
    value_target = torch.zeros(batch_size)

    model = _make_mini_model()
    components = alphazero_loss(policy_logits, policy_target, value_pred, value_target, model)
    # Avec logits "parfaits", policy_loss should be quasi-zero.
    assert components.policy.item() < 0.01


# ---------------------------------------------------------------------------
# value loss — cross-entropy WDL
# ---------------------------------------------------------------------------


def test_value_loss_low_when_logits_match_outcome() -> None:
    """Logits prédisant fortement la bonne classe WDL -> value CE quasi-nulle."""
    policy_logits, policy_target, _, _ = _make_inputs()
    value_target = torch.tensor([1.0, -1.0])  # Win, Loss
    # logits forts sur la bonne classe (Win=0, Loss=2)
    value_pred = torch.tensor([[10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]])
    model = _make_mini_model()
    components = alphazero_loss(policy_logits, policy_target, value_pred, value_target, model)
    assert components.value.item() < 0.01


def test_value_loss_high_when_logits_oppose_outcome() -> None:
    """Logits prédisant la classe opposée -> value CE élevée."""
    policy_logits, policy_target, _, _ = _make_inputs()
    value_target = torch.tensor([1.0, -1.0])  # Win, Loss
    # logits inversés : prédit Loss pour un Win, Win pour un Loss
    wrong = torch.tensor([[-10.0, -10.0, 10.0], [10.0, -10.0, -10.0]])
    right = torch.tensor([[10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]])
    model = _make_mini_model()
    comp_wrong = alphazero_loss(policy_logits, policy_target, wrong, value_target, model)
    comp_right = alphazero_loss(policy_logits, policy_target, right, value_target, model)
    assert comp_wrong.value.item() > 5.0
    assert comp_wrong.value.item() > comp_right.value.item()


# ---------------------------------------------------------------------------
# L2 metric (monitoring) — inchangé par le switch WDL
# ---------------------------------------------------------------------------


def test_alphazero_loss_l2_excludes_bn() -> None:
    """L2 doit être identique avec BN gamma=100 ou BN gamma=1 (exclusion stricte)."""
    policy_logits, policy_target, value_pred, value_target = _make_inputs()
    torch.manual_seed(123)
    model_a = _make_mini_model()
    model_b = _make_mini_model()
    model_b.load_state_dict(model_a.state_dict())

    for m in model_a.modules():
        if isinstance(m, nn.BatchNorm2d):
            with torch.no_grad():
                m.weight.fill_(1.0)
                m.bias.fill_(0.0)
    for m in model_b.modules():
        if isinstance(m, nn.BatchNorm2d):
            with torch.no_grad():
                m.weight.fill_(100.0)
                m.bias.fill_(100.0)

    comp_a = alphazero_loss(policy_logits, policy_target, value_pred, value_target, model_a)
    comp_b = alphazero_loss(policy_logits, policy_target, value_pred, value_target, model_b)
    assert comp_a.l2.item() == pytest.approx(comp_b.l2.item(), rel=1e-5)


def test_alphazero_loss_l2_zero_with_zero_weights() -> None:
    """All weights == 0 -> L2 == 0."""
    policy_logits, policy_target, value_pred, value_target = _make_inputs()
    model = _make_mini_model()
    with torch.no_grad():
        for name, p in model.named_parameters():
            if not ("bn" in name.lower() or name.endswith(".bias")):
                p.zero_()
    components = alphazero_loss(policy_logits, policy_target, value_pred, value_target, model)
    assert components.l2.item() == pytest.approx(0.0, abs=1e-7)


def test_alphazero_loss_decomposition_sums() -> None:
    """total ≈ policy + value (L2 decoupled via AdamW, pas dans le gradient)."""
    policy_logits, policy_target, value_pred, value_target = _make_inputs()
    model = _make_mini_model()
    components = alphazero_loss(policy_logits, policy_target, value_pred, value_target, model)
    expected = components.policy.item() + components.value.item()
    assert components.total.item() == pytest.approx(expected, rel=1e-5)


# ---------------------------------------------------------------------------
# label smoothing WDL
# ---------------------------------------------------------------------------


def test_value_smoothing_zero_is_noop() -> None:
    """smoothing=0.0 -> one-hot dur, identique à l'appel par défaut."""
    policy_logits, policy_target, value_pred, _ = _make_inputs()
    value_target = torch.tensor([1.0, -1.0])
    base = alphazero_loss(
        policy_logits, policy_target, value_pred, value_target, model=_make_mini_model()
    )
    smoothed0 = alphazero_loss(
        policy_logits,
        policy_target,
        value_pred,
        value_target,
        model=_make_mini_model(),
        value_target_smoothing=0.0,
    )
    assert smoothed0.value.item() == pytest.approx(base.value.item(), rel=1e-6)


def test_value_smoothing_increases_loss_for_confident_correct() -> None:
    """Une prédiction confiante-correcte ne peut pas matcher une cible lissée.

    Logits one-hot parfaits -> CE ≈ 0 contre one-hot dur, mais > 0 contre la cible
    lissée (la cible n'est plus atteignable parfaitement). C'est l'effet
    anti-sur-confiance recherché.
    """
    policy_logits, policy_target, _, _ = _make_inputs()
    value_target = torch.tensor([1.0, 1.0])  # Win
    value_pred = torch.tensor([[20.0, -20.0, -20.0], [20.0, -20.0, -20.0]])  # confiant Win
    no_smooth = alphazero_loss(
        policy_logits, policy_target, value_pred, value_target, _make_mini_model()
    )
    smooth = alphazero_loss(
        policy_logits,
        policy_target,
        value_pred,
        value_target,
        _make_mini_model(),
        value_target_smoothing=0.1,
    )
    assert smooth.value.item() > no_smooth.value.item()
