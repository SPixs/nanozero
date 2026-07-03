"""Tests for init_fixup_gamma_zero — parity init (matches nn fixture).

Validate that the Fixup init reproduces the nn reference structure exactly:
- 2nd BN of each residual block has gamma=0 (Fixup trick).
- 1st BN of each residual block has gamma != 0 (random from N(1, 0.1)).
- Reproducible with same seed.
- Distinct from init_kaiming_standard on same seed.
"""

from __future__ import annotations

import torch
from nanozero_training.network.init import init_fixup_gamma_zero, init_kaiming_standard
from nanozero_training.network.resnet import NanoZeroResNet


def test_init_fixup_reproducible_with_same_seed() -> None:
    """Same seed -> identical state_dict bit-pour-bit."""
    a = NanoZeroResNet()
    b = NanoZeroResNet()
    init_fixup_gamma_zero(a, seed=42)
    init_fixup_gamma_zero(b, seed=42)
    for (n_a, p_a), (n_b, p_b) in zip(a.state_dict().items(), b.state_dict().items(), strict=True):
        assert n_a == n_b
        assert torch.equal(p_a, p_b), f"Mismatch on {n_a}"


def test_init_fixup_different_from_kaiming_standard() -> None:
    """Fixup and Kaiming standard produce different weights even with same seed.

    Sanity: les deux inits sont structurellement distincts. Au moins quelques
    parameters doivent différer (Kaiming uses mode='fan_out', Fixup doesn't;
    Kaiming sets BN gamma=1, Fixup samples from N(1, 0.1)).
    """
    a = NanoZeroResNet()
    b = NanoZeroResNet()
    init_fixup_gamma_zero(a, seed=42)
    init_kaiming_standard(b, seed=42)
    diffs = [
        not torch.equal(p_a, p_b) for p_a, p_b in zip(a.parameters(), b.parameters(), strict=True)
    ]
    assert any(diffs), "Fixup and Kaiming standard produced identical state_dicts"


def test_init_fixup_second_bn_gamma_is_zero() -> None:
    """2nd BN of each residual block must have gamma == 0 after Fixup init."""
    model = NanoZeroResNet()
    init_fixup_gamma_zero(model, seed=42)
    for i, block in enumerate(model.blocks):
        gamma = block.bn2.weight
        assert torch.equal(gamma, torch.zeros_like(gamma)), (
            f"Block {i} bn2.weight expected all zeros (Fixup), got "
            f"min={gamma.min().item()} max={gamma.max().item()}"
        )


def test_init_fixup_first_bn_gamma_non_zero() -> None:
    """1st BN of each residual block has gamma != 0 (sampled from N(1, 0.1))."""
    model = NanoZeroResNet()
    init_fixup_gamma_zero(model, seed=42)
    for i, block in enumerate(model.blocks):
        gamma = block.bn1.weight
        # Sampled from N(1, 0.1) — extremely unlikely to be exactly zero.
        assert not torch.equal(
            gamma, torch.zeros_like(gamma)
        ), f"Block {i} bn1.weight should not be all zeros (only bn2 is Fixup'd)"


def test_init_fixup_input_bn_gamma_non_zero() -> None:
    """input_bn (outside residual blocks) is not Fixup'd."""
    model = NanoZeroResNet()
    init_fixup_gamma_zero(model, seed=42)
    gamma = model.input_bn.weight
    assert not torch.equal(
        gamma, torch.zeros_like(gamma)
    ), "input_bn.weight should not be all zeros — only block.bn2 are Fixup'd"


def test_init_fixup_module_invariant_after_init() -> None:
    """No module added/removed by init — only weights change."""
    model = NanoZeroResNet()
    modules_before = list(model.modules())
    init_fixup_gamma_zero(model, seed=42)
    modules_after = list(model.modules())
    assert len(modules_before) == len(modules_after)
    for m_b, m_a in zip(modules_before, modules_after, strict=True):
        assert type(m_b) is type(m_a)
