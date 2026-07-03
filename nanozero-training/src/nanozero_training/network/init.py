"""Network weight initialization helpers (ADR-003)."""

from __future__ import annotations

import torch
from torch import nn


def init_fixup_gamma_zero(model: nn.Module, seed: int = 42) -> None:
    """Initialize weights matching the nn parity-model.npz fixture (ADR-003 parity test).

    Reproduit BIT-POUR-BIT la logique d'init de
    `nanozero-nn/scripts/python/nanozero_resnet.py::init_with_seed`, utilisée
    pour générer `nanozero-nn/src/test/resources/npz/parity-model.npz` et
    `parity-fixtures.npz`.

    **DO NOT** use this init for self-play training — utiliser
    `init_kaiming_standard` à la place. Cette fonction existe UNIQUEMENT pour
    la validation de parité Python ↔ Python en phase 1.0.0-3.

    Pattern (strictement aligné nn) :
    - `torch.manual_seed(seed)` consomme le RNG global.
    - Generator séparé `rng_for_bn` initialisé avec `seed + 1` pour running_mean
      et running_var de BN (sinon le seed global est consommé dans un ordre
      non-déterministe vs torch.randn calls).
    - Pour chaque module (ordre d'itération `model.modules()`) :
      - Conv2d : kaiming_normal_(weight, nonlinearity="relu") (PAS de mode="fan_out") + bias zero.
      - Linear : kaiming_normal_(weight, nonlinearity="relu") + bias zero.
      - BatchNorm2d : weight ~ N(1, 0.1), bias ~ N(0, 0.1),
                       running_mean ~ N(0, 0.1), running_var ~ |N(0, 1)| + 0.5.
    - Trick Fixup : zero-init du gamma (poids) du 2ᵉ BN de CHAQUE bloc résiduel.
      Stabilise les activations à l'init (cf. SPEC-nn §5.2.2, Fixup Initialization
      Zhang et al. 2019).

    Args:
        model: NanoZeroResNet (ou compatible) à initialiser in-place.
        seed: RNG seed (défaut 42, doit matcher nn fixture).

    Note critique : l'ordre des opérations consomme le RNG. Toute permutation
    casse la parité bit-pour-bit avec parity-model.npz. NE PAS refactorer pour
    "simplifier" sans tester rigoureusement contre les fixtures nn.
    """
    torch.manual_seed(seed)
    rng_for_bn = torch.Generator().manual_seed(seed + 1)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, mean=1.0, std=0.1)
            nn.init.normal_(m.bias, mean=0.0, std=0.1)
            # running_mean ~ N(0, 0.1), running_var = |N(0,1)| + 0.5 (toujours > 0).
            # BN avec track_running_stats=True (défaut) garantit running_mean/var
            # non-None — narrowing pour mypy strict.
            running_mean = m.running_mean
            running_var = m.running_var
            assert running_mean is not None
            assert running_var is not None
            m.running_mean = torch.randn(running_mean.shape, generator=rng_for_bn) * 0.1
            m.running_var = torch.abs(torch.randn(running_var.shape, generator=rng_for_bn)) + 0.5

    # Trick Fixup : zero-init du second BN de chaque bloc résiduel.
    # `model.blocks` est typé Tensor|Module par mypy strict ; on cast vers
    # nn.ModuleList qui est itérable.
    blocks: nn.ModuleList = model.blocks  # type: ignore[assignment]
    for block in blocks:
        # block.bn2.weight n'est pas statiquement typable depuis ModuleList
        # (modules dynamiques) — narrowing manuel + suppression mypy ciblée.
        nn.init.zeros_(block.bn2.weight)  # type: ignore[union-attr,arg-type]


def init_kaiming_standard(model: nn.Module, seed: int = 42) -> None:
    """Initialize model weights with standard Kaiming + standard PyTorch BN.

    Differs from nanozero-nn parity fixture (Fixup gamma=0 sur le 2e BN de
    chaque bloc résiduel) : ici init standard BN (gamma=1, beta=0) plus
    adaptée pour self-play training (la regularisation Fixup est utile pour
    inférence dès l'init random, pas pour training où le réseau apprend).

    Args:
        model: PyTorch model to initialize in-place.
        seed: RNG seed for reproducibility.
    """
    torch.manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
