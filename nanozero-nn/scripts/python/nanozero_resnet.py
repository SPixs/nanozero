"""Implémentation PyTorch de l'architecture NanoZeroResNet (cf. SPEC-nn §3.3, §5.1, §6.5).

ResNet 8 blocs × 96 channels, BatchNorm partout (folded à l'export, cf. ADR-008).
Pattern post-activation strict §5.1.2.

Cette définition est la SOURCE DE VÉRITÉ Python de l'architecture. Toute évolution
casserait la compatibilité avec les .npz existants ; voir ADR-009 (versioning
_meta_architecture_version).

Conventions critiques :
- Bias dans toutes les conv (le bias initial est absorbé par fold_conv_bn à l'export).
- BN UNIQUEMENT après input_conv et après chaque conv des blocs résiduels (17 BN
  total). PAS de BN sur policy/value heads.
- Transposition policy §3.5.1 :
      logits[n, square*73 + plane] = nchw[n, plane, square // 8, square % 8]
  équivalent à reshape(N, 73, 64) -> transpose(1, 2) -> reshape(N, 4672).
- Value head : tanh AVANT squeeze final, scalaire ∈ [-1, 1].
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------------
# Architecture (§3.3, §5.1.2)
# ---------------------------------------------------------------------------------


class ResidualBlock(nn.Module):
    """Bloc résiduel ResNet conforme §5.1.2.

    Pattern :
        skip = x
        tmp  = ReLU(BN1(conv1(x)))
        out  = BN2(conv2(tmp))
        out  = ReLU(out + skip)
    """

    def __init__(self, channels: int = 96):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        out = out + skip
        out = F.relu(out)
        return out


class NanoZeroResNet(nn.Module):
    """ResNet 8 blocs × 96 channels conforme SPEC §3.3.

    Input  : [N, 119, 8, 8] float32
    Output : (policy_logits [N, 4672], value [N]) float32

    Note : la transposition policy NCHW -> logits flat est faite ici dans le forward
    Python pour que la sortie corresponde bit-pour-bit au format §3.5.1 (et au
    Java Network.transposeNCHWtoLogits, qui applique la même convention).
    """

    INPUT_CHANNELS = 119
    BODY_CHANNELS = 96
    NUM_BLOCKS = 8
    POLICY_PLANES = 73
    VALUE_HIDDEN = 64
    VALUE_WDL_CLASSES = 3  # Win / Draw / Loss (v1.5.0)
    POLICY_LOGITS = POLICY_PLANES * 8 * 8  # 4672

    def __init__(self):
        super().__init__()
        # Input conv + BN.
        self.input_conv = nn.Conv2d(
            self.INPUT_CHANNELS, self.BODY_CHANNELS, kernel_size=3, padding=1, bias=True
        )
        self.input_bn = nn.BatchNorm2d(self.BODY_CHANNELS)

        # Tour résiduelle (8 blocs).
        self.blocks = nn.ModuleList(
            [ResidualBlock(self.BODY_CHANNELS) for _ in range(self.NUM_BLOCKS)]
        )

        # Policy head : conv 1x1 96 -> 73, PAS DE BN (cf. §6.5 commentaire "pas de
        # BN, biais déjà présent").
        self.policy_conv = nn.Conv2d(
            self.BODY_CHANNELS, self.POLICY_PLANES, kernel_size=1, bias=True
        )

        # Value head (WDL v1.5.0) : conv 1x1 96 -> 1, PAS DE BN, puis 2 FC ;
        # fc2 sort 3 logits Win/Draw/Loss (au lieu d'un scalaire tanh).
        self.value_conv = nn.Conv2d(self.BODY_CHANNELS, 1, kernel_size=1, bias=True)
        self.value_fc1 = nn.Linear(self.VALUE_HIDDEN, self.VALUE_HIDDEN, bias=True)
        self.value_fc2 = nn.Linear(self.VALUE_HIDDEN, self.VALUE_WDL_CLASSES, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Body.
        out = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            out = block(out)

        N = out.shape[0]

        # Policy head + transposition NCHW -> logits flat.
        # Convention §3.5.1 :
        #   logits[n, square*73 + plane] = nchw[n, plane, square//8, square%8].
        # square = h*8 + w (row-major) ; on reshape NCHW [N, 73, 8, 8] vers
        # [N, 73, 64] (linéarisation row-major du HW), puis transpose(1, 2) pour
        # passer en [N, 64, 73] (axe square avant axe plane), puis flatten.
        policy_nchw = self.policy_conv(out)  # [N, 73, 8, 8]
        policy_3d = policy_nchw.reshape(N, self.POLICY_PLANES, 64)  # [N, 73, 64]
        policy_transposed = policy_3d.transpose(1, 2)  # [N, 64, 73]
        policy_logits = policy_transposed.reshape(N, self.POLICY_LOGITS)  # [N, 4672]

        # Value head (WDL v1.5.0) : 3 logits Win/Draw/Loss, PAS de softmax/tanh
        # (logits bruts ; softmax appliqué côté loss / inference Java).
        value_conv_out = self.value_conv(out)  # [N, 1, 8, 8]
        value_flat = value_conv_out.reshape(N, self.VALUE_HIDDEN)  # [N, 64]
        value_hidden = F.relu(self.value_fc1(value_flat))  # [N, 64]
        value_logits = self.value_fc2(value_hidden)  # [N, 3] W/D/L logits

        return policy_logits, value_logits


# ---------------------------------------------------------------------------------
# Initialisation déterministe non-triviale (BN gamma/beta/mean/var ≠ identité)
# ---------------------------------------------------------------------------------


def init_with_seed(model: NanoZeroResNet, seed: int = 42) -> None:
    """Initialisation déterministe avec BN non-triviaux pour exercer fold_conv_bn.

    Init Fixup-like : gamma=0 sur le SECOND BN de chaque bloc résiduel pour
    stabiliser les activations dès l'init. Sans cela, l'accumulation positive
    ~1.5x par bloc (Var(conv+ReLU) ≈ Var(x), puis +skip, puis ReLU imparfait)
    sature tanh après 8 blocs (value=−1 constant).

    Avec gamma_bn2=0 : chaque bloc devient initialement ReLU(0 + skip) = skip,
    bornant les activations au niveau post-input-conv. beta, running_mean et
    running_var de bn2 restent perturbés, donc fold_conv_bn(conv2, bn2) produit
    b'_folded = beta non-trivial (mais W'_folded = 0). Le path bias-init +
    scatter du kernel Java reste exercé sur les conv2.

    Référence : Fixup Initialization (Zhang et al. 2019) ; pratique standard
    pour les ResNet random-init évalués en mode eval.
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
            m.running_mean = torch.randn(
                m.running_mean.shape, generator=rng_for_bn
            ) * 0.1
            m.running_var = (
                torch.abs(torch.randn(m.running_var.shape, generator=rng_for_bn)) + 0.5
            )

    # Fixup-like : zero-init du second BN de chaque bloc résiduel.
    for block in model.blocks:
        nn.init.zeros_(block.bn2.weight)


# ---------------------------------------------------------------------------------
# Fold BN dans la conv précédente (cf. ADR-008, §6.5)
# ---------------------------------------------------------------------------------


def fold_conv_bn(
    conv: nn.Conv2d, bn: nn.BatchNorm2d
) -> tuple[np.ndarray, np.ndarray]:
    """Fold BatchNorm dans la conv précédente, retour (W', b') float32.

    En mode eval, BN(x) = gamma * (x - mean) / sqrt(var + eps) + beta.
    Soit conv(x) = W * x + b (broadcasting sur les channels). Alors
        BN(conv(x)) = W' * x + b'
    avec :
        scale = gamma / sqrt(var + eps)
        W'    = W * scale.view(-1, 1, 1, 1)
        b'    = (b - mean) * scale + beta

    Le résultat est strictement identique numériquement à BN(conv(x)) en mode eval.
    """
    eps = bn.eps
    mean = bn.running_mean.detach()
    var = bn.running_var.detach()
    gamma = bn.weight.detach()
    beta = bn.bias.detach()

    scale = gamma / torch.sqrt(var + eps)

    w_folded = conv.weight.detach() * scale.view(-1, 1, 1, 1)
    b_conv = conv.bias.detach() if conv.bias is not None else torch.zeros_like(beta)
    b_folded = (b_conv - mean) * scale + beta

    return (
        w_folded.cpu().numpy().astype(np.float32),
        b_folded.cpu().numpy().astype(np.float32),
    )
