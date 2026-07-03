"""ResNet 8 blocs x 96 channels — duplication contrôlée vs nanozero-nn (ADR-003).

Source de vérité de l'architecture : `nanozero-nn/scripts/python/nanozero_resnet.py`.
La structure (layers, hyperparamètres, forward) est identique. La divergence
intentionnelle est l'initialisation : ici init Kaiming standard (cf.
`network/init.py`), pas le trick Fixup gamma=0 de nn (réservé fixture parity-model).

Conventions critiques (héritées SPEC-nn §3.3, §5.1.2) :
- Bias dans toutes les conv (le bias initial sera absorbé par fold_conv_bn à
  l'export pour engine, cf. ADR-008 nn — folding fait en phase 1.0.0-3).
- BatchNorm UNIQUEMENT après input_conv et après chaque conv des blocs
  résiduels (17 BN total). PAS de BN sur policy / value heads.
- Transposition policy §3.5.1 :
      logits[n, square*73 + plane] = nchw[n, plane, square // 8, square % 8]
  équivalent à reshape(N, 73, 64) -> transpose(1, 2) -> reshape(N, 4672).
- Value head (v1.5.0, WDL) : 3 logits Win/Draw/Loss, shape [N, 3], PAS de
  softmax ni tanh (logits bruts — softmax appliqué dans la loss côté training
  et côté inference Java pour calculer V = P(W) - P(L)). Cf. SPEC-nn §6 amendé.
  Remplace l'ancien value head scalaire tanh ∈ [-1, 1] (≤ v1.4.0).

Note phase 1.0.0-1 : skeleton fonctionnel. Test de parité numérique Java/Python
phase 1.0.0-3 (parité priorisée seulement après ADR-002 + export_npz complet).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812 — convention PyTorch (F majuscule)
from torch import nn


class ResidualBlock(nn.Module):
    """Bloc résiduel ResNet conforme SPEC-nn §5.1.2.

    Pattern :
        skip = x
        tmp  = ReLU(BN1(conv1(x)))
        out  = BN2(conv2(tmp))
        out  = ReLU(out + skip)
    """

    def __init__(self, channels: int = 96) -> None:
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
        return F.relu(out)


class NanoZeroResNet(nn.Module):
    """ResNet 8 blocs x 96 channels conforme SPEC-nn §3.3.

    Input  : [N, 119, 8, 8] float32
    Output : (policy_logits [N, 4672], value_logits [N, 3]) float32 — value =
             3 logits Win/Draw/Loss (WDL, v1.5.0). Softmax NON appliqué ici
             (logits bruts) : la loss fait la cross-entropy, l'inference Java
             applique softmax puis V = P(W) - P(L) pour le PUCT.

    La transposition policy NCHW -> logits flat est faite dans le forward Python
    pour que la sortie corresponde bit-pour-bit au format §3.5.1 (et au Java
    Network.transposeNCHWtoLogits).
    """

    INPUT_CHANNELS = 119
    BODY_CHANNELS = 96
    NUM_BLOCKS = 8
    POLICY_PLANES = 73
    VALUE_HIDDEN = 64
    VALUE_WDL_CLASSES = 3  # Win / Draw / Loss (v1.5.0)
    POLICY_LOGITS = POLICY_PLANES * 8 * 8  # 4672

    def __init__(
        self,
        n_blocks: int = NUM_BLOCKS,
        channels: int = BODY_CHANNELS,
    ) -> None:
        super().__init__()
        # Input conv + BN.
        self.input_conv = nn.Conv2d(
            self.INPUT_CHANNELS, channels, kernel_size=3, padding=1, bias=True
        )
        self.input_bn = nn.BatchNorm2d(channels)

        # Tour résiduelle.
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(n_blocks)])

        # Policy head : conv 1x1 channels -> 73, PAS DE BN.
        self.policy_conv = nn.Conv2d(channels, self.POLICY_PLANES, kernel_size=1, bias=True)

        # Value head (WDL v1.5.0) : conv 1x1 channels -> 1, PAS DE BN, puis 2 FC.
        # fc2 sort 3 logits (Win/Draw/Loss) au lieu d'un scalaire tanh.
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1, bias=True)
        self.value_fc1 = nn.Linear(self.VALUE_HIDDEN, self.VALUE_HIDDEN, bias=True)
        self.value_fc2 = nn.Linear(self.VALUE_HIDDEN, self.VALUE_WDL_CLASSES, bias=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Body.
        out = F.relu(self.input_bn(self.input_conv(x)))
        for block in self.blocks:
            out = block(out)

        batch_size = out.shape[0]

        # Policy head + transposition NCHW -> logits flat (§3.5.1).
        policy_nchw = self.policy_conv(out)  # [N, 73, 8, 8]
        policy_3d = policy_nchw.reshape(batch_size, self.POLICY_PLANES, 64)  # [N, 73, 64]
        policy_transposed = policy_3d.transpose(1, 2)  # [N, 64, 73]
        policy_logits = policy_transposed.reshape(batch_size, self.POLICY_LOGITS)  # [N, 4672]

        # Value head (WDL v1.5.0) : sort 3 logits Win/Draw/Loss, PAS de softmax
        # ni tanh ici (logits bruts, comme la policy).
        value_conv_out = self.value_conv(out)  # [N, 1, 8, 8]
        value_flat = value_conv_out.reshape(batch_size, self.VALUE_HIDDEN)  # [N, 64]
        value_hidden = F.relu(self.value_fc1(value_flat))  # [N, 64]
        value_logits = self.value_fc2(value_hidden)  # [N, 3] — W/D/L logits

        return policy_logits, value_logits
