"""AlphaZero training loss (policy + value).

Reference : SPEC-training §8.1 (amendé 2026-05-21 — drop L2-in-loss).

Loss formula (value head WDL depuis v1.5.0) :
    L = policy_loss + value_loss

    policy_loss = -mean(sum(policy_target * log_softmax(policy_logits), axis=1))
    value_loss  = -mean(sum(wdl_target * log_softmax(value_logits), axis=1))

où ``value_logits`` est [N, 3] (Win/Draw/Loss) et ``wdl_target`` la one-hot
dérivée de la cible scalaire z ∈ {-1, 0, +1} (cf. ``_z_to_wdl_target``). C'est
la même forme de cross-entropy soft-target que la policy. Remplace l'ancien
``value_loss = mean((tanh_pred - z)^2)`` (MSE scalaire, ≤ v1.4.0).

Le weight_decay est désormais appliqué *decoupled* par AdamW (cf. trainer
``make_optimizer``), ce qui élimine le bug Adam+L2-in-loss documenté par Lc0
("invisible weight blowup" — le L2 est divisé par le second-moment Adam,
régularisation non uniforme).

Le calcul ``LossComponents.l2`` (somme des carrés des poids conv/linear) est
conservé pour monitoring uniquement, il n'est plus inclus dans le gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812 — convention PyTorch
from torch import nn


@dataclass(frozen=True)
class LossComponents:
    """Decomposed AlphaZero loss for monitoring.

    Attributes:
        total: scalar tensor (with grad, backprop target).
        policy: policy cross-entropy term (scalar).
        value: value MSE term (scalar).
        l2: L2 regularization sum (scalar, BEFORE multiplication par l2_reg coef).
    """

    total: torch.Tensor
    policy: torch.Tensor
    value: torch.Tensor
    l2: torch.Tensor


def alphazero_loss(
    policy_logits: torch.Tensor,
    policy_target: torch.Tensor,
    value_pred: torch.Tensor,
    value_target: torch.Tensor,
    model: nn.Module,
    value_target_smoothing: float = 0.0,
) -> LossComponents:
    """Compute AlphaZero training loss with decomposition.

    Args:
        policy_logits: (B, 4672) raw logits from model.
        policy_target: (B, 4672) target distribution (sum to 1.0 per row, ou
                       all zeros pour position terminale).
        value_pred: (B, 3) value logits Win/Draw/Loss (PAS de softmax — WDL
                    v1.5.0, cf. NanoZeroResNet).
        value_target: (B,) target value scalaire in {-1.0, 0.0, +1.0} (convertie
                    en one-hot WDL en interne par ``_z_to_wdl_target``).
        model: NanoZeroResNet (pour L2 metric monitoring uniquement —
               weight_decay est désormais decoupled via AdamW).
        value_target_smoothing: facteur de label smoothing dans [0, 1) appliqué à
               la cible WDL ((1-eps)*one_hot + eps/3). Calibration anti-sur-confiance
               du value head (cf. diagnostic gen-24 king-safety + Lc0 WDL). Défaut
               0.0 = one-hot dur.

    Returns:
        LossComponents avec total (backprop target), policy, value, l2 (l2
        est calculé pour monitoring mais N'EST PLUS inclus dans total).

    Notes:
        - policy_loss utilise log_softmax pour stabilité numérique.
        - value_pred shape (B,) — squeezed (cohérent ADR-003 NanoZeroResNet).
        - L2 metric exclut BN params et biases (cohérent avec exclusion
          AdamW dans trainer.make_optimizer).
    """
    # Policy loss : cross-entropy soft target.
    log_probs = F.log_softmax(policy_logits, dim=1)
    policy_loss = -(policy_target * log_probs).sum(dim=1).mean()

    # Value loss (WDL v1.5.0) : cross-entropy 3-classes Win/Draw/Loss.
    # value_pred = logits [N, 3] (PAS de softmax), value_target = scalaire z [N]
    # in {-1, 0, +1} converti en distribution one-hot WDL. Le label smoothing
    # (value_target_smoothing) lisse le one-hot ((1-eps)*onehot + eps/3) — décourage
    # la sur-confiance du value head (diagnostic gen-24 + calibration Lc0 WDL).
    wdl_target = _z_to_wdl_target(
        value_target, num_classes=value_pred.shape[-1], smoothing=value_target_smoothing
    )
    value_log_probs = F.log_softmax(value_pred, dim=1)
    value_loss = -(wdl_target * value_log_probs).sum(dim=1).mean()

    # L2 (monitoring only) : exclude BN params et biases.
    l2 = torch.zeros((), device=policy_logits.device)
    for name, p in model.named_parameters():
        if "bn" in name.lower() or name.endswith(".bias"):
            continue
        l2 = l2 + p.pow(2).sum()

    total = policy_loss + value_loss
    return LossComponents(total=total, policy=policy_loss, value=value_loss, l2=l2)


def _z_to_wdl_target(
    z: torch.Tensor,
    num_classes: int = 3,
    smoothing: float = 0.0,
) -> torch.Tensor:
    """Convertit la cible value scalaire z en distribution WDL (Win/Draw/Loss).

    Mapping (POV du joueur au trait) : z = +1 -> Win (classe 0), z = 0 -> Draw
    (classe 1), z = -1 -> Loss (classe 2). Les seuils ±0.5 sont robustes à la
    représentation float (z ∈ {-1, 0, +1} exactement).

    Args:
        z: cible scalaire [N] in {-1.0, 0.0, +1.0}.
        num_classes: 3 (W/D/L).
        smoothing: label smoothing standard dans [0, 1) — la cible devient
            (1-eps)*one_hot + eps/num_classes. 0.0 = one-hot dur.

    Returns:
        Tensor [N, num_classes] (distribution cible, somme 1.0 par ligne).
    """
    cls = torch.ones_like(z, dtype=torch.long)  # défaut Draw (classe 1)
    cls[z > 0.5] = 0  # Win
    cls[z < -0.5] = 2  # Loss
    wdl = F.one_hot(cls, num_classes=num_classes).to(dtype=z.dtype)  # [N, 3]
    if smoothing != 0.0:
        wdl = wdl * (1.0 - smoothing) + smoothing / num_classes
    return wdl
