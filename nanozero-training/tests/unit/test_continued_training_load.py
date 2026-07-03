"""Test du load tolérant Phase A3 (transfert WDL v1.5.0).

load_model_for_continued_training doit charger les tenseurs dont la shape
matche (body + policy + value_conv + value_fc1) et laisser à leur init courante
ceux qui mismatch (value_fc2 : ancien scalaire [1, 64] -> WDL [3, 64]).
"""

from __future__ import annotations

from pathlib import Path

import torch
from nanozero_training.network.resnet import NanoZeroResNet
from nanozero_training.train.checkpoint import load_model_for_continued_training


def _save_legacy_scalar_checkpoint(path: Path) -> dict:
    """Construit un checkpoint .pt mimant l'ancien value head scalaire (fc2 64->1).

    Repart d'un modèle WDL, remplace value_fc2 par une version [1, 64]/[1], et
    sauve le state_dict dans extra['model_state'] (format attendu par le load).
    Retourne le state_dict 'legacy' pour comparaison.
    """
    legacy = NanoZeroResNet()
    sd = legacy.state_dict()
    # Simule l'ancien value head scalaire : fc2 -> 1 sortie.
    sd["value_fc2.weight"] = torch.randn(1, NanoZeroResNet.VALUE_HIDDEN)
    sd["value_fc2.bias"] = torch.randn(1)
    torch.save({"extra": {"model_state": sd}}, path)
    return sd


def test_continued_load_transfers_body_reinits_value_head(tmp_path: Path) -> None:
    ckpt = tmp_path / "train-state-legacy.pt"
    legacy_sd = _save_legacy_scalar_checkpoint(ckpt)

    model = NanoZeroResNet()  # WDL : value_fc2 [3, 64]
    fresh_value_fc2 = model.state_dict()["value_fc2.weight"].clone()

    loaded, skipped = load_model_for_continued_training(ckpt, model)

    # value_fc2 (shape mismatch [1,64] vs [3,64]) ré-initialisé, pas chargé.
    assert "value_fc2.weight" in skipped
    assert "value_fc2.bias" in skipped
    assert "value_fc2.weight" not in loaded
    # Le reste (body + policy + value_conv + value_fc1) chargé.
    assert "input_conv.weight" in loaded
    assert "policy_conv.weight" in loaded
    assert "value_conv.weight" in loaded
    assert "value_fc1.weight" in loaded

    new_sd = model.state_dict()
    # value_fc2 reste à son init fraîche (PAS écrasé par le legacy [1,64]).
    assert new_sd["value_fc2.weight"].shape == (NanoZeroResNet.VALUE_WDL_CLASSES, 64)
    assert torch.equal(new_sd["value_fc2.weight"], fresh_value_fc2)
    # Un tenseur du body matche bien le checkpoint legacy (transfert effectif).
    assert torch.equal(new_sd["input_conv.weight"], legacy_sd["input_conv.weight"])
    assert torch.equal(new_sd["policy_conv.weight"], legacy_sd["policy_conv.weight"])


def test_continued_load_same_arch_skips_nothing(tmp_path: Path) -> None:
    """Sur une gen SANS changement d'archi (WDL->WDL), rien n'est skippé."""
    src = NanoZeroResNet()
    ckpt = tmp_path / "train-state-wdl.pt"
    torch.save({"extra": {"model_state": src.state_dict()}}, ckpt)

    model = NanoZeroResNet()
    loaded, skipped = load_model_for_continued_training(ckpt, model)
    assert skipped == []
    # tous les poids chargés -> model == src
    for k, v in src.state_dict().items():
        assert torch.equal(model.state_dict()[k], v)
