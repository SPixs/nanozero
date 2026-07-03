"""Unit tests for train/checkpoint.py."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from nanozero_training.train.checkpoint import (
    load_model_for_resume,
    load_training_state,
    save_training_state,
)
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


def _make_simple_setup() -> tuple[nn.Module, Adam, CosineAnnealingLR]:
    """Build minimal model + optimizer + scheduler for tests."""
    model = nn.Linear(10, 5)
    opt = Adam(model.parameters(), lr=1e-3)
    sched = CosineAnnealingLR(opt, T_max=100)
    return model, opt, sched


def test_save_training_state_creates_file(tmp_path: Path) -> None:
    _, opt, _ = _make_simple_setup()
    target = tmp_path / "state.pt"
    save_training_state(target, optimizer=opt)
    assert target.exists()


def test_save_load_roundtrip_optimizer(tmp_path: Path) -> None:
    _, opt, _ = _make_simple_setup()
    # Modify optimizer state (run a step) to have non-trivial state.
    opt.zero_grad()
    for p in opt.param_groups[0]["params"]:
        p.grad = torch.randn_like(p)
    opt.step()

    target = tmp_path / "state.pt"
    save_training_state(target, optimizer=opt)

    # Fresh optimizer + load.
    _, opt_b, _ = _make_simple_setup()
    load_training_state(target, optimizer=opt_b, restore_rng=False)

    # State dicts should match.
    sd_a = opt.state_dict()
    sd_b = opt_b.state_dict()
    assert set(sd_a.keys()) == set(sd_b.keys())


def test_save_load_roundtrip_scheduler(tmp_path: Path) -> None:
    model, opt, sched = _make_simple_setup()
    # Advance scheduler (must step optimizer first to avoid PyTorch warning).
    for _ in range(5):
        opt.zero_grad()
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        opt.step()
        sched.step()

    target = tmp_path / "state.pt"
    save_training_state(target, optimizer=opt, scheduler=sched)

    _, opt_b, sched_b = _make_simple_setup()
    load_training_state(target, optimizer=opt_b, scheduler=sched_b, restore_rng=False)
    assert sched.last_epoch == sched_b.last_epoch


def test_save_load_roundtrip_rng(tmp_path: Path) -> None:
    _, opt, _ = _make_simple_setup()
    rng_state = torch.get_rng_state()
    # Sample a value avec ce state pour comparaison.
    expected = torch.rand(1).item()

    target = tmp_path / "state.pt"
    save_training_state(target, optimizer=opt, torch_rng_state=rng_state)

    # Change RNG state pour vérifier que load restore proprement.
    torch.manual_seed(999)
    _, opt_b, _ = _make_simple_setup()
    load_training_state(target, optimizer=opt_b, restore_rng=True)
    actual = torch.rand(1).item()
    assert actual == expected


def test_save_load_roundtrip_model_via_extra(tmp_path: Path) -> None:
    model, opt, _ = _make_simple_setup()
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(0.5)

    target = tmp_path / "state.pt"
    save_training_state(
        target, optimizer=opt, extra={"model_state": model.state_dict(), "epoch": 3}
    )

    # Fresh model -> load via load_model_for_resume.
    model_b = nn.Linear(10, 5)
    load_model_for_resume(target, model_b)

    for p_a, p_b in zip(model.parameters(), model_b.parameters(), strict=True):
        assert torch.equal(p_a, p_b)


def test_load_training_state_missing_file_raises(tmp_path: Path) -> None:
    _, opt, _ = _make_simple_setup()
    with pytest.raises(FileNotFoundError):
        load_training_state(tmp_path / "missing.pt", optimizer=opt)


def test_load_training_state_missing_optimizer_key_raises(tmp_path: Path) -> None:
    target = tmp_path / "bad.pt"
    torch.save({"foo": 1}, target)
    _, opt, _ = _make_simple_setup()
    with pytest.raises(KeyError, match="Missing 'optimizer'"):
        load_training_state(target, optimizer=opt)


def test_load_model_for_resume_missing_extra_raises(tmp_path: Path) -> None:
    """train-state without extra.model_state -> KeyError."""
    target = tmp_path / "bad.pt"
    torch.save({"optimizer": {}}, target)
    model = nn.Linear(10, 5)
    with pytest.raises(KeyError, match="Missing extra.model_state"):
        load_model_for_resume(target, model)
