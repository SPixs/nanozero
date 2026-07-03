"""Unit tests for Trainer.train_generation (mini-model + mocks state_manager)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch
from nanozero_training.train.config import TrainConfig
from nanozero_training.train.trainer import Trainer
from torch.utils.data import DataLoader, Dataset


class _MiniDictDataset(Dataset):
    """Tiny dataset yielding dicts compatibles avec NanoZeroResNet shape."""

    def __init__(self, n_samples: int = 4) -> None:
        self.n = n_samples
        self.input_planes = torch.randn(n_samples, 119, 8, 8, dtype=torch.float32)
        self.policy_target = torch.zeros(n_samples, 4672, dtype=torch.float32)
        self.policy_target[:, 0] = 1.0
        self.value_target = torch.zeros(n_samples, dtype=torch.float32)
        self.turn = torch.zeros(n_samples, dtype=torch.long)
        self.ply = torch.zeros(n_samples, dtype=torch.long)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_planes": self.input_planes[idx],
            "policy_target": self.policy_target[idx],
            "value_target": self.value_target[idx],
            "turn": self.turn[idx],
            "ply": self.ply[idx],
        }


def _make_mock_state_manager(current_epoch: int = 0, phase: str = "idle") -> MagicMock:
    """RunStateManager mock with state mutation tracking."""
    mgr = MagicMock()
    state = MagicMock()
    state.phase = phase
    state.current_gen = 0
    state.status = "in_progress"
    train_state = MagicMock()
    train_state.current_epoch = current_epoch
    train_state.last_checkpoint = None
    train_state.optimizer_state_file = None
    train_state.total_epochs = 0
    state.train = train_state
    mgr.state = state

    def fake_update(**kwargs):
        for k, v in kwargs.items():
            if "__" in k:
                section, field = k.split("__", 1)
                setattr(getattr(state, section), field, v)
            else:
                setattr(state, k, v)

    mgr.update.side_effect = fake_update
    return mgr


def _make_setup(
    tmp_path: Path,
    total_epochs: int = 2,
    current_epoch: int = 0,
) -> tuple[Trainer, DataLoader, MagicMock, Path]:
    """Build Trainer + DataLoader + mocked state + models_dir."""
    from nanozero_training.network.resnet import NanoZeroResNet

    model = NanoZeroResNet()
    config = TrainConfig(
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        total_epochs=total_epochs,
        l2_reg=1e-4,
    )
    trainer = Trainer(model=model, config=config, device="cpu")
    dataset = _MiniDictDataset(n_samples=4)
    dl = DataLoader(dataset, batch_size=2, shuffle=False)
    state_mgr = _make_mock_state_manager(current_epoch=current_epoch)
    models_dir = tmp_path / "models"
    return trainer, dl, state_mgr, models_dir


def test_train_generation_runs_total_epochs(tmp_path: Path) -> None:
    """config.total_epochs=3 -> 3 epochs joués."""
    trainer, dl, state_mgr, models_dir = _make_setup(tmp_path, total_epochs=3)
    trainer.train_generation(gen=1, dataloader=dl, models_dir=models_dir, state_manager=state_mgr)
    assert state_mgr.state.train.current_epoch == 3


def test_train_generation_writes_per_epoch_pt(tmp_path: Path) -> None:
    """3 epochs -> 3 .pt files."""
    trainer, dl, state_mgr, models_dir = _make_setup(tmp_path, total_epochs=3)
    trainer.train_generation(gen=1, dataloader=dl, models_dir=models_dir, state_manager=state_mgr)
    pt_files = sorted(models_dir.glob("train-state-gen-001-epoch-*.pt"))
    assert len(pt_files) == 3


def test_train_generation_promotes_final_npz(tmp_path: Path) -> None:
    """À la fin, gen-001-trained.npz existe (renommé de l'epoch finale)."""
    trainer, dl, state_mgr, models_dir = _make_setup(tmp_path, total_epochs=2)
    final = trainer.train_generation(
        gen=1, dataloader=dl, models_dir=models_dir, state_manager=state_mgr
    )
    assert final.name == "gen-001-trained.npz"
    assert final.exists()


def test_train_generation_cleanups_intermediates(tmp_path: Path) -> None:
    """Après cleanup, intermediates .npz absents (sauf final)."""
    trainer, dl, state_mgr, models_dir = _make_setup(tmp_path, total_epochs=3)
    trainer.train_generation(gen=1, dataloader=dl, models_dir=models_dir, state_manager=state_mgr)
    # Aucun intermediate gen-001-trained-epoch-*.npz.
    intermediates = list(models_dir.glob("gen-001-trained-epoch-*.npz"))
    assert intermediates == []
    # Seul gen-001-trained.npz.
    npz_files = list(models_dir.glob("gen-001-trained*.npz"))
    assert len(npz_files) == 1
    assert npz_files[0].name == "gen-001-trained.npz"


def test_train_generation_updates_state_per_epoch(tmp_path: Path) -> None:
    """3 epochs -> state.train.current_epoch passe par 1, 2, 3."""
    trainer, dl, state_mgr, models_dir = _make_setup(tmp_path, total_epochs=3)
    trainer.train_generation(gen=1, dataloader=dl, models_dir=models_dir, state_manager=state_mgr)
    # Track les updates pour current_epoch.
    epoch_updates = [
        c.kwargs.get("train__current_epoch")
        for c in state_mgr.update.call_args_list
        if "train__current_epoch" in c.kwargs
    ]
    assert epoch_updates == [1, 2, 3]


def test_train_generation_sets_phase_train(tmp_path: Path) -> None:
    """state.phase pré='idle' -> après run, phase='train'."""
    trainer, dl, state_mgr, models_dir = _make_setup(tmp_path, total_epochs=1)
    trainer.train_generation(gen=1, dataloader=dl, models_dir=models_dir, state_manager=state_mgr)
    assert state_mgr.state.phase == "train"


def test_train_generation_abort_before_epoch_breaks(tmp_path: Path) -> None:
    """abort_flag set initially -> 0 epochs ran, state.status='aborted'."""
    trainer, dl, state_mgr, models_dir = _make_setup(tmp_path, total_epochs=3)
    abort = {"requested": True}
    trainer.train_generation(
        gen=1, dataloader=dl, models_dir=models_dir, state_manager=state_mgr, abort_flag=abort
    )
    assert state_mgr.state.status == "aborted"
    # Aucun .pt produit.
    pt_files = list(models_dir.glob("train-state-gen-001-epoch-*.pt"))
    assert pt_files == []


def test_train_generation_resume_from_state(tmp_path: Path) -> None:
    """state.train.current_epoch=2 -> resume depuis epoch 2 + ne refait pas 0, 1.

    Setup : run 2 epochs first, save checkpoints. Then build fresh trainer
    and run with state.current_epoch=2 + total_epochs=4. Expect: 2 additional
    epochs run (epochs 2, 3), final state.current_epoch=4.
    """
    # Step 1 : run 2 epochs.
    trainer1, dl, state_mgr, models_dir = _make_setup(tmp_path, total_epochs=2)
    trainer1.train_generation(gen=1, dataloader=dl, models_dir=models_dir, state_manager=state_mgr)
    assert state_mgr.state.train.current_epoch == 2

    # Cleanup final .npz (replicates state mid-training avant final cleanup).
    # En réalité train_generation final cleanup, mais .pt préservés.
    # Step 2 : "fake" mid-training continuing — bump total_epochs to 4.
    # Note: dans un vrai resume cross-process, le state serait lu depuis disque.
    # Ici on simule en gardant le mock state_mgr.

    from nanozero_training.network.resnet import NanoZeroResNet

    config_4 = TrainConfig(
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        total_epochs=4,
        l2_reg=1e-4,
    )
    # Build fresh trainer with new total_epochs.
    trainer2 = Trainer(model=NanoZeroResNet(), config=config_4, device="cpu")
    # state.train.current_epoch == 2 déjà mis par step 1.
    # Note: la final.npz du step 1 est renommée gen-001-trained.npz. Pour resume,
    # il faut le .pt epoch 1, qui DOIT exister.
    pt_path = models_dir / "train-state-gen-001-epoch-001.pt"
    assert pt_path.exists(), "Step 1 should have left epoch 1 .pt for resume"

    trainer2.train_generation(gen=1, dataloader=dl, models_dir=models_dir, state_manager=state_mgr)
    # Final current_epoch == 4.
    assert state_mgr.state.train.current_epoch == 4


def test_train_generation_passes_writer_to_each_epoch(tmp_path: Path) -> None:
    """Phase 10 : metrics_writer forwarded to train_epoch for each epoch."""
    from unittest.mock import MagicMock

    trainer, dl, state_mgr, models_dir = _make_setup(tmp_path, total_epochs=3)
    writer = MagicMock()
    trainer.train_generation(
        gen=1,
        dataloader=dl,
        models_dir=models_dir,
        state_manager=state_mgr,
        metrics_writer=writer,
    )
    # 3 epochs -> 3 append_train_row calls (1 per epoch).
    assert writer.append_train_row.call_count == 3
    # Verify epoch indices 0, 1, 2
    epochs = [c.kwargs["epoch"] for c in writer.append_train_row.call_args_list]
    assert epochs == [0, 1, 2]
