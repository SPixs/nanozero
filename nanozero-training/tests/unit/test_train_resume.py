"""Resume scenarios SPEC §11.5 pour training (scenarios 3 et 4).

Scenario 3 (abort mid-epoch — current epoch lost) :
- v1.0.0 design : abort est checké BETWEEN epochs uniquement. Si un crash
  survient mid-epoch, state.train.current_epoch reflète les K epochs
  TERMINÉS (pas K+1 — l'update se fait après save_training_state).
- Au resume, on relit state.current_epoch=K et on reprend à l'epoch K
  (pas K+1). L'epoch K perdue mid-flight est ré-exécutée.

Scenario 4 (abort entre epochs — clean state) :
- State.train.current_epoch=K après save propre.
- Au resume, continue à epoch K+1...total_epochs.

Différence pratique entre 3 et 4 : aucune côté code. La distinction est
sémantique : 3 = "crash perdu une epoch en cours", 4 = "shutdown propre
entre epochs". Les deux résultent dans le même state file et même resume
behavior. On teste les deux pour démontrer que le pattern résiste aux
deux cas (cf. SPEC §11.5).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import torch
from nanozero_training.network.resnet import NanoZeroResNet
from nanozero_training.train.config import TrainConfig
from nanozero_training.train.trainer import Trainer
from torch.utils.data import DataLoader, Dataset


class _MiniDictDataset(Dataset):
    """Tiny dataset for tests (4 samples)."""

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


def _make_mock_state_manager(current_epoch: int = 0) -> MagicMock:
    """RunStateManager mock."""
    mgr = MagicMock()
    state = MagicMock()
    state.phase = "idle"
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


def test_resume_scenario_4_between_epochs(tmp_path: Path) -> None:
    """SPEC §11.5 scenario 4 : state.current_epoch=2 (clean), total=4 -> 2 more.

    Setup :
    - Run 2 epochs propres (state.current_epoch=2, 2 .pt files).
    - Fresh Trainer instance (simule resume cross-process).
    - state.current_epoch=2, total_epochs=4 -> 2 nouveaux epochs run.

    Expected :
    - 2 nouveaux epochs (2 et 3 0-indexed).
    - 4 .pt files au total.
    - state final current_epoch=4.
    - final gen-001-trained.npz produit.
    """
    # Step 1 : Run 2 epochs propres.
    model_1 = NanoZeroResNet()
    config_2 = TrainConfig(
        batch_size=2, num_workers=0, pin_memory=False, total_epochs=2, l2_reg=1e-4
    )
    trainer_1 = Trainer(model=model_1, config=config_2, device="cpu")
    dataset = _MiniDictDataset(n_samples=4)
    dl = DataLoader(dataset, batch_size=2, shuffle=False)
    state_mgr = _make_mock_state_manager(current_epoch=0)
    models_dir = tmp_path / "models"

    trainer_1.train_generation(gen=1, dataloader=dl, models_dir=models_dir, state_manager=state_mgr)
    assert state_mgr.state.train.current_epoch == 2

    # Step 2 : Fresh Trainer avec total_epochs=4, simule resume.
    model_2 = NanoZeroResNet()
    config_4 = TrainConfig(
        batch_size=2, num_workers=0, pin_memory=False, total_epochs=4, l2_reg=1e-4
    )
    trainer_2 = Trainer(model=model_2, config=config_4, device="cpu")

    # state.current_epoch = 2 (déjà set par step 1).
    # Vérifier que le .pt epoch 1 existe pour resume.
    pt_path = models_dir / "train-state-gen-001-epoch-001.pt"
    assert pt_path.exists()

    final_npz = trainer_2.train_generation(
        gen=1, dataloader=dl, models_dir=models_dir, state_manager=state_mgr
    )

    # Final state.
    assert state_mgr.state.train.current_epoch == 4
    assert final_npz.name == "gen-001-trained.npz"
    assert final_npz.exists()

    # 4 .pt files (epochs 0, 1, 2, 3).
    pt_files = sorted(models_dir.glob("train-state-gen-001-epoch-*.pt"))
    assert len(pt_files) == 4


def test_resume_scenario_3_mid_epoch_lost(tmp_path: Path) -> None:
    """SPEC §11.5 scenario 3 : simule crash mid-epoch -> resume re-fait l'epoch.

    Setup :
    - Run 1 epoch propre (state.current_epoch=1, 1 .pt file).
    - Simule un crash mid-epoch 2 : state n'est PAS mis à 2 (l'update se fait
      après save_training_state, donc un crash AVANT save laisse state=1).
    - Resume : Fresh Trainer, state.current_epoch=1, total_epochs=3 -> 2 more
      epochs (1 et 2 0-indexed) — l'epoch 1 perdue est ré-exécutée.

    Expected :
    - 2 epochs ré-exécutés/exécutés (epochs 1 et 2 0-indexed).
    - .pt epoch 1 RE-écrit (re-fait).
    - state final current_epoch=3.
    """
    # Step 1 : 1 epoch propre.
    config_1 = TrainConfig(
        batch_size=2, num_workers=0, pin_memory=False, total_epochs=1, l2_reg=1e-4
    )
    trainer_1 = Trainer(model=NanoZeroResNet(), config=config_1, device="cpu")
    dataset = _MiniDictDataset(n_samples=4)
    dl = DataLoader(dataset, batch_size=2, shuffle=False)
    state_mgr = _make_mock_state_manager(current_epoch=0)
    models_dir = tmp_path / "models"

    trainer_1.train_generation(gen=1, dataloader=dl, models_dir=models_dir, state_manager=state_mgr)
    assert state_mgr.state.train.current_epoch == 1

    # Vérifier l'état après step 1 :
    # - 1 .pt file (epoch 0).
    # - Final cleanup a renommé gen-001-trained-epoch-000.npz en gen-001-trained.npz.
    # Note : avant resume, le final .npz du step 1 sera ré-écrasé par
    # atomic_rename au step 2 (final epoch = total_epochs-1).
    pt_path = models_dir / "train-state-gen-001-epoch-000.pt"
    assert pt_path.exists()

    # Step 2 : "simule mid-epoch 2 crash" -> state.current_epoch reste 1.
    # Resume avec total_epochs=3.
    config_3 = TrainConfig(
        batch_size=2, num_workers=0, pin_memory=False, total_epochs=3, l2_reg=1e-4
    )
    trainer_2 = Trainer(model=NanoZeroResNet(), config=config_3, device="cpu")

    trainer_2.train_generation(gen=1, dataloader=dl, models_dir=models_dir, state_manager=state_mgr)

    # Final state : current_epoch=3.
    assert state_mgr.state.train.current_epoch == 3
    # 3 .pt files (epochs 0, 1, 2).
    pt_files = sorted(models_dir.glob("train-state-gen-001-epoch-*.pt"))
    assert len(pt_files) == 3
