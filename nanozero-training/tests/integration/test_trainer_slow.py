"""Slow integration : mini-training 2 epochs end-to-end on tiny on-the-fly dataset.

Valide la chaîne complète :
- Dataset .npz produit on-the-fly (write_batch + NpzDataset + load_replay_buffer)
- NanoZeroResNet
- Trainer.train_generation avec 2 epochs
- Checkpoints atomiques (npz + pt)
- Final gen-NNN-trained.npz produit
- RunStateManager updates persistés

Wall-clock target : < 2 min CPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from nanozero_training.data.npz_writer import write_batch
from nanozero_training.data.sample import Sample, make_valid_sample_arrays
from nanozero_training.network.resnet import NanoZeroResNet
from nanozero_training.state.manager import RunStateManager
from nanozero_training.train.config import TrainConfig
from nanozero_training.train.dataloader import make_train_dataloader_for_gen
from nanozero_training.train.trainer import Trainer


def _make_dataset_on_the_fly(datasets_dir: Path, n_samples: int = 32) -> None:
    """Generate tiny selfplay-gen001-batch-000.npz avec n_samples valid Samples."""
    samples = []
    for i in range(n_samples):
        kwargs = make_valid_sample_arrays()
        kwargs["ply"] = i  # vary ply pour diversité
        samples.append(Sample(**kwargs))
    write_batch(
        samples,
        datasets_dir / "selfplay-gen001-batch-000.npz",
        gen=1,
        batch_idx=0,
        n_games=4,
    )


@pytest.mark.slow()
def test_mini_training_2_epochs_end_to_end(tmp_path: Path) -> None:
    """Run 2 epochs sur 32 samples on-the-fly. Validation full chain."""
    datasets_dir = tmp_path / "datasets"
    datasets_dir.mkdir()
    _make_dataset_on_the_fly(datasets_dir, n_samples=32)

    models_dir = tmp_path / "models"
    monitoring_dir = tmp_path / "monitoring"

    state_mgr = RunStateManager(monitoring_dir=monitoring_dir)
    state_mgr.create_new(
        run_id="test-train-001",
        config_path="dummy.yaml",
        config_hash="dummy",
        max_generations=1,
        target_games_per_gen=4,
    )

    config = TrainConfig(
        batch_size=8,
        total_epochs=2,
        replay_window=1,
        num_workers=0,
        pin_memory=False,
        learning_rate=1e-3,
    )

    model = NanoZeroResNet()
    trainer = Trainer(model=model, config=config, device="cpu")

    dataloader = make_train_dataloader_for_gen(
        datasets_dir=datasets_dir,
        current_gen=1,
        config=config,
    )

    final_npz = trainer.train_generation(
        gen=1,
        dataloader=dataloader,
        models_dir=models_dir,
        state_manager=state_mgr,
    )

    # Assertions.
    assert final_npz.exists()
    assert final_npz.name == "gen-001-trained.npz"
    assert state_mgr.state.train.current_epoch == 2
    # 2 .pt files (1 par epoch).
    pt_files = list(models_dir.glob("train-state-gen-001-epoch-*.pt"))
    assert len(pt_files) == 2
    # Intermediates .npz nettoyés (seul final reste).
    npz_files = list(models_dir.glob("gen-001-trained*.npz"))
    assert len(npz_files) == 1
    assert npz_files[0].name == "gen-001-trained.npz"
    # State persisté on-disk.
    assert (monitoring_dir / "run_state.yaml").exists()
