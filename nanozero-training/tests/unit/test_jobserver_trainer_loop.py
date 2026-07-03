"""Unit tests for jobserver_client.trainer_loop (Phase 13.5b).

Mocks the TrainerClient and the existing Trainer so we can exercise the
orchestration logic without firing up PyTorch + GPU + a real server.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from nanozero_training.jobserver_client.trainer import DecodedPosition, ShouldTrainStatus
from nanozero_training.jobserver_client.trainer_loop import (
    JobserverReplayDataset,
    TrainerStreamingLoop,
    compute_sha256,
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _make_position(value: float = 0.0, ply: int = 0, model_version: int = 1) -> DecodedPosition:
    return DecodedPosition(
        input_planes=np.zeros((119, 8, 8), dtype=np.float32),
        policy_target=np.zeros((4672,), dtype=np.float32),
        value_target=value,
        ply=ply,
        model_version=model_version,
    )


def _make_loop(tmp_path: Path, *, threshold: int = 10, poll_interval: float = 0.0):
    """Build a TrainerStreamingLoop with all collaborators mocked."""
    client = MagicMock()
    trainer = MagicMock()
    trainer.train_epoch.return_value = {"total_loss": 1.0}
    trainer.make_optimizer.return_value = MagicMock()
    trainer.make_scheduler.return_value = MagicMock()

    loop = TrainerStreamingLoop(
        client=client,
        trainer=trainer,
        models_dir=tmp_path,
        threshold=threshold,
        window=5,
        sample_size=threshold,
        batch_size=2,
        poll_interval_seconds=poll_interval,
    )
    return loop, client, trainer


# -----------------------------------------------------------------------------
# JobserverReplayDataset
# -----------------------------------------------------------------------------


def test_dataset_length_matches_positions():
    positions = [_make_position(ply=i) for i in range(10)]
    ds = JobserverReplayDataset(positions)
    assert len(ds) == 10


def test_dataset_returns_correct_keys():
    positions = [_make_position(value=0.5, ply=3)]
    ds = JobserverReplayDataset(positions)
    item = ds[0]
    assert set(item.keys()) == {"input_planes", "policy_target", "value_target", "turn", "ply"}
    assert item["input_planes"].shape == (119, 8, 8)
    assert item["policy_target"].shape == (4672,)
    assert item["value_target"].item() == 0.5
    assert item["ply"].item() == 3


def test_dataset_supports_iteration():
    positions = [_make_position(ply=i) for i in range(5)]
    ds = JobserverReplayDataset(positions)
    items = [ds[i] for i in range(len(ds))]
    assert len(items) == 5
    assert [it["ply"].item() for it in items] == [0, 1, 2, 3, 4]


# -----------------------------------------------------------------------------
# compute_sha256
# -----------------------------------------------------------------------------


def test_sha256_known_content(tmp_path: Path):
    """SHA-256 of "hello\n" is a well-known constant — verify byte-for-byte read."""
    p = tmp_path / "hello.txt"
    p.write_bytes(b"hello\n")
    expected = "5891b5b522d5df086d0ff0b110fbd9d21bb4fc7163af34d08286a2e846f6be03"
    assert compute_sha256(p) == expected


def test_sha256_handles_large_file_streaming(tmp_path: Path):
    """compute_sha256 must work on >64 KB files (streaming read with chunks)."""
    big_path = tmp_path / "big.bin"
    big_path.write_bytes(b"\x00" * (1024 * 1024))  # 1 MB
    digest = compute_sha256(big_path)
    assert len(digest) == 64


# -----------------------------------------------------------------------------
# TrainerStreamingLoop — should_train trigger
# -----------------------------------------------------------------------------


def test_loop_skips_when_should_train_false(tmp_path: Path):
    """should_train=False → _run_one_iteration returns False, no promotion.

    Tests the single-iteration path directly (calling loop.run with
    max_promotions=1 would spin forever because promotions never increments,
    and MagicMock would leak call_args_list — observed 25 MB/s growth).
    """
    loop, client, _trainer = _make_loop(tmp_path)
    client.should_train.return_value = ShouldTrainStatus(
        should_train=False, new_positions=5, threshold=10
    )

    result = loop._run_one_iteration()

    assert result is False
    client.should_train.assert_called_once()


def test_loop_train_and_promote_when_should_train_true(tmp_path: Path):
    loop, client, trainer = _make_loop(tmp_path)
    client.should_train.return_value = ShouldTrainStatus(
        should_train=True, new_positions=20, threshold=10
    )
    client.current_model.return_value = {"version": 3, "name": "gen-003-trained"}
    client.fetch_sample.return_value = [_make_position(ply=i) for i in range(4)]

    with (
        patch("nanozero_training.network.export_npz.export_to_npz"),
        patch("nanozero_training.network.export_onnx.export_onnx_companion") as mock_onnx,
    ):
        # Make export_onnx_companion produce a real file we can sha256.
        onnx_path = tmp_path / "gen-004-trained.onnx"
        onnx_path.write_bytes(b"FAKE_ONNX")
        mock_onnx.return_value = onnx_path

        # Simulate one iteration directly.
        result = loop._run_one_iteration()

    assert result is True
    client.fetch_sample.assert_called_once()
    trainer.train_epoch.assert_called_once()
    client.register_model.assert_called_once()
    client.promote_model.assert_called_once_with(version=4)


def test_loop_uses_version_zero_when_no_current_model(tmp_path: Path):
    """First training round : no promoted model exists yet."""
    loop, client, trainer = _make_loop(tmp_path)
    client.should_train.return_value = ShouldTrainStatus(
        should_train=True, new_positions=20, threshold=10
    )
    client.current_model.return_value = None  # nothing promoted yet
    client.fetch_sample.return_value = [_make_position(ply=0)]

    with (
        patch("nanozero_training.network.export_npz.export_to_npz"),
        patch("nanozero_training.network.export_onnx.export_onnx_companion") as mock_onnx,
    ):
        onnx_path = tmp_path / "gen-001-trained.onnx"
        onnx_path.write_bytes(b"x")
        mock_onnx.return_value = onnx_path
        loop._run_one_iteration()

    # First version becomes 1 (current_version=0 + 1).
    client.promote_model.assert_called_once_with(version=1)
    register_call = client.register_model.call_args
    # parent_version should be omitted/None for the first model.
    assert register_call.kwargs.get("parent_version") is None


def test_loop_skips_when_fetch_returns_empty(tmp_path: Path):
    """Should_train says yes, but fetch returns no positions → skip."""
    loop, client, trainer = _make_loop(tmp_path)
    client.should_train.return_value = ShouldTrainStatus(
        should_train=True, new_positions=20, threshold=10
    )
    client.current_model.return_value = {"version": 1}
    client.fetch_sample.return_value = []

    result = loop._run_one_iteration()

    assert result is False
    trainer.train_epoch.assert_not_called()
    client.register_model.assert_not_called()


# -----------------------------------------------------------------------------
# Run loop with max_promotions
# -----------------------------------------------------------------------------


def test_run_stops_at_max_promotions(tmp_path: Path):
    """If we set max_promotions=2 and trigger fires every iteration, stop at 2."""
    loop, client, trainer = _make_loop(tmp_path)
    client.should_train.return_value = ShouldTrainStatus(True, 20, 10)
    client.current_model.return_value = {"version": 0}
    client.fetch_sample.return_value = [_make_position()]

    with (
        patch("nanozero_training.network.export_npz.export_to_npz"),
        patch("nanozero_training.network.export_onnx.export_onnx_companion") as mock_onnx,
    ):
        onnx_path = tmp_path / "x.onnx"
        onnx_path.write_bytes(b"y")
        mock_onnx.return_value = onnx_path

        promotions = loop.run(max_promotions=2)

    assert promotions == 2
    assert client.register_model.call_count == 2
    assert client.promote_model.call_count == 2


def test_run_recovers_from_iteration_exception(tmp_path: Path):
    """If train_epoch raises, run() catches, sleeps, and continues."""
    loop, client, trainer = _make_loop(tmp_path, poll_interval=0.0)
    # First call : raise. Second call : succeed.
    call_count = {"n": 0}

    def _flaky_train(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated training crash")
        return {"loss": 0.5}

    trainer.train_epoch.side_effect = _flaky_train
    client.should_train.return_value = ShouldTrainStatus(True, 20, 10)
    client.current_model.return_value = {"version": 0}
    client.fetch_sample.return_value = [_make_position()]

    with (
        patch("nanozero_training.network.export_npz.export_to_npz"),
        patch("nanozero_training.network.export_onnx.export_onnx_companion") as mock_onnx,
    ):
        onnx_path = tmp_path / "y.onnx"
        onnx_path.write_bytes(b"z")
        mock_onnx.return_value = onnx_path

        promotions = loop.run(max_promotions=1)

    assert promotions == 1
    assert call_count["n"] == 2  # one crash + one success
