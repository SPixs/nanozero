"""Unit tests for monitoring/csv_writer — MetricsCSVWriter."""

from __future__ import annotations

import csv
from pathlib import Path

from nanozero_training.monitoring.csv_writer import (
    EVAL_HEADERS,
    SELFPLAY_HEADERS,
    TRAIN_HEADERS,
    MetricsCSVWriter,
)


def test_writer_creates_directory(tmp_path: Path) -> None:
    target = tmp_path / "monitoring" / "deep"
    writer = MetricsCSVWriter(monitoring_dir=target)
    assert target.exists()
    assert writer.monitoring_dir == target


def test_append_train_row_creates_file_with_header(tmp_path: Path) -> None:
    writer = MetricsCSVWriter(monitoring_dir=tmp_path)
    writer.append_train_row(
        gen=1,
        epoch=0,
        policy_loss=2.5,
        value_loss=0.3,
        l2=0.01,
        total_loss=2.81,
        lr=1e-3,
        grad_norm=5.2,
    )
    path = tmp_path / "train-gen-001.csv"
    assert path.exists()
    with path.open() as f:
        lines = f.readlines()
    assert len(lines) == 2  # header + 1 row
    assert lines[0].strip().split(",") == TRAIN_HEADERS


def test_append_train_row_appends_subsequent_rows(tmp_path: Path) -> None:
    writer = MetricsCSVWriter(monitoring_dir=tmp_path)
    for epoch in range(3):
        writer.append_train_row(
            gen=1,
            epoch=epoch,
            policy_loss=2.5 - epoch * 0.1,
            value_loss=0.3,
            l2=0.01,
            total_loss=2.81 - epoch * 0.1,
            lr=1e-3,
            grad_norm=5.2,
        )
    path = tmp_path / "train-gen-001.csv"
    with path.open() as f:
        lines = f.readlines()
    assert len(lines) == 4  # header + 3 rows


def test_append_selfplay_row_separate_file(tmp_path: Path) -> None:
    writer = MetricsCSVWriter(monitoring_dir=tmp_path)
    writer.append_train_row(
        gen=1,
        epoch=0,
        policy_loss=1.0,
        value_loss=0.1,
        l2=0.01,
        total_loss=1.11,
        lr=1e-3,
        grad_norm=2.0,
    )
    writer.append_selfplay_row(gen=1, batch_idx=0, completed_games=100, target_games=250)
    assert (tmp_path / "train-gen-001.csv").exists()
    assert (tmp_path / "selfplay-gen-001.csv").exists()


def test_append_eval_row_handles_none_values(tmp_path: Path) -> None:
    """Phase 10 Q3 : None values written as empty string."""
    writer = MetricsCSVWriter(monitoring_dir=tmp_path)
    writer.append_eval_row(
        gen=2,
        games_played=10,
        llr=None,
        elo_diff=None,
        sprt_status="running",
    )
    path = tmp_path / "eval-gen-002.csv"
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["llr"] == ""
    assert rows[0]["elo_diff"] == ""
    assert rows[0]["sprt_status"] == "running"


def test_append_train_row_wall_clock_auto(tmp_path: Path) -> None:
    """wall_clock_s defaults to elapsed since writer creation."""
    writer = MetricsCSVWriter(monitoring_dir=tmp_path)
    writer.append_train_row(
        gen=1,
        epoch=0,
        policy_loss=1.0,
        value_loss=0.1,
        l2=0.01,
        total_loss=1.11,
        lr=1e-3,
        grad_norm=2.0,
    )
    path = tmp_path / "train-gen-001.csv"
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    wall_clock = float(rows[0]["wall_clock_s"])
    assert wall_clock >= 0.0


def test_append_train_row_wall_clock_override(tmp_path: Path) -> None:
    writer = MetricsCSVWriter(monitoring_dir=tmp_path)
    writer.append_train_row(
        gen=1,
        epoch=0,
        policy_loss=1.0,
        value_loss=0.1,
        l2=0.01,
        total_loss=1.11,
        lr=1e-3,
        grad_norm=2.0,
        wall_clock_s=42.5,
    )
    path = tmp_path / "train-gen-001.csv"
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert float(rows[0]["wall_clock_s"]) == 42.5


def test_filenames_use_3_digit_padding(tmp_path: Path) -> None:
    writer = MetricsCSVWriter(monitoring_dir=tmp_path)
    writer.append_train_row(
        gen=2,
        epoch=0,
        policy_loss=1.0,
        value_loss=0.1,
        l2=0.01,
        total_loss=1.11,
        lr=1e-3,
        grad_norm=2.0,
    )
    writer.append_train_row(
        gen=42,
        epoch=0,
        policy_loss=1.0,
        value_loss=0.1,
        l2=0.01,
        total_loss=1.11,
        lr=1e-3,
        grad_norm=2.0,
    )
    assert (tmp_path / "train-gen-002.csv").exists()
    assert (tmp_path / "train-gen-042.csv").exists()


def test_csv_parseable_with_dictreader(tmp_path: Path) -> None:
    writer = MetricsCSVWriter(monitoring_dir=tmp_path)
    writer.append_eval_row(
        gen=1,
        games_played=10,
        wins=4,
        losses=4,
        draws=2,
        llr=0.5,
        elo_diff=2.3,
        sprt_status="running",
    )
    writer.append_eval_row(
        gen=1,
        games_played=20,
        wins=8,
        losses=8,
        draws=4,
        llr=0.7,
        elo_diff=2.5,
        sprt_status="running",
    )
    path = tmp_path / "eval-gen-001.csv"
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert len(rows) == 2
    assert set(rows[0].keys()) == set(EVAL_HEADERS)
    assert int(rows[1]["games_played"]) == 20


def test_separate_csvs_per_gen(tmp_path: Path) -> None:
    """gen=1 et gen=2 -> deux fichiers distincts."""
    writer = MetricsCSVWriter(monitoring_dir=tmp_path)
    writer.append_train_row(
        gen=1,
        epoch=0,
        policy_loss=1.0,
        value_loss=0.1,
        l2=0.01,
        total_loss=1.11,
        lr=1e-3,
        grad_norm=2.0,
    )
    writer.append_train_row(
        gen=2,
        epoch=0,
        policy_loss=0.9,
        value_loss=0.08,
        l2=0.01,
        total_loss=0.99,
        lr=1e-3,
        grad_norm=1.8,
    )
    assert (tmp_path / "train-gen-001.csv").exists()
    assert (tmp_path / "train-gen-002.csv").exists()


def test_selfplay_headers_match_module_constant(tmp_path: Path) -> None:
    writer = MetricsCSVWriter(monitoring_dir=tmp_path)
    writer.append_selfplay_row(gen=1, batch_idx=0, completed_games=50, target_games=250)
    path = tmp_path / "selfplay-gen-001.csv"
    with path.open() as f:
        header_line = f.readline().strip()
    assert header_line.split(",") == SELFPLAY_HEADERS
