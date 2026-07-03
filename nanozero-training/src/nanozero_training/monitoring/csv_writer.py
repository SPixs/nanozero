"""CSV writers for training metrics (atomic append-only).

Three CSV files per generation (schéma SPEC §10.1):
  - train-gen-NNN.csv      : per-epoch training metrics
  - selfplay-gen-NNN.csv   : per-batch self-play metrics
  - eval-gen-NNN.csv       : per-poll SPRT eval metrics

Format design (Q3 décision actée):
  - Schéma fixe par fichier (colonnes définies au header).
  - Writer tolérant None : champs absents écrits "" dans la ligne.
  - Atomic append via utils.atomic_io.atomic_append_csv (existing helper).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanozero_training.utils.atomic_io import atomic_append_csv

# Schémas SPEC §10.1
TRAIN_HEADERS: list[str] = [
    "timestamp",
    "gen",
    "epoch",
    "policy_loss",
    "value_loss",
    "l2",
    "total_loss",
    "lr",
    "grad_norm",
    "wall_clock_s",
]
SELFPLAY_HEADERS: list[str] = [
    "timestamp",
    "gen",
    "batch_idx",
    "completed_games",
    "target_games",
    "wall_clock_s",
]
EVAL_HEADERS: list[str] = [
    "timestamp",
    "gen",
    "games_played",
    "wins",
    "losses",
    "draws",
    "llr",
    "elo_diff",
    "sprt_status",
    "wall_clock_s",
]


def _row_dict_to_list(row: dict[str, Any], headers: list[str]) -> list[Any]:
    """Convert a row dict to a list ordered per headers. None -> "" (Q3 actée)."""
    out: list[Any] = []
    for h in headers:
        v = row.get(h)
        out.append("" if v is None else v)
    return out


@dataclass
class MetricsCSVWriter:
    """Append-only CSV writer for training metrics.

    Usage:
        writer = MetricsCSVWriter(monitoring_dir=Path("nano-runs/run-001/monitoring"))
        writer.append_train_row(gen=2, epoch=3, policy_loss=2.3, value_loss=0.4, ...)
        writer.append_selfplay_row(gen=2, batch_idx=0, completed_games=47, target_games=250)
        writer.append_eval_row(gen=2, games_played=120, llr=1.2, sprt_status="running")

    Files created lazily as gen-NNN files at first append for a gen.
    Each append goes through atomic_append_csv (handles header on first write).
    """

    monitoring_dir: Path
    _run_start: float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())

    def __post_init__(self) -> None:
        self.monitoring_dir = Path(self.monitoring_dir)
        self.monitoring_dir.mkdir(parents=True, exist_ok=True)

    def append_train_row(
        self,
        gen: int,
        epoch: int,
        policy_loss: float,
        value_loss: float,
        l2: float,
        total_loss: float,
        lr: float,
        grad_norm: float,
        wall_clock_s: float | None = None,
    ) -> None:
        """Append one row to train-gen-NNN.csv."""
        row: dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "gen": gen,
            "epoch": epoch,
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "l2": l2,
            "total_loss": total_loss,
            "lr": lr,
            "grad_norm": grad_norm,
            "wall_clock_s": wall_clock_s if wall_clock_s is not None else self._elapsed(),
        }
        path = self.monitoring_dir / f"train-gen-{gen:03d}.csv"
        atomic_append_csv(path, _row_dict_to_list(row, TRAIN_HEADERS), TRAIN_HEADERS)

    def append_selfplay_row(
        self,
        gen: int,
        batch_idx: int,
        completed_games: int,
        target_games: int,
        wall_clock_s: float | None = None,
    ) -> None:
        """Append one row to selfplay-gen-NNN.csv."""
        row: dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "gen": gen,
            "batch_idx": batch_idx,
            "completed_games": completed_games,
            "target_games": target_games,
            "wall_clock_s": wall_clock_s if wall_clock_s is not None else self._elapsed(),
        }
        path = self.monitoring_dir / f"selfplay-gen-{gen:03d}.csv"
        atomic_append_csv(path, _row_dict_to_list(row, SELFPLAY_HEADERS), SELFPLAY_HEADERS)

    def append_eval_row(
        self,
        gen: int,
        games_played: int,
        wins: int = 0,
        losses: int = 0,
        draws: int = 0,
        llr: float | None = None,
        elo_diff: float | None = None,
        sprt_status: str = "running",
        wall_clock_s: float | None = None,
    ) -> None:
        """Append one row to eval-gen-NNN.csv. None values written as empty string."""
        row: dict[str, Any] = {
            "timestamp": _utc_now_iso(),
            "gen": gen,
            "games_played": games_played,
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "llr": llr,
            "elo_diff": elo_diff,
            "sprt_status": sprt_status,
            "wall_clock_s": wall_clock_s if wall_clock_s is not None else self._elapsed(),
        }
        path = self.monitoring_dir / f"eval-gen-{gen:03d}.csv"
        atomic_append_csv(path, _row_dict_to_list(row, EVAL_HEADERS), EVAL_HEADERS)

    def _elapsed(self) -> float:
        return datetime.now(timezone.utc).timestamp() - self._run_start


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
