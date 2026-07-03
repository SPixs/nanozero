"""Version manager for tracking model generations (ADR-007).

versions.yaml schema (SPEC §11.2, Reading A — 3 entries per promoted gen):
    current: gen-005-promoted
    latest_trained: gen-006-trained
    all:
      - name: gen-001-init
        type: init
        promoted: true
        seed: 42
        init_method: kaiming_standard
        created: "2026-05-12T08:00:00Z"
      - name: gen-002-trained
        type: trained
        promoted: true
        parent: gen-001-init
        sprt_result: h1_accepted
        sprt_games: 142
        sprt_elo_diff: 11.3
        sprt_llr: 2.95
        created: "2026-05-12T18:35:00Z"
      - name: gen-002-promoted
        type: promoted
        promoted: true
        parent: gen-002-trained
        sprt_result: h1_accepted
        sprt_games: 142
        sprt_elo_diff: 11.3
        sprt_llr: 2.95
        created: "2026-05-12T18:36:00Z"

Notes:
    `current` is the name of the currently-promoted champion (used as baseline for next SPRT).
    `latest_trained` is the most recent trained model (may not yet be promoted).
    `all` is the full history — append-only in normal flow.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from nanozero_training.utils.atomic_io import atomic_write_yaml

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelEntry:
    """One model entry in versions.yaml all[] list.

    Fields:
        name: model identifier, eg. "gen-002-trained" or "gen-005-promoted".
        type: one of "init", "trained", "promoted".
        promoted: whether this model was promoted (true once SPRT H1_ACCEPTED).
        parent: name of the parent model (None for init).
        seed: init seed (only set for type="init").
        init_method: init method name (only set for type="init").
        sprt_result: SPRT status string ("h1_accepted", "h0_accepted", "max_games")
            for trained/promoted models post-SPRT.
        sprt_games: number of games played in the SPRT.
        sprt_elo_diff: Elo difference estimate from SPRT.
        sprt_llr: final LLR value from SPRT.
        created: ISO 8601 UTC timestamp of entry creation.
    """

    name: str
    type: str  # "init" | "trained" | "promoted"
    promoted: bool = False
    parent: str | None = None
    seed: int | None = None
    init_method: str | None = None
    sprt_result: str | None = None
    sprt_games: int | None = None
    sprt_elo_diff: float | None = None
    sprt_llr: float | None = None
    created: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for YAML serialization. Excludes None/empty fields."""
        d: dict[str, Any] = {
            "name": self.name,
            "type": self.type,
            "promoted": self.promoted,
            "created": self.created,
        }
        if self.parent is not None:
            d["parent"] = self.parent
        if self.seed is not None:
            d["seed"] = self.seed
        if self.init_method is not None:
            d["init_method"] = self.init_method
        if self.sprt_result is not None:
            d["sprt_result"] = self.sprt_result
        if self.sprt_games is not None:
            d["sprt_games"] = self.sprt_games
        if self.sprt_elo_diff is not None:
            d["sprt_elo_diff"] = self.sprt_elo_diff
        if self.sprt_llr is not None:
            d["sprt_llr"] = self.sprt_llr
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ModelEntry:
        """Build from YAML dict. Tolerant to missing optional fields."""
        return cls(
            name=d["name"],
            type=d["type"],
            promoted=d.get("promoted", False),
            parent=d.get("parent"),
            seed=d.get("seed"),
            init_method=d.get("init_method"),
            sprt_result=d.get("sprt_result"),
            sprt_games=d.get("sprt_games"),
            sprt_elo_diff=d.get("sprt_elo_diff"),
            sprt_llr=d.get("sprt_llr"),
            created=d.get("created", ""),
        )


@dataclass
class Versions:
    """In-memory state of versions.yaml. Mutable for builder pattern."""

    current: str | None = None
    latest_trained: str | None = None
    all: list[ModelEntry] = field(default_factory=list)

    def find(self, name: str) -> ModelEntry | None:
        """Find an entry by name."""
        for e in self.all:
            if e.name == name:
                return e
        return None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"all": [e.to_dict() for e in self.all]}
        if self.current is not None:
            d["current"] = self.current
        if self.latest_trained is not None:
            d["latest_trained"] = self.latest_trained
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Versions:
        return cls(
            current=d.get("current"),
            latest_trained=d.get("latest_trained"),
            all=[ModelEntry.from_dict(e) for e in d.get("all", [])],
        )


class VersionManager:
    """Manage versions.yaml file with atomic writes (ADR-007).

    Usage:
        vm = VersionManager(versions_yaml_path="state/versions.yaml", models_dir=...)
        vm.create_new()  # or vm.load() if file exists
        vm.add_init(name="gen-001-init", seed=42, init_method="kaiming_standard")
        vm.save()
        # ... after training gen 2 ...
        vm.add_trained(name="gen-002-trained", parent="gen-001-init")
        vm.save()
        # ... after SPRT, see promotion.py for promote_if_h1 ...
    """

    def __init__(
        self,
        versions_yaml_path: str | Path,
        models_dir: str | Path,
    ) -> None:
        """Initialize manager.

        Args:
            versions_yaml_path: path to versions.yaml file.
            models_dir: directory where .npz model files reside.
        """
        self.path = Path(versions_yaml_path)
        self.models_dir = Path(models_dir)
        self._versions: Versions | None = None

    @property
    def versions(self) -> Versions:
        if self._versions is None:
            raise RuntimeError("Versions not loaded. Call load() or create_new() first.")
        return self._versions

    def create_new(self) -> None:
        """Initialize empty versions (no file yet on disk)."""
        self._versions = Versions()

    def load(self, auto_reconcile: bool = True) -> None:
        """Load versions.yaml from disk.

        Args:
            auto_reconcile: if True (default), automatically detect+repair
                mid-promote crashes (SPEC §11.5 scenario 6) via
                reconcile_on_load(). Set False for inspection without
                filesystem mutations.

        Raises:
            FileNotFoundError: if file doesn't exist (use create_new for fresh).
            ValueError: if YAML malformed or missing required structure.
            RuntimeError: if auto_reconcile detects inconsistency too deep
                to recover automatically.
        """
        if not self.path.exists():
            raise FileNotFoundError(
                f"versions.yaml not found at {self.path}. " "Use create_new() for a fresh run."
            )
        with self.path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        if not isinstance(data, dict):
            raise ValueError(
                f"Malformed versions.yaml at {self.path}: "
                f"expected dict, got {type(data).__name__}"
            )
        self._versions = Versions.from_dict(data)

        if auto_reconcile:
            # Local import to break circular dep (promotion imports manager).
            from nanozero_training.version.promotion import reconcile_on_load

            reconciled = reconcile_on_load(self)
            if reconciled:
                self.save()

    def save(self) -> None:
        """Atomically write versions.yaml to disk."""
        atomic_write_yaml(self.path, self.versions.to_dict())

    def add_init(
        self,
        name: str,
        seed: int,
        init_method: str = "kaiming_standard",
    ) -> ModelEntry:
        """Register the initial (gen-0) model.

        Args:
            name: model name, eg. "gen-001-init".
            seed: init RNG seed.
            init_method: init method name.

        Returns:
            The created ModelEntry.

        Raises:
            ValueError: if a model with this name already exists.
        """
        if self.versions.find(name) is not None:
            raise ValueError(f"Model already registered: {name}")
        entry = ModelEntry(
            name=name,
            type="init",
            promoted=True,  # init is implicitly the first champion
            seed=seed,
            init_method=init_method,
            created=_utc_now_iso(),
        )
        self.versions.all.append(entry)
        if self.versions.current is None:
            self.versions.current = name
        self.versions.latest_trained = name
        return entry

    def add_trained(
        self,
        name: str,
        parent: str,
    ) -> ModelEntry:
        """Register a trained-but-not-yet-evaluated model.

        Idempotent sur ré-entraînement d'un slot NON promu : si une entrée
        ``name`` existe déjà mais n'a pas été promue (rejetée au SPRT ou jamais
        évaluée), elle est REMPLACÉE — ce qui permet de refaire une gen rejetée
        sans éditer ``versions.yaml`` à la main (cf. premier rejet SPRT gen-026,
        mémoire rejected-gen-continuation-gap). Un slot déjà PROMU ne peut jamais
        être écrasé (lève).

        Args:
            name: model name, eg. "gen-002-trained".
            parent: parent model name (must exist).

        Returns:
            The created ModelEntry.

        Raises:
            ValueError: if name already exists AND was promoted, or parent unknown.
        """
        existing = self.versions.find(name)
        if existing is not None and existing.promoted:
            raise ValueError(f"Cannot re-train an already-promoted slot: {name}")
        if self.versions.find(parent) is None:
            raise ValueError(f"Parent model not found: {parent}")
        if existing is not None:
            LOG.warning(
                "Re-training non-promoted slot %s — replacing previous entry "
                "(statut SPRT précédent: %s)",
                name,
                existing.sprt_result,
            )
            self.versions.all = [e for e in self.versions.all if e.name != name]
        entry = ModelEntry(
            name=name,
            type="trained",
            promoted=False,
            parent=parent,
            created=_utc_now_iso(),
        )
        self.versions.all.append(entry)
        self.versions.latest_trained = name
        return entry

    def get_current(self) -> str:
        """Return name of current champion model."""
        if self.versions.current is None:
            raise RuntimeError("No current champion (versions empty)")
        return self.versions.current

    def get_latest_trained(self) -> str | None:
        """Return name of most recent trained model, or None."""
        return self.versions.latest_trained

    def get_path_for(self, model_name: str) -> Path:
        """Return absolute path to a model's .npz file.

        Args:
            model_name: model identifier (eg. "gen-002-promoted").

        Returns:
            Path to {models_dir}/{model_name}.npz.

        Notes:
            Does NOT check existence — caller must verify if needed.
        """
        return self.models_dir / f"{model_name}.npz"


def _utc_now_iso() -> str:
    """Current UTC time as ISO 8601 string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
