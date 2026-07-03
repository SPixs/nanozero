"""YAML config loading + validation + CLI overrides merging."""

from __future__ import annotations

import hashlib
from dataclasses import fields, replace
from pathlib import Path
from typing import Any

import yaml

from nanozero_training.config.run_config import (
    MonitorConfig,
    PathsConfig,
    RunConfig,
)
from nanozero_training.eval.fastchess_runner import FastchessConfig
from nanozero_training.selfplay.config import SelfplayConfig
from nanozero_training.train.config import TrainConfig


def load_config(path: str | Path) -> RunConfig:
    """Load a YAML config file into a validated RunConfig.

    Args:
        path: path to YAML config file.

    Returns:
        Validated RunConfig (all __post_init__ checks passed).

    Raises:
        FileNotFoundError: file doesn't exist.
        ValueError: malformed YAML or validation failure.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Malformed YAML at {path}: {e}") from e

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a dict at {path}, got {type(data).__name__}")

    return _build_run_config(data)


def compute_config_hash(yaml_content: str) -> str:
    """Compute sha256 hex digest of YAML content for audit/reproducibility.

    Args:
        yaml_content: raw YAML text (as read from file).

    Returns:
        Hex string of sha256 digest (64 chars).

    Notes:
        Hash is over RAW content, including whitespace/comments.
        Two functionally-identical configs with different whitespace will have
        DIFFERENT hashes — intentional, audit reflects user's input verbatim.
        For functional equality check, normalize via yaml.safe_load + dump first.
    """
    return hashlib.sha256(yaml_content.encode("utf-8")).hexdigest()


def merge_cli_overrides(config: RunConfig, **overrides: Any) -> RunConfig:
    """Apply CLI overrides to a loaded RunConfig.

    Args:
        config: base RunConfig (from YAML).
        **overrides: keyword args like batch_size=256, mcts_sims=400, etc.
            None values are ignored (click default for absent options).
            Keys can be flat (mcts_sims) or dotted (selfplay.mcts_sims).

    Returns:
        New RunConfig with overrides applied.

    Notes:
        Flat keys are matched against any sub-config field. If multiple sub-configs
        have the same field name, use dotted syntax to disambiguate.
    """
    overrides = {k: v for k, v in overrides.items() if v is not None}
    if not overrides:
        return config

    new_selfplay = _apply_overrides(config.selfplay, overrides, prefix="selfplay")
    new_train = _apply_overrides(config.train, overrides, prefix="train")
    new_eval = _apply_overrides(config.eval_fastchess, overrides, prefix="eval")
    new_monitor = _apply_overrides(config.monitor, overrides, prefix="monitor")
    new_paths = _apply_overrides(config.paths, overrides, prefix="paths")

    top_overrides = {k: v for k, v in overrides.items() if k in {"max_generations", "run_id"}}
    return replace(
        config,
        selfplay=new_selfplay,
        train=new_train,
        eval_fastchess=new_eval,
        monitor=new_monitor,
        paths=new_paths,
        **top_overrides,
    )


def _build_run_config(data: dict[str, Any]) -> RunConfig:
    """Build RunConfig from parsed YAML dict.

    Each sub-section is built via dataclass(**section_dict), respecting defaults
    for missing fields. Unknown fields raise TypeError.
    """
    selfplay = SelfplayConfig(**data.get("selfplay", {}))
    train = TrainConfig(**data.get("train", {}))
    eval_cfg = FastchessConfig(**data.get("eval", {}))
    monitor = MonitorConfig(**data.get("monitor", {}))
    paths = PathsConfig(**data.get("paths", {}))

    top_kwargs: dict[str, Any] = {}
    if "max_generations" in data:
        top_kwargs["max_generations"] = data["max_generations"]
    if "run_id" in data:
        top_kwargs["run_id"] = data["run_id"]

    return RunConfig(
        selfplay=selfplay,
        train=train,
        eval_fastchess=eval_cfg,
        monitor=monitor,
        paths=paths,
        **top_kwargs,
    )


def _apply_overrides(sub_config: Any, overrides: dict[str, Any], prefix: str) -> Any:
    """Apply overrides to a sub-config dataclass.

    Handles both flat (key=mcts_sims) and dotted (key=selfplay.mcts_sims) overrides.
    Flat overrides match by field name ; dotted overrides match by prefix.field_name.
    """
    sub_fields = {f.name for f in fields(sub_config)}
    to_apply: dict[str, Any] = {}
    for key, value in overrides.items():
        if "." in key:
            section, field_name = key.split(".", 1)
            if section == prefix and field_name in sub_fields:
                to_apply[field_name] = value
        elif key in sub_fields:
            to_apply[key] = value

    if not to_apply:
        return sub_config
    return replace(sub_config, **to_apply)
