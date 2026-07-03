"""Unit tests for storage/models.py — registry + promotion lifecycle."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest
from nanozero_jobserver.storage.db import init_schema
from nanozero_jobserver.storage.models import (
    current_model,
    get_model,
    list_models,
    promote_model,
    register_model,
    set_current_champion,
)


@pytest.fixture()
def db(tmp_path: Path) -> Path:
    p = tmp_path / "models.db"
    init_schema(p)
    return p


def test_register_model_inserts_record(db: Path) -> None:
    register_model(db, version=1, name="gen-001-init", onnx_path="/p/v1.onnx", sha256_onnx="aaa")
    model = get_model(db, 1)
    assert model is not None
    assert model.version == 1
    assert model.name == "gen-001-init"
    assert model.onnx_path == "/p/v1.onnx"
    assert model.sha256_onnx == "aaa"
    assert model.promoted_at is None  # not promoted yet
    assert model.parent_version is None  # init has no parent


def test_register_with_parent(db: Path) -> None:
    register_model(db, version=1, name="v1", onnx_path="/p", sha256_onnx="a")
    register_model(db, version=2, name="v2", onnx_path="/p2", sha256_onnx="b", parent_version=1)
    model = get_model(db, 2)
    assert model is not None
    assert model.parent_version == 1


def test_register_duplicate_version_raises(db: Path) -> None:
    register_model(db, version=1, name="v1", onnx_path="/p", sha256_onnx="a")
    with pytest.raises(sqlite3.IntegrityError):
        register_model(db, version=1, name="v1-duplicate", onnx_path="/p", sha256_onnx="b")


def test_register_duplicate_name_raises(db: Path) -> None:
    """name is UNIQUE — humans shouldn't see two 'gen-005-trained' in the registry."""
    register_model(db, version=1, name="dupname", onnx_path="/p", sha256_onnx="a")
    with pytest.raises(sqlite3.IntegrityError):
        register_model(db, version=2, name="dupname", onnx_path="/p2", sha256_onnx="b")


def test_register_with_unknown_parent_raises(db: Path) -> None:
    """Foreign-key violation when parent_version doesn't exist."""
    with pytest.raises(sqlite3.IntegrityError):
        register_model(db, version=2, name="v2", onnx_path="/p", sha256_onnx="a", parent_version=99)


def test_get_model_returns_none_when_absent(db: Path) -> None:
    assert get_model(db, 42) is None


# -----------------------------------------------------------------------------
# Promotion
# -----------------------------------------------------------------------------


def test_promote_model_sets_promoted_at(db: Path) -> None:
    register_model(db, version=1, name="v1", onnx_path="/p", sha256_onnx="a")
    assert promote_model(db, 1) is True
    model = get_model(db, 1)
    assert model is not None
    assert model.promoted_at is not None


def test_promote_unknown_model_returns_false(db: Path) -> None:
    assert promote_model(db, 99) is False


def test_promote_twice_returns_false_second_time(db: Path) -> None:
    register_model(db, version=1, name="v1", onnx_path="/p", sha256_onnx="a")
    assert promote_model(db, 1) is True
    assert promote_model(db, 1) is False  # already promoted


def test_current_model_returns_most_recent_promotion(db: Path) -> None:
    register_model(db, version=1, name="v1", onnx_path="/p1", sha256_onnx="a")
    register_model(db, version=2, name="v2", onnx_path="/p2", sha256_onnx="b", parent_version=1)
    register_model(db, version=3, name="v3", onnx_path="/p3", sha256_onnx="c", parent_version=2)

    assert current_model(db) is None  # nothing promoted yet
    promote_model(db, 1)
    assert current_model(db).version == 1  # type: ignore[union-attr]
    time.sleep(0.01)
    promote_model(db, 3)
    assert current_model(db).version == 3  # type: ignore[union-attr]


def test_current_model_none_when_nothing_promoted(db: Path) -> None:
    register_model(db, version=1, name="v1", onnx_path="/p", sha256_onnx="a")
    assert current_model(db) is None


# -----------------------------------------------------------------------------
# Listing
# -----------------------------------------------------------------------------


def test_list_models_returns_descending(db: Path) -> None:
    register_model(db, version=1, name="v1", onnx_path="/1", sha256_onnx="a")
    register_model(db, version=2, name="v2", onnx_path="/2", sha256_onnx="b", parent_version=1)
    register_model(db, version=3, name="v3", onnx_path="/3", sha256_onnx="c", parent_version=2)
    items = list_models(db)
    assert [m.version for m in items] == [3, 2, 1]


def test_list_models_respects_limit(db: Path) -> None:
    for v in range(1, 8):
        parent = v - 1 if v > 1 else None
        register_model(
            db,
            version=v,
            name=f"v{v}",
            onnx_path=f"/p{v}",
            sha256_onnx=f"h{v}",
            parent_version=parent,
        )
    items = list_models(db, limit=3)
    assert len(items) == 3
    assert [m.version for m in items] == [7, 6, 5]


def test_set_current_champion_rollback(db: Path) -> None:
    """Rollback : re-promouvoir un ancien champion déjà promu → il redevient current."""
    register_model(db, version=1, name="g1", onnx_path="/1", sha256_onnx="a")
    register_model(db, version=2, name="g2", onnx_path="/2", sha256_onnx="b")
    promote_model(db, 1)
    time.sleep(0.01)
    promote_model(db, 2)
    assert current_model(db).version == 2  # g2 champion
    # promote refuse (déjà promu) ; set_current force le retour à g1.
    assert promote_model(db, 1) is False
    time.sleep(0.01)
    assert set_current_champion(db, 1) is True
    assert current_model(db).version == 1  # g1 de retour comme champion


def test_set_current_champion_unknown_version(db: Path) -> None:
    assert set_current_champion(db, 999) is False
