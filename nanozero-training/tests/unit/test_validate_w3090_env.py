"""Unit tests for scripts/validate_w3090_env.py (phase 1.0.0-12-prep).

The script is loaded via importlib so it stays a standalone executable
under scripts/ (no package nesting), yet remains unit-testable.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT_PATH = Path(__file__).resolve().parent.parent.parent / "scripts" / "validate_w3090_env.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("validate_w3090_env", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # Register in sys.modules BEFORE exec so @dataclass can resolve its module.
    sys.modules["validate_w3090_env"] = mod
    spec.loader.exec_module(mod)
    return mod


VEM = _load_module()


def test_check_python_version_passes() -> None:
    """Test runs on Python 3.10+ so the check must report ok=True."""
    result = VEM.check_python_version()
    assert result.ok is True
    assert result.critical is True
    assert str(sys.version_info.major) in result.message


def test_check_fastchess_in_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """When shutil.which('fastchess') resolves, result is ok=True with path."""
    monkeypatch.setattr(VEM.shutil, "which", lambda _name: "/usr/bin/fastchess")
    result = VEM.check_fastchess()
    assert result.ok is True
    assert "/usr/bin/fastchess" in result.message


def test_check_fastchess_not_in_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """When shutil.which returns None, result is ok=False critical."""
    monkeypatch.setattr(VEM.shutil, "which", lambda _name: None)
    result = VEM.check_fastchess()
    assert result.ok is False
    assert result.critical is True
    assert "not found" in result.message


def test_check_uci_jar_exists(tmp_path: Path) -> None:
    """JAR present at path → ok=True. JAR absent → ok=False critical."""
    jar = tmp_path / "nanozero-uci-1.2.0.jar"
    jar.write_bytes(b"PK\x03\x04 fake jar")
    result_ok = VEM.check_uci_jar(jar)
    assert result_ok.ok is True

    missing = tmp_path / "absent.jar"
    result_fail = VEM.check_uci_jar(missing)
    assert result_fail.ok is False
    assert result_fail.critical is True
    assert "not found" in result_fail.message


def test_check_run_root_writable(tmp_path: Path) -> None:
    """Writable run_root → ok=True. Inexistent parent we can mkdir → ok=True."""
    result = VEM.check_run_root_writable(tmp_path / "new-run-root")
    assert result.ok is True
    assert result.critical is True
    assert (tmp_path / "new-run-root").exists()
