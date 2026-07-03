"""Tests for state/signals.py — graceful abort handlers (ADR-009)."""

from __future__ import annotations

import signal
from collections.abc import Iterator
from typing import Any
from unittest.mock import MagicMock

import pytest
from nanozero_training.state.signals import install_signal_handlers


@pytest.fixture(autouse=True)
def _restore_signal_handlers() -> Iterator[None]:
    """Save and restore SIGINT/SIGTERM/SIGHUP handlers around each test.

    Autouse so every test in this module gets the restore for free without
    declaring the fixture as a parameter (ruff PT019 compliance).
    """
    saved: dict[int, Any] = {}
    for sig_name in ("SIGINT", "SIGTERM", "SIGHUP"):
        if hasattr(signal, sig_name):
            sig = getattr(signal, sig_name)
            saved[sig] = signal.getsignal(sig)
    try:
        yield
    finally:
        for sig, h in saved.items():
            signal.signal(sig, h)


def test_abort_flag_starts_false() -> None:
    abort_flag = install_signal_handlers()
    assert abort_flag == {"requested": False}


def test_handler_sets_abort_flag() -> None:
    abort_flag = install_signal_handlers()
    signal.raise_signal(signal.SIGINT)
    assert abort_flag["requested"] is True


def test_callback_invoked_on_first_signal() -> None:
    callback = MagicMock()
    abort_flag = install_signal_handlers(on_first_signal=callback)
    signal.raise_signal(signal.SIGTERM)
    callback.assert_called_once()
    assert abort_flag["requested"] is True


def test_callback_exception_is_swallowed(capsys: pytest.CaptureFixture[str]) -> None:
    def bad_callback() -> None:
        raise RuntimeError("callback boom")

    abort_flag = install_signal_handlers(on_first_signal=bad_callback)
    # Should not propagate the callback exception
    signal.raise_signal(signal.SIGINT)
    assert abort_flag["requested"] is True
    captured = capsys.readouterr()
    assert "on_first_signal callback failed" in captured.err
    assert "callback boom" in captured.err


def test_second_signal_calls_sys_exit(capsys: pytest.CaptureFixture[str]) -> None:
    abort_flag = install_signal_handlers()
    signal.raise_signal(signal.SIGINT)  # First → set flag
    assert abort_flag["requested"] is True
    with pytest.raises(SystemExit) as exc_info:
        signal.raise_signal(signal.SIGINT)  # Second → hard exit
    assert exc_info.value.code == 130
    captured = capsys.readouterr()
    assert "Second signal" in captured.err
