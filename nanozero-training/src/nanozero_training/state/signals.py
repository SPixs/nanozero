"""Signal handlers for graceful abort (ADR-009).

Catches SIGINT (Ctrl+C), SIGTERM (kill), SIGHUP (SSH disconnect).
Sets an abort flag the workers check between work units (batches/epochs).
Second consecutive Ctrl+C triggers immediate hard exit.
"""

from __future__ import annotations

import signal
import sys
from collections.abc import Callable
from types import FrameType


def install_signal_handlers(
    on_first_signal: Callable[[], None] | None = None,
) -> dict[str, bool]:
    """Install handlers for SIGINT/SIGTERM/SIGHUP.

    Args:
        on_first_signal: Optional callback invoked on first signal received
                         (eg. log message, partial state flush).

    Returns:
        abort_flag dict {requested: bool}. Workers should check
        abort_flag["requested"] between units and finish gracefully if True.
    """
    abort_flag = {"requested": False}

    def handler(signum: int, _frame: FrameType | None) -> None:
        if abort_flag["requested"]:
            print(
                f"\n[abort] Second signal {signum} received — hard exit.",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(130)
        print(
            f"\n[abort] Signal {signum} received — finishing current unit then aborting...",
            file=sys.stderr,
            flush=True,
        )
        abort_flag["requested"] = True
        if on_first_signal is not None:
            try:
                on_first_signal()
            except Exception as e:  # — best effort
                print(f"[abort] on_first_signal callback failed: {e}", file=sys.stderr)

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)
    if hasattr(signal, "SIGHUP"):  # POSIX only
        signal.signal(signal.SIGHUP, handler)

    return abort_flag
