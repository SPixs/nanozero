"""Persistent UCI subprocess client for self-play (ADR-001).

Pattern : 1 subprocess par worker. Lifecycle :
  start() once -> {go_nodes(), new_game()} many times -> quit() | restart().

Reader thread stdout : producer-consumer pour drainer stdout sans bloquer.
Le thread lit ligne par ligne et append à un buffer thread-safe.

Thread safety : single-threaded usage v1.0.0. Le reader thread est background
mais l'API send/read n'est PAS concurrent-safe (2 calls go_nodes simultanés
casseraient le buffer).
"""

from __future__ import annotations

import contextlib
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import IO

from nanozero_training.selfplay.visits_parser import parse_visits_line


class UciTimeoutError(RuntimeError):
    """Raised when UCI subprocess doesn't respond within timeout."""


class UciCrashError(RuntimeError):
    """Raised when UCI subprocess exits unexpectedly."""


@dataclass(frozen=True)
class UciResult:
    """Result of a single search.

    Attributes:
        visits: {uci_move: count}, empty dict if terminal position.
        bestmove: UCI bestmove string, or None if terminal ('bestmove (none)').
    """

    visits: dict[str, int]
    bestmove: str | None


_POLL_INTERVAL_S = 0.01  # 10 ms poll sur le buffer stdout


class UciClient:
    """Persistent UCI subprocess client (cf. ADR-001 selfplay-subprocess-uci).

    NOT thread-safe pour des appels concurrents send/read. Le reader thread
    interne est strictement le seul consommateur de stdout du subprocess.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stdout_buffer: deque[str] = deque()
        self._stdout_lock = threading.Lock()
        # Last successful start() params (for restart()).
        self._uci_jar: Path | None = None
        self._model_path: Path | None = None
        self._dirichlet_alpha: int = 300
        self._dirichlet_epsilon: int = 250
        self._dirichlet_seed: int = 42
        self._handshake_timeout: float = 30.0

    # ------------------------------------------------------------------ lifecycle

    def start(
        self,
        uci_jar: str | Path,
        model_path: str | Path,
        dirichlet_alpha: int = 300,
        dirichlet_epsilon: int = 250,
        dirichlet_seed: int = 42,
        handshake_timeout: float = 30.0,
    ) -> None:
        """Start the UCI subprocess and perform handshake.

        Raises:
            FileNotFoundError: jar or model not found.
            UciTimeoutError: handshake timeout exceeded.
            UciCrashError: subprocess exited during handshake.
            RuntimeError: subprocess already started (must quit() first).
        """
        if self._proc is not None and self._proc.poll() is None:
            raise RuntimeError("UciClient already started — call quit() first")

        uci_jar_path = Path(uci_jar)
        model_path_p = Path(model_path)
        if not uci_jar_path.exists():
            raise FileNotFoundError(f"UCI JAR not found: {uci_jar_path}")
        if not model_path_p.exists():
            raise FileNotFoundError(f"Model file not found: {model_path_p}")

        # Phase 12+ ONNX backend (~17x speedup CPU) : si .onnx existe next to .npz,
        # passer le .onnx au UCI subprocess. NetworkLoader.loadAuto() côté Java
        # dispatche sur extension (NetworkVectorApi pour .npz, NetworkOnnx pour .onnx).
        if model_path_p.suffix == ".npz":
            onnx_path = model_path_p.with_suffix(".onnx")
            if onnx_path.exists():
                model_path_p = onnx_path

        # Save params for restart().
        self._uci_jar = uci_jar_path
        self._model_path = model_path_p
        self._dirichlet_alpha = dirichlet_alpha
        self._dirichlet_epsilon = dirichlet_epsilon
        self._dirichlet_seed = dirichlet_seed
        self._handshake_timeout = handshake_timeout

        self._proc = subprocess.Popen(
            [
                "java",
                "--add-modules",
                "jdk.incubator.vector",
                "-jar",
                str(uci_jar_path),
                "--network",
                str(model_path_p),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )

        # Start reader thread.
        self._stdout_buffer = deque()
        self._stdout_thread = threading.Thread(
            target=self._stdout_reader_loop, daemon=False, name="uci-stdout-reader"
        )
        self._stdout_thread.start()

        # Handshake : uci -> uciok, setoption Dirichlet, isready -> readyok.
        try:
            self.send_line("uci")
            self.read_until(("uciok",), timeout=handshake_timeout)

            # Hidden options Dirichlet (cf. SPEC-uci §6.4, ADR-003 hidden options).
            self.send_line(f"setoption name DirichletAlpha value {dirichlet_alpha}")
            self.send_line(f"setoption name DirichletEpsilon value {dirichlet_epsilon}")
            self.send_line(f"setoption name DirichletSeed value {dirichlet_seed}")

            self.send_line("isready")
            self.read_until(("readyok",), timeout=handshake_timeout)
        except (UciTimeoutError, UciCrashError):
            # Cleanup on handshake failure.
            self.quit(timeout=1.0)
            raise

    def quit(self, timeout: float = 5.0) -> None:
        """Send `quit`, wait for exit, cleanup. Idempotent."""
        if self._proc is None:
            return
        try:
            if self._proc.poll() is None:
                # Subprocess still alive — send quit politely.
                with contextlib.suppress(BrokenPipeError, OSError):
                    self.send_line("quit")
                try:
                    self._proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
                    self._proc.wait(timeout=1.0)
        finally:
            # Cleanup reader thread.
            if self._stdout_thread is not None and self._stdout_thread.is_alive():
                self._stdout_thread.join(timeout=1.0)
            # Close pipes to avoid ResourceWarning on subprocess pipes left open.
            if self._proc is not None:
                for stream in (self._proc.stdin, self._proc.stdout, self._proc.stderr):
                    if stream is not None:
                        with contextlib.suppress(OSError, ValueError):
                            stream.close()
            self._proc = None
            self._stdout_thread = None
            self._stdout_buffer = deque()

    def restart(self) -> None:
        """Quit + start with last-used params."""
        if self._uci_jar is None or self._model_path is None:
            raise RuntimeError("Cannot restart: start() was never called successfully")
        uci_jar = self._uci_jar
        model_path = self._model_path
        alpha = self._dirichlet_alpha
        eps = self._dirichlet_epsilon
        seed = self._dirichlet_seed
        timeout = self._handshake_timeout
        self.quit()
        self.start(
            uci_jar=uci_jar,
            model_path=model_path,
            dirichlet_alpha=alpha,
            dirichlet_epsilon=eps,
            dirichlet_seed=seed,
            handshake_timeout=timeout,
        )

    def is_alive(self) -> bool:
        """True if subprocess running."""
        return self._proc is not None and self._proc.poll() is None

    # ------------------------------------------------------------------ I/O

    def send_line(self, line: str) -> None:
        """Send a line to UCI stdin (auto-append newline)."""
        if self._proc is None or self._proc.stdin is None:
            raise RuntimeError("UciClient not started")
        if not self.is_alive():
            raise UciCrashError("Subprocess exited before send_line could complete")
        self._proc.stdin.write(line + "\n")
        self._proc.stdin.flush()

    def read_until(
        self,
        predicates: tuple[str, ...] = ("bestmove",),
        timeout: float = 60.0,
    ) -> list[str]:
        """Drain stdout buffer until any line starts with a prefix in `predicates`.

        Returns:
            All lines read since the previous read_until call, INCLUDING the
            matching line.

        Raises:
            UciTimeoutError: no matching line within `timeout` seconds.
            UciCrashError: subprocess exited before match.
        """
        if self._proc is None:
            raise RuntimeError("UciClient not started")

        deadline = time.monotonic() + timeout
        collected: list[str] = []
        while True:
            # Drain ligne par ligne et stop au premier match (lignes
            # post-match restent dans le buffer pour la prochaine read_until,
            # ce qui matche le flux UCI naturel — pas de "perte" de lignes).
            with self._stdout_lock:
                while self._stdout_buffer:
                    line = self._stdout_buffer.popleft()
                    collected.append(line)
                    if any(line.startswith(p) for p in predicates):
                        return collected

            # Check subprocess alive (post-drain).
            if self._proc.poll() is not None:
                # Drain remaining stdout via thread join (en cas de lignes
                # bufferisées non encore transférées par le reader thread).
                if self._stdout_thread is not None:
                    self._stdout_thread.join(timeout=0.5)
                with self._stdout_lock:
                    while self._stdout_buffer:
                        line = self._stdout_buffer.popleft()
                        collected.append(line)
                        if any(line.startswith(p) for p in predicates):
                            return collected
                raise UciCrashError(
                    f"Subprocess exited with code {self._proc.returncode} "
                    f"before matching predicates {predicates}. "
                    f"Last lines: {collected[-5:]}"
                )

            # Check timeout.
            if time.monotonic() >= deadline:
                raise UciTimeoutError(
                    f"No line starting with {predicates} within {timeout}s. "
                    f"Last lines: {collected[-5:]}"
                )

            time.sleep(_POLL_INTERVAL_S)

    # ------------------------------------------------------------------ commands

    def new_game(self, timeout: float = 5.0) -> None:
        """Send `ucinewgame` + `isready`, wait for `readyok`."""
        self.send_line("ucinewgame")
        self.send_line("isready")
        self.read_until(("readyok",), timeout=timeout)

    def go_nodes(
        self,
        position_cmd: str,
        nodes: int,
        timeout: float = 60.0,
    ) -> UciResult:
        """Send `position ...` + `go nodes N`, parse visits + bestmove.

        Args:
            position_cmd: 'position startpos' ou 'position startpos moves e2e4 ...'.
            nodes: nombre de MCTS simulations.
            timeout: max seconds.

        Returns:
            UciResult(visits, bestmove). bestmove est None si terminal
            ('bestmove (none)' UCI).
        """
        self.send_line(position_cmd)
        self.send_line(f"go nodes {nodes}")
        lines = self.read_until(("bestmove",), timeout=timeout)

        visits: dict[str, int] = {}
        bestmove: str | None = None
        for line in lines:
            if line.startswith("info string visits"):
                visits = parse_visits_line(line)
            elif line.startswith("bestmove"):
                parts = line.split()
                if len(parts) >= 2 and parts[1] != "(none)":
                    bestmove = parts[1]

        return UciResult(visits=visits, bestmove=bestmove)

    # ------------------------------------------------------------------ internals

    def _stdout_reader_loop(self) -> None:
        """Background thread: drain subprocess stdout line by line."""
        assert self._proc is not None
        stdout: IO[str] | None = self._proc.stdout
        if stdout is None:
            return
        try:
            for line in stdout:
                stripped = line.rstrip("\n")
                with self._stdout_lock:
                    self._stdout_buffer.append(stripped)
        except (ValueError, OSError):
            # stdout closed (subprocess exited).
            pass

    # ------------------------------------------------------------------ context

    def __enter__(self) -> UciClient:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.quit()
