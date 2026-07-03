"""Unit tests for selfplay/uci_client.py — UciClient subprocess client (mocks).

Pattern de test : on patche `subprocess.Popen` au niveau du module
`nanozero_training.selfplay.uci_client` (où il est utilisé). Le mock retourne
un objet avec stdin (write/flush), stdout (iterable de lignes mockées),
stderr, poll(), wait(), kill(), returncode.
"""

from __future__ import annotations

from pathlib import Path
from queue import Queue
from typing import Any
from unittest.mock import MagicMock

import pytest
from nanozero_training.selfplay.uci_client import (
    UciClient,
    UciCrashError,
    UciResult,
    UciTimeoutError,
)

# ----- Helpers : fake subprocess --------------------------------------------


class _FakeStdout:
    """Iterable simulant subprocess.stdout : drain depuis une queue de lignes."""

    def __init__(self) -> None:
        self.queue: Queue[str | None] = Queue()

    def __iter__(self) -> Any:
        while True:
            line = self.queue.get()
            if line is None:
                return
            yield line + "\n"

    def push(self, line: str) -> None:
        self.queue.put(line)

    def close(self) -> None:
        self.queue.put(None)


class _FakeProc:
    """Simule subprocess.Popen avec stdin/stdout/poll/wait/kill."""

    def __init__(self) -> None:
        self.stdin = MagicMock()
        self.stdin.write = MagicMock()
        self.stdin.flush = MagicMock()
        self._stdin_lines: list[str] = []
        self.stdin.write.side_effect = lambda s: self._stdin_lines.append(s)
        self.stdout = _FakeStdout()
        self.stderr = MagicMock()
        self.returncode: int | None = None
        self._wait_called = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self._wait_called = True
        # Si returncode déjà set, retourne immédiatement.
        if self.returncode is not None:
            return self.returncode
        # Sinon, simule exit propre.
        self.returncode = 0
        self.stdout.close()
        return 0

    def kill(self) -> None:
        self.returncode = -9
        self.stdout.close()

    def get_stdin_lines(self) -> list[str]:
        return list(self._stdin_lines)


@pytest.fixture()
def fake_proc() -> _FakeProc:
    return _FakeProc()


@pytest.fixture()
def tmp_files(tmp_path: Path) -> tuple[Path, Path]:
    """Returns (jar_path, model_path), both existing dummy files."""
    jar = tmp_path / "nanozero-uci-1.2.0.jar"
    jar.write_bytes(b"dummy-jar")
    model = tmp_path / "model.npz"
    model.write_bytes(b"dummy-npz")
    return jar, model


def _patch_popen(mocker: Any, fake_proc: _FakeProc) -> MagicMock:
    """Patch subprocess.Popen in uci_client module to return fake_proc."""
    return mocker.patch(
        "nanozero_training.selfplay.uci_client.subprocess.Popen",
        return_value=fake_proc,
    )


def _start_with_handshake(
    client: UciClient,
    fake_proc: _FakeProc,
    jar: Path,
    model: Path,
    **kwargs: Any,
) -> None:
    """Push uciok + readyok onto fake_proc.stdout, then call client.start().

    Runs start() in a thread so we can push lines AFTER the reader thread starts.
    Simpler: push BEFORE start since the reader thread polls.
    """
    fake_proc.stdout.push("id name NanoZero")
    fake_proc.stdout.push("uciok")
    fake_proc.stdout.push("readyok")
    client.start(jar, model, handshake_timeout=2.0, **kwargs)


# ----- Tests start() ---------------------------------------------------------


def test_start_invokes_java_with_correct_args(
    mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]
) -> None:
    jar, model = tmp_files
    popen_mock = _patch_popen(mocker, fake_proc)
    client = UciClient()
    _start_with_handshake(client, fake_proc, jar, model)

    args, _kwargs = popen_mock.call_args
    cmd = args[0]
    assert "java" in cmd[0]
    assert "--add-modules" in cmd
    assert "jdk.incubator.vector" in cmd
    assert "-jar" in cmd
    assert str(jar) in cmd
    assert "--network" in cmd
    assert str(model) in cmd
    client.quit()


def test_start_fails_if_jar_missing(tmp_path: Path) -> None:
    client = UciClient()
    with pytest.raises(FileNotFoundError, match="UCI JAR not found"):
        client.start(tmp_path / "missing.jar", tmp_path / "model.npz")


def test_start_fails_if_model_missing(tmp_path: Path) -> None:
    jar = tmp_path / "uci.jar"
    jar.write_bytes(b"dummy")
    client = UciClient()
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        client.start(jar, tmp_path / "missing.npz")


def test_start_handshake_sends_uci_setoption_isready(
    mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]
) -> None:
    jar, model = tmp_files
    _patch_popen(mocker, fake_proc)
    client = UciClient()
    _start_with_handshake(client, fake_proc, jar, model)
    stdin_lines = fake_proc.get_stdin_lines()
    # Sequence : uci, setoption (x3), isready.
    assert stdin_lines[0] == "uci\n"
    assert "setoption name DirichletAlpha value 300" in stdin_lines[1]
    assert "setoption name DirichletEpsilon value 250" in stdin_lines[2]
    assert "setoption name DirichletSeed value 42" in stdin_lines[3]
    assert stdin_lines[4] == "isready\n"
    client.quit()


def test_start_custom_dirichlet_params(
    mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]
) -> None:
    jar, model = tmp_files
    _patch_popen(mocker, fake_proc)
    client = UciClient()
    _start_with_handshake(
        client,
        fake_proc,
        jar,
        model,
        dirichlet_alpha=500,
        dirichlet_epsilon=100,
        dirichlet_seed=999,
    )
    stdin_lines = fake_proc.get_stdin_lines()
    assert "setoption name DirichletAlpha value 500" in stdin_lines[1]
    assert "setoption name DirichletEpsilon value 100" in stdin_lines[2]
    assert "setoption name DirichletSeed value 999" in stdin_lines[3]
    client.quit()


def test_start_timeout_no_uciok(
    mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]
) -> None:
    jar, model = tmp_files
    _patch_popen(mocker, fake_proc)
    client = UciClient()
    # Pas de uciok pushed -> handshake timeout.
    with pytest.raises(UciTimeoutError, match="uciok"):
        client.start(jar, model, handshake_timeout=0.1)
    # Subprocess cleaned up.
    assert not client.is_alive()


def test_start_crash_during_handshake(
    mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]
) -> None:
    """Subprocess exited before uciok -> UciCrashError."""
    jar, model = tmp_files
    _patch_popen(mocker, fake_proc)
    client = UciClient()
    # Simulate subprocess crash early : returncode set before any line pushed.
    fake_proc.returncode = 1
    fake_proc.stdout.close()
    with pytest.raises(UciCrashError, match="Subprocess exited"):
        client.start(jar, model, handshake_timeout=0.5)


# ----- Tests send_line / read_until -----------------------------------------


def test_send_line_appends_newline(
    mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]
) -> None:
    jar, model = tmp_files
    _patch_popen(mocker, fake_proc)
    client = UciClient()
    _start_with_handshake(client, fake_proc, jar, model)
    fake_proc._stdin_lines.clear()  # Reset depuis handshake
    client.send_line("hello")
    assert fake_proc.get_stdin_lines() == ["hello\n"]
    client.quit()


def test_read_until_captures_matching_line(
    mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]
) -> None:
    jar, model = tmp_files
    _patch_popen(mocker, fake_proc)
    client = UciClient()
    _start_with_handshake(client, fake_proc, jar, model)
    fake_proc.stdout.push("info depth 1")
    fake_proc.stdout.push("bestmove e2e4")
    lines = client.read_until(("bestmove",), timeout=2.0)
    assert any(line.startswith("bestmove e2e4") for line in lines)
    assert any(line.startswith("info depth 1") for line in lines)
    client.quit()


def test_read_until_timeout(
    mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]
) -> None:
    jar, model = tmp_files
    _patch_popen(mocker, fake_proc)
    client = UciClient()
    _start_with_handshake(client, fake_proc, jar, model)
    # Pas de ligne 'bestmove' poussée.
    with pytest.raises(UciTimeoutError, match="bestmove"):
        client.read_until(("bestmove",), timeout=0.1)
    client.quit()


# ----- Tests go_nodes -------------------------------------------------------


def test_go_nodes_parses_visits_and_bestmove(
    mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]
) -> None:
    jar, model = tmp_files
    _patch_popen(mocker, fake_proc)
    client = UciClient()
    _start_with_handshake(client, fake_proc, jar, model)
    fake_proc.stdout.push("info string visits e2e4 12 d2d4 8 g1f3 4")
    fake_proc.stdout.push("bestmove e2e4")
    result = client.go_nodes("position startpos", nodes=100, timeout=2.0)
    assert isinstance(result, UciResult)
    assert result.visits == {"e2e4": 12, "d2d4": 8, "g1f3": 4}
    assert result.bestmove == "e2e4"
    # Vérifier la commande envoyée.
    stdin_lines = fake_proc.get_stdin_lines()
    assert "position startpos\n" in stdin_lines
    assert "go nodes 100\n" in stdin_lines
    client.quit()


def test_go_nodes_terminal_position(
    mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]
) -> None:
    """Position terminale : visits vide + bestmove (none) → UciResult(empty, None)."""
    jar, model = tmp_files
    _patch_popen(mocker, fake_proc)
    client = UciClient()
    _start_with_handshake(client, fake_proc, jar, model)
    fake_proc.stdout.push("info string visits")
    fake_proc.stdout.push("bestmove (none)")
    result = client.go_nodes("position startpos moves e2e4 e7e5", nodes=10, timeout=2.0)
    assert result.visits == {}
    assert result.bestmove is None
    client.quit()


# ----- Tests new_game / quit / restart --------------------------------------


def test_new_game_sends_ucinewgame_and_isready(
    mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]
) -> None:
    jar, model = tmp_files
    _patch_popen(mocker, fake_proc)
    client = UciClient()
    _start_with_handshake(client, fake_proc, jar, model)
    fake_proc._stdin_lines.clear()
    fake_proc.stdout.push("readyok")
    client.new_game(timeout=2.0)
    stdin_lines = fake_proc.get_stdin_lines()
    assert stdin_lines == ["ucinewgame\n", "isready\n"]
    client.quit()


def test_quit_sends_quit_and_waits(
    mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]
) -> None:
    jar, model = tmp_files
    _patch_popen(mocker, fake_proc)
    client = UciClient()
    _start_with_handshake(client, fake_proc, jar, model)
    fake_proc._stdin_lines.clear()
    client.quit()
    stdin_lines = fake_proc.get_stdin_lines()
    assert "quit\n" in stdin_lines
    assert not client.is_alive()


def test_quit_idempotent(mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]) -> None:
    jar, model = tmp_files
    _patch_popen(mocker, fake_proc)
    client = UciClient()
    _start_with_handshake(client, fake_proc, jar, model)
    client.quit()
    # 2e quit ne raise pas.
    client.quit()
    assert not client.is_alive()


def test_context_manager_calls_quit_on_exit(
    mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]
) -> None:
    jar, model = tmp_files
    _patch_popen(mocker, fake_proc)
    with UciClient() as client:
        _start_with_handshake(client, fake_proc, jar, model)
        assert client.is_alive()
    # quit appelé au __exit__.
    assert not client.is_alive()


def test_restart_without_prior_start_raises() -> None:
    client = UciClient()
    with pytest.raises(RuntimeError, match="never called"):
        client.restart()


def test_send_line_without_start_raises() -> None:
    client = UciClient()
    with pytest.raises(RuntimeError, match="not started"):
        client.send_line("foo")


def test_is_alive_initially_false() -> None:
    client = UciClient()
    assert client.is_alive() is False


def test_double_start_raises(
    mocker: Any, fake_proc: _FakeProc, tmp_files: tuple[Path, Path]
) -> None:
    """start() twice without quit() raises."""
    jar, model = tmp_files
    _patch_popen(mocker, fake_proc)
    client = UciClient()
    _start_with_handshake(client, fake_proc, jar, model)
    # 2nd start tente Popen mais notre patch retourne toujours fake_proc — la check
    # is_alive() côté UciClient doit catcher AVANT le Popen.
    with pytest.raises(RuntimeError, match="already started"):
        client.start(jar, model)
    client.quit()
