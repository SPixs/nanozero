"""Unit tests for eval/fastchess_runner — FastchessRunner subprocess + polling.

Approach : mock subprocess.Popen avec un fake .stdout iterable.
Le polling thread est testé en injectant un state_manager mock.
"""

from __future__ import annotations

import logging
import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from nanozero_training.eval.fastchess_runner import FastchessConfig, FastchessRunner
from nanozero_training.eval.sprt_result import SPRTStatus
from pytest_mock import MockerFixture


def _make_fake_jar(tmp_path: Path) -> Path:
    jar = tmp_path / "nanozero-uci-1.2.0.jar"
    jar.write_bytes(b"fake-jar")
    return jar


def _make_fake_npz(tmp_path: Path, name: str) -> Path:
    npz = tmp_path / name
    npz.write_bytes(b"fake-npz")
    return npz


def _make_config(tmp_path: Path, **overrides: object) -> FastchessConfig:
    jar = _make_fake_jar(tmp_path)
    defaults: dict[str, object] = {
        "fastchess_path": "fastchess",
        "uci_jar": str(jar),
        "pgn_output": str(tmp_path / "sprt.pgn"),
        "max_games": 30,
        "poll_interval_seconds": 0.05,
    }
    defaults.update(overrides)
    return FastchessConfig(**defaults)  # type: ignore[arg-type]


def _make_mock_proc(stdout_lines: list[str], returncode: int = 0) -> MagicMock:
    proc = MagicMock(spec=subprocess.Popen)
    proc.stdout = iter(stdout_lines)
    proc.wait.return_value = returncode
    proc.kill.return_value = None
    proc.returncode = returncode
    return proc


def _make_mock_state_manager() -> MagicMock:
    mgr = MagicMock()
    mgr.update = MagicMock()
    return mgr


def test_fastchess_config_poll_interval_default_5_seconds() -> None:
    """Phase 10 : poll_interval_seconds default 5.0 (was 30.0) for SSE reactivity."""
    cfg = FastchessConfig()
    assert cfg.poll_interval_seconds == 5.0


def test_run_sprt_raises_if_jar_missing(tmp_path: Path) -> None:
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    config = FastchessConfig(uci_jar=str(tmp_path / "missing.jar"))
    runner = FastchessRunner(config)
    with pytest.raises(FileNotFoundError, match="UCI JAR"):
        runner.run_sprt(challenger, baseline)


def test_run_sprt_raises_if_challenger_missing(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    baseline = _make_fake_npz(tmp_path, "b.npz")
    runner = FastchessRunner(config)
    with pytest.raises(FileNotFoundError, match="Challenger"):
        runner.run_sprt(tmp_path / "missing.npz", baseline)


def test_run_sprt_raises_if_baseline_missing(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    runner = FastchessRunner(config)
    with pytest.raises(FileNotFoundError, match="Baseline"):
        runner.run_sprt(challenger, tmp_path / "missing.npz")


def test_run_sprt_raises_if_fastchess_not_in_path(tmp_path: Path, mocker: MockerFixture) -> None:
    config = _make_config(tmp_path)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    mocker.patch("subprocess.Popen", side_effect=FileNotFoundError("no fastchess"))
    runner = FastchessRunner(config)
    with pytest.raises(FileNotFoundError, match="fastchess binary not found"):
        runner.run_sprt(challenger, baseline)


def test_run_sprt_builds_correct_command(tmp_path: Path, mocker: MockerFixture) -> None:
    config = _make_config(tmp_path, elo_low=0.0, elo_high=5.0, max_games=40)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")

    popen_mock = mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(["H0 accepted\n", "Score of c vs b: 0 - 0 - 0  [0.500] 0\n"]),
    )
    runner = FastchessRunner(config)
    runner.run_sprt(challenger, baseline)

    args = popen_mock.call_args.args[0]
    # Engines: java + --add-modules jdk.incubator.vector + --network <model>
    assert any("--add-modules jdk.incubator.vector" in a for a in args)
    assert any(f"--network {challenger}" in a for a in args)
    assert any(f"--network {baseline}" in a for a in args)
    # SPRT bounds
    assert "elo0=0.0" in args
    assert "elo1=5.0" in args
    # Repeat + pgnout (fastchess >= 1.7 key=value form, breaking change 12-hotfix-002)
    assert "-repeat" in args
    # Phase 12 hotfix-010 — -recover must be present to avoid tournament aborting on
    # first "engine not responsive" event (which was masking real H0/H1 verdicts).
    assert "-recover" in args, "fastchess cmd missing -recover flag (hotfix-010)"
    assert "-pgnout" in args
    pgnout_idx = args.index("-pgnout")
    assert args[pgnout_idx + 1].startswith(
        "file="
    ), f"Expected 'file=PATH' after -pgnout, got {args[pgnout_idx + 1]!r}"
    # Rounds = max_games // 2 = 20
    rounds_idx = args.index("-rounds")
    assert args[rounds_idx + 1] == "20"
    # No opening book in default config -> no -openings flag (12-hotfix-003)
    assert "-openings" not in args


def test_run_sprt_same_jar_for_both_engines_by_default(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Sans baseline_uci_jar, les deux moteurs utilisent uci_jar (rétrocompat)."""
    config = _make_config(tmp_path)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    popen_mock = mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(["H0 accepted\n", "Score of c vs b: 0 - 0 - 0  [0.5] 0\n"]),
    )
    runner = FastchessRunner(config)
    runner.run_sprt(challenger, baseline)
    args = popen_mock.call_args.args[0]
    # Le jar challenger apparaît dans les deux -engine args (challenger + baseline).
    jar = str(config.uci_jar)
    engine_args = [a for a in args if a.startswith("args=")]
    assert len(engine_args) == 2
    assert all(jar in a for a in engine_args)


def test_run_sprt_cross_jar_baseline_uses_separate_jar(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """SPRT cross-jar : baseline_uci_jar défini -> baseline tourne sur un autre jar.

    Cas WDL v1.5.0 : challenger = jar WDL, baseline = jar scalaire (gen-025).
    """
    scalar_jar = tmp_path / "nanozero-uci-1.4.0-scalar.jar"
    scalar_jar.write_bytes(b"fake-scalar-jar")
    config = _make_config(tmp_path, baseline_uci_jar=str(scalar_jar))
    challenger = _make_fake_npz(tmp_path, "wdl.npz")
    baseline = _make_fake_npz(tmp_path, "gen025.npz")
    popen_mock = mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(["H1 accepted\n", "Score of w vs g: 1 - 0 - 0  [1.0] 1\n"]),
    )
    runner = FastchessRunner(config)
    runner.run_sprt(challenger, baseline)
    args = popen_mock.call_args.args[0]
    # Le moteur challenger porte le jar WDL + le réseau WDL ; le baseline le jar
    # scalaire + gen-025. Les deux jars sont distincts.
    challenger_arg = next(a for a in args if f"--network {challenger}" in a)
    baseline_arg = next(a for a in args if f"--network {baseline}" in a)
    assert str(config.uci_jar) in challenger_arg
    assert str(scalar_jar) in baseline_arg
    assert str(scalar_jar) not in challenger_arg


def test_run_sprt_nodes_limit_uses_fixed_nodes(tmp_path, mocker):
    """nodes_limit>0 -> go nodes N (recherche à budget de sims fixe) + clock généreux."""
    config = _make_config(tmp_path, nodes_limit=800)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    popen_mock = mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(["H0 accepted\n", "Score of c vs b: 0 - 0 - 0  [0.5] 0\n"]),
    )
    FastchessRunner(config).run_sprt(challenger, baseline)
    args = popen_mock.call_args.args[0]
    assert "nodes=800" in args
    assert any(a.startswith("tc=1200") for a in args), "clock de secours généreux attendu"
    assert "tc=10+0.1" not in args


def test_run_sprt_includes_opening_book_when_path_set(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Phase 12-hotfix-003 : opening_book_path triggers -openings file= ... in cmd.

    Without opening book, deterministic NN match plays the same line every
    game -> 3-fold draws systematic, SPRT LLR stays tiny, no verdict.
    """
    book = tmp_path / "UHO_4060_v2.epd"
    book.write_text("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\n")
    config = _make_config(
        tmp_path,
        opening_book_path=str(book),
        opening_book_plies=16,
    )
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    popen_mock = mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(["H0 accepted\n"]),
    )
    FastchessRunner(config).run_sprt(challenger, baseline)

    args = popen_mock.call_args.args[0]
    assert "-openings" in args
    openings_idx = args.index("-openings")
    assert args[openings_idx + 1] == f"file={book}"
    assert "format=epd" in args
    assert "order=random" in args
    assert "plies=16" in args


def test_run_sprt_existing_pgn_overwrites_with_warning(
    tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    config = _make_config(tmp_path)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    pgn = Path(config.pgn_output)
    pgn.write_text(
        '[Result "1-0"]\n[Result "0-1"]\n[Result "1/2-1/2"]\n[Result "1-0"]\n[Result "0-1"]\n',
        encoding="utf-8",
    )

    mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(["H1 accepted\n"]),
    )
    runner = FastchessRunner(config)
    with caplog.at_level(logging.WARNING):
        runner.run_sprt(challenger, baseline)

    # PGN supprimé
    assert not pgn.exists() or pgn.read_text() == ""
    # Warning loggée
    assert any("restarting SPRT from scratch" in r.message for r in caplog.records)


def test_run_sprt_parses_h1_accepted_output(tmp_path: Path, mocker: MockerFixture) -> None:
    config = _make_config(tmp_path)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(
            [
                "Score of challenger vs baseline: 80 - 20 - 30  [0.730] 130\n",
                "LLR: 2.95 (-2.94, 2.94) [0.00, 5.00]\n",
                "H1 accepted\n",
            ]
        ),
    )
    runner = FastchessRunner(config)
    result = runner.run_sprt(challenger, baseline)
    assert result.status == SPRTStatus.H1_ACCEPTED
    assert result.games_played == 130


def test_run_sprt_parses_h0_accepted_output(tmp_path: Path, mocker: MockerFixture) -> None:
    config = _make_config(tmp_path)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(
            [
                "Score of challenger vs baseline: 20 - 80 - 30  [0.270] 130\n",
                "H0 accepted\n",
            ]
        ),
    )
    runner = FastchessRunner(config)
    result = runner.run_sprt(challenger, baseline)
    assert result.status == SPRTStatus.H0_ACCEPTED


def test_run_sprt_parses_max_games_reached(tmp_path: Path, mocker: MockerFixture) -> None:
    config = _make_config(tmp_path)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(
            [
                "Score of challenger vs baseline: 15 - 12 - 3  [0.550] 30\n",
                "LLR: 0.42 (-2.94, 2.94) [0.00, 5.00]\n",
            ]
        ),
    )
    runner = FastchessRunner(config)
    result = runner.run_sprt(challenger, baseline)
    assert result.status == SPRTStatus.MAX_GAMES_REACHED
    assert result.games_played == 30


def test_run_sprt_timeout_kills_subprocess(tmp_path: Path, mocker: MockerFixture) -> None:
    config = _make_config(tmp_path)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")

    def slow_lines() -> object:
        # Generator yielding indefinitely — will trigger timeout.
        # Each iteration sleeps to push time.monotonic forward.
        import time as _time

        while True:
            _time.sleep(0.05)
            yield "Score of c vs b: 1 - 0 - 0  [1.000] 1\n"

    proc = MagicMock(spec=subprocess.Popen)
    proc.stdout = slow_lines()
    proc.wait.return_value = -9
    proc.kill = MagicMock()
    proc.returncode = -9
    mocker.patch("subprocess.Popen", return_value=proc)

    runner = FastchessRunner(config)
    runner.run_sprt(challenger, baseline, timeout_seconds=0.1)
    proc.kill.assert_called()


def test_run_sprt_initial_state_update_uses_eval_pgn_path(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """initial state.update doit utiliser `eval__pgn_path` (existant), pas `eval__current_pgn`."""
    config = _make_config(tmp_path)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(["H1 accepted\n"]),
    )
    state_mgr = _make_mock_state_manager()
    runner = FastchessRunner(config)
    runner.run_sprt(challenger, baseline, state_manager=state_mgr)

    # Inspect all .update calls and find one with eval__pgn_path
    found_pgn_path = any("eval__pgn_path" in c.kwargs for c in state_mgr.update.call_args_list)
    found_current_pgn = any(
        "eval__current_pgn" in c.kwargs for c in state_mgr.update.call_args_list
    )
    assert found_pgn_path
    assert not found_current_pgn


def test_run_sprt_initial_state_update_clears_last_decision(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    config = _make_config(tmp_path)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(["H1 accepted\n"]),
    )
    state_mgr = _make_mock_state_manager()
    runner = FastchessRunner(config)
    runner.run_sprt(challenger, baseline, state_manager=state_mgr)

    # First update call clears last_decision to None.
    initial_call = state_mgr.update.call_args_list[0]
    assert initial_call.kwargs.get("eval__last_decision") is None


def test_run_sprt_final_state_update_sets_last_decision(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    config = _make_config(tmp_path)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(
            [
                "Score of c vs b: 80 - 20 - 30  [0.730] 130\n",
                "H1 accepted\n",
            ]
        ),
    )
    state_mgr = _make_mock_state_manager()
    runner = FastchessRunner(config)
    runner.run_sprt(challenger, baseline, state_manager=state_mgr)

    # Last update call has last_decision = "h1_accepted".
    final_call = state_mgr.update.call_args_list[-1]
    assert final_call.kwargs.get("eval__last_decision") == "h1_accepted"


def test_polling_thread_stops_on_subprocess_exit(tmp_path: Path, mocker: MockerFixture) -> None:
    config = _make_config(tmp_path, poll_interval_seconds=0.5)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(["H1 accepted\n"]),
    )
    state_mgr = _make_mock_state_manager()
    runner = FastchessRunner(config)
    runner.run_sprt(challenger, baseline, state_manager=state_mgr)
    # Polling thread must have stopped within join timeout (5s).
    assert runner._poll_thread is not None
    assert not runner._poll_thread.is_alive()


def test_run_sprt_handles_polling_exception_gracefully(
    tmp_path: Path, mocker: MockerFixture, caplog: pytest.LogCaptureFixture
) -> None:
    config = _make_config(tmp_path, poll_interval_seconds=0.05)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")

    # Make state_manager.update raise on polling-loop calls only.
    state_mgr = MagicMock()
    call_count = {"n": 0}

    def update_side_effect(**kwargs: object) -> None:
        # First call is initial state update (challenger/baseline/etc) -> ok
        # Subsequent polling-loop calls only have eval__games_played_at_last_save -> raise
        call_count["n"] += 1
        if "eval__games_played_at_last_save" in kwargs and len(kwargs) == 1 and call_count["n"] > 1:
            raise RuntimeError("simulated state.update failure")

    state_mgr.update.side_effect = update_side_effect

    # Slow stdout to let polling fire at least once
    def slow_stdout() -> object:
        import time as _time

        for line in ["Score of c vs b: 5 - 5 - 0  [0.500] 10\n", "H0 accepted\n"]:
            _time.sleep(0.15)
            yield line

    proc = MagicMock(spec=subprocess.Popen)
    proc.stdout = slow_stdout()
    proc.wait.return_value = 0
    proc.returncode = 0
    mocker.patch("subprocess.Popen", return_value=proc)

    runner = FastchessRunner(config)
    with caplog.at_level(logging.WARNING):
        runner.run_sprt(challenger, baseline, state_manager=state_mgr)

    # Doit avoir logué un warning de polling error, sans planter
    assert any("SPRT polling error" in r.message for r in caplog.records)


def test_polling_loop_updates_games_played(tmp_path: Path, mocker: MockerFixture) -> None:
    """Le polling loop appelle bien state.update(eval__games_played_at_last_save=N)."""
    pgn = tmp_path / "sprt.pgn"
    pgn.write_text('[Result "1-0"]\n[Result "0-1"]\n', encoding="utf-8")
    config = _make_config(tmp_path, poll_interval_seconds=0.05)
    state_mgr = _make_mock_state_manager()
    runner = FastchessRunner(config)
    # Run the polling loop manually for ~150ms then stop
    runner._poll_stop.clear()
    t = threading.Thread(target=runner._polling_loop, args=(pgn, state_mgr), daemon=False)
    t.start()
    import time as _time

    _time.sleep(0.15)
    runner._poll_stop.set()
    t.join(timeout=2.0)
    # update called at least once with eval__games_played_at_last_save=2
    matching = [
        c
        for c in state_mgr.update.call_args_list
        if c.kwargs.get("eval__games_played_at_last_save") == 2
    ]
    assert len(matching) >= 1


def test_run_sprt_writes_final_csv_when_writer_provided(
    tmp_path: Path, mocker: MockerFixture
) -> None:
    """Phase 10 : metrics_writer + gen -> at least 1 final append_eval_row call."""
    config = _make_config(tmp_path)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(
            ["Score of c vs b: 80 - 20 - 30  [0.730] 130\n", "H1 accepted\n"]
        ),
    )
    writer = MagicMock()
    runner = FastchessRunner(config)
    runner.run_sprt(challenger, baseline, metrics_writer=writer, gen=2)

    # Final row should be called with terminal status h1_accepted
    writer.append_eval_row.assert_called()
    final_call = writer.append_eval_row.call_args_list[-1]
    assert final_call.kwargs["gen"] == 2
    assert final_call.kwargs["sprt_status"] == "h1_accepted"
    assert final_call.kwargs["games_played"] == 130


def test_polling_loop_writes_csv_per_poll(tmp_path: Path) -> None:
    """Phase 10 : polling loop appelle append_eval_row par poll (running status)."""
    import threading
    import time as _time

    pgn = tmp_path / "sprt.pgn"
    pgn.write_text('[Result "1-0"]\n[Result "0-1"]\n', encoding="utf-8")
    config = _make_config(tmp_path, poll_interval_seconds=0.05)
    state_mgr = _make_mock_state_manager()
    writer = MagicMock()

    runner = FastchessRunner(config)
    runner._poll_stop.clear()
    t = threading.Thread(
        target=runner._polling_loop,
        args=(pgn, state_mgr, writer, 3),
        daemon=False,
    )
    t.start()
    _time.sleep(0.15)
    runner._poll_stop.set()
    t.join(timeout=2.0)

    # append_eval_row called at least once with sprt_status="running"
    eval_calls = [
        c for c in writer.append_eval_row.call_args_list if c.kwargs.get("sprt_status") == "running"
    ]
    assert len(eval_calls) >= 1
    assert all(c.kwargs.get("gen") == 3 for c in eval_calls)


def test_run_sprt_csv_failure_does_not_break(tmp_path: Path, mocker: MockerFixture) -> None:
    """Phase 10 : if writer.append_eval_row raises, run_sprt completes anyway."""
    config = _make_config(tmp_path)
    challenger = _make_fake_npz(tmp_path, "c.npz")
    baseline = _make_fake_npz(tmp_path, "b.npz")
    mocker.patch(
        "subprocess.Popen",
        return_value=_make_mock_proc(["H1 accepted\n"]),
    )
    writer = MagicMock()
    writer.append_eval_row.side_effect = OSError("simulated CSV failure")
    runner = FastchessRunner(config)

    # Should not raise — append exception is caught and logged
    result = runner.run_sprt(challenger, baseline, metrics_writer=writer, gen=1)
    assert result.status == SPRTStatus.H1_ACCEPTED
