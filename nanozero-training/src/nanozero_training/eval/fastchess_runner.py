"""Fastchess subprocess runner for SPRT evaluation (ADR-005)."""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from nanozero_training.eval.pgn_utils import count_games_in_pgn, parse_sprt_counts_from_pgn
from nanozero_training.eval.sprt_result import (
    SPRTResult,
    SPRTStatus,
    parse_sprt_stdout,
)

if TYPE_CHECKING:
    from nanozero_training.monitoring.csv_writer import MetricsCSVWriter
    from nanozero_training.state.manager import RunStateManager

LOG = logging.getLogger(__name__)

# Clock de secours pour le mode nœuds-fixes (nodes_limit > 0) : 20 min base +
# 20 s/coup. Jamais binding (un backend ORT lent à N sims reste très en dessous),
# mais requis car fastchess exige un TC ; la recherche est bornée par `nodes`.
_NODES_MODE_FALLBACK_TC = "1200+20"


@dataclass(frozen=True)
class FastchessConfig:
    """Fastchess SPRT configuration. Defaults per SPEC-training §9.2."""

    fastchess_path: str = "fastchess"
    uci_jar: str | Path = ""
    # Jar UCI du moteur BASELINE. Vide (défaut) => réutilise ``uci_jar`` (SPRT
    # intra-version classique). Permet un SPRT CROSS-JAR : challenger et baseline
    # tournent sur des jars différents, ne dialoguant qu'en UCI (texte, agnostique
    # à la version). Cas d'usage v1.5.0 : challenger = jar WDL (value head 3
    # logits), baseline = jar scalaire (value head tanh, charge gen-025-promoted).
    # Le code engine/uci/board étant identique entre les deux jars, seule la
    # représentation du value head diffère -> comparaison équitable du réseau.
    baseline_uci_jar: str | Path = ""
    elo_low: float = 0.0
    elo_high: float = 5.0
    alpha: float = 0.05
    beta: float = 0.05
    max_games: int = 2000
    time_control: str = "10+0.1"
    # Limite de NŒUDS (sims MCTS) par coup. 0 = désactivé (TC seul, défaut). Si > 0,
    # fastchess envoie `go nodes N` -> le moteur fait exactement N sims (priorité 2
    # > time-control, cf. TimeManagementPolicy). Annule le confond VITESSE : un
    # backend lent (ORT CPU pour un net rebumpé 10×120) fait le MÊME nombre de sims
    # qu'un backend rapide (SIMD), donc on compare la QUALITÉ du réseau à budget de
    # recherche égal. Le TC devient un clock de secours généreux (jamais binding).
    nodes_limit: int = 0
    concurrency: int = 1
    pgn_output: str | Path = "sprt.pgn"
    poll_interval_seconds: float = 5.0  # phase 10 : SSE dashboard reactivity (was 30.0)
    # Phase 12-hotfix-003 : opening book for SPRT variance. Empty path = no book
    # (engines play from startpos, deterministic NN match = 3-fold draws systematic,
    # SPRT statistically meaningless). UHO_4060_v2.epd recommended (242k positions).
    opening_book_path: str | Path = ""
    opening_book_plies: int = 16  # plies replayed from each opening before SPRT start


class FastchessRunner:
    """Orchestrate SPRT via fastchess subprocess.

    Workflow run_sprt(challenger, baseline):
      1. Detect existing PGN -- if present, log WARNING and OVERWRITE (restart from scratch).
         Decision Q5 SPEC §11.5 scenario 5: fastchess has no official resume-from-PGN API.
         Acceptable v1.0.0: max ~3-4h perdues at 2000 games x 10+0.1s.
      2. Launch fastchess subprocess with full SPRT params (SPEC §9.2).
      3. Spawn polling thread (daemon=False, pattern UciClient phase 4-b):
         - Every poll_interval_seconds: count_games_in_pgn(pgn_output)
         - state.update(eval__games_played_at_last_save=N)
      4. Drain stdout into buffer (stderr merged via subprocess.STDOUT).
      5. On subprocess exit: parse_sprt_stdout(buffer) -> SPRTResult.
      6. Stop polling thread, final state.update(eval__last_decision=...).
      7. Return result.

    Thread safety: single-threaded usage v1.0.0. The polling thread is internal,
    not concurrent with run_sprt main flow.
    """

    def __init__(self, config: FastchessConfig) -> None:
        self.config = config
        self._proc: subprocess.Popen[str] | None = None
        self._poll_thread: threading.Thread | None = None
        self._poll_stop = threading.Event()

    def run_sprt(  # noqa: PLR0912, PLR0915  # subprocess orchestration + polling : refactor pending
        self,
        challenger_npz: str | Path,
        baseline_npz: str | Path,
        state_manager: RunStateManager | None = None,
        timeout_seconds: float | None = None,
        metrics_writer: MetricsCSVWriter | None = None,
        gen: int | None = None,
    ) -> SPRTResult:
        """Run SPRT between two models. Blocks until completion or timeout.

        Args:
            challenger_npz: path to challenger model .npz.
            baseline_npz: path to baseline model .npz.
            state_manager: optional RunStateManager for progress updates.
            timeout_seconds: optional global timeout. If exceeded, kill subprocess.
            metrics_writer: optional MetricsCSVWriter for live monitoring CSV.
                If provided AND gen is also provided, polling thread appends 1 row
                per poll + final row with terminal SPRT status. Graceful on failure.
            gen: generation number for CSV filename (required if metrics_writer).

        Returns:
            SPRTResult with final decision.

        Raises:
            FileNotFoundError: jar, models, or fastchess binary not found.
        """
        challenger_npz = Path(challenger_npz)
        baseline_npz = Path(baseline_npz)
        pgn_path = Path(self.config.pgn_output)

        if not Path(self.config.uci_jar).exists():
            raise FileNotFoundError(f"UCI JAR not found: {self.config.uci_jar}")
        if not challenger_npz.exists():
            raise FileNotFoundError(f"Challenger model not found: {challenger_npz}")
        if not baseline_npz.exists():
            raise FileNotFoundError(f"Baseline model not found: {baseline_npz}")

        # Phase 12+ ONNX backend : swap .npz → .onnx si companion existe (~17x speedup CPU).
        if challenger_npz.suffix == ".npz":
            challenger_onnx = challenger_npz.with_suffix(".onnx")
            if challenger_onnx.exists():
                LOG.info("Using .onnx companion for challenger: %s", challenger_onnx)
                challenger_npz = challenger_onnx
        if baseline_npz.suffix == ".npz":
            baseline_onnx = baseline_npz.with_suffix(".onnx")
            if baseline_onnx.exists():
                LOG.info("Using .onnx companion for baseline: %s", baseline_onnx)
                baseline_npz = baseline_onnx

        # Scenario 5 SPEC §11.5 v1.0.0: restart from scratch if existing PGN.
        existing_games = count_games_in_pgn(pgn_path)
        if existing_games > 0:
            LOG.warning(
                "Existing PGN at %s with %d games -- restarting SPRT from scratch "
                "(v1.0.0: fastchess does not support resume from PGN, ADR-005).",
                pgn_path,
                existing_games,
            )
            pgn_path.unlink(missing_ok=True)

        # Initial state update : clear last_decision, set pgn_path + names.
        if state_manager is not None:
            state_manager.update(
                eval__challenger=str(challenger_npz.name),
                eval__baseline=str(baseline_npz.name),
                eval__pgn_path=str(pgn_path),
                eval__games_played_at_last_save=0,
                eval__last_decision=None,
            )

        cmd = self._build_fastchess_cmd(challenger_npz, baseline_npz, pgn_path)
        LOG.info("Launching fastchess: %s", " ".join(cmd))

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"fastchess binary not found: {self.config.fastchess_path}. "
                "Install per scripts/install-fastchess-w3090.md."
            ) from e

        if state_manager is not None:
            self._poll_stop.clear()
            self._poll_thread = threading.Thread(
                target=self._polling_loop,
                args=(pgn_path, state_manager, metrics_writer, gen),
                daemon=False,
            )
            self._poll_thread.start()

        stdout_buffer: list[str] = []
        start_time = time.monotonic()
        try:
            assert self._proc.stdout is not None
            for line in self._proc.stdout:
                stdout_buffer.append(line)
                if (
                    timeout_seconds is not None
                    and (time.monotonic() - start_time) > timeout_seconds
                ):
                    LOG.error("Fastchess timeout reached after %.0fs", timeout_seconds)
                    self._proc.kill()
                    break
            self._proc.wait(timeout=10.0)
        finally:
            self._poll_stop.set()
            if self._poll_thread is not None:
                self._poll_thread.join(timeout=5.0)

        full_output = "".join(stdout_buffer)

        # Phase 12 hotfix-013 — Persist raw fastchess stdout to disk for forensic debug.
        # Without this, the buffer lives only in memory and is lost after parsing.
        # In hotfix-010/011/012 sessions we couldn't tell whether fastchess emitted
        # an "H1 accepted" line that our regex missed — having the raw stdout on disk
        # lets us update regexes retroactively and confirm SPRT verdict semantics.
        if gen is not None:
            stdout_log_path = pgn_path.parent / f"fastchess-stdout-gen-{gen:03d}.log"
            try:
                stdout_log_path.write_text(full_output, encoding="utf-8", errors="replace")
            except OSError as e:
                LOG.warning("Failed to persist fastchess stdout to %s: %s", stdout_log_path, e)

        result = parse_sprt_stdout(full_output)

        # Phase 12 hotfix-008 — PGN fallback when stdout parsing fails to extract
        # W/L/D and games_played (regex mismatch on newer fastchess output format).
        # The PGN is the source of truth for game outcomes; if stdout summary is
        # incomplete/absent, recount from PGN directly. Preserves stdout-extracted
        # llr and elo_diff (those regexes still work) but rebuilds counters.
        if result.games_played == 0:
            counts = parse_sprt_counts_from_pgn(pgn_path, challenger_name="challenger")
            if counts.total > 0:
                LOG.warning(
                    "SPRT stdout parse incomplete (games_played=0); "
                    "PGN fallback: W=%d L=%d D=%d total=%d",
                    counts.challenger_wins,
                    counts.challenger_losses,
                    counts.challenger_draws,
                    counts.total,
                )
                # Upgrade status: with games played but no H0/H1 in stdout, this
                # is functionally MAX_GAMES_REACHED (or interrupted SPRT).
                upgraded_status = result.status
                if result.status == SPRTStatus.RUNNING:
                    upgraded_status = SPRTStatus.MAX_GAMES_REACHED
                result = SPRTResult(
                    status=upgraded_status,
                    llr=result.llr,
                    games_played=counts.total,
                    wins=counts.challenger_wins,
                    losses=counts.challenger_losses,
                    draws=counts.challenger_draws,
                    elo_diff=result.elo_diff,
                    raw_output=result.raw_output,
                )

        LOG.info(
            "SPRT complete: status=%s games=%d llr=%s",
            result.status.value,
            result.games_played,
            result.llr,
        )

        if state_manager is not None:
            state_manager.update(
                eval__games_played_at_last_save=result.games_played,
                eval__last_decision=result.status.value,
            )

        self._write_final_eval_csv(result, metrics_writer, gen)
        return result

    def _write_final_eval_csv(
        self,
        result: SPRTResult,
        metrics_writer: MetricsCSVWriter | None,
        gen: int | None,
    ) -> None:
        """Phase 10 : append final eval row with terminal SPRT status. Graceful on failure."""
        if metrics_writer is None or gen is None:
            return
        try:
            metrics_writer.append_eval_row(
                gen=gen,
                games_played=result.games_played,
                wins=result.wins,
                losses=result.losses,
                draws=result.draws,
                llr=result.llr,
                elo_diff=result.elo_diff,
                sprt_status=result.status.value,
            )
        except Exception as e:
            LOG.warning("CSV final eval row failed: %s", e)

    def _build_fastchess_cmd(
        self,
        challenger_npz: Path,
        baseline_npz: Path,
        pgn_path: Path,
    ) -> list[str]:
        """Build fastchess command line per SPEC-training §9.2."""
        cfg = self.config
        rounds = max(cfg.max_games // 2, 1)
        # Jar par moteur : baseline_uci_jar si défini, sinon le même que challenger
        # (rétrocompat SPRT intra-version). Cf. docstring FastchessConfig.
        baseline_jar = cfg.baseline_uci_jar if cfg.baseline_uci_jar else cfg.uci_jar
        cmd: list[str] = [
            cfg.fastchess_path,
            "-engine",
            "cmd=java",
            f"args=--add-modules jdk.incubator.vector -jar {cfg.uci_jar} "
            f"--network {challenger_npz}",
            "name=challenger",
            "-engine",
            "cmd=java",
            f"args=--add-modules jdk.incubator.vector -jar {baseline_jar} "
            f"--network {baseline_npz}",
            "name=baseline",
            "-each",
            # Mode nœuds-fixes : clock de secours TRÈS généreux (jamais binding, même
            # pour un backend ORT lent à N sims) + nodes=N qui borne réellement la
            # recherche (prio > tc). Sinon : TC normal.
            *(
                [f"tc={_NODES_MODE_FALLBACK_TC}", f"nodes={cfg.nodes_limit}"]
                if cfg.nodes_limit > 0
                else [f"tc={cfg.time_control}"]
            ),
            "proto=uci",
            "-rounds",
            str(rounds),
            "-games",
            "2",
            "-repeat",
            "-sprt",
            f"elo0={cfg.elo_low}",
            f"elo1={cfg.elo_high}",
            f"alpha={cfg.alpha}",
            f"beta={cfg.beta}",
            "-concurrency",
            str(cfg.concurrency),
            # Phase 12 hotfix-010 — Without -recover, fastchess STOPS the tournament
            # on the first "engine not responsive" event (UCI ping timeout). Observed
            # in prod : SPRT terminating prematurely at 230/551 games with status=running,
            # giving false H0-implicit rejection. With -recover, fastchess restarts the
            # stalled engine and continues until H0/H1/MAX_GAMES is genuinely reached.
            "-recover",
            # fastchess >= 1.7 requires key=value form for -pgnout (breaking change).
            # Detected phase 12-deploy 2026-05-15 with fastchess 1.8.0-alpha :
            # "Error while reading option \"-pgnout\" with value \"...\"
            #  Reason: Option \"-pgnout\" expects key=value pairs, got \"...\"."
            "-pgnout",
            f"file={pgn_path}",
        ]
        # Phase 12-hotfix-003 : add opening book if configured.
        # Without one, deterministic NN match plays the same line every game -> 3-fold
        # draws systematic, SPRT LLR stays tiny, no H1/H0 verdict reachable.
        if cfg.opening_book_path:
            cmd.extend(
                [
                    "-openings",
                    f"file={cfg.opening_book_path}",
                    "format=epd",
                    "order=random",
                    f"plies={cfg.opening_book_plies}",
                ]
            )
        return cmd

    def _polling_loop(
        self,
        pgn_path: Path,
        state_manager: RunStateManager,
        metrics_writer: MetricsCSVWriter | None = None,
        gen: int | None = None,
    ) -> None:
        """Polling thread: every poll_interval_seconds, update state.eval.games_played.

        Phase 10 : also writes 1 CSV row per poll if metrics_writer + gen provided.
        Status field always "running" (final terminal status written by run_sprt).
        """
        while not self._poll_stop.is_set():
            try:
                games = count_games_in_pgn(pgn_path)
                state_manager.update(eval__games_played_at_last_save=games)
                LOG.debug("SPRT polling: %d games", games)

                if metrics_writer is not None and gen is not None:
                    try:
                        metrics_writer.append_eval_row(
                            gen=gen,
                            games_played=games,
                            sprt_status="running",
                        )
                    except Exception as csv_e:
                        LOG.warning("CSV poll row failed: %s", csv_e)
            except Exception as e:
                LOG.warning("SPRT polling error: %s", e)
            self._poll_stop.wait(self.config.poll_interval_seconds)


__all__ = ["FastchessConfig", "FastchessRunner", "SPRTResult", "SPRTStatus"]
