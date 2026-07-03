"""FlusherService — background thread that flushes positions to NPZ shards (ADR-015).

Design (cf. ADR-015 §Workflow) :
  - Thread daemon=False (cohérent monitoring/Flask SSE pattern phase 10).
  - tick() method : public, synchronous. Performs ONE flush + purge cycle.
    Used both by the background loop and by tests.
  - _loop() : private, calls tick() on tick_interval seconds.
  - force_flush() : flush all unflushed positions even below threshold.
    Called at shutdown to avoid data loss.
  - stop() : signal + force_flush + join.

Thread safety : SQLite stdlib + WAL mode = single writer, multiple readers
safely from multiple connections. FlusherService is a single writer for
positions/batches tables ; FastAPI request handlers (workers POST) are also
writers but on different rows. SQLite serializes writes — no application
locking needed.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from nanozero_jobserver.storage.batches import (
    insert_batch,
    next_batch_idx,
    sum_positions_by_version,
)
from nanozero_jobserver.storage.control import (
    get_selfplay_target,
    set_selfplay_paused,
    set_selfplay_target,
)
from nanozero_jobserver.storage.jobs import delete_stale_claims
from nanozero_jobserver.storage.npz_writer import atomic_write_npz_shard
from nanozero_jobserver.storage.replay_buffer import (
    count_unflushed_by_version,
    count_unflushed_positions,
    delete_flushed_old,
    iter_unflushed_positions,
    list_unflushed_model_versions,
    mark_positions_flushed,
)

if TYPE_CHECKING:
    pass

LOG = logging.getLogger(__name__)


class FlusherService:
    """Background flusher : positions (HOT SQLite) -> NPZ shards (COLD archive).

    Args:
        output_dir: directory for NPZ shards (created if absent).
        flush_threshold: minimum unflushed positions per model_version to
            trigger a shard write. Default 100_000.
        retention_window: keep flushed positions for the last N model_versions
            in HOT cache, then DELETE. Default 5.
        tick_interval_seconds: how often the background loop runs tick().
            Default 30.0s.
        get_current_model_version: callable that returns the current promoted
            model version (or None if no model). Used for purge cutoff.
            None disables purge (only flush, no DELETE).
    """

    def __init__(
        self,
        output_dir: Path,
        flush_threshold: int = 100_000,
        retention_window: int = 5,
        tick_interval_seconds: float = 30.0,
        get_current_model_version: Callable[[], int | None] | None = None,
        stale_claim_timeout_seconds: int = 3600,
        stale_claim_max_per_tick: int = 500_000,
        flush_browser: bool = True,
        browser_flush_threshold: int = 25_000,
    ) -> None:
        if flush_threshold <= 0:
            raise ValueError(f"flush_threshold must be > 0, got {flush_threshold}")
        if retention_window < 0:
            raise ValueError(f"retention_window must be >= 0, got {retention_window}")
        if tick_interval_seconds <= 0:
            raise ValueError(f"tick_interval_seconds must be > 0, got {tick_interval_seconds}")
        self.output_dir = Path(output_dir)
        self.flush_threshold = flush_threshold
        self.retention_window = retention_window
        self.tick_interval_seconds = tick_interval_seconds
        self._get_current_version = get_current_model_version
        # Watchdog claims périmés : un worker qui meurt/redémarre en cours de partie laisse son job
        # en 'claimed' à vie. 0 désactive. max_per_tick draine un gros backlog progressivement.
        self.stale_claim_timeout_seconds = stale_claim_timeout_seconds
        self.stale_claim_max_per_tick = stale_claim_max_per_tick
        # Chantier 1 cloisonnement : flushe AUSSI la cohorte 'browser' vers des shards SÉPARÉS
        # ('browser-gen*', hors corpus d'entraînement). flush_browser=False = quarantine stricte
        # historique (browser jamais flushé → reste dans le HOT cache). Seuil browser plus bas car
        # corpus plus petit. La quarantine reste assurée par le NOMMAGE (le training ne lit que
        # 'selfplay-gen{N}'). Bonus : libère le HOT cache du bloat browser (jamais flushé jusqu'ici).
        self.flush_browser = flush_browser
        self.browser_flush_threshold = browser_flush_threshold
        # Santé : nb de ticks consécutifs en échec (timeout SQLite, ENOSPC sur le write
        # NPZ, etc.). Exposé via /health — un flusher bloqué était avalé en LOG.warning
        # sans signal (la DB grossit alors silencieusement). 0 = sain.
        self.consecutive_tick_failures = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ public

    def tick(self) -> int:
        """Run ONE flush + purge cycle. Returns shards written.

        Logic :
            For each model_version with unflushed positions (FIFO oldest first) :
              if count_unflushed(version) >= flush_threshold :
                build NPZ shard with first `flush_threshold` rows
                insert_batch + mark_positions_flushed
            Then run purge (delete flushed positions outside retention window).
        """
        shards_written = self._flush_full_shards()
        self._purge_old_flushed()
        self._purge_stale_claims()
        self._check_selfplay_target()
        return shards_written

    def force_flush(self) -> int:
        """Flush ALL unflushed positions, ignoring threshold. Returns shards.

        Use at shutdown to avoid losing the in-memory buffer.
        Each model_version produces ONE shard regardless of size.
        """
        shards_written = 0
        for source, prefix, _threshold in self._flush_sources():
            for version in list_unflushed_model_versions(source=source):
                n = count_unflushed_positions(model_version=version, source=source)
                if n == 0:
                    continue
                if self._flush_one_shard(version, n, source, prefix):
                    shards_written += 1
        return shards_written

    def start(self) -> None:
        """Spawn the background thread (idempotent : no-op if already running)."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="FlusherService", daemon=False)
        self._thread.start()
        LOG.info(
            "FlusherService started : threshold=%d retention=%d tick=%.1fs",
            self.flush_threshold,
            self.retention_window,
            self.tick_interval_seconds,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal stop, force_flush remaining, join thread.

        Args:
            timeout: max wall-clock to wait for thread.join.
        """
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        # Drain remaining unflushed before declaring done.
        try:
            self.force_flush()
        except Exception as e:  # — best effort at shutdown
            LOG.warning("force_flush at stop() failed : %s", e)
        LOG.info("FlusherService stopped")

    # ---------------------------------------------------------------- internal

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._tick_guarded()
            self._stop.wait(self.tick_interval_seconds)

    def _tick_guarded(self) -> None:
        """Un tick protégé : exécute tick() en suivant les échecs CONSÉCUTIFS.

        Un échec (timeout SQLite, ENOSPC sur le write NPZ…) était avalé en LOG.warning
        sans signal ; le compteur ``consecutive_tick_failures`` le rend visible via /health.
        Extrait de ``_loop`` pour être testable sans lancer le thread.
        """
        try:
            self.tick()
            self.consecutive_tick_failures = 0
        except Exception as e:
            self.consecutive_tick_failures += 1
            LOG.warning(
                "FlusherService tick failed (%d consécutif·s) : %s",
                self.consecutive_tick_failures,
                e,
            )

    def _flush_sources(self) -> list[tuple[str, str, int]]:
        """(source, name_prefix, threshold) à flusher. Fleet toujours ; browser si activé.

        Chantier 1 cloisonnement : fleet → corpus 'selfplay-gen*' (entraînement), browser →
        'browser-gen*' (séparé, hors corpus — quarantine par le nommage).
        """
        sources = [("fleet", "selfplay", self.flush_threshold)]
        if self.flush_browser:
            sources.append(("browser", "browser", self.browser_flush_threshold))
        return sources

    def _flush_full_shards(self) -> int:
        """Flush each (source, model_version) that has >= its threshold unflushed."""
        shards = 0
        for source, prefix, threshold in self._flush_sources():
            for version in list_unflushed_model_versions(source=source):
                while (
                    count_unflushed_positions(model_version=version, source=source)
                    >= threshold
                ):
                    if self._flush_one_shard(version, threshold, source, prefix):
                        shards += 1
                    else:
                        break  # safety : shouldn't happen but avoid tight loop
        return shards

    def _flush_one_shard(
        self,
        model_version: int,
        limit: int,
        source: str = "fleet",
        name_prefix: str = "selfplay",
    ) -> bool:
        """Flush exactly one NPZ shard for (model_version, source). Returns success."""
        rows = iter_unflushed_positions(model_version, limit, source=source)
        if not rows:
            return False
        idx = next_batch_idx(model_version)  # séquence GLOBALE par version (option c)
        path = atomic_write_npz_shard(
            output_dir=self.output_dir,
            model_version=model_version,
            batch_idx=idx,
            positions=rows,
            name_prefix=name_prefix,
        )
        batch_id = insert_batch(
            model_version=model_version,
            batch_idx=idx,
            npz_path=str(path),
            n_positions=len(rows),
            source=source,
        )
        mark_positions_flushed([r.id for r in rows], batch_id)
        LOG.info(
            "Flushed %s shard %s (%d positions, model_version=%d, batch_idx=%d)",
            source,
            path.name,
            len(rows),
            model_version,
            idx,
        )
        return True

    def _purge_old_flushed(self) -> int:
        """Purge flushed positions older than retention window. Returns deleted count."""
        if self._get_current_version is None:
            return 0
        current = self._get_current_version()
        if current is None:
            return 0
        max_version = current - self.retention_window
        if max_version < 1:
            return 0
        n = delete_flushed_old(max_model_version=max_version)
        if n > 0:
            LOG.info(
                "Purged %d flushed positions (model_version <= %d, retention=%d)",
                n,
                max_version,
                self.retention_window,
            )
        return n

    def _purge_stale_claims(self) -> int:
        """Watchdog : delete jobs stuck in 'claimed' past the timeout. Returns deleted.

        Recovers from crashed/restarted workers that leave jobs in 'claimed' forever
        (the queue is auto-refilled so they never block, but accumulate as dead rows).
        Capped per tick so a large backlog drains gradually instead of in one long
        write-lock-holding sweep. `requeue_stale_jobs` was designed for this but was
        never wired into the background loop — this is that wiring (as a DELETE).
        """
        if self.stale_claim_timeout_seconds <= 0:
            return 0
        n = delete_stale_claims(
            timeout_seconds=self.stale_claim_timeout_seconds,
            max_total=self.stale_claim_max_per_tick,
        )
        if n > 0:
            LOG.info(
                "Watchdog : deleted %d stale 'claimed' jobs (claimed > %ds ago)",
                n,
                self.stale_claim_timeout_seconds,
            )
        return n

    def _check_selfplay_target(self) -> None:
        """Auto-pause (ou notify) quand le volume FLEET entraînable de la gen courante atteint
        la cible posée via ``POST /selfplay/target``. One-shot : la cible est effacée après le
        déclenchement (sinon un ``resume`` re-paserait aussitôt). Décharge l'opérateur du
        « pause manuel au volume » (footgun récurrent du cycle)."""
        if self._get_current_version is None:
            return
        target, action = get_selfplay_target()
        if target is None:
            return
        mv = self._get_current_version()
        if mv is None:
            return
        fleet = sum_positions_by_version(source="fleet").get(mv, 0) + (
            count_unflushed_by_version(source="fleet").get(mv, 0)
        )
        if fleet < target:
            return
        if action == "pause":
            set_selfplay_paused(True)
            LOG.info("Cible self-play atteinte (%d ≥ %d, gen %d) → AUTO-PAUSE", fleet, target, mv)
        else:
            LOG.info("Cible self-play atteinte (%d ≥ %d, gen %d) → notify", fleet, target, mv)
        set_selfplay_target(None)  # one-shot
