"""Self-play orchestrator : N parties par batch, M batches par génération.

Integrates :
- UciClient (subprocess UCI persistent, ADR-001).
- SelfplayConfig (hyperparams).
- RunStateManager (resilience, ADR-009).
- install_signal_handlers (Ctrl+C graceful abort).
- data/npz_writer.write_batch (atomic flush, ADR-002).
- selfplay/worker.play_one_game (single-game logic).

Pattern (ADR-004 sequential v1.0.0) : 1 worker, 1 partie à la fois,
batches de N parties flushés vers .npz. Resume support via RunState
(selfplay.completed_games comme cursor).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from nanozero_training.data.sample import Sample
from nanozero_training.selfplay.config import SelfplayConfig
from nanozero_training.selfplay.uci_client import UciCrashError, UciTimeoutError
from nanozero_training.selfplay.worker import play_one_game

if TYPE_CHECKING:
    from nanozero_training.selfplay.uci_client import UciClient
    from nanozero_training.state.manager import RunStateManager

LOG = logging.getLogger(__name__)


def make_game_rngs(
    worker_seed: int,
    n_games: int,
) -> list[np.random.Generator]:
    """Génère n_games RNGs décorelés depuis un seul worker_seed.

    Utilise `numpy.random.SeedSequence.spawn()` (pattern recommandé pour
    parent-child RNGs cf. numpy docs).

    Critique pour la reproducibilité au resume : pour tout target_games,
    `make_game_rngs(worker_seed, target_games)[k]` retourne TOUJOURS le même
    RNG pour le k-ème jeu, peu importe quand cette fonction est appelée
    (au démarrage initial OU lors d'un resume).

    Args:
        worker_seed: parent seed.
        n_games: nombre de RNGs enfants.

    Returns:
        Liste de np.random.Generator décorelés.
    """
    seed_seq = np.random.SeedSequence(worker_seed)
    child_seeds = seed_seq.spawn(n_games)
    return [np.random.default_rng(s) for s in child_seeds]


def derive_dirichlet_seed(rng: np.random.Generator) -> int:
    """Dérive un int32 positif depuis un Generator pour l'option UCI DirichletSeed.

    UCI option DirichletSeed accepte un int positif. On sample uniform
    [0, 2^31 - 1].

    Args:
        rng: générateur source.

    Returns:
        int dans [0, 2^31 - 1).
    """
    return int(rng.integers(0, 2**31 - 1))


class SelfplayOrchestrator:
    """Orchestrate self-play pour une génération.

    Workflow `run_generation(gen, target_games)` :
      1. Détermine resume point depuis state.selfplay.completed_games.
      2. Pour chaque partie restante :
         a. Check abort_flag → break si demandé.
         b. Génère RNG déterministe pour cette partie.
         c. Maybe restart worker (tous worker_restart_every parties).
         d. play_one_game avec recovery (jusqu'à max_consecutive_crashes).
         e. Ajoute samples au buffer in-memory.
         f. Si buffer atteint games_per_batch : flush via write_batch.
         g. Update state.
      3. Flush samples restants (dernier batch partiel).
      4. Mark state fin de génération.

    NOT thread-safe v1.0.0 (sequential ADR-004).
    """

    def __init__(
        self,
        uci_client: UciClient,
        config: SelfplayConfig,
        state_manager: RunStateManager,
        abort_flag: dict[str, bool],
        datasets_dir: Path,
    ) -> None:
        """Initialize orchestrator.

        Args:
            uci_client: UciClient déjà démarré (handshake fait, options Dirichlet
                        set au start).
            config: SelfplayConfig avec tous hyperparams.
            state_manager: RunStateManager pour persistance state.
            abort_flag: dict {requested: bool} de install_signal_handlers.
            datasets_dir: répertoire pour les .npz batchés.
        """
        self.client = uci_client
        self.config = config
        self.state = state_manager
        self.abort_flag = abort_flag
        self.datasets_dir = Path(datasets_dir)
        self.datasets_dir.mkdir(parents=True, exist_ok=True)

        self._games_since_restart = 0
        self._consecutive_crashes = 0

    def run_generation(self, gen: int, target_games: int | None = None) -> None:
        """Run self-play pour une génération.

        Args:
            gen: index de génération (1-based).
            target_games: total de parties à jouer. None -> utilise
                          config.target_games_per_gen.
        """
        target = target_games if target_games is not None else self.config.target_games_per_gen

        # Set phase=selfplay + current_gen au démarrage (cohérent SPEC §11.5).
        # Phase 12 hotfix-015 — also persist selfplay.target_games so the dashboard
        # can display "X/N games" (else field stays at default 0 → "X/?" in UI).
        state_obj = self.state.state
        if state_obj.phase != "selfplay":
            self.state.update(
                phase="selfplay",
                phase_started=datetime.now(timezone.utc),
                current_gen=gen,
                selfplay__target_games=target,
            )
        elif state_obj.selfplay.target_games != target:
            # Phase already selfplay (resume) but target_games not yet persisted.
            self.state.update(selfplay__target_games=target)

        # Determine resume point.
        completed_games = state_obj.selfplay.completed_games
        completed_batches = state_obj.selfplay.completed_batches

        if completed_games >= target:
            LOG.info("Generation %d already complete (%d games)", gen, completed_games)
            return

        LOG.info(
            "Resuming generation %d at game %d/%d (batch %d)",
            gen,
            completed_games,
            target,
            completed_batches,
        )

        # Génère TOUS les RNGs upfront pour reproducibilité bit-perfect au resume.
        # all_rngs[k] est le RNG de la k-ème partie, peu importe quand la fonction est appelée.
        all_rngs = make_game_rngs(self.config.worker_seed, target)

        # Buffer in-memory pour le batch courant.
        buffer: list[Sample] = []
        games_in_current_batch = state_obj.selfplay.current_batch_games

        for game_idx in range(completed_games, target):
            # Check abort AVANT de démarrer une nouvelle partie (jamais mid-game).
            if self._check_abort():
                LOG.info("Abort requested — flushing partial batch and exiting")
                if buffer:
                    self._flush_batch(gen, completed_batches, buffer, games_in_current_batch)
                self.state.update(status="aborted")
                return

            # Restart worker si seuil atteint.
            self._maybe_restart_worker()

            # Play one game (recovery détaillée commit d ; ici simple pass-through).
            samples = self._run_one_game_with_recovery(
                rng=all_rngs[game_idx],
                game_idx=game_idx,
            )
            if samples is None:
                # Partie écartée après crashes — passer à la suivante.
                LOG.warning("Game %d discarded — continuing", game_idx)
                continue

            buffer.extend(samples)
            games_in_current_batch += 1
            self._games_since_restart += 1

            # Flush si seuil atteint.
            if games_in_current_batch >= self.config.games_per_batch:
                self._flush_batch(gen, completed_batches, buffer, games_in_current_batch)
                completed_batches += 1
                buffer = []
                games_in_current_batch = 0

            # Update state après chaque partie (raffiné commit e).
            self.state.update(
                selfplay__completed_games=game_idx + 1,
                selfplay__current_batch_idx=completed_batches,
                selfplay__current_batch_games=games_in_current_batch,
            )

        # Flush du dernier batch partiel s'il reste des samples.
        if buffer:
            self._flush_batch(gen, completed_batches, buffer, games_in_current_batch)
            completed_batches += 1

        # Mark generation done.
        self.state.update(
            selfplay__completed_batches=completed_batches,
            selfplay__completed_games=target,
        )
        LOG.info("Generation %d complete (%d games, %d batches)", gen, target, completed_batches)

    def _flush_batch(
        self,
        gen: int,
        batch_idx: int,
        samples: list[Sample],
        n_games: int,
    ) -> None:
        """Atomic flush du buffer vers un fichier .npz batché (ADR-002)."""
        from nanozero_training.data.npz_writer import make_batch_filename, write_batch

        if not samples:
            return
        filename = make_batch_filename(gen, batch_idx)
        path = self.datasets_dir / filename
        write_batch(samples, path, gen=gen, batch_idx=batch_idx, n_games=n_games)
        LOG.info(
            "Flushed batch %d (%d samples, %d games) -> %s",
            batch_idx,
            len(samples),
            n_games,
            filename,
        )
        self.state.update(selfplay__last_batch_file=filename)

    def _check_abort(self) -> bool:
        """Check si signal abort reçu (Ctrl+C / SIGTERM / SIGHUP)."""
        return self.abort_flag.get("requested", False)

    def _maybe_restart_worker(self) -> None:
        """Restart UCI worker tous les worker_restart_every parties (robustesse mémoire)."""
        if self._games_since_restart >= self.config.worker_restart_every:
            LOG.info("Restarting UCI worker after %d games", self._games_since_restart)
            self.client.restart()
            self._games_since_restart = 0

    def _run_one_game_with_recovery(
        self,
        rng: np.random.Generator,
        game_idx: int,
    ) -> list[Sample] | None:
        """Play one game avec crash recovery (SPEC §6.4).

        Stratégie :
        - Sur UciCrashError ou UciTimeoutError : log + restart UCI + retry
          la MÊME partie (game_idx) depuis le début. NE PAS passer à la
          partie suivante (la partie est ré-essayée avec le même RNG).
        - Tracker _consecutive_crashes ; abort la génération si seuil
          max_consecutive_crashes atteint -> RuntimeError + state.status="error".
        - Reset _consecutive_crashes à 0 sur succès.

        Args:
            rng: générateur pour cette partie (réutilisé entre retries).
            game_idx: index pour logs uniquement.

        Returns:
            list[Sample] si la partie a réussi (éventuellement après retry).
            None n'est pas retourné en pratique (boucle exit via return ou raise).

        Raises:
            RuntimeError: si max_consecutive_crashes consécutifs atteint.
        """
        for attempt in range(self.config.max_consecutive_crashes + 1):
            try:
                samples = play_one_game(self.client, self.config, rng)
            except (UciCrashError, UciTimeoutError) as e:
                self._consecutive_crashes += 1
                LOG.warning(
                    "Game %d attempt %d failed: %s (consecutive crashes: %d/%d)",
                    game_idx,
                    attempt,
                    type(e).__name__,
                    self._consecutive_crashes,
                    self.config.max_consecutive_crashes,
                )

                if self._consecutive_crashes >= self.config.max_consecutive_crashes:
                    LOG.error(
                        "Max consecutive crashes (%d) reached — aborting generation",
                        self.config.max_consecutive_crashes,
                    )
                    self.state.update(
                        status="error",
                        last_error=(
                            f"UCI crashed {self._consecutive_crashes} consecutive times "
                            f"at game {game_idx}"
                        ),
                        crash_count=self._consecutive_crashes,
                    )
                    raise RuntimeError(
                        f"UCI crashed {self._consecutive_crashes} consecutive times "
                        f"at game {game_idx} — aborting generation"
                    ) from e

                # Restart UCI + retry (même game_idx, même rng — sans rejouer le tirage).
                try:
                    self.client.restart()
                except Exception as restart_err:
                    LOG.error("UCI restart failed: %s", restart_err)
                    self.state.update(status="error", last_error=str(restart_err))
                    raise
                self._games_since_restart = 0
            else:
                # Success : reset compteur consecutive crashes.
                self._consecutive_crashes = 0
                return samples

        # Should not reach here (loop exits via return ou raise).
        return None
