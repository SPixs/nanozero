"""Server configuration sourced from environment variables.

Pattern: explicit env-var keys with sensible defaults. Loaded once at startup
via load_config() and frozen for the process lifetime.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# -----------------------------------------------------------------------------
# STORY-007 — normalisation + blocklist du pseudo « adresse ouverte »
# -----------------------------------------------------------------------------

# Longueur max d'un pseudo (clé d'unicité en lowercase). AC-1.
PSEUDO_MAX_LEN = 24

# Caractères autorisés APRÈS normalisation (NFC + lowercase + strip + troncature). AC-1.
_PSEUDO_RE = re.compile(r"^[a-z0-9_-]{1,24}$")

# Blocklist MINIMALE de termes offensants évidents (T6). Match SOUS-CHAÎNE sur le pseudo
# normalisé (lowercase) → un pseudo qui CONTIENT l'un de ces termes est rejeté (→ NULL). Liste
# volontairement courte et conservatrice (un faux négatif = pseudo cosmétique inoffensif ; la
# modération exhaustive est V1.5, cf. canon §10). Étendable sans changer le contrat.
PSEUDO_BLOCKLIST: frozenset[str] = frozenset(
    {
        "nigger",
        "nigga",
        "faggot",
        "fuck",
        "shit",
        "cunt",
        "bitch",
        "rape",
        "nazi",
        "hitler",
        "retard",
        "slut",
        "whore",
        "pedo",
    }
)


def normalize_pseudo(raw: str | None) -> str | None:
    """Normalise un pseudo « adresse ouverte » côté SERVEUR (autorité). STORY-007 T4.2 / AC-1.

    Pipeline : ``None``/absent → ``None`` ; sinon NFC Unicode → ``strip().lower()`` → troncature à
    ``PSEUDO_MAX_LEN`` → regex ``^[a-z0-9_-]{1,24}$`` → blocklist (sous-chaîne). Tout échec
    (vide, hors charset, offensant) renvoie ``None`` SILENCIEUSEMENT — JAMAIS d'exception : un
    pseudo invalide ne doit pas faire échouer la soumission (AC-6). Le pseudo est purement
    cosmétique : ``None`` = anonyme (compte au collectif seulement).

    Args:
        raw: pseudo brut tel que reçu du header ``X-Pseudo`` (peut être ``None``).

    Returns:
        Le pseudo normalisé (lowercase, ≤ 24 chars) ou ``None`` si absent / invalide / offensant.
    """
    if raw is None:
        return None
    # NFC : forme canonique composée (un é tapé en deux codepoints == un seul) avant troncature,
    # pour que la clé d'unicité lowercase soit stable quelle que soit la saisie.
    normalized = unicodedata.normalize("NFC", raw).strip().lower()[:PSEUDO_MAX_LEN]
    if not _PSEUDO_RE.match(normalized):
        return None
    if any(bad in normalized for bad in PSEUDO_BLOCKLIST):
        return None
    return normalized


@dataclass(frozen=True)
class ServerConfig:
    """Immutable server config holder.

    Attributes:
        host: bind host (0.0.0.0 to accept Tailscale/LAN connections).
        port: bind port.
        api_key: required auth token for protected endpoints. Empty string in
            development mode disables auth (allowed for local-only setups).
        database_url: PostgreSQL libpq URL (postgresql://user:pwd@host:5432/db).
        pg_pool_min/pg_pool_max: connection pool sizing (HTTP handlers + flusher
            thread share this pool ; pg_pool_max must stay < PG max_connections).
        flusher_enabled: ADR-015 — enable the background NPZ flusher thread.
            False disables write-through cache (positions stay in SQLite BLOBs).
        npz_output_dir: ADR-015 — destination for NPZ shards. None defaults to
            ``./datasets`` (the /data volume in prod).
        flush_threshold_positions: ADR-015 — min unflushed positions per
            model_version to trigger a shard write. Default 100_000.
        flush_retention_window: ADR-015 — keep flushed positions for the last N
            model_versions in HOT cache before DELETE. Default 2 (was 5 jusqu'au
            2026-05-22). Réduit suite incident DB 128 GB : avec retention=5 on
            gardait 2.5M+ positions flushed inutilement, paralysant les writes
            sous contention. Le replay buffer training utilise les NPZ shards
            sur disque, pas le SQLite HOT cache → retention=2 suffit largement.
        flush_tick_interval_seconds: ADR-015 — flusher loop tick period.
            Default 30.0s.
        flush_browser: D.3 chantier 1 — flushe AUSSI la cohorte browser vers des
            shards ``browser-gen*`` séparés (hors corpus training). Default True.
        browser_flush_threshold: seuil de flush de la cohorte browser (plus bas car
            corpus plus petit). Default 25_000.
        stale_claim_timeout_seconds: watchdog — un job 'claimed' au-delà de ce
            délai est supprimé (worker mort). 0 désactive. Default 3600.
        stale_claim_max_per_tick: cap de suppression de claims périmés par tick
            (draine un gros backlog progressivement). Default 500_000.
        season_target: gamification (STORY-001) — objectif de positions vérifiées de la
            « saison » (génération courante), pour la jauge collective ``collective_pct``.
            Constante de config (pas un champ DB) car c'est un seuil de présentation, pas
            de la donnée durable. Default 1_000_000.
    """

    host: str
    port: int
    api_key: str
    database_url: str
    pg_pool_min: int = 2
    pg_pool_max: int = 10
    flusher_enabled: bool = True
    npz_output_dir: Path | None = None
    flush_threshold_positions: int = 100_000
    flush_retention_window: int = 2  # was 5, réduit 2026-05-22 (cf. docstring)
    flush_tick_interval_seconds: float = 30.0
    flush_browser: bool = True
    browser_flush_threshold: int = 25_000
    stale_claim_timeout_seconds: int = 3600
    stale_claim_max_per_tick: int = 500_000
    season_target: int = 1_000_000

    def resolve_npz_output_dir(self) -> Path:
        """Return the effective NPZ output directory (default ``./datasets`` = the /data volume)."""
        return self.npz_output_dir or Path("datasets")


def load_config() -> ServerConfig:
    """Read environment variables and return a frozen ServerConfig."""
    npz_dir_env = os.environ.get("NANOZERO_JOBSERVER_NPZ_OUTPUT_DIR")
    return ServerConfig(
        host=os.environ.get("NANOZERO_JOBSERVER_HOST", "0.0.0.0"),
        # Default 8090 chosen to avoid collision with common dev servers on 8080
        # (Tomcat, yarn, etc.). Override via NANOZERO_JOBSERVER_PORT.
        port=int(os.environ.get("NANOZERO_JOBSERVER_PORT", "8090")),
        api_key=os.environ.get("NANOZERO_JOBSERVER_API_KEY", ""),
        database_url=os.environ.get(
            "NANOZERO_JOBSERVER_DATABASE_URL",
            "postgresql://nanozero:nanozero@localhost:5432/nanozero",
        ),
        pg_pool_min=int(os.environ.get("NANOZERO_JOBSERVER_PG_POOL_MIN", "2")),
        pg_pool_max=int(os.environ.get("NANOZERO_JOBSERVER_PG_POOL_MAX", "10")),
        flusher_enabled=os.environ.get("NANOZERO_JOBSERVER_FLUSHER_ENABLED", "1")
        not in ("0", "false", "False", ""),
        npz_output_dir=Path(npz_dir_env) if npz_dir_env else None,
        flush_threshold_positions=int(
            os.environ.get("NANOZERO_JOBSERVER_FLUSH_THRESHOLD", "100000")
        ),
        flush_retention_window=int(
            os.environ.get("NANOZERO_JOBSERVER_FLUSH_RETENTION_WINDOW", "2")
        ),
        flush_tick_interval_seconds=float(
            os.environ.get("NANOZERO_JOBSERVER_FLUSH_TICK_INTERVAL", "30.0")
        ),
        flush_browser=os.environ.get("NANOZERO_JOBSERVER_FLUSH_BROWSER", "1")
        not in ("0", "false", "False", ""),
        browser_flush_threshold=int(
            os.environ.get("NANOZERO_JOBSERVER_BROWSER_FLUSH_THRESHOLD", "25000")
        ),
        stale_claim_timeout_seconds=int(
            os.environ.get("NANOZERO_JOBSERVER_STALE_CLAIM_TIMEOUT", "3600")
        ),
        stale_claim_max_per_tick=int(
            os.environ.get("NANOZERO_JOBSERVER_STALE_CLAIM_MAX_PER_TICK", "500000")
        ),
        season_target=int(os.environ.get("NANOZERO_JOBSERVER_SEASON_TARGET", "1000000")),
    )
