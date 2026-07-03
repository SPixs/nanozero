"""Server control flags — tiny key/value store in the `server_control` table.

Used for runtime switches that must survive a restart, the first being the
self-play pause flag : when set, /jobs/claim returns 204 so the whole fleet
goes idle without losing the pending queue (reversible clean stop).

Kept deliberately minimal (string values, one row per key). Read on the hot
claim path, so queries stay single-row and indexed by the PRIMARY KEY.

Storage : PostgreSQL via psycopg3 (cf. db.py). Le pool fournit `connect()` —
plus de paramètre `db_path`. Accès aux colonnes PAR NOM (dict_row).
"""

from __future__ import annotations

from nanozero_jobserver.storage.db import connect

SELFPLAY_PAUSED_KEY = "selfplay_paused"


def get_control(key: str) -> str | None:
    """Return the stored value for `key`, or None if unset."""
    with connect() as conn:
        cur = conn.execute("SELECT value FROM server_control WHERE key = %s", (key,))
        row = cur.fetchone()
        return row["value"] if row else None


def set_control(key: str, value: str) -> None:
    """Upsert `key` = `value`, refreshing updated_at."""
    with connect() as conn:
        conn.execute(
            "INSERT INTO server_control (key, value) VALUES (%s, %s)"
            " ON CONFLICT (key) DO UPDATE SET"
            "  value = EXCLUDED.value,"
            "  updated_at = NOW()",
            (key, value),
        )


def is_selfplay_paused() -> bool:
    """True when the self-play pause flag is set (claims should be refused)."""
    return get_control(SELFPLAY_PAUSED_KEY) == "1"


def set_selfplay_paused(paused: bool) -> None:
    """Set or clear the self-play pause flag."""
    set_control(SELFPLAY_PAUSED_KEY, "1" if paused else "0")


# -----------------------------------------------------------------------------
# On-demand autorefill — keep the pending queue "never empty"
# -----------------------------------------------------------------------------
# When enabled, /jobs/claim mints a fresh job for the current champion whenever
# the pending queue is empty (instead of returning 204), so the fleet never
# idles for lack of work. Still gated by the pause flag and by a champion being
# promoted. Config persists across restarts in server_control.

AUTOREFILL_ENABLED_KEY = "autorefill_enabled"
AUTOREFILL_SIMS_KEY = "autorefill_sims"
DEFAULT_AUTOREFILL_SIMS = 800


def get_autorefill() -> tuple[bool, int]:
    """Return (enabled, num_sims) for on-demand job generation.

    num_sims falls back to DEFAULT_AUTOREFILL_SIMS when never configured.
    """
    enabled = get_control(AUTOREFILL_ENABLED_KEY) == "1"
    sims_raw = get_control(AUTOREFILL_SIMS_KEY)
    sims = int(sims_raw) if sims_raw is not None else DEFAULT_AUTOREFILL_SIMS
    return enabled, sims


def set_autorefill(enabled: bool, num_sims: int | None = None) -> None:
    """Enable/disable on-demand generation; optionally update its num_sims."""
    set_control(AUTOREFILL_ENABLED_KEY, "1" if enabled else "0")
    if num_sims is not None:
        set_control(AUTOREFILL_SIMS_KEY, str(int(num_sims)))


# -----------------------------------------------------------------------------
# Cible de positions self-play → auto-pause (décharge l'opérateur du pause manuel)
# -----------------------------------------------------------------------------
# Quand une cible est posée, le flusher pause (ou notifie) dès que le volume FLEET
# entraînable de la gen courante l'atteint, puis EFFACE la cible (one-shot : sinon un
# resume re-paserait aussitôt). Persiste au restart dans server_control.

SELFPLAY_TARGET_KEY = "selfplay_target_positions"
SELFPLAY_TARGET_ACTION_KEY = "selfplay_target_action"


def get_selfplay_target() -> tuple[int | None, str]:
    """Return (target_positions, action). target None si non posée / 0. action='pause'|'notify'."""
    raw = get_control(SELFPLAY_TARGET_KEY)
    target = int(raw) if raw is not None and int(raw) > 0 else None
    action = get_control(SELFPLAY_TARGET_ACTION_KEY) or "pause"
    return target, action


def set_selfplay_target(target_positions: int | None, action: str = "pause") -> None:
    """Pose (ou efface si None/<=0) la cible d'arrêt. action='pause' (défaut) ou 'notify'."""
    valid = target_positions if target_positions and target_positions > 0 else 0
    set_control(SELFPLAY_TARGET_KEY, str(int(valid)))
    set_control(SELFPLAY_TARGET_ACTION_KEY, "notify" if action == "notify" else "pause")
