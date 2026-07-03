"""Décompte des contributeurs (STORY-016).

Deux cohortes, comptées **au submit** (jamais au claim — un claim sans submit = partie
abandonnée, non comptée) :

- ``browser`` : identité stable ``X-Contributor`` (token pseudonyme dérivé d'un UUID local
  au navigateur, STORY-015). Une entrée par appareil/navigateur, partagée par les K Web
  Workers d'un onglet et stable entre redémarrages.
- ``fleet`` : workers natifs Java. Leur ``worker_id`` (``jobs.claimed_by``) est déjà stable
  (``--worker-id``). On compte **par machine** : le suffixe numérique est retiré
  (``ovh-gpu-4`` → ``ovh-gpu``) pour ne pas reproduire le biais ×slots (un launcher = N
  slots sur 1 machine). Décision user 2026-06-28.

La table ``contributors`` (clé = ``token``) donne deux compteurs O(1)/index :
- cumul lifetime = ``COUNT(*)`` ;
- actifs maintenant = ``last_seen`` dans la dernière heure (index ``idx_contributors_last_seen``).

Aucun pseudo n'est exposé ici (seulement des comptes) — canon §10.
"""

from __future__ import annotations

import re

import psycopg

from nanozero_jobserver.storage.db import connect

# Suffixe numérique d'un worker_id fleet (slot/thread) : retiré pour compter par MACHINE.
# ``ovh-gpu-4`` → ``ovh-gpu`` ; ``bugatti-night-10`` → ``bugatti-night`` ; ``ovh-21`` → ``ovh``.
_FLEET_SLOT_SUFFIX = re.compile(r"-\d+$")

# Token contributeur navigateur : sha256(nz_client_id) tronqué à 16 hex (cf. STORY-015 client).
_CONTRIBUTOR_TOKEN_RE = re.compile(r"^[a-f0-9]{16}$")

_TOUCH_SQL = (
    "INSERT INTO contributors (token, source, first_seen, last_seen, named) "
    "VALUES (%s, %s, NOW(), NOW(), %s) "
    "ON CONFLICT (token) DO UPDATE SET "
    "  last_seen = EXCLUDED.last_seen, "
    "  named = contributors.named OR EXCLUDED.named"
)


def normalize_contributor_token(raw: str | None) -> str | None:
    """Valide le token ``X-Contributor`` (16 hex lowercase). Invalide/absent → ``None``.

    Ne lève jamais : un token malformé ne doit pas faire échouer la soumission.
    """
    if not raw:
        return None
    token = raw.strip().lower()
    return token if _CONTRIBUTOR_TOKEN_RE.match(token) else None


def fleet_machine_token(worker_id: str) -> str:
    """Réduit un ``worker_id`` fleet à sa MACHINE (retrait du suffixe numérique de slot)."""
    return _FLEET_SLOT_SUFFIX.sub("", worker_id)


def touch_contributor(
    conn: psycopg.Connection, token: str, source: str, named: bool
) -> None:
    """Upsert ``last_seen`` (et ``named`` monotone) pour un contributeur — dans la transaction ``conn``.

    Appelé depuis le submit ATOMIQUE (``submit_job_with_positions``) → même transaction que la
    complétion du job + l'insert des positions. ``named`` ne repasse jamais à FALSE (OR logique).
    """
    conn.execute(_TOUCH_SQL, (token, source, named))


def contributor_counts(window_seconds: int = 3600) -> dict:
    """Compteurs cumul + actifs (toutes sources) avec ventilation par source — 1 requête.

    Returns:
        ``{'total': int, 'online': int,
           'total_by_source': {'browser': int, 'fleet': int},
           'online_by_source': {'browser': int, 'fleet': int}}``
    """
    sql = (
        "SELECT source, "
        "  COUNT(*) AS total, "
        "  COUNT(*) FILTER (WHERE last_seen >= NOW() - make_interval(secs => %s)) AS online "
        "FROM contributors GROUP BY source"
    )
    total_by = {"browser": 0, "fleet": 0}
    online_by = {"browser": 0, "fleet": 0}
    total = online = 0
    with connect() as conn:
        for row in conn.execute(sql, (window_seconds,)).fetchall():
            t, o = int(row["total"]), int(row["online"])
            total += t
            online += o
            if row["source"] in total_by:  # cohortes connues ; toute autre compte au global
                total_by[row["source"]] = t
                online_by[row["source"]] = o
    return {
        "total": total,
        "online": online,
        "total_by_source": total_by,
        "online_by_source": online_by,
    }
