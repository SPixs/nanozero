"""SPRT results storage — historique durable des décisions de promotion (vs AGENTS.md/mémoire).

Chaque ligne = un test SPRT d'un candidat (``candidate_version``) contre une baseline
(``baseline_version``) : décision accepted/rejected/inconclusive + Elo + W/D/L. C'est la
seule donnée qui manquait pour avoir un historique complet du cycle (gen-026 rejeté à
combien d'Elo ? delta moyen from-scratch vs continued ?), et elle sert de garde naturelle
contre une re-promotion accidentelle d'un candidat rejeté.

Storage : PostgreSQL via psycopg3 (cf. db.py). Le pool fournit `connect()` — plus de
paramètre `db_path`. `id` vient d'un BIGSERIAL via `INSERT ... RETURNING id` (psycopg3 ne
remplit pas `cursor.lastrowid`). `created_at` (TIMESTAMPTZ) revient en `datetime` →
reconverti en ISO 8601 dans `_row_to_sprt` pour préserver le contrat `str` de la dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanozero_jobserver.storage.db import connect

_DECISIONS = ("accepted", "rejected", "inconclusive")


@dataclass(frozen=True)
class SprtResult:
    """Un résultat SPRT archivé."""

    id: int
    candidate_version: int
    baseline_version: int
    decision: str
    elo_estimate: float | None
    games_played: int | None
    wins: int | None
    draws: int | None
    losses: int | None
    notes: str | None
    created_at: str


def record_sprt(
    candidate_version: int,
    baseline_version: int,
    decision: str,
    elo_estimate: float | None = None,
    games_played: int | None = None,
    wins: int | None = None,
    draws: int | None = None,
    losses: int | None = None,
    notes: str | None = None,
) -> int:
    """Enregistre un résultat SPRT. Returns l'id inséré. Raise sur décision invalide."""
    if decision not in _DECISIONS:
        raise ValueError(f"decision must be one of {_DECISIONS}, got {decision!r}")
    with connect() as conn:
        cur = conn.execute(
            "INSERT INTO sprt_results"
            " (candidate_version, baseline_version, decision, elo_estimate, games_played,"
            "  wins, draws, losses, notes)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
            " RETURNING id",
            (
                candidate_version,
                baseline_version,
                decision,
                elo_estimate,
                games_played,
                wins,
                draws,
                losses,
                notes,
            ),
        )
        return int(cur.fetchone()["id"])


def _row_to_sprt(row: dict[str, Any]) -> SprtResult:
    created_at = row["created_at"]
    return SprtResult(
        id=row["id"],
        candidate_version=row["candidate_version"],
        baseline_version=row["baseline_version"],
        decision=row["decision"],
        elo_estimate=row["elo_estimate"],
        games_played=row["games_played"],
        wins=row["wins"],
        draws=row["draws"],
        losses=row["losses"],
        notes=row["notes"],
        created_at=created_at.isoformat() if created_at else None,
    )


def list_sprt(limit: int = 50, model_version: int | None = None) -> list[SprtResult]:
    """Historique SPRT, plus récent d'abord. ``model_version`` filtre les tests où il est
    candidat OU baseline (toute l'implication d'une gen)."""
    where = ""
    params: list[object] = []
    if model_version is not None:
        where = " WHERE candidate_version = %s OR baseline_version = %s"
        params = [model_version, model_version]
    params.append(int(limit))
    with connect() as conn:
        cur = conn.execute(
            "SELECT id, candidate_version, baseline_version, decision, elo_estimate,"
            "  games_played, wins, draws, losses, notes, created_at"
            f" FROM sprt_results{where} ORDER BY id DESC LIMIT %s",
            tuple(params),
        )
        return [_row_to_sprt(row) for row in cur.fetchall()]


def latest_decision_for_candidate(candidate_version: int) -> str | None:
    """Décision SPRT la plus récente où ``candidate_version`` était candidat (garde anti-promote)."""
    with connect() as conn:
        row = conn.execute(
            "SELECT decision FROM sprt_results WHERE candidate_version = %s"
            " ORDER BY id DESC LIMIT 1",
            (candidate_version,),
        ).fetchone()
    return row["decision"] if row else None


def latest_accepted_promotion() -> SprtResult | None:
    """Le dernier SPRT ``accepted`` (= dernière promotion) — pour le bandeau gamification.

    Source du gain Elo de la dernière génération promue : on lit le ``sprt_results``
    durable (déjà peuplé par ``POST /sprt/record`` à chaque cycle) plutôt que d'ajouter
    une colonne ``elo_gain`` sur ``models`` + une migration sur la DB live (cf. STORY-001,
    décision laissée à l'implémentation). ``candidate_version`` = la gen promue, ``elo_estimate``
    = son gain vs baseline. Retourne None si aucune promotion n'a jamais été archivée.
    """
    with connect() as conn:
        row = conn.execute(
            "SELECT id, candidate_version, baseline_version, decision, elo_estimate,"
            "  games_played, wins, draws, losses, notes, created_at"
            " FROM sprt_results WHERE decision = 'accepted'"
            " ORDER BY id DESC LIMIT 1"
        ).fetchone()
    return _row_to_sprt(row) if row else None
