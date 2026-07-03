"""Replay buffer storage : insert positions + sample window.

Pattern: positions are appended one-by-one (per game) or batched (per job submit).
Sampling reads N random rows from the last `window` model versions, matching
the AlphaZero replay buffer convention (cf. ADR-014 §Rolling window).

Migré SQLite → PostgreSQL (psycopg3, cf. docs/PG-MIGRATION-plan.md §3.3) : plus de
paramètre `db_path` (le pool global de `db.py` est la source de connexions), BLOBs
``input_planes``/``policy_target`` mappés en ``BYTEA`` (``bytes`` Python natif, sans
cast), ``flushed_to_npz`` est un BOOLEAN (``FALSE``/``TRUE``), accès colonnes PAR NOM
uniquement (``dict_row``).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

import numpy as np

from nanozero_jobserver.storage.db import connect

_POLICY_LEN = 4672  # ADR-002 — cible policy dense (distribution de visites normalisée)

# L2 (2026-06-06) — compression zlib des BLOBs du HOT cache.
# Les positions sont du float32 dense ~94 % zéros (input_planes) / ~99,7 % zéros
# (policy) → zlib niveau 6 ≈ 130× (même ratio que np.savez_compressed des NPZ).
# Compat : colonne `compressed` (0 = brut legacy, 1 = zlib). On décompresse au
# READ pour que Position.input_planes/policy_target restent du brut côté
# consommateurs (npz_writer, api/replay) — aucun changement de leur côté.
_BLOB_ZLIB_LEVEL = 6


def _compress_blob(raw: bytes) -> bytes:
    return zlib.compress(raw, _BLOB_ZLIB_LEVEL)


def _decompress_blob(stored: bytes, compressed: int) -> bytes:
    return zlib.decompress(stored) if compressed else stored


@dataclass(frozen=True)
class Position:
    """One training sample (one position from one selfplay game).

    Attributes:
        game_id: unique id of the source game (for traceability / dedup).
        model_version: version of the model that played this game.
        ply: 0-based move index within the game.
        fen: position FEN (audit/debug; not used for training input).
        input_planes: serialized network input planes (BYTEA, e.g., float32 array bytes).
        policy_target: serialized policy target distribution (BYTEA).
        outcome: game outcome from current side-to-move POV (-1, 0, +1).
        pseudo: STORY-007 — label PUBLIC cosmétique du soumetteur (crédits « adresse ouverte »),
            NORMALISÉ côté serveur, ou ``None`` (anonyme / pseudo absent ou invalide). Jamais utilisé
            pour l'entraînement ni l'anti-poison : pure annotation. Défaut ``None`` → les chemins de
            lecture (sample/iter, qui n'ont pas besoin du pseudo) construisent Position sans l'exiger.
    """

    game_id: str
    model_version: int
    ply: int
    fen: str
    input_planes: bytes
    policy_target: bytes
    outcome: float
    pseudo: str | None = None


# INSERT partagé entre l'insert simple et le submit ATOMIQUE (cf. submit_job_with_positions).
# STORY-007 : la colonne `pseudo` (nullable) est ajoutée en queue → l'ordre des colonnes existantes
# est inchangé. Un pseudo None est inséré comme NULL (anonyme) — rétro-compat flotte native.
INSERT_POSITIONS_SQL = (
    "INSERT INTO positions"
    " (game_id, model_version, ply, fen, input_planes, policy_target,"
    " outcome, compressed, source, pseudo)"
    " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
)


def build_position_rows(positions: list[Position], source: str = "fleet") -> list[tuple[object, ...]]:
    """Compress + tuple-ise positions pour ``INSERT_POSITIONS_SQL``.

    Factorisé pour que l'insert simple ET le submit atomique (``submit_job_with_positions``,
    qui doit insérer dans la MÊME transaction que la complétion du job) partagent exactement
    le même encodage de ligne (BLOBs compressés zlib, ``compressed=1``, tag ``source``).
    Les BLOBs sont des ``bytes`` Python directs → mappés en ``BYTEA`` par psycopg3 sans cast.
    """
    return [
        (
            p.game_id,
            p.model_version,
            p.ply,
            p.fen,
            _compress_blob(p.input_planes),
            _compress_blob(p.policy_target),
            p.outcome,
            1,
            source,
            p.pseudo,  # STORY-007 — None → NULL (anonyme) ; déjà normalisé côté handler
        )
        for p in positions
    ]


def insert_positions(positions: list[Position], source: str = "fleet") -> int:
    """Bulk-insert positions. Returns count inserted.

    Args:
        positions: list of Position records to append.
        source: provenance tag applied to every row of this submit (B.4 quarantaine).
            'fleet' = flotte de confiance (workers Java, défaut) ; 'browser' =
            contribution bénévole navigateur (exclue de l'entraînement par défaut).
            **Server-authoritative** : décidé par le handler selon le Content-Type,
            jamais lu du payload client (un browser pourrait se déclarer 'fleet').

    Returns:
        Number of rows actually inserted (== len(positions)).
    """
    if not positions:
        return 0
    rows = build_position_rows(positions, source)
    with connect() as conn:
        conn.cursor().executemany(INSERT_POSITIONS_SQL, rows)
    return len(rows)


def count_positions(min_model_version: int | None = None) -> int:
    """Total positions in the buffer, optionally filtered by min model_version.

    Args:
        min_model_version: include only positions where model_version >= this.
            None (default) counts everything.

    Returns:
        Row count.
    """
    with connect() as conn:
        if min_model_version is None:
            cur = conn.execute("SELECT COUNT(*) AS n FROM positions")
        else:
            cur = conn.execute(
                "SELECT COUNT(*) AS n FROM positions WHERE model_version >= %s",
                (min_model_version,),
            )
        return int(cur.fetchone()["n"])


def count_positions_by_source(
    model_version: int | None = None,
) -> dict[str, int]:
    """Live position count per provenance tag ('fleet' / 'browser') — B.4 visibility.

    Counts rows currently in the ``positions`` table (HOT cache). Browser rows are
    never flushed (quarantine), so for them this is the true total ; fleet rows
    under-report by the purge volume (durable fleet counts come from ``batches``).

    Args:
        model_version: if set, count only this version. None = all versions.

    Returns:
        Mapping {source: count}. Sources with zero live rows are omitted.
    """
    with connect() as conn:
        if model_version is None:
            cur = conn.execute("SELECT source, COUNT(*) AS n FROM positions GROUP BY source")
        else:
            cur = conn.execute(
                "SELECT source, COUNT(*) AS n FROM positions"
                " WHERE model_version = %s GROUP BY source",
                (model_version,),
            )
        return {str(row["source"]): int(row["n"]) for row in cur.fetchall()}


def source_distribution(model_version: int, source: str) -> dict[str, float]:
    """Cheap distributional features of a (model_version, source) cohort — for drift detection (D.2).

    Computed via SQL aggregates ONLY (no BLOB decode) : outcome W/D/L balance, mean game length,
    positions per game. Ces features distinguent un empoisonnement grossier / un contributeur buggé
    de la flotte de confiance SANS faire confiance aux cibles policy/value (falsifiables).

    Les buckets d'outcome utilisent un seuil ±0.33 pour tolérer le label smoothing (z ∈ {-1,0,+1}
    lissé, cf. value_target_smoothing). Renvoie n=0 (et zéros) pour une cohorte absente.

    Args:
        model_version: génération à analyser.
        source: 'browser' ou 'fleet'.

    Returns:
        {n, n_games, positions_per_game, outcome_mean, win_frac, draw_frac, loss_frac, mean_ply, max_ply}.
    """
    sql = (
        "SELECT COUNT(*) AS n, COUNT(DISTINCT game_id) AS n_games, AVG(outcome) AS outcome_mean,"
        " AVG(CASE WHEN outcome > 0.33 THEN 1.0 ELSE 0.0 END) AS win_frac,"
        " AVG(CASE WHEN outcome BETWEEN -0.33 AND 0.33 THEN 1.0 ELSE 0.0 END) AS draw_frac,"
        " AVG(CASE WHEN outcome < -0.33 THEN 1.0 ELSE 0.0 END) AS loss_frac,"
        " AVG(ply) AS mean_ply, MAX(ply) AS max_ply"
        " FROM positions WHERE model_version = %s AND source = %s"
    )
    with connect() as conn:
        row = conn.execute(sql, (model_version, source)).fetchone()
    n = int(row["n"] or 0)
    n_games = int(row["n_games"] or 0)
    return {
        "n": n,
        "n_games": n_games,
        "positions_per_game": (n / n_games) if n_games else 0.0,
        "outcome_mean": float(row["outcome_mean"] or 0.0),
        "win_frac": float(row["win_frac"] or 0.0),
        "draw_frac": float(row["draw_frac"] or 0.0),
        "loss_frac": float(row["loss_frac"] or 0.0),
        "mean_ply": float(row["mean_ply"] or 0.0),
        "max_ply": int(row["max_ply"] or 0),
    }


def policy_resolution(
    model_version: int,
    source: str,
    sample_size: int = 256,
) -> dict[str, float]:
    """Estime le nombre de sims MCTS *effectifs* derrière les cibles policy d'une cohorte (D.2).

    La cible policy est une distribution de visites NORMALISÉE : ``π_i = N_i / T`` avec ``T`` le total
    des visites (≈ num_sims). Les entrées non-nulles sont donc des multiples de ``1/T`` → ``T`` se
    récupère par ``sum(π) / min(π>0)`` (robuste au stockage normalisé OU en comptes bruts). Peu de
    sims ⇒ π grossière ⇒ estimation basse. C'est le **pendant vérification** du num_sims dicté côté
    serveur (cf. browser-worker run-loop) : ça trahit un client qui a ignoré la consigne **sans rien
    croire d'une métadonnée auto-déclarée**.

    Coût : décode le BLOB policy d'un échantillon BORNÉ (la queue récente, assistée par l'index sur
    ``model_version``), PAS toute la cohorte. ⚠️ Contournable par un attaquant sophistiqué qui
    rembourrerait π à grain fin ; attrape le client naïf / mal configuré. L'arbitre de la VALEUR reste
    le SPRT (D.3), jamais cette sonde.

    Returns ``{n_sampled, median_eff_sims, p10_eff_sims, mean_support, mean_max_mass}`` (zéros si vide).
    """
    sql = (
        "SELECT policy_target, compressed FROM positions"
        " WHERE model_version = %s AND source = %s"
        " ORDER BY id DESC LIMIT %s"  # queue récente : cheap + index-assisté ; eff_sims est position-indépendant
    )
    effs: list[float] = []
    supports: list[int] = []
    max_masses: list[float] = []
    with connect() as conn:
        rows = conn.execute(sql, (model_version, source, sample_size)).fetchall()
    for row in rows:
        raw = _decompress_blob(row["policy_target"], row["compressed"])
        arr = np.frombuffer(raw, dtype="<f4")
        if arr.size != _POLICY_LEN:
            continue  # BLOB non-conforme (ex. fixture de test) → ignoré, pas d'erreur
        nz = arr[arr > 0.0]
        total = float(arr.sum())
        if nz.size == 0 or total <= 0.0:
            continue
        mn = float(nz.min())
        if mn <= 0.0:
            continue
        effs.append(total / mn)
        supports.append(int(nz.size))
        max_masses.append(float(arr.max()))
    n = len(effs)
    if n == 0:
        return {
            "n_sampled": 0,
            "median_eff_sims": 0.0,
            "p10_eff_sims": 0.0,
            "mean_support": 0.0,
            "mean_max_mass": 0.0,
        }
    eff_arr = np.array(effs)
    return {
        "n_sampled": n,
        "median_eff_sims": float(np.median(eff_arr)),
        "p10_eff_sims": float(np.percentile(eff_arr, 10)),
        "mean_support": float(np.mean(supports)),
        "mean_max_mass": float(np.mean(max_masses)),
    }


def sample_positions(
    n: int,
    current_model_version: int,
    window: int = 5,
    include_browser: bool = False,
) -> list[Position]:
    """Uniform random sample of N positions from the last `window` model versions.

    Args:
        n: number of positions to sample.
        current_model_version: latest model version (defines the upper end).
        window: number of consecutive model versions to include (rolling window).
            For window=5 and current_model_version=12, returns positions with
            model_version in [8, 12].
        include_browser: B.4 quarantaine. False (défaut) = la data ``source='browser'``
            est EXCLUE de l'échantillon d'entraînement. True = l'inclut (opérateur-only,
            route gated ; hook pour le gate Epic D / inspection). L'admission bornée-
            par-fraction (« robinet ») reste D.3 ; ici c'est un simple on/off.

    Returns:
        List of Position records, up to n entries. May return fewer if buffer is
        smaller than n positions matching the window.

    Notes:
        Uses ORDER BY RANDOM() LIMIT n (syntaxe identique en PG). Fine for our scale
        (≤ a few hundred MB) ; sur une grosse table BYTEA c'est un seqscan complet —
        pour de très gros buffers, envisager reservoir sampling ou un id-range trick.
    """
    if n <= 0:
        return []
    lower = max(1, current_model_version - window + 1)
    quarantine = "" if include_browser else " AND source != 'browser'"
    with connect() as conn:
        cur = conn.execute(
            "SELECT game_id, model_version, ply, fen, input_planes, policy_target,"
            " outcome, compressed"
            " FROM positions"
            " WHERE model_version >= %s AND model_version <= %s" + quarantine + ""
            " ORDER BY RANDOM()"
            " LIMIT %s",
            (lower, current_model_version, n),
        )
        return [
            Position(
                game_id=row["game_id"],
                model_version=row["model_version"],
                ply=row["ply"],
                fen=row["fen"],
                input_planes=_decompress_blob(row["input_planes"], row["compressed"]),
                policy_target=_decompress_blob(row["policy_target"], row["compressed"]),
                outcome=row["outcome"],
            )
            for row in cur.fetchall()
        ]


def count_new_positions(since_model_version: int) -> int:
    """Count positions generated by models strictly newer than `since_model_version`.

    Used by the trainer "should we train?" trigger : when this count exceeds a
    configurable threshold (e.g., 25000), run an epoch.

    Args:
        since_model_version: positions with model_version > this are counted.

    Returns:
        Row count.
    """
    with connect() as conn:
        cur = conn.execute(
            "SELECT COUNT(*) AS n FROM positions WHERE model_version > %s",
            (since_model_version,),
        )
        return int(cur.fetchone()["n"])


# -----------------------------------------------------------------------------
# ADR-015 : flush + purge queries for write-through cache.
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class PositionRow(Position):
    """Position + row id (needed for mark_positions_flushed).

    Subclass of Position adds the row id so the flusher can target updates
    without re-querying by content.
    """

    # STORY-007 : `Position.pseudo` a un défaut (None) → tout champ AJOUTÉ par cette sous-classe doit
    # AUSSI avoir un défaut (règle dataclass : pas de champ non-défaut après un champ défaut hérité).
    # `id` est toujours fourni en keyword par les appelants → le défaut 0 n'est jamais utilisé.
    id: int = 0


def _source_clause(source: str | None) -> str:
    """Fragment SQL de filtre de provenance (chantier 1 cloisonnement).

    None = toutes sources (comportement historique) ; 'fleet' = flotte de confiance
    (``source != 'browser'``, robuste à d'éventuelles autres provenances) ; 'browser' =
    cohorte navigateur uniquement.
    """
    if source == "browser":
        return " AND source = 'browser'"
    if source == "fleet":
        return " AND source != 'browser'"
    return ""


def count_unflushed_positions(
    model_version: int | None = None,
    source: str | None = None,
) -> int:
    """Count positions not yet flushed to NPZ.

    Args:
        model_version: if set, count only unflushed positions for this version.
        source: 'fleet' / 'browser' pour cibler une cohorte (chantier 1). None = toutes.

    Returns:
        Number of positions with flushed_to_npz = FALSE.
    """
    sql = "SELECT COUNT(*) AS n FROM positions WHERE flushed_to_npz = FALSE"
    params: list[object] = []
    if model_version is not None:
        sql += " AND model_version = %s"
        params.append(model_version)
    sql += _source_clause(source)
    with connect() as conn:
        return int(conn.execute(sql, params).fetchone()["n"])


def iter_unflushed_positions(
    model_version: int,
    limit: int,
    source: str = "fleet",
) -> list[PositionRow]:
    """Read up to `limit` unflushed positions for a given model_version and source.

    Returned in id-order (FIFO insertion order). Caller uses these to build an
    NPZ shard, then calls mark_positions_flushed(ids, batch_id) to update.

    Args:
        model_version: target model_version (one shard = one version).
        limit: max rows to return.
        source: 'fleet' (défaut, = ``source != 'browser'``) ou 'browser'. Chantier 1 :
            le flusher écrit des shards PAR-SOURCE (fleet → corpus 'selfplay-gen*',
            browser → 'browser-gen*' séparés). La quarantine de la data browser hors du
            corpus d'entraînement est désormais assurée par le NOMMAGE des shards (le glob
            training ne ramasse que 'selfplay-gen{N}'), plus par l'exclusion en dur ici.
    """
    if limit <= 0:
        return []
    with connect() as conn:
        cur = conn.execute(
            "SELECT id, game_id, model_version, ply, fen, input_planes,"
            " policy_target, outcome, compressed"
            " FROM positions"
            " WHERE flushed_to_npz = FALSE AND model_version = %s" + _source_clause(source) + ""
            " ORDER BY id ASC"
            " LIMIT %s",
            (model_version, limit),
        )
        return [
            PositionRow(
                id=row["id"],
                game_id=row["game_id"],
                model_version=row["model_version"],
                ply=row["ply"],
                fen=row["fen"],
                input_planes=_decompress_blob(row["input_planes"], row["compressed"]),
                policy_target=_decompress_blob(row["policy_target"], row["compressed"]),
                outcome=row["outcome"],
            )
            for row in cur.fetchall()
        ]


def mark_positions_flushed(
    position_ids: list[int],
    batch_id: int,
) -> int:
    """Mark positions as flushed to a given NPZ batch.

    Args:
        position_ids: list of positions.id rows to mark.
        batch_id: batches.id of the produced NPZ shard.

    Returns:
        Number of rows updated.
    """
    if not position_ids:
        return 0
    with connect() as conn:
        cur = conn.execute(
            "UPDATE positions"
            " SET flushed_to_npz = TRUE, batch_id = %s"
            " WHERE id = ANY(%s)",  # psycopg3 adapte la liste Python en array PG
            (batch_id, position_ids),
        )
        return int(cur.rowcount)


def delete_flushed_old(
    max_model_version: int,
    batch_size: int = 50_000,
    source: str | None = None,
) -> int:
    """Delete flushed positions with model_version <= max_model_version (purge).

    Used to bound the hot PostgreSQL cache : positions older than the retention
    window are removed after they've been safely flushed to NPZ shards.

    Args:
        max_model_version: upper bound (inclusive). Positions with
            model_version <= this AND flushed_to_npz=TRUE are deleted.
        batch_size: rows per DELETE batch (default 50_000). Un DELETE monolithique
            sur > 100k rows de BLOB BYTEA génère un volume massif de dead tuples
            d'un coup ; le batch loop ouvre une NOUVELLE transaction (et donc un
            commit) par itération pour borner le bloat MVCC entre deux passes
            d'autovacuum (cf. PG-MIGRATION-plan.md §3.3 / R2). En SQLite ce
            découpage servait à libérer le write lock (incident 2026-05-22, DB
            gonflée à 128 GB sur 2.5M rows) ; en PG il limite l'impact autovacuum.
        source: B4-D1 — restreint la purge à une cohorte ('browser' / 'fleet').
            None (défaut) = toutes sources (comportement de rétention inchangé).
            ``source='browser'`` permet de purger la cohorte navigateur flushée à la
            gen courante (que la rétention par version ne touche pas) sans toucher
            au fleet — le levier maintenance qui manquait.

    Returns:
        Number of rows deleted (cumulé sur tous les batches).
    """
    total = 0
    while True:
        # Une NOUVELLE connexion (= une transaction) PAR batch : le commit à la
        # sortie du `with` borne le bloat MVCC entre itérations.
        with connect() as conn:
            cur = conn.execute(
                "DELETE FROM positions WHERE id IN ("
                "  SELECT id FROM positions"
                f"  WHERE flushed_to_npz = TRUE AND model_version <= %s{_source_clause(source)}"
                f"  LIMIT {batch_size}"
                ")",
                (max_model_version,),
            )
            n = int(cur.rowcount)
        if n == 0:
            break
        total += n
    return total


def count_unflushed_by_version(source: str | None = None) -> dict[int, int]:
    """Live (not-yet-flushed) position count per model_version.

    Complements `batches.sum_positions_by_version` (durable/flushed) : the true
    total a version has produced so far is durable[v] + live[v]. The live tail
    is whatever the FlusherService hasn't packed into an NPZ shard yet.

    Args:
        source: provenance filter (B4-D2). None (défaut) = toutes sources
            (observabilité : l'opérateur voit tout). ``"fleet"`` exclut la cohorte
            browser quarantinée — À UTILISER pour le déclencheur d'entraînement
            (``should_train``), sinon du volume browser fait fire le training tôt.

    Returns:
        Mapping {model_version: unflushed position count}. Versions with zero
        unflushed rows are omitted.
    """
    with connect() as conn:
        cur = conn.execute(
            "SELECT model_version, COUNT(*) AS n"
            f" FROM positions WHERE flushed_to_npz = FALSE{_source_clause(source)}"
            " GROUP BY model_version"
        )
        return {int(row["model_version"]): int(row["n"]) for row in cur.fetchall()}


def list_unflushed_model_versions(source: str | None = None) -> list[int]:
    """Return sorted (asc) list of distinct model_versions having unflushed positions.

    The flusher iterates these to produce one NPZ shard per (version, batch_idx).

    Args:
        source: 'fleet' / 'browser' pour ne lister que les versions d'une cohorte
            (chantier 1). None = toutes sources (comportement historique).

    Returns:
        List of model_version integers (ascending).
    """
    with connect() as conn:
        cur = conn.execute(
            "SELECT DISTINCT model_version FROM positions"
            " WHERE flushed_to_npz = FALSE" + _source_clause(source) + ""
            " ORDER BY model_version ASC"
        )
        return [int(row["model_version"]) for row in cur.fetchall()]
