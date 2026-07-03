"""D.2 — détection de drift de la data ``source='browser'`` vs la flotte de confiance.

Compare statistiquement les features distributionnelles de la cohorte browser (équilibre W/D/L des
outcomes, longueur de partie, positions/partie) à celles de la flotte — à la manière de la détection
de pollution NPU (cf. memory ``npu-quantization-pollutes-replay``). Un drift au-delà de seuils
flague la cohorte ; D.3 (ouverture du robinet vers l'entraînement) n'admettra que les cohortes ``ok``.

⚠️ **Seuils LÂCHES, par conception.** Le self-play navigateur tourne légitimement à MOINS de sims
que la flotte (ex. 256 vs 1600) → cibles plus bruitées, parties potentiellement plus courtes. Ce
n'est PAS un empoisonnement. On vise donc le drift GROSSIER (cohorte all-draws, outcomes aberrants,
parties absurdement courtes/longues), pas l'écart de qualité attendu d'un budget de recherche moindre.
"""

from __future__ import annotations

from dataclasses import dataclass

from nanozero_jobserver.storage.replay_buffer import policy_resolution, source_distribution

_VERDICTS = ("ok", "warn", "drift")


@dataclass(frozen=True)
class DriftThresholds:
    """Seuils de drift (lâches). ``warn`` = à surveiller ; ``drift`` = flaguer/quarantiner."""

    draw_frac_warn: float = 0.20
    draw_frac_drift: float = 0.40
    outcome_mean_warn: float = 0.20
    outcome_mean_drift: float = 0.40
    ppg_ratio_warn_lo: float = 0.5  # positions/partie browser / fleet
    ppg_ratio_warn_hi: float = 2.0
    min_browser_positions: int = 1000  # en-dessous : échantillon insuffisant, pas de verdict
    # Résolution policy = sims MCTS effectifs estimés depuis la granularité de la cible normalisée
    # (le pendant VÉRIFICATION du num_sims dicté côté serveur). Contrairement aux features ci-dessus
    # (lâches car le low-sims est légitime), CELLE-CI cible justement le low-sims pour le QUANTIFIER.
    eff_sims_floor_drift: float = 64.0  # sims effectifs absurdement bas → flaguer drift
    eff_sims_floor_warn: float = 200.0  # < seuil de claim typique → à surveiller
    eff_sims_ratio_warn: float = 0.5  # browser < 0.5 × fleet (sims effectifs) → à surveiller
    policy_sample_size: int = 256  # taille de l'échantillon de BLOBs décodés par cohorte
    min_resolution_sample: int = 50  # en-dessous : pas de verdict de résolution


def drift_report(
    model_version: int,
    thresholds: DriftThresholds | None = None,
    include_policy_resolution: bool = True,
) -> dict[str, object]:
    """Compare la cohorte browser à la flotte pour ``model_version`` et rend un verdict.

    Returns un dict sérialisable : ``model_version``, ``verdict`` (ok|warn|drift|insufficient),
    ``browser``/``fleet`` (features distributionnelles), ``resolution`` (sims effectifs browser/fleet,
    ou ``None`` si désactivé), ``deltas`` (écarts mesurés), ``reasons``. Le verdict est le MAX de
    sévérité sur toutes les règles (une seule règle ``drift`` → ``drift``).

    ``include_policy_resolution`` (défaut True) décode un échantillon borné de BLOBs policy pour
    estimer les sims effectifs (cf. ``policy_resolution``) ; False = garde le chemin SQL-only cheap.
    """
    t = thresholds or DriftThresholds()
    browser = source_distribution(model_version, "browser")
    fleet = source_distribution(model_version, "fleet")
    resolution: dict[str, object] | None = None
    if include_policy_resolution:
        resolution = {
            "browser": policy_resolution(model_version, "browser", t.policy_sample_size),
            "fleet": policy_resolution(model_version, "fleet", t.policy_sample_size),
        }

    base = {
        "model_version": model_version,
        "browser": browser,
        "fleet": fleet,
        "resolution": resolution,
    }
    if browser["n"] < t.min_browser_positions or fleet["n"] == 0:
        return {
            **base,
            "verdict": "insufficient",
            "deltas": {},
            "reasons": [
                f"échantillon insuffisant (browser n={browser['n']} < {t.min_browser_positions}"
                f" ou fleet n={fleet['n']})"
            ],
        }

    draw_delta = abs(browser["draw_frac"] - fleet["draw_frac"])
    outcome_delta = abs(browser["outcome_mean"] - fleet["outcome_mean"])
    ppg_ratio = (
        browser["positions_per_game"] / fleet["positions_per_game"]
        if fleet["positions_per_game"]
        else 0.0
    )

    level = 0
    reasons: list[str] = []

    def bump(new_level: int, reason: str) -> None:
        nonlocal level
        level = max(level, new_level)
        reasons.append(reason)

    if draw_delta > t.draw_frac_drift:
        bump(2, f"draw_frac Δ={draw_delta:.2f} > {t.draw_frac_drift} (drift)")
    elif draw_delta > t.draw_frac_warn:
        bump(1, f"draw_frac Δ={draw_delta:.2f} > {t.draw_frac_warn} (warn)")

    if outcome_delta > t.outcome_mean_drift:
        bump(2, f"outcome_mean Δ={outcome_delta:.2f} > {t.outcome_mean_drift} (drift)")
    elif outcome_delta > t.outcome_mean_warn:
        bump(1, f"outcome_mean Δ={outcome_delta:.2f} > {t.outcome_mean_warn} (warn)")

    if not (t.ppg_ratio_warn_lo <= ppg_ratio <= t.ppg_ratio_warn_hi):
        bump(1, f"positions/partie ratio={ppg_ratio:.2f} hors [{t.ppg_ratio_warn_lo}, {t.ppg_ratio_warn_hi}] (warn)")

    # Résolution policy = sims effectifs (pendant vérification du num_sims dicté). On ne juge que si
    # l'échantillon décodé est suffisant (BLOBs conformes au format dense 4672).
    if resolution is not None:
        b_res = resolution["browser"]
        f_res = resolution["fleet"]
        if isinstance(b_res, dict) and b_res["n_sampled"] >= t.min_resolution_sample:
            b_eff = float(b_res["median_eff_sims"])
            if 0.0 < b_eff < t.eff_sims_floor_drift:
                bump(2, f"policy ~{b_eff:.0f} sims effectifs < {t.eff_sims_floor_drift:.0f} (drift : recherche trop pauvre)")
            elif 0.0 < b_eff < t.eff_sims_floor_warn:
                bump(1, f"policy ~{b_eff:.0f} sims effectifs < {t.eff_sims_floor_warn:.0f} (warn)")
            f_eff = float(f_res["median_eff_sims"]) if isinstance(f_res, dict) else 0.0
            if b_eff > 0.0 and f_eff > 0.0 and b_eff < t.eff_sims_ratio_warn * f_eff:
                bump(1, f"sims effectifs browser {b_eff:.0f} < {t.eff_sims_ratio_warn:g}×fleet {f_eff:.0f} (warn)")

    return {
        **base,
        "verdict": _VERDICTS[level],
        "deltas": {
            "draw_frac": draw_delta,
            "outcome_mean": outcome_delta,
            "positions_per_game_ratio": ppg_ratio,
        },
        "reasons": reasons or ["dans les seuils"],
    }
