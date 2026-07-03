"""validate_beta.py — LE gate : β prédit-il les gaffes humaines ? (DESIGN §6)

Entrée : CSV de compute_beta.py (colonnes beta_*, top12_gap, blunder).
Pour chaque variante de β : AUC(β → gaffe), comparaison à la baseline top1−top2,
test placebo, calibration par décile. Applique les gates chiffrés et rend un verdict.

Note d'honnêteté : β est une fonction FIXE du réseau (aucun paramètre ajusté sur
les labels) → son AUC in-sample n'est pas sur-ajusté. Le hold-out par
partie+joueur+mois compte surtout pour la calibration WDL (P1), pas pour
discriminer β. On rapporte quand même l'AUC groupée-partie (split game_id) comme
garde-fou anti-fuite intra-partie.

Sortie : rapport JSON + verdict console. Gates : AUC≥0.70, skill vs baseline≥+0.03,
placebo≤0.02.

Run : ~/.gatevenv/bin/python validate_beta.py IN.csv [--out report.json --seed 7]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

GATE_AUC = 0.70
GATE_SKILL = 0.03  # β doit battre la baseline top1−top2 de +0.03 AUC
GATE_PLACEBO = 0.02


def _auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    return float(roc_auc_score(y, score))


def _grouped_auc(y: np.ndarray, score: np.ndarray, groups: np.ndarray, seed: int) -> float | None:
    """AUC sur un split 70/30 par groupe (game_id) — garde-fou anti-fuite intra-partie.
    β étant sans paramètre, on n'entraîne rien : on mesure juste l'AUC sur le sous-ensemble test."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    test_groups = set(uniq[: max(1, int(0.3 * len(uniq)))])
    mask = np.array([g in test_groups for g in groups])
    if mask.sum() < 10:
        return None
    return _auc(y[mask], score[mask])


def _calibration_deciles(y: np.ndarray, beta: np.ndarray, nbins: int = 10) -> list[dict]:
    """Taux de gaffe réel par décile de β (interprétabilité + détecteur sur-confiance)."""
    order = np.argsort(beta)
    rows = []
    for b in np.array_split(order, nbins):
        if len(b) == 0:
            continue
        rows.append(
            {
                "n": int(len(b)),
                "beta_mean": round(float(beta[b].mean()), 4),
                "blunder_rate": round(float(y[b].mean()), 4),
            }
        )
    return rows


def _ece_brier(y: np.ndarray, beta: np.ndarray) -> tuple[float, float]:
    """ECE + Brier-skill via calibration isotonique β→P(gaffe) (in-sample, indicatif)."""
    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    iso = IsotonicRegression(out_of_bounds="clip")
    p = iso.fit_transform(beta, y)
    # ECE (10 bins équi-effectifs).
    order = np.argsort(p)
    ece = 0.0
    for b in np.array_split(order, 10):
        if len(b) == 0:
            continue
        ece += (len(b) / len(y)) * abs(p[b].mean() - y[b].mean())
    base = y.mean()
    brier = float(np.mean((p - y) ** 2))
    brier_base = float(np.mean((base - y) ** 2))
    brier_skill = 1.0 - brier / brier_base if brier_base > 0 else 0.0
    return float(ece), float(brier_skill)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("--out", default="")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--placebo-runs", type=int, default=200, help="nb de shuffles (moyenne ≈0.5 si pas de fuite)")
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("CSV vide", file=sys.stderr)
        sys.exit(1)

    y = np.array([int(r["blunder"]) for r in rows])
    groups = np.array([r.get("game_id", str(i)) for i, r in enumerate(rows)])
    beta_cols = [c for c in rows[0] if c.startswith("beta_")]
    base = -np.array([float(r["top12_gap"]) for r in rows])  # grand gap = facile → −gap prédit la gaffe
    rng = np.random.default_rng(args.seed)

    n, npos = len(y), int(y.sum())
    print(f"\n=== GATE β — {n} positions, {npos} gaffes ({100*npos/n:.1f}%) ===")
    auc_base = _auc(y, base)
    print(f"baseline top1−top2 : AUC={auc_base if auc_base is None else round(auc_base,3)}\n")
    if auc_base is None:
        print("⚠️  une seule classe (0 ou 100% gaffes) → AUC indéfini. Besoin de données réelles équilibrées.")

    report: dict = {"n": n, "n_blunder": npos, "baseline_auc": auc_base, "variants": {}}
    best = None
    for c in beta_cols:
        beta = np.array([float(r[c]) for r in rows])
        auc = _auc(y, beta)
        gauc = _grouped_auc(y, beta, groups, args.seed)
        # Placebo : labels mélangés, MOYENNÉ sur N shuffles → doit ≈ 0.5 (β sans paramètre).
        # Un seul shuffle = trop bruité (±1.5 SE) ; la moyenne révèle une fuite systématique du harnais.
        placebo_aucs = []
        for _ in range(args.placebo_runs):
            yp = y.copy()
            rng.shuffle(yp)
            a = _auc(yp, beta)
            if a is not None:
                placebo_aucs.append(a)
        placebo = float(np.mean(placebo_aucs)) if placebo_aucs else None
        ece, bskill = _ece_brier(y, beta)
        skill = (auc - auc_base) if (auc is not None and auc_base is not None) else None
        placebo_dev = abs(placebo - 0.5) if placebo is not None else None
        passed = (
            auc is not None
            and auc >= GATE_AUC
            and skill is not None
            and skill >= GATE_SKILL
            and placebo_dev is not None
            and placebo_dev <= GATE_PLACEBO
        )
        report["variants"][c] = {
            "auc": auc,
            "grouped_auc": gauc,
            "skill_vs_baseline": skill,
            "placebo_auc_mean": None if placebo is None else round(placebo, 4),
            "placebo_dev": None if placebo_dev is None else round(placebo_dev, 4),
            "ece": None if ece != ece else round(ece, 4),  # noqa: PLR0124 (NaN check)
            "brier_skill": None if bskill != bskill else round(bskill, 4),
            "passed": bool(passed),
        }
        aucs = "N/A" if auc is None else f"{auc:.3f}"
        sk = "N/A" if skill is None else f"{skill:+.3f}"
        gs = "N/A" if gauc is None else f"{gauc:.3f}"
        pl = "N/A" if placebo_dev is None else f"{placebo_dev:.3f}"
        print(f"{c:20s} AUC={aucs} (grpd {gs})  skill={sk}  placebo±{pl}  {'✅' if passed else '·'}")
        if auc is not None and (best is None or auc > best[1]):
            best = (c, auc)
        if c == beta_cols[0] or (auc is not None and npos > 0):
            report["variants"][c]["deciles"] = _calibration_deciles(y, beta)

    if best:
        print(f"\nMeilleure variante : {best[0]} (AUC {best[1]:.3f})")
        verdict = "PASSED ✅" if report["variants"][best[0]]["passed"] else "NON PASSÉ (gate non atteint)"
        print(f"Verdict gate : {verdict}")
        report["best_variant"] = best[0]
        report["gate_passed"] = report["variants"][best[0]]["passed"]

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nrapport → {args.out}")


if __name__ == "__main__":
    main()
