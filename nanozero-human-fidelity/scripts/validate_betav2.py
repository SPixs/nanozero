"""validate_betav2.py — gate à DOUBLE cible (cf. docs/EXP-beta-v2-preregistration.md).

T_obj (validité de MESURE) : β mesure-t-il la netteté objective Stockfish ? → testé UNIQUEMENT
  avec β_v1 (Nano, indépendant de SF). gap2/n_within_50 = SF → β_strong/β_depth les prédiraient
  de façon ~circulaire (tous deux SF-dérivés) : on ne les y teste PAS.
T_hum (prédiction COMPORTEMENT) : gaffe humaine → testé avec toutes les variantes + baselines.

Sortie : matrice AUC + Spearman + placebo + verdict H1-H4.
Run : ~/.gatevenv/bin/python validate_betav2.py IN.csv [--gap 100 --seed 7]
"""

from __future__ import annotations

import argparse
import csv

import chess
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def board_feats(fen: str) -> dict[str, float]:
    b = chess.Board(fen)
    legal = list(b.legal_moves)
    mat = sum(_VAL[p] * (len(b.pieces(p, chess.WHITE)) + len(b.pieces(p, chess.BLACK))) for p in _VAL)
    return {
        "n_legal": len(legal),
        "n_cap": sum(1 for m in legal if b.is_capture(m)),
        "n_chk": sum(1 for m in legal if b.gives_check(m)),
        "queens": len(b.pieces(chess.QUEEN, chess.WHITE)) + len(b.pieces(chess.QUEEN, chess.BLACK)),
        "material": mat,
        "in_check": int(b.is_check()),
    }


def _auc(y, s):
    return float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None


def _cv_auc(X, y, groups, seed):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    p = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]
    return _auc(y, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("--gap", type=int, default=100, help="seuil gap2 pour 'critique' (T_obj)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.inp, encoding="utf-8")))
    n = len(rows)
    f = lambda k: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows])  # noqa: E731
    y_hum = np.array([int(r["blunder"]) for r in rows])
    groups = np.array([r["game_id"] for r in rows])
    beta_v1, beta_strong = f("beta_v1"), f("beta_strong")
    depth_swing, gap2, n50 = f("depth_swing"), f("gap2"), f("n_within_50")
    best_cp = f("best_cp")
    bf = [board_feats(r["fen"]) for r in rows]
    feat_names = ["n_legal", "n_cap", "n_chk", "queens", "material", "in_check"]
    Xfree = np.array([[d[k] for k in feat_names] for d in bf])

    print(f"\n=== β-v2 — {n} positions, {y_hum.sum()} gaffes ({100*y_hum.mean():.1f}%) ===")

    # ---- T_obj : validité de mesure (β_v1 SEUL, indépendant de SF) ----
    y_obj = (gap2 >= args.gap).astype(int)
    print(f"\n[T_obj] criticité objective (gap2≥{args.gap}cp) : {100*y_obj.mean():.1f}% critiques")
    print("  β_v1 (Nano) prédit la netteté Stockfish — test de VALIDITÉ DE MESURE (propre, non circulaire) :")
    print(f"    AUC(β_v1 → critique)          = {_auc(y_obj, beta_v1)}")
    print(f"    Spearman(β_v1, gap2)          = {spearmanr(beta_v1, gap2, nan_policy='omit').correlation:+.3f}")
    print(f"    Spearman(β_v1, −n_within_50)  = {spearmanr(beta_v1, -n50, nan_policy='omit').correlation:+.3f}")

    # ---- T_hum : prédiction de la gaffe humaine ----
    print("\n[T_hum] prédiction de la gaffe humaine (AUC) :")
    fixed = {
        "β_v1 (Nano)": beta_v1,
        "β_strong (trap_mass SF×policy)": beta_strong,
        "β_depth (volatilité prof.)": depth_swing,
        "gap2 (SF)": gap2,
        "−n_within_50 (only-move)": -n50,
        "best_cp signé (perdant=+gaffe)": -best_cp,
    }
    results = {}
    for name, s in fixed.items():
        mask = ~np.isnan(s)
        a = _auc(y_hum[mask], s[mask])
        results[name] = a
        print(f"    {name:34s} AUC={'N/A' if a is None else f'{a:.3f}'}")
    # modèles ajustés (CV groupée par partie)
    auc_free = _cv_auc(Xfree, y_hum, groups, args.seed)
    Xcombo = np.column_stack([Xfree, np.nan_to_num(beta_strong), np.nan_to_num(depth_swing), np.nan_to_num(best_cp)])
    auc_combo = _cv_auc(Xcombo, y_hum, groups, args.seed)
    print(f"    {'features libres (logreg CV)':34s} AUC={auc_free:.3f}  ← baseline à battre")
    print(f"    {'combo +β_strong+depth+best (CV)':34s} AUC={auc_combo:.3f}")

    # placebo sur le meilleur fixe
    best_name = max((k for k in results if results[k] is not None), key=lambda k: results[k])
    s = fixed[best_name]; mask = ~np.isnan(s)
    rng = np.random.default_rng(args.seed)
    plac = np.mean([_auc(rng.permutation(y_hum[mask]), s[mask]) for _ in range(200)])
    print(f"\n  placebo (meilleur fixe={best_name}) : AUC moyenne labels permutés = {plac:.3f} (doit ≈0.5)")

    # ---- Verdict vs hypothèses pré-enregistrées ----
    print("\n=== VERDICT vs hypothèses pré-enregistrées ===")
    a_obj = _auc(y_obj, beta_v1)
    h1 = "(β_strong se juge sur T_hum, pas T_obj — voir ci-dessous)"
    h3 = (a_obj is not None and a_obj >= 0.70 and (results["β_v1 (Nano)"] or 0) < 0.55)
    print(f"  H1 (value=maillon faible) : AUC(β_strong→T_hum)={results['β_strong (trap_mass SF×policy)']} vs β_v1={results['β_v1 (Nano)']}")
    print(f"  H2 (profondeur révèle piège) : AUC(β_depth→T_hum)={results['β_depth (volatilité prof.)']} (vs β_v1, +0.03 ?)")
    print(f"  H3 (β=netteté PAS erreur) : AUC(β_v1→T_obj)={a_obj} ≥0.70 ET β_v1→T_hum≈0.5 → {'CONFIRMÉ ✅' if h3 else 'NON'}")
    print(f"  H4 (plafond comportement) : meilleur AUC→T_hum = {max(v for v in list(results.values())+[auc_free,auc_combo] if v is not None):.3f} (<0.70 ?)")


if __name__ == "__main__":
    main()
