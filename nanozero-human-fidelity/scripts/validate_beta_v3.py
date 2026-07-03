"""Gate β-v3 : teste les features MCTS Stockfish-free (mcts_features.mjs) contre les labels Stockfish
du dataset 15k. Réutilise la machinerie v2 : GroupKFold(game_id), placebo, AUC vs T_obj ET T_hum.
Ancré β_v1 (0.40/0.58) et trap_mass (0.75). Vérifie H1-H5 du pré-enregistrement v3.

Usage : python validate_beta_v3.py --feat /tmp/mcts_feat_all.csv [--ds .../betav2_hist.csv]
"""
import argparse, csv
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

MCTS_FEATS = ["beta_mcts_32", "beta_mcts_128", "beta_mcts_512", "beta_mcts_800", "beta_v1_js",
              "beta_swing_root", "beta_swing_child", "q_var", "pv_flip", "n_pv", "beta_kl",
              "beta_wdl", "decisiveness", "beta_var", "ent_visits", "ent_policy", "beta_nanompv"]


def fnum(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def load(ds_path, feat_path):
    ds = {(r["game_id"], r["ply"]): r for r in csv.DictReader(open(ds_path))}
    rows = []
    for r in csv.DictReader(open(feat_path)):
        if r.get("ok") not in ("1", "1.000000") or (r["game_id"], r["ply"]) not in ds:
            continue
        d = ds[(r["game_id"], r["ply"])]
        row = {"game_id": r["game_id"],
               "drop_cp": fnum(d.get("drop_cp")), "gap2": fnum(d.get("gap2")),
               "n_within_50": fnum(d.get("n_within_50")), "beta_v1": fnum(d.get("beta_v1")),
               "beta_strong": fnum(d.get("beta_strong")),
               "depth_swing": abs(fnum(d.get("cp_d18", np.nan)) - fnum(d.get("cp_d8", np.nan)))}
        for f in MCTS_FEATS:
            row[f] = fnum(r.get(f))
        rows.append(row)
    return rows


def auc(y, s):
    m = ~np.isnan(s)
    if m.sum() < 50 or len(np.unique(y[m])) < 2:
        return None
    return float(roc_auc_score(y[m], s[m]))


def cvauc(X, y, g, seed=0):
    ok = ~np.isnan(X).any(axis=1)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    p = cross_val_predict(clf, X[ok], y[ok], cv=GroupKFold(5), groups=g[ok], method="predict_proba")[:, 1]
    return auc(y[ok], p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", required=True)
    ap.add_argument("--ds", default="../data/lichess_sample/betav2_hist.csv")
    a = ap.parse_args()
    rows = load(a.ds, a.feat)
    n = len(rows)
    print(f"joint {n} positions (features MCTS ∩ dataset labellisé)\n")
    col = lambda k: np.array([r[k] for r in rows], dtype=float)
    groups = np.array([r["game_id"] for r in rows])
    drop = col("drop_cp"); gap2 = col("gap2"); n50 = col("n_within_50")

    # cibles
    y_obj = (n50 <= 2).astype(int)                      # T_obj : peu de bons coups = tranchant
    Y_HUM = {t: (drop > t).astype(int) for t in (100, 300, 500, 800)}
    print(f"T_obj (n_within_50≤2) : {100*y_obj.mean():.1f}%   T_hum drop>100 : {100*Y_HUM[100].mean():.1f}%  "
          f">500 : {100*Y_HUM[500].mean():.1f}%  >800 : {100*Y_HUM[800].mean():.1f}%\n")

    ANCHORS = {"β_v1 (dataset)": col("beta_v1"), "trap_mass (SF)": col("beta_strong"),
               "β_depth (SF)": col("depth_swing")}
    CANDS = {f: col(f) for f in MCTS_FEATS if not np.isnan(col(f)).all()}

    print(f"{'feature':22s} {'T_obj':>7} {'d>100':>7} {'d>300':>7} {'d>500':>7} {'d>800':>7}   (Stockfish-free sauf ancres)")
    print("-" * 78)
    def line(name, s, tag=""):
        a_obj = auc(y_obj, s)
        cells = [f"{a_obj:.3f}" if a_obj else "  -  "]
        for t in (100, 300, 500, 800):
            av = auc(Y_HUM[t], s); cells.append(f"{av:.3f}" if av else "  -  ")
        print(f"{name:22s} " + " ".join(f"{c:>7}" for c in cells) + f"   {tag}")
    for nm, s in ANCHORS.items():
        line(nm, s, "★ ancre")
    print("-" * 78)
    for nm, s in CANDS.items():
        line(nm, s)

    # ---- combos Stockfish-free (CV logreg) ----
    print("\n=== Combos Stockfish-free (logreg GroupKFold-5) ===")
    free = [f for f in ("beta_mcts_800", "beta_swing_root", "beta_swing_child", "q_var", "pv_flip",
                        "n_pv", "beta_kl", "beta_wdl", "decisiveness", "beta_var", "ent_visits") if f in CANDS]
    if "beta_nanompv" in CANDS and not np.isnan(CANDS["beta_nanompv"]).all():
        free.append("beta_nanompv")
    X = np.column_stack([CANDS[f] for f in free])
    for t in (100, 500, 800):
        print(f"  combo MCTS → T_hum drop>{t:<4} : AUC={cvauc(X, Y_HUM[t], groups)}")
    print(f"  combo MCTS → T_obj            : AUC={cvauc(X, y_obj, groups)}")
    print(f"  (features du combo : {', '.join(free)})")

    # ---- placebo ----
    rng = np.random.default_rng(0)
    s = CANDS.get("beta_mcts_800")
    m = ~np.isnan(s)
    plac = np.mean([auc(rng.permutation(Y_HUM[100][m]), s[m]) for _ in range(100)])
    print(f"\nplacebo (β_mcts_800, labels permutés) : {plac:.3f} (doit ≈0.5)")

    # ---- verdict pré-enregistré ----
    print("\n=== Verdict H1-H5 (pré-enregistrement v3) ===")
    bv1_hum = auc(Y_HUM[100], col("beta_v1")); bmcts_hum = auc(Y_HUM[100], CANDS["beta_mcts_800"])
    bv1_obj = auc(y_obj, col("beta_v1")); bmcts_obj = auc(y_obj, CANDS["beta_mcts_800"])
    print(f"  H1 (recherche répare value) : β_mcts T_obj={bmcts_obj:.3f} (vs β_v1 {bv1_obj:.3f}, cible ≥0.65) ; "
          f"T_hum={bmcts_hum:.3f} (vs {bv1_hum:.3f}, cible +0.03)")
    if "beta_swing_root" in CANDS:
        print(f"  H2 (volatilité→catastrophes): β_swing d>500={auc(Y_HUM[500], CANDS['beta_swing_root']):.3f} "
              f"d>800={auc(Y_HUM[800], CANDS['beta_swing_root']):.3f} (cible ≥0.65)")
    if "beta_nanompv" in CANDS and not np.isnan(CANDS["beta_nanompv"]).all():
        print(f"  H3 (nano-multipv≈trap_mass) : β_nanompv T_hum={auc(Y_HUM[100], CANDS['beta_nanompv']):.3f} "
              f"(trap_mass 0.75 ; cible ≥0.68)")
    else:
        print("  H3 : β_nanompv non calculé (phase 1 sans --multipv)")
    sweep = [auc(Y_HUM[100], CANDS[f]) for f in ("beta_mcts_32", "beta_mcts_128", "beta_mcts_512", "beta_mcts_800")]
    print(f"  H4 (plus de sims→mieux)     : AUC(T_hum) 32→800 = " + " → ".join(f"{x:.3f}" if x else "-" for x in sweep))


if __name__ == "__main__":
    main()
