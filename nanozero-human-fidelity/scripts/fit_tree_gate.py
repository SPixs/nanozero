"""fit_tree_gate.py — modèle de jouabilité FINAL (v2) : table empirique (phase×mat, 105k) + raffinement
ARBRE intra-case (12k), et gate curated re-passé avec la MÊME source d'éval (arbre 800 sims).

Features runtime (100% Nano + échiquier) : cell_base (table 105k), refv_800 (éval arbre), criticité
(beta_mcts/beta_wdl/beta_var/ent_policy/decisiveness), mat_stm, total_mat (trade-down Kaufman),
late_disc = |q800−q128| (découverte tardive Guid-Bratko).
Gate : plages ATTENDUES issues de la table empirique (plus de jugement de fauteuil).
Pré-requis : strat_tree3.shard*.csv (extraction) + /tmp/gate_tree_feats.json (gate_tree_feats.mjs).
"""
from __future__ import annotations

import csv, glob, json, math

import chess
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import os as _os
BASE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "lichess_sample")
OUT = {"W": 1.0, "D": 0.5, "L": 0.0}
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
CRIT = ["beta_mcts_800", "beta_wdl", "beta_var", "ent_policy", "decisiveness"]
FEATS = ["cell_base", "g_tree", "g2", "mat_stm", "total_mat", "late_disc"] + CRIT


def fnum(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def board_stats(fen):
    b = chess.Board(fen)
    mat = 0; tot = 0
    for pt, v in VAL.items():
        wn, bn = len(b.pieces(pt, chess.WHITE)), len(b.pieces(pt, chess.BLACK))
        mat += v * (wn - bn); tot += v * (wn + bn)
    mat = mat if b.turn == chess.WHITE else -mat
    n = chess.popcount(b.occupied)
    phase = "opening" if n >= 26 else ("middlegame" if n >= 12 else "endgame")
    bucket = "m3" if mat <= -3 else "m1" if mat <= -1 else "eq" if mat == 0 else "p1" if mat <= 2 else "p3"
    return mat, tot, phase, f"{phase}:{bucket}"


def cp_of(g):
    v = max(-0.98, min(0.98, 2 * g - 1))
    return 111.71 * math.tan(1.5621 * v)


# ── 1. table empirique (105k) ────────────────────────────────────────────────
cell_sum, cell_n = {}, {}
for r in csv.DictReader(open(f"{BASE}/strat_105k.csv")):
    c = r["cell"]
    cell_sum[c] = cell_sum.get(c, 0.0) + OUT[r["result_stm"]]
    cell_n[c] = cell_n.get(c, 0) + 1
cell_base = {c: cell_sum[c] / cell_n[c] for c in cell_sum}

# ── 2. join 12k : meta (strat_12k) + arbre (strat_tree2) ─────────────────────
meta = {(r["game_id"], r["ply"]): r for r in csv.DictReader(open(f"{BASE}/strat_12k.csv"))}
rows = []
for f in sorted(glob.glob(f"{BASE}/strat_tree3.shard*.csv")):
    for t in csv.DictReader(open(f)):
        k = (t["game_id"], t["ply"])
        if t.get("ok") not in ("1", "1.000000") or k not in meta:
            continue
        m = meta[k]
        mat, tot, phase, cell = board_stats(m["fen"])
        refv = fnum(t["refv_800"])
        g = (refv + 1) / 2
        rows.append({
            "game_id": m["game_id"], "y": OUT[m["result_stm"]], "cell": cell,
            "cell_base": cell_base.get(cell, 0.5), "g_tree": g, "g2": g * g,
            "mat_stm": float(mat), "total_mat": float(tot),
            "late_disc": abs(fnum(t["q_root_800"]) - fnum(t["q_root_128"])),
            **{c: fnum(t[c]) for c in CRIT},
        })
print(f"12k arbre joint : {len(rows)} positions")

y = np.array([r["y"] for r in rows])
X = np.array([[r[f] for f in FEATS] for r in rows])
grp = np.array([r["game_id"] for r in rows])
ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
X, y, grp = X[ok], y[ok], grp[ok]

# ── 3. fits empilés (contribution de chaque étage) ───────────────────────────
def fit_eval(cols, label):
    idx = [FEATS.index(c) for c in cols]
    clf = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    p = cross_val_predict(clf, X[:, idx], y, cv=GroupKFold(5), groups=grp)
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    clf.fit(X[:, idx], y)
    print(f"  {label:44s} R²(OOF)={r2:.4f}")
    return clf, r2

print("\n=== contribution des étages (cible = score humain réalisé) ===")
fit_eval(["cell_base"], "① table empirique seule")
fit_eval(["cell_base", "g_tree", "g2"], "② + éval ARBRE")
fit_eval(["cell_base", "g_tree", "g2", "mat_stm", "total_mat"], "③ + trade-down (Kaufman)")
fit_eval(["cell_base", "g_tree", "g2", "mat_stm", "total_mat", "late_disc"], "④ + découverte tardive (Guid-Bratko)")
clf, r2 = fit_eval(FEATS, "⑤ + criticité (COMPLET)")

sc_ = clf.named_steps["standardscaler"]; rg = clf.named_steps["ridge"]
print("\ncoefs standardisés :", {f: round(c, 4) for f, c in zip(FEATS, rg.coef_)})
model = {"feats": FEATS, "mean": sc_.mean_.tolist(), "std": sc_.scale_.tolist(),
         "coef": rg.coef_.tolist(), "intercept": float(rg.intercept_),
         "cell_base": cell_base, "r2_oof": float(r2)}
json.dump(model, open(f"{BASE}/jouab_model_v2_tree.json", "w"), indent=1)
print("modèle → jouab_model_v2_tree.json")

# ── 4. GATE CURATED — plages attendues EMPIRIQUES ────────────────────────────
# plage = ce que la table + le bon sens échiquéen ajusté aux données autorisent pour ĝ_H
RANGES = {
    "K+Q vs K (trivial)":         (0.85, 1.00),
    "K+R vs K (trivial)":         (0.80, 1.00),
    "R+P vs R Lucena":            (0.55, 0.85),
    "Fous opposes +1P":           (0.40, 0.68),
    "K+P opposition":             (0.40, 0.72),
    "Egale calme (Italienne)":    (0.42, 0.62),
    "Milieu, +un Cavalier calme": (0.72, 0.95),
    "Milieu, +une Tour calme":    (0.75, 0.98),
    "Milieu, +une Dame":          (0.42, 0.65),   # en réalité égale (SF +20)
    "Attaque tranchante gagnante": (0.55, 0.85),
}

def predict(x):
    v = model["intercept"]
    for i, f in enumerate(model["feats"]):
        v += model["coef"][i] * ((x[f] - model["mean"][i]) / model["std"][i])
    return min(1.0, max(0.0, v))

gate = json.load(open("/tmp/gate_tree_feats.json"))
print(f"\n=== GATE CURATED (éval ARBRE, plages empiriques) ===")
print(f"{'position':30s} {'SF':>7s} {'ordi':>8s} {'humain':>8s} {'plage attendue':>15s} ok?")
print("-" * 85)
n_ok = n_tot = 0
for r in gate:
    if not r.get("ok"):
        print(f"{r['name']:30s}  (features indispo : {r.get('err','?')})"); continue
    mat, tot, phase, cell = board_stats(r["fen"])
    x = {"cell_base": cell_base.get(cell, 0.5), "g_tree": (r["refv"] + 1) / 2,
         "g2": ((r["refv"] + 1) / 2) ** 2, "mat_stm": float(mat), "total_mat": float(tot),
         "late_disc": abs(r["q_root_800"] - r["q_root_128"]),
         **{c: r[c] for c in CRIT}}
    gh = predict(x)
    lo, hi = RANGES[r["name"]]
    good = lo <= gh <= hi
    n_tot += 1; n_ok += int(good)
    print(f"{r['name']:30s} {r['sf_cp']:>+6d} {cp_of(x['g_tree']):>+7.0f} {cp_of(gh):>+7.0f} "
          f"[{lo:.2f},{hi:.2f}] g={gh:.2f} {'✅' if good else '❌'}")
print(f"\nGATE : {n_ok}/{n_tot} dans la plage empirique. "
      f"{'PASS — câblage autorisé' if n_ok >= 8 else 'PARTIEL — itérer' if n_ok >= 6 else 'FAIL'}")
