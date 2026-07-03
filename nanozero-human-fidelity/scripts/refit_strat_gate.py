"""refit_strat_gate.py — re-fit jouabilité sur le corpus STRATIFIÉ + gate curated automatique (piste A).

Hypothèse testée : le corpus stratifié (couvre matériel/phases) rend le coefficient matériel significatif
et fait passer le test de validité de construction (curated). Features 100% Nano-only 1-ply + matériel + phase
(toutes dérivables au runtime de l'arbre + échiquier).
"""
from __future__ import annotations

import csv, glob, json, math, sys

import chess
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from beta_engine import BetaEngine, board_with_history

import os as _os
BASE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "lichess_sample")
ONNX = _os.path.expanduser("~/nanozero-night/models/gen-031-promoted.onnx")
NANO1 = ["g_nano", "g2", "beta1", "betavar1", "ent_policy", "p_max"]
FEATS = NANO1 + ["material", "phase_mid", "phase_end"]
OUT = {"W": 1.0, "D": 0.5, "L": 0.0}
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def fnum(x):
    try:
        return float(x)
    except Exception:
        return np.nan


def mat_stm_of(board: chess.Board) -> int:
    s = 0
    for pt, v in VAL.items():
        s += v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return s if board.turn == chess.WHITE else -s


def phase_of(board: chess.Board) -> str:
    n = chess.popcount(board.occupied)
    return "opening" if n >= 26 else ("middlegame" if n >= 12 else "endgame")


def cp_of(g: float) -> float:
    v = max(-0.98, min(0.98, 2 * g - 1))
    return 111.71 * math.tan(1.5621 * v)


# ── 1. join corpus stratifié + features 1-ply ────────────────────────────────
feats1 = {}
for f in sorted(glob.glob(f"{BASE}/strat_1ply.shard*.csv")):
    for r in csv.DictReader(open(f)):
        if r["ok"] == "1":
            feats1[(r["game_id"], r["ply"])] = r

rows = []
for r in csv.DictReader(open(f"{BASE}/strat_105k.csv")):
    k = (r["game_id"], r["ply"])
    if k not in feats1:
        continue
    f1 = feats1[k]
    o = {"game_id": r["game_id"], "y": OUT.get(r["result_stm"], np.nan),
         "material": fnum(r["mat_stm"]), "cell": r["cell"],
         "phase_mid": 1.0 if r["phase"] == "middlegame" else 0.0,
         "phase_end": 1.0 if r["phase"] == "endgame" else 0.0}
    for f in ["g_nano", "beta1", "betavar1", "ent_policy", "p_max"]:
        o[f] = fnum(f1[f])
    o["g2"] = o["g_nano"] ** 2
    rows.append(o)
print(f"corpus stratifié joint : {len(rows)} positions", flush=True)

y = np.array([o["y"] for o in rows])
X = np.array([[o[f] for f in FEATS] for o in rows])
grp = np.array([o["game_id"] for o in rows])
ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
X, y, grp = X[ok], y[ok], grp[ok]

# ── 2. fits comparés : amplitude seule / +criticité(1-ply) / +matériel ───────
def fit_eval(cols, label):
    idx = [FEATS.index(c) for c in cols]
    clf = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    p = cross_val_predict(clf, X[:, idx], y, cv=GroupKFold(5), groups=grp)
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    clf.fit(X[:, idx], y)
    print(f"  {label:34s} R²(OOF)={r2:.4f}")
    return clf, r2

print("\n=== fits (cible = score humain réalisé W/D/L) ===")
fit_eval(["g_nano", "g2"], "amplitude seule")
fit_eval(["g_nano", "g2", "beta1", "betavar1", "ent_policy", "p_max"], "+ criticité 1-ply")
clf_full, r2_full = fit_eval(FEATS, "+ MATÉRIEL + phase (complet)")

sc = clf_full.named_steps["standardscaler"]; rg = clf_full.named_steps["ridge"]
coefs = dict(zip(FEATS, [round(c, 4) for c in rg.coef_]))
print(f"\ncoefs standardisés : {coefs}")
print(f"→ coef matériel = {coefs['material']} (corpus 15k : 0.028 ≈ nul)")

model = {"feats": FEATS, "mean": sc.mean_.tolist(), "std": sc.scale_.tolist(),
         "coef": rg.coef_.tolist(), "intercept": float(rg.intercept_), "r2_oof": float(r2_full)}
json.dump(model, open(f"{BASE}/jouab_model_v3_strat.json", "w"), indent=1)
print(f"modèle → jouab_model_v3_strat.json")

# ── 3. GATE CURATED (validité de construction) ───────────────────────────────
curated = json.load(open(f"{BASE}/curated_gate.json"))
eng = BetaEngine(ONNX, intra_op=2)

def feats_of(fen):
    b = chess.Board(fen)
    moves, P, vm = eng.root_expand(board_with_history(fen, None))
    P = np.asarray(P, dtype=float); P = P / (P.sum() or 1.0)
    vm = np.asarray(vm, dtype=float)
    vmax = float(vm.max()); mean = float((P * vm).sum())
    return {"g_nano": (vmax + 1) / 2, "g2": ((vmax + 1) / 2) ** 2,
            "beta1": float((P * np.maximum(0, vmax - vm)).sum()),
            "betavar1": float((P * (vm - mean) ** 2).sum()),
            "ent_policy": float(-(P[P > 0] * np.log2(P[P > 0])).sum()),
            "p_max": float(P.max()), "material": float(mat_stm_of(b)),
            "phase_mid": 1.0 if phase_of(b) == "middlegame" else 0.0,
            "phase_end": 1.0 if phase_of(b) == "endgame" else 0.0}

def predict(x):
    v = model["intercept"]
    for i, f in enumerate(model["feats"]):
        v += model["coef"][i] * ((x[f] - model["mean"][i]) / model["std"][i])
    return min(1.0, max(0.0, v))

print(f"\n=== GATE CURATED ===")
print(f"{'position':30s} {'attendu':7s} {'SF':>7s} {'ordi(Nano)':>11s} {'humain':>8s} {'contr':>6s} {'verdict':12s} ok?")
print("-" * 95)
n_ok = n_tot = 0
for r in curated:
    x = feats_of(r["fen"])
    gh = predict(x); contr = x["g_nano"] - gh
    lvl = "🔴 glissant" if contr > 0.12 else ("🟡 précision" if contr > 0.05 else "🟢 net")
    cat = r["cat"]
    good = ((cat == "FACILE" and lvl.startswith("🟢")) or (cat == "DUR" and not lvl.startswith("🟢"))
            or (cat == "EGALE" and abs(cp_of(gh)) < 60))
    n_tot += 1; n_ok += int(good)
    print(f"{r['name']:30s} {cat:7s} {r['sf_cp']:>+6d} {cp_of(x['g_nano']):>+9.0f}cp {cp_of(gh):>+7.0f}cp "
          f"{contr:>+6.2f} {lvl:12s} {'✅' if good else '❌'}")
print(f"\nGATE : {n_ok}/{n_tot} corrects (15k-corpus : 4/10). Hypothèse corpus {'CONFIRMÉE' if n_ok >= 8 else 'PARTIELLE' if n_ok >= 6 else 'RÉFUTÉE'}")
