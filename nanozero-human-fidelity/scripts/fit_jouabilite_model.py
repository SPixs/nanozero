"""Fige le modèle de jouabilité combiné (amplitude WDL + criticité) → jouab_model.json.
Cible = score humain réalisé y (W/D/L). Features Nano : g_nano, g_nano^2 (amplitude) + 5 features criticité.
Prédit ĝ_H = éval-humaine réalisée (bornée). Sert au test de validité de construction (positions connues)."""
import csv, json, numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict

import os as _os
BASE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "lichess_sample")
CRIT = ["decisiveness", "beta_mcts_800", "beta_wdl", "beta_var", "ent_policy"]
FEATS = ["g_nano", "g2"] + CRIT
OUT = {"W": 1.0, "D": 0.5, "L": 0.0}
def fnum(x):
    try: return float(x)
    except Exception: return np.nan

rows = [r for r in csv.DictReader(open(f"{BASE}/play_train.csv"))]
for r in rows: r["g2"] = str(fnum(r["g_nano"]) ** 2)
y = np.array([OUT.get(r["result_stm"], np.nan) for r in rows])
X = np.array([[fnum(r[f]) for f in FEATS] for r in rows])
g = np.array([r["game_id"] for r in rows])
ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
X, y, g = X[ok], y[ok], g[ok]

clf = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
sc = clf.named_steps["standardscaler"]; rg = clf.named_steps["ridge"]
# amplitude seule (g, g2) pour comparaison
amp = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[:, :2], y)
p_full = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=g)
p_amp = cross_val_predict(amp, X[:, :2], y, cv=GroupKFold(5), groups=g)
r2 = lambda p: 1 - np.sum((y-p)**2)/np.sum((y-y.mean())**2)
model = {"feats": FEATS, "mean": sc.mean_.tolist(), "std": sc.scale_.tolist(),
         "coef": rg.coef_.tolist(), "intercept": float(rg.intercept_)}
json.dump(model, open(f"{BASE}/jouab_model.json", "w"), indent=1)
print(f"{ok.sum()} positions | R²(amplitude)={r2(p_amp):.4f}  R²(amplitude+criticité)={r2(p_full):.4f}")
print(f"coef (standardisés): {dict(zip(FEATS, [round(c,3) for c in model['coef']]))}")
print(f"modèle → jouab_model.json")
