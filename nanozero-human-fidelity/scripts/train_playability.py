"""train_playability.py — modèle de jouabilité v1 (logreg Nano-only, format complexity).
Cible OFFLINE y = drop_cp>=tau. Sweep tau. GroupKFold(game_id). Sortie playability_v1_model.json."""
import csv, json, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import os as _os
BASE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "lichess_sample")
FEATS = ["decisiveness", "beta_mcts_800", "beta_wdl", "beta_var", "ent_policy", "eval_stm", "abs_eval",
         "g_nano", "ix_eval_betawdl", "ix_abseval_entpol", "ix_sign_betamcts", "ix_g_decis", "phase_mid", "phase_end"]

def fnum(x):
    try: return float(x)
    except Exception: return np.nan

rows = [r for r in csv.DictReader(open(f"{BASE}/play_train.csv"))]
X = np.array([[fnum(r[f]) for f in FEATS] for r in rows])
drop = np.array([fnum(r["drop_cp"]) for r in rows])
g = np.array([r["game_id"] for r in rows])
ok = ~np.isnan(X).any(axis=1) & ~np.isnan(drop)
X, drop, g = X[ok], drop[ok], g[ok]
print(f"{len(rows)} lignes → {ok.sum()} complètes | features={len(FEATS)}", file=sys.stderr)

def cv_auc(y):
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
    p = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=g, method="predict_proba")[:, 1]
    return roc_auc_score(y, p), p

print(f"{'tau':>5} {'positifs':>9} {'AUC':>7}")
best = None
for tau in (80, 100, 150):
    y = (drop >= tau).astype(int)
    a, p = cv_auc(y)
    print(f"{tau:>5} {100*y.mean():>8.1f}% {a:>7.4f}")
    if tau == 100: best = (tau, y, p, a)   # tau primaire

tau, y, p, auc = best
clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)).fit(X, y)
sc = clf.named_steps["standardscaler"]; lr = clf.named_steps["logisticregression"]
tense, sharp = np.quantile(p, [1/3, 2/3])
model = {"feats": FEATS, "mean": sc.mean_.tolist(), "std": sc.scale_.tolist(),
         "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0]),
         "tau": tau, "tense": float(tense), "sharp": float(sharp), "auc_cv": float(auc)}
json.dump(model, open(f"{BASE}/playability_v1_model.json", "w"), indent=1)
print(f"\nmodèle → playability_v1_model.json (tau={tau}, AUC_cv={auc:.4f}, seuils {tense:.3f}/{sharp:.3f})")
print("coef:", {f: round(c, 3) for f, c in zip(FEATS, model["coef"])})
