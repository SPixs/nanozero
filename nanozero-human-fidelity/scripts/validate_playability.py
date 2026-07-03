"""validate_playability.py — gates go/no-go du modèle de jouabilité v1.
Cible y=drop_cp>=100. GroupKFold(game_id). Gates : AUC, incrément vs criticité (anti-doublon),
ECE, Brier-skill, placebo, + TEST DE MORT partialling Stockfish (coefs Nano survivent-ils au contrôle SF ?)."""
import csv, sys
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import os as _os
BASE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "lichess_sample")
FULL = ["decisiveness", "beta_mcts_800", "beta_wdl", "beta_var", "ent_policy", "eval_stm", "abs_eval",
        "g_nano", "ix_eval_betawdl", "ix_abseval_entpol", "ix_sign_betamcts", "ix_g_decis", "phase_mid", "phase_end"]
CRIT = ["decisiveness", "beta_mcts_800", "beta_wdl", "beta_var", "ent_policy"]   # ~ criticité seule
SF = ["best_cp", "n_within_50", "gap2"]                                          # contrôles Stockfish (OFFLINE)
TAU = 100

def fnum(x):
    try: return float(x)
    except Exception: return np.nan

rows = [r for r in csv.DictReader(open(f"{BASE}/play_train.csv"))]
def col(f): return np.array([fnum(r.get(f)) for r in rows])
drop = col("drop_cp"); g = np.array([r["game_id"] for r in rows])
Xall = {f: col(f) for f in set(FULL + SF)}
ok = ~np.isnan(drop)
for f in FULL + SF: ok &= ~np.isnan(Xall[f])
y = (drop[ok] >= TAU).astype(int); g = g[ok]
mat = lambda feats: np.column_stack([Xall[f][ok] for f in feats])
print(f"{ok.sum()} lignes | y=drop_cp>={TAU} : {100*y.mean():.1f}% positifs\n")

def cvp(feats, yy=None):
    yy = y if yy is None else yy
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
    return cross_val_predict(clf, mat(feats), yy, cv=GroupKFold(5), groups=g, method="predict_proba")[:, 1]

def ece(yv, p, bins=10):
    ed = np.linspace(0, 1, bins + 1); e = 0.0; n = len(yv)
    for i in range(bins):
        m = (p >= ed[i]) & (p < ed[i+1]) if i < bins-1 else (p >= ed[i]) & (p <= ed[i+1])
        if m.sum(): e += m.sum()/n * abs(p[m].mean() - yv[m].mean())
    return e

p_full = cvp(FULL); auc_full = roc_auc_score(y, p_full)
p_crit = cvp(CRIT); auc_crit = roc_auc_score(y, p_crit)
p_eval = Xall["eval_stm"][ok]; auc_eval = roc_auc_score(y, np.abs(p_eval))
p_sf = cvp(SF); auc_sf = roc_auc_score(y, p_sf)
p_full_sf = cvp(FULL + SF); auc_full_sf = roc_auc_score(y, p_full_sf)
rng = np.random.default_rng(0)
plac = np.mean([roc_auc_score(yp := rng.permutation(y), cvp(FULL, yp)) for _ in range(5)])
e = ece(y, p_full)
brier = brier_score_loss(y, p_full); brier_base = brier_score_loss(y, np.full_like(p_full, y.mean()))
bskill = 1 - brier / brier_base

def g_(x, ok_): return "PASS" if ok_ else "FAIL"
print("=== GATES ===")
print(f"  AUC modèle complet          {auc_full:.4f}   {g_(0, auc_full>=0.70)} (>=0.70)")
print(f"  AUC criticité seule         {auc_crit:.4f}")
print(f"  → INCRÉMENT vs criticité    {auc_full-auc_crit:+.4f}   {g_(0, auc_full-auc_crit>=0.03)} (>=+0.03 = signal NEUF, anti-doublon)")
print(f"  AUC |eval_stm| seul         {auc_eval:.4f}")
print(f"  ECE                         {e:.4f}   {g_(0, e<=0.05)} (<=0.05)")
print(f"  Brier-skill vs taux base    {bskill:.4f}   {g_(0, bskill>=0.15)} (>=0.15)")
print(f"  Placebo (y permuté)         {plac:.4f}   {g_(0, plac<=0.52)} (~0.5)")
print("\n=== TEST DE MORT — partialling Stockfish (OFFLINE) ===")
print(f"  AUC Stockfish-controls seul {auc_sf:.4f}")
print(f"  AUC Nano+SF                 {auc_full_sf:.4f}")
print(f"  → Nano AJOUTE sur Stockfish {auc_full_sf-auc_sf:+.4f}   {g_(0, auc_full_sf-auc_sf>=0.01)} (>0 = capte la difficulté au-delà de la netteté SF)")
# rétention des coefs Nano quand on ajoute les contrôles SF (features standardisées)
m1 = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)).fit(mat(FULL), y)
m2 = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000)).fit(mat(FULL + SF), y)
c1 = m1.named_steps["logisticregression"].coef_[0]
c2 = m2.named_steps["logisticregression"].coef_[0][:len(FULL)]
print(f"  {'feature':20s} {'coef M1':>9} {'coef M2':>9} {'rétention':>10} {'signe':>6}")
key = ["eval_stm", "ix_sign_betamcts", "ix_eval_betawdl", "beta_wdl", "decisiveness"]
for f in key:
    i = FULL.index(f); ret = abs(c2[i]) / (abs(c1[i]) + 1e-9); sg = "ok" if np.sign(c1[i]) == np.sign(c2[i]) else "FLIP"
    print(f"  {f:20s} {c1[i]:>9.3f} {c2[i]:>9.3f} {ret:>9.0%} {sg:>6}")
print("\n(PASS test de mort ≈ Nano ajoute sur SF ET coefs directionnels gardent signe + >~60% magnitude)")
