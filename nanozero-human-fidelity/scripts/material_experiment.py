"""Hypothèse : ajouter la BALANCE MATÉRIELLE (échiquier, runtime-OK) corrige les cas matériel-en-plus.
Re-fitte le modèle avec 'material' et repasse le test de validité de construction."""
import csv, json, math, chess, numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupKFold, cross_val_predict

import os as _os
BASE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "lichess_sample")
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
def material_stm(fen):
    b = chess.Board(fen); s = 0
    for pt, v in VAL.items():
        s += v * (len(b.pieces(pt, chess.WHITE)) - len(b.pieces(pt, chess.BLACK)))
    return s if b.turn == chess.WHITE else -s   # POV camp au trait

def fnum(x):
    try: return float(x)
    except Exception: return np.nan

# fen par (game_id,ply) depuis betav2_hist
fen_of = {(r["game_id"], r["ply"]): r["fen"] for r in csv.DictReader(open(f"{BASE}/betav2_hist.csv"))}
rows = [r for r in csv.DictReader(open(f"{BASE}/play_train.csv"))]
for r in rows:
    r["g2"] = fnum(r["g_nano"]) ** 2
    f = fen_of.get((r["game_id"], r["ply"]))
    r["material"] = material_stm(f) if f else np.nan
OUT = {"W": 1.0, "D": 0.5, "L": 0.0}
CRIT = ["decisiveness", "beta_mcts_800", "beta_wdl", "beta_var", "ent_policy"]

def fit(FEATS, label):
    y = np.array([OUT.get(r["result_stm"], np.nan) for r in rows])
    X = np.array([[fnum(r[f]) if not isinstance(r[f], float) else r[f] for f in FEATS] for r in rows])
    g = np.array([r["game_id"] for r in rows])
    ok = ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    X, y, g = X[ok], y[ok], g[ok]
    clf = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X, y)
    p = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=g)
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    sc = clf.named_steps["standardscaler"]; rg = clf.named_steps["ridge"]
    m = {"feats": FEATS, "mean": sc.mean_.tolist(), "std": sc.scale_.tolist(), "coef": rg.coef_.tolist(), "intercept": float(rg.intercept_)}
    print(f"  {label:28s} R²={r2:.4f}  coef material={dict(zip(FEATS,[round(c,3) for c in m['coef']])).get('material','—')}")
    return m

print("=== re-fit avec vs sans matériel ===")
m_base = fit(["g_nano", "g2"] + CRIT, "amplitude + criticité")
m_mat = fit(["g_nano", "g2", "material"] + CRIT, "+ MATÉRIEL")
json.dump(m_mat, open(f"{BASE}/jouab_model_v2.json", "w"), indent=1)

# --- repasse le test curated avec le modèle v2 (matériel) ---
sfd = {r["name"]: r for r in json.load(open("/tmp/curated_sf.json"))}
feats = json.load(open("/tmp/jouab_feats.json"))
for r in feats:
    if r.get("ok"): r["material"] = material_stm(r["fen"]); r["g2"] = r["g_nano"] ** 2
def predict(r, M):
    x = {**{k: r.get(k) for k in M["feats"]}}
    v = M["intercept"]
    for i, f in enumerate(M["feats"]): v += M["coef"][i] * ((x[f] - M["mean"][i]) / M["std"][i])
    return min(1.0, max(0.0, v))
def cp(g):
    v = max(-0.98, min(0.98, 2 * g - 1)); return 111.71 * math.tan(1.5621 * v)
print(f"\n{'position':30s} {'attendu':7s} {'SF':>7s} {'mat':>4s} {'humain(v1)':>11s} {'humain(+mat)':>13s} {'verdict(+mat)':13s} ok?")
print("-" * 100)
for r in feats:
    if not r.get("ok"): continue
    sf = sfd.get(r["name"], {}); cat = r["cat"]
    gh1 = predict(r, m_base); gh2 = predict(r, m_mat); contr = r["g_nano"] - gh2
    lvl = "🔴 glissant" if contr > 0.12 else "🟡 précision" if contr > 0.05 else "🟢 net"
    good = "✅" if ((cat == "FACILE" and lvl.startswith("🟢")) or (cat == "DUR" and not lvl.startswith("🟢")) or (cat == "EGALE" and abs(cp(gh2)) < 60)) else "❌"
    print(f"{r['name']:30s} {cat:7s} {sf.get('sf_cp'):>+6d} {r['material']:>+4d} {cp(gh1):>+9.0f}cp {cp(gh2):>+11.0f}cp {lvl:13s} {good}")
