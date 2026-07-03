"""Test de validité de construction : SF (vérité-terrain) vs Nano (éval) vs jouabilité prédite,
sur positions curated (finales vs milieux, facile/dur à convertir)."""
import json, math

import os as _os
BASE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "lichess_sample")
M = json.load(open(f"{BASE}/jouab_model.json"))
sfd = {r["name"]: r for r in json.load(open("/tmp/curated_sf.json"))}
feats = json.load(open("/tmp/jouab_feats.json"))

def predict(row):
    x = {"g_nano": row["g_nano"], "g2": row["g_nano"] ** 2,
         **{k: row[k] for k in ["decisiveness", "beta_mcts_800", "beta_wdl", "beta_var", "ent_policy"]}}
    v = M["intercept"]
    for i, f in enumerate(M["feats"]):
        v += M["coef"][i] * ((x[f] - M["mean"][i]) / M["std"][i])
    return min(1.0, max(0.0, v))

def cp(g):
    v = max(-0.98, min(0.98, 2 * g - 1)); return 111.71 * math.tan(1.5621 * v)

print(f"{'position':30s} {'phase':7s} {'attendu':7s} {'SF(vrai)':>9s} {'Nano':>8s} {'compress':>9s} {'humain':>8s} {'verdict':11s} {'ok?':3s}")
print("-" * 100)
for r in feats:
    sf = sfd.get(r["name"], {}); phase = sf.get("phase", "?"); cat = r["cat"]; sfcp = sf.get("sf_cp")
    if not r.get("ok"):
        print(f"{r['name']:30s} {phase:7s} {cat:7s}  (éval Nano indispo)"); continue
    gn = cp(r["g_nano"]); gh = predict(r); ghcp = cp(gh); contr = r["g_nano"] - gh
    lvl = "🔴 glissant" if contr > 0.12 else "🟡 précision" if contr > 0.05 else "🟢 net"
    # correct ? FACILE→net attendu ; DUR→glissant/précision ; EGALE→net & éval ~0
    good = ("✅" if (cat == "FACILE" and lvl.startswith("🟢")) or
            (cat == "DUR" and not lvl.startswith("🟢")) or
            (cat == "EGALE" and abs(ghcp) < 60) else "❌")
    comp = (gn - sfcp) if sfcp is not None else None
    print(f"{r['name']:30s} {phase:7s} {cat:7s} {sfcp:>+8d}cp {gn:>+7.0f}cp {comp:>+8.0f}cp {ghcp:>+7.0f}cp {lvl:11s} {good:3s}")
print("\nSF(vrai)=Stockfish (vérité) · Nano=éval Nano · compress=Nano−SF (négatif = Nano SOUS-évalue) · humain=jouabilité prédite")
