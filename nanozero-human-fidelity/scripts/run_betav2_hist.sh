#!/usr/bin/env bash
# Cycle β-v2 AVEC historique : joint la colonne `line` dans le CSV multipv, calcule β-v2
# (β_v1, β_strong=trap_mass, + β_depth=depth_swing déjà présent), valide en double cible,
# puis LE TEST DÉCISIF : β_depth attrape-t-il les catastrophes (drop>500/800) où β_v1 chute à 0.38 ?
# À lancer quand lichess_2024-12_15k.mpv.csv est prêt.
set -euo pipefail
DIR="$(dirname "$0")/../data/lichess_sample"
SC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; PY="$HOME/.gatevenv/bin/python"

[ -f "$DIR/lichess_2024-12_15k.mpv.csv" ] || { echo "multipv pas prêt"; exit 1; }

echo "[1/4] jointure de l'historique (line) dans le CSV multipv"
"$PY" - <<'PYEOF'
import csv
D=""$(dirname "$0")/../data/lichess_sample""
line={r["fen"]:r["line"] for r in csv.DictReader(open(f"{D}/lichess_2024-12_15k_hist.csv"))}
rows=list(csv.DictReader(open(f"{D}/lichess_2024-12_15k.mpv.csv")))
fields=list(rows[0].keys())+["line"]
with open(f"{D}/mpv_hist.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    miss=0
    for r in rows:
        r["line"]=line.get(r["fen"],""); miss+=(r["fen"] not in line); w.writerow(r)
print(f"  joint {len(rows)} lignes, {miss} sans historique")
PYEOF

echo "[2/4] compute_beta_v2 AVEC historique (β_v1 + β_strong=trap_mass)"
"$PY" "$SC/compute_beta_v2.py" "$DIR/mpv_hist.csv" "$DIR/betav2_hist.csv"

echo "[3/4] validation double cible (T_obj criticité / T_hum gaffe)"
"$PY" "$SC/validate_betav2.py" "$DIR/betav2_hist.csv" --out "$DIR/betav2_hist.report.json"

echo "[4/4] TEST DÉCISIF — β_depth & β_strong vs seuil de gaffe (les catastrophes)"
"$PY" - <<'PYEOF'
import csv, numpy as np
from sklearn.metrics import roc_auc_score
D=""$(dirname "$0")/../data/lichess_sample""
rows=list(csv.DictReader(open(f"{D}/betav2_hist.csv")))
drop=np.array([float(r["drop_cp"]) for r in rows])
preds={"β_v1(hist)":"beta_v1","β_strong(trap_mass)":"beta_strong","β_depth(volatilité)":"depth_swing"}
arr={k:np.array([float(r[c]) for r in rows]) for k,c in preds.items()}
def auc(y,s):
    return roc_auc_score(y,s) if len(np.unique(y))>1 else float("nan")
print(f"   {'seuil':>9} {'%':>5} "+" ".join(f"{k:>20}" for k in preds))
for t in (100,300,500,800):
    y=(drop>t).astype(int)
    if y.sum()<20: continue
    print(f"   drop>{t:>4} {100*y.mean():4.1f}% "+" ".join(f"{auc(y,arr[k]):>20.3f}" for k in preds))
print("   (rappel β_v1 SANS hist : 100→0.49, 800→0.40 ; cible = β_depth >0.5 sur drop>800)")
PYEOF
echo "DONE"
