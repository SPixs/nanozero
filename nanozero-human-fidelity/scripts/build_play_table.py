"""build_play_table.py — table d'entraînement du modèle de JOUABILITÉ (niveau 2).
Join labels Stockfish (betav2_hist, OFFLINE) ∩ features Nano (mcts_features_v3) + eval_stm signé (root_expand).
Features runtime = 100% Nano. Stockfish = labels/validation seulement. Sortie : play_train.csv."""
import csv, sys
import numpy as np
from beta_engine import BetaEngine, board_with_history

import os as _os
BASE = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "lichess_sample")
ONNX = _os.path.expanduser("~/nanozero-night/models/gen-031-promoted.onnx")
OUT = f"{BASE}/play_train.csv"
NANO = ["decisiveness", "beta_mcts_800", "beta_wdl", "beta_var", "ent_policy"]

def fnum(x):
    try: return float(x)
    except Exception: return np.nan

# labels Stockfish + fen/line/phase (OFFLINE)
ds = {(r["game_id"], r["ply"]): r for r in csv.DictReader(open(f"{BASE}/betav2_hist.csv"))}
# features Nano
rows = []
for r in csv.DictReader(open(f"{BASE}/mcts_features_v3.csv")):
    if r.get("ok") not in ("1", "1.000000"): continue
    k = (r["game_id"], r["ply"])
    if k not in ds: continue
    d = ds[k]
    o = {"game_id": r["game_id"], "ply": r["ply"], "fen": d["fen"], "line": d.get("line", ""),
         "drop_cp": fnum(d.get("drop_cp")), "result_stm": d.get("result_stm"),
         "best_cp": fnum(d.get("best_cp")), "n_within_50": fnum(d.get("n_within_50")),
         "gap2": fnum(d.get("gap2")), "mover_elo": fnum(d.get("mover_elo")), "phase": d.get("phase", "")}
    for f in NANO: o[f] = fnum(r.get(f))
    rows.append(o)
print(f"{len(rows)} positions jointes. Calcul eval_stm (Nano, root_expand)…", file=sys.stderr)

eng = BetaEngine(ONNX, intra_op=1)
for i, o in enumerate(rows):
    _m, _P, vm = eng.root_expand(board_with_history(o["fen"], o["line"] or None))
    o["eval_stm"] = float(np.max(vm)) if vm.size else 0.0
    if (i + 1) % 2000 == 0: print(f"  {i+1}/{len(rows)}", file=sys.stderr)

# features dérivées + interactions (toutes Nano) + phase one-hot + label
FIELDS = ["game_id", "ply"] + NANO + ["eval_stm", "g_nano", "abs_eval",
         "ix_eval_betawdl", "ix_abseval_entpol", "ix_sign_betamcts", "ix_g_decis",
         "phase_mid", "phase_end", "drop_cp", "result_stm", "best_cp", "n_within_50", "gap2", "mover_elo"]
with open(OUT, "w", newline="") as fo:
    w = csv.DictWriter(fo, fieldnames=FIELDS); w.writeheader()
    for o in rows:
        e = o["eval_stm"]; g = (e + 1) / 2
        o["g_nano"] = g; o["abs_eval"] = abs(e)
        o["ix_eval_betawdl"] = e * (o["beta_wdl"] or 0)
        o["ix_abseval_entpol"] = abs(e) * (o["ent_policy"] or 0)
        o["ix_sign_betamcts"] = np.sign(e) * (o["beta_mcts_800"] or 0)
        o["ix_g_decis"] = g * (o["decisiveness"] or 0)
        o["phase_mid"] = 1 if o["phase"] == "middlegame" else 0
        o["phase_end"] = 1 if o["phase"] == "endgame" else 0
        w.writerow({k: o.get(k) for k in FIELDS})
print(f"écrit {OUT} ({len(rows)} lignes)", file=sys.stderr)
