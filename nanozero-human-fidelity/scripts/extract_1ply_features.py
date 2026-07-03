"""extract_1ply_features.py — features Nano 1-ply (root_expand) pour le corpus stratifié (piste A).

Par position (AVEC historique via line) : eval_stm = max(valmover) POV trait, g_nano, ent_policy,
p_max, beta1 (regret pondéré policy vs meilleur coup), betavar1 (variance pondérée policy).
Toutes dérivables au runtime de l'arbre MCTS (profondeur 1 = priors + Q enfants) → Nano-only, sim-stable.

Shardé : --shard i --nshards N (ligne j traitée si j % N == i). Sortie : <out>.shard<i>.csv.
Run : PYTHONPATH=.../nanozero-training/src ~/.gatevenv/bin/python extract_1ply_features.py IN.csv OUT --shard 0 --nshards 8
"""
from __future__ import annotations

import argparse
import csv, os
import math
import sys

import numpy as np

from beta_engine import BetaEngine, board_with_history

ONNX = os.path.expanduser("~/nanozero-night/models/gen-031-promoted.onnx")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--onnx", default=ONNX)
    args = ap.parse_args()

    rows = [r for j, r in enumerate(csv.DictReader(open(args.inp))) if j % args.nshards == args.shard]
    eng = BetaEngine(args.onnx, intra_op=1)
    out_path = f"{args.out}.shard{args.shard}.csv"
    fields = ["game_id", "ply", "eval_stm", "g_nano", "ent_policy", "p_max", "beta1", "betavar1", "ok"]
    n_ok = 0
    with open(out_path, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows):
            rec = {"game_id": r["game_id"], "ply": r["ply"], "ok": 0}
            try:
                board = board_with_history(r["fen"], r.get("line"))
                moves, P, vm = eng.root_expand(board)
                if len(moves) >= 2:
                    P = np.asarray(P, dtype=float)
                    P = P / (P.sum() or 1.0)
                    vm = np.asarray(vm, dtype=float)
                    vmax = float(vm.max())
                    mean = float((P * vm).sum())
                    rec.update(
                        eval_stm=vmax,
                        g_nano=(vmax + 1.0) / 2.0,
                        ent_policy=float(-(P[P > 0] * np.log2(P[P > 0])).sum()),
                        p_max=float(P.max()),
                        beta1=float((P * np.maximum(0.0, vmax - vm)).sum()),
                        betavar1=float((P * (vm - mean) ** 2).sum()),
                        ok=1,
                    )
                    n_ok += 1
            except Exception:
                pass
            w.writerow(rec)
            if (i + 1) % 1000 == 0:
                print(f"[shard {args.shard}] {i + 1}/{len(rows)} (ok={n_ok})", file=sys.stderr)
    print(f"[shard {args.shard}] FINI {n_ok}/{len(rows)} ok → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
