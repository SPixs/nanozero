"""compute_beta.py — calcule les variantes de β pour chaque position étiquetée.

Entrée : CSV de label_blunders.py (fen, ..., blunder).
Pour chaque position : expansion 1-ply du champion ONNX → (P, valmover) →
toutes les variantes candidates (architecture §6.2.1) + la baseline top1−top2.

Sortie : même CSV + colonnes beta_* et top12_gap (la baseline que β doit battre).

Run : ~/.gatevenv/bin/python compute_beta.py IN.csv OUT.csv \
        [--onnx models/gen-031-promoted.onnx --threads 1]
"""

from __future__ import annotations

import argparse
import csv, os
import sys

import chess
import numpy as np

from beta_engine import BetaEngine, compute_beta

# Variantes calculées (nom → kwargs de compute_beta). Le gate choisit la meilleure par AUC.
VARIANTS: dict[str, dict] = {
    "beta_swing_maxq": dict(form="swing", ref="maxq"),
    "beta_swing_policy": dict(form="swing", ref="policy"),
    "beta_swing_nat25": dict(form="swing", ref="maxq", naturalness=0.25),
    "beta_thr05": dict(form="threshold", tau=0.05, ref="maxq"),
    "beta_thr10": dict(form="threshold", tau=0.10, ref="maxq"),
    "beta_thr20": dict(form="threshold", tau=0.20, ref="maxq"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--onnx", default=os.path.expanduser("~/nanozero-night/models/gen-031-promoted.onnx"))
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    eng = BetaEngine(args.onnx, intra_op=args.threads)
    with open(args.inp, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    extra = list(VARIANTS.keys()) + ["top12_gap"]
    out_fields = list(rows[0].keys()) + extra if rows else []

    with open(args.out, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=out_fields)
        w.writeheader()
        for i, row in enumerate(rows):
            board = chess.Board(row["fen"])
            _moves, P, valmover = eng.root_expand(board)
            for name, kw in VARIANTS.items():
                row[name] = round(compute_beta(P, valmover, **kw), 6)
            # Baseline top1−top2 : écart de valeur best vs 2e best (grand = position facile).
            if valmover.size >= 2:
                s = np.sort(valmover)[::-1]
                row["top12_gap"] = round(float(s[0] - s[1]), 6)
            else:
                row["top12_gap"] = 0.0
            w.writerow(row)
            if (i + 1) % 200 == 0:
                print(f"  beta {i + 1}/{len(rows)}", file=sys.stderr)
    print(f"positions={len(rows)} variantes={len(VARIANTS)} → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
