"""compute_beta_v2.py — ajoute β_v1 et β_strong (trap_mass) à chaque position labellisée multipv.

Entrée : CSV de label_multipv.py (colonnes sf_topk, best_cp, depth_swing, n_within_50, gap2, blunder...).
- β_v1     = swing/maxq sur les VALEURS NANO (1-ply) — témoin, INDÉPENDANT de Stockfish.
- β_strong = trap_mass = Σ_b policy_Nano(b)·max(0, best_cp_SF − cp_SF(b)) / Σ_b policy_Nano(b),
             sur les coups du top-k Stockfish (regret par réf forte, plausibilité par policy Nano).
β_depth est déjà dans le CSV (= depth_swing). Sortie : CSV + colonnes beta_v1, beta_strong.

Run : ~/.gatevenv/bin/python compute_beta_v2.py IN.mpv.csv OUT.csv [--onnx ...gen-031-promoted.onnx]
"""

from __future__ import annotations

import argparse
import csv, os
import sys

import numpy as np

from beta_engine import BetaEngine, board_with_history, compute_beta


def _parse_topk(s: str) -> list[tuple[str, float]]:
    out = []
    for tok in s.split("|"):
        if ":" in tok:
            u, cp = tok.rsplit(":", 1)
            try:
                out.append((u, float(cp)))
            except ValueError:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--onnx", default=os.path.expanduser("~/nanozero-night/models/gen-031-promoted.onnx"))
    args = ap.parse_args()

    eng = BetaEngine(args.onnx, intra_op=1)
    with open(args.inp, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out_fields = list(rows[0].keys()) + ["beta_v1", "beta_strong"] if rows else []

    with open(args.out, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=out_fields)
        w.writeheader()
        for i, row in enumerate(rows):
            board = board_with_history(row["fen"], row.get("line"))
            moves, P, valmover = eng.root_expand(board)
            # β_v1 : valeurs Nano (indépendant de Stockfish).
            row["beta_v1"] = round(compute_beta(P, valmover, form="swing", ref="maxq"), 6)
            # β_strong (trap_mass) : regret Stockfish × policy Nano sur le top-k SF.
            pol = {m.uci(): float(P[j]) for j, m in enumerate(moves)}
            topk = _parse_topk(row.get("sf_topk", ""))
            best_cp = float(row["best_cp"]) if row.get("best_cp") not in (None, "") else 0.0
            num = den = 0.0
            for u, cp in topk:
                p = pol.get(u, 0.0)
                num += p * max(0.0, best_cp - cp)
                den += p
            row["beta_strong"] = round(num / den, 6) if den > 0 else 0.0
            w.writerow(row)
            if (i + 1) % 500 == 0:
                print(f"  beta_v2 {i + 1}/{len(rows)}", file=sys.stderr)
    print(f"positions={len(rows)} → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
