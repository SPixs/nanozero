import os
"""Smoke test du moteur β 1-ply sur le champion ONNX.

Sanité (pas une validation — c'est le rôle du gate) : signe des valeurs, gestion
des terminaux, débit. Run : ~/.gatevenv/bin/python smoke_beta.py [onnx_path]
"""

from __future__ import annotations

import sys
import time

import chess

from beta_engine import BetaEngine, compute_beta, level_of

ONNX = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/nanozero-night/models/gen-031-promoted.onnx")

CASES = [
    ("startpos (equilibre attendu)", chess.STARTING_FEN),
    ("blancs SANS dame noire (gros + matériel)", "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
    ("mat en 1 : Ra8# (test terminal)", "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1"),
    ("Italienne (positions ouvertes)", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
    ("milieu de jeu tendu (Najdorf)", "rnbqkb1r/1p2pppp/p2p1n2/8/3NP3/2N5/PPP2PPP/R1BQKB1R w KQkq - 0 6"),
]


def main() -> None:
    print(f"ONNX = {ONNX}\n")
    eng = BetaEngine(ONNX, intra_op=1)
    t0 = time.time()
    npos = 0
    for name, fen in CASES:
        board = chess.Board(fen)
        moves, P, valmover = eng.root_expand(board)
        npos += 1
        if len(moves) < 2:
            print(f"[{name}] <2 coups legaux\n")
            continue
        order = sorted(range(len(moves)), key=lambda i: P[i], reverse=True)
        best_i = max(range(len(moves)), key=lambda i: valmover[i])
        b_swing = compute_beta(P, valmover, form="swing", ref="maxq")
        b_thr = compute_beta(P, valmover, form="threshold", tau=0.10, ref="maxq")
        print(f"[{name}]  ({len(moves)} coups)")
        print(f"    meilleur coup (valeur) : {moves[best_i].uci()}  val={valmover[best_i]:+.3f}")
        top = ", ".join(f"{moves[i].uci()} P={P[i]:.2f} v={valmover[i]:+.2f}" for i in order[:4])
        print(f"    top policy : {top}")
        print(f"    β swing={b_swing:.4f} ({level_of(b_swing)})  |  β masse@τ0.10={b_thr:.4f}\n")
    dt = time.time() - t0
    print(f"{npos} positions en {dt:.2f}s  →  {npos / dt:.1f} pos/s (1 thread, expansion 1-ply complete)")


if __name__ == "__main__":
    main()
