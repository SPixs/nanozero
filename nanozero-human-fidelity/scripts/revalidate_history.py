"""Re-test AVEC historique (2026-06-30) : la value/β étaient calculées sur des FEN nus
(hors-distribution → value sous-évaluée). On recalcule β_v1 ET le score-machine g AVEC
l'historique 8-plies (board_with_history), en UNE passe root_expand par position.

Répond à deux questions :
  1) β_v1 → gaffe humaine : β=0.50 était-il réel, ou l'artefact du bug d'historique ?
  2) jouabilité corrigée : score-machine (avec hist) vs issue humaine réelle.

Entrées : labeled.csv (blunder, result_stm, phase — mêmes positions) + hist.csv (colonne line).
Sortie : AUC β + courbe jouabilité + CSV per-position (fen,blunder,result_stm,phase,beta_v1,g).
"""

from __future__ import annotations

import csv
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

from beta_engine import BetaEngine, board_with_history, compute_beta

import os as _os
D = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "lichess_sample")
ONNX = _os.path.expanduser("~/nanozero-night/models/gen-031-promoted.onnx")
OUTCOME = {"W": 1.0, "D": 0.5, "L": 0.0}


def main() -> None:
    labeled = list(csv.DictReader(open(f"{D}/lichess_2024-12_15k.labeled.csv")))
    hist = list(csv.DictReader(open(f"{D}/lichess_2024-12_15k_hist.csv")))
    line_of = {h["fen"]: h["line"] for h in hist}

    eng = BetaEngine(ONNX, intra_op=1)
    beta = np.empty(len(labeled)); g = np.empty(len(labeled))
    blunder = np.array([int(r["blunder"]) for r in labeled])
    res = np.array([OUTCOME[r["result_stm"]] for r in labeled])
    phase = np.array([r["phase"] for r in labeled])

    for i, r in enumerate(labeled):
        b = board_with_history(r["fen"], line_of.get(r["fen"]))
        _m, P, vm = eng.root_expand(b)
        beta[i] = compute_beta(P, vm, form="swing", ref="maxq")
        g[i] = (float(np.max(vm)) + 1) / 2 if vm.size else 0.5
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(labeled)}", file=sys.stderr)

    # sauvegarde per-position
    with open(f"{D}/revalid_hist.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["fen", "blunder", "result_stm", "phase", "beta_v1_hist", "g_hist"])
        for i, r in enumerate(labeled):
            w.writerow([r["fen"], blunder[i], r["result_stm"], phase[i], round(beta[i], 6), round(g[i], 4)])

    def auc(y, s):
        return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")

    print(f"\n=== 1) RE-TEST β_v1 AVEC historique → gaffe ({len(labeled)} pos, {blunder.mean()*100:.1f}% gaffes) ===")
    print(f"   AUC(β_v1 SANS hist) = 0.504  (mémoire, le verdict 'réfuté')")
    print(f"   AUC(β_v1 AVEC hist) = {auc(blunder, beta):.3f}   ← décisif")
    for ph in ("opening", "middlegame", "endgame"):
        m = phase == ph
        if m.sum() > 50:
            print(f"      {ph:11s} AUC={auc(blunder[m], beta[m]):.3f} (n={m.sum()})")

    print(f"\n=== 2) JOUABILITÉ corrigée (score-machine AVEC hist vs issue humaine) ===")
    print(f"   score-machine moyen = {g.mean():.3f}  (doit être ~0.5 ; 0.31 = symptôme du bug)")
    edges = np.linspace(0, 1, 11); bb = np.clip(np.digitize(g, edges) - 1, 0, 9)
    ece = 0.0
    print(f"   {'machine':>8} {'humain':>8} {'contraction':>12} {'n':>6}")
    for k in range(10):
        m = bb == k; n = int(m.sum())
        if n == 0:
            continue
        mm, hh = g[m].mean(), res[m].mean(); ece += n * abs(mm - hh)
        print(f"   {mm:8.2f} {hh:8.2f} {mm-hh:+12.2f} {n:6d}")
    print(f"   ECE = {ece/len(g):.3f}")
    from scipy.stats import spearmanr
    print(f"   Spearman(g, issue) = {spearmanr(g, res).correlation:+.3f}")


if __name__ == "__main__":
    main()
