"""investigate_history.py — cycle d'investigation AVEC historique (sans multipv).

Teste 3 hypothèses pré-enregistrées sur les données déjà recalculées avec historique :
  H-A : β_v1(hist) + features plateau (combo CV) ≥ 0.63 ?
  H-B : la valeur-machine g prédit-elle la gaffe ? (et best_cp inverse)
  H-C : l'inversion "grosses gaffes dans le calme" était-elle l'artefact d'historique ?
        (sweep du seuil de gaffe ; β reste-t-il ≥0.5 sur les grosses gaffes ?)

Entrées : revalid_hist.csv (beta_v1_hist, g_hist) + labeled.csv (drop_cp, best_cp, game_id) — join par fen.
"""

from __future__ import annotations

import csv

import chess
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import os as _os
D = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "data", "lichess_sample")
_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def board_feats(fen):
    b = chess.Board(fen); legal = list(b.legal_moves)
    mat = sum(_VAL[p] * (len(b.pieces(p, True)) + len(b.pieces(p, False))) for p in _VAL)
    return [len(legal), sum(b.is_capture(m) for m in legal), sum(b.gives_check(m) for m in legal),
            len(b.pieces(chess.QUEEN, True)) + len(b.pieces(chess.QUEEN, False)), mat, int(b.is_check())]


def auc(y, s):
    return roc_auc_score(y, s) if len(np.unique(y)) > 1 else float("nan")


def cv_auc(X, y, groups, clf):
    p = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]
    return auc(y, p)


def main():
    lab = {r["fen"]: r for r in csv.DictReader(open(f"{D}/lichess_2024-12_15k.labeled.csv"))}
    rev = list(csv.DictReader(open(f"{D}/revalid_hist.csv")))
    beta, g, drop, best, blunder, groups, phase, Xf = [], [], [], [], [], [], [], []
    for r in rev:
        l = lab.get(r["fen"])
        if not l:
            continue
        beta.append(float(r["beta_v1_hist"])); g.append(float(r["g_hist"]))
        drop.append(float(l["drop_cp"])); best.append(float(l["best_cp"]))
        blunder.append(int(l["blunder"])); groups.append(l["game_id"]); phase.append(r["phase"])
        Xf.append(board_feats(r["fen"]))
    beta, g, drop, best = map(np.array, (beta, g, drop, best))
    blunder = np.array(blunder); groups = np.array(groups); phase = np.array(phase); Xf = np.array(Xf)
    n = len(beta)
    print(f"=== INVESTIGATION AVEC HISTORIQUE — {n} positions, {blunder.mean()*100:.1f}% gaffes ===")

    # H-C : sweep seuil de gaffe — l'inversion persiste-t-elle avec historique ?
    print("\n[H-C] β_v1(hist) → gaffe, selon le seuil (inversion d'avant = AUC<0.5 sur grosses gaffes) :")
    print(f"   {'seuil':>8} {'%gaffe':>7} {'AUC(β hist)':>12}  (rappel SANS hist : 100→0.49, 300→0.46, 800→0.40)")
    for t in (100, 200, 300, 500, 800):
        y = (drop > t).astype(int)
        if y.sum() < 20:
            continue
        print(f"   drop>{t:>4} {100*y.mean():6.1f}% {auc(y, beta):12.3f}")

    # H-B : valeur-machine
    print("\n[H-B] la valeur-machine prédit-elle la gaffe ? (cible drop>100)")
    y = (drop > 100).astype(int)
    print(f"   AUC(g_hist → gaffe)        = {auc(y, g):.3f}")
    print(f"   AUC(|g−0.5| → gaffe)       = {auc(y, np.abs(g-0.5)):.3f}  (extrémité = tranché)")
    print(f"   AUC(−g → gaffe)            = {auc(y, -g):.3f}  (perdant = +gaffe ?)")
    print(f"   AUC(−best_cp → gaffe)      = {auc(y, -best):.3f}")

    # H-A : combos (CV groupée par partie)
    print("\n[H-A] combos (logistique L2 + GBM, CV groupée game_id, cible drop>100) :")
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000))
    gb = GradientBoostingClassifier(max_depth=3, n_estimators=200, learning_rate=0.05)
    print(f"   features plateau seules (logreg) = {cv_auc(Xf, y, groups, lr):.3f}  (ancien plafond ~0.62)")
    print(f"   β_v1(hist) seul                  = {auc(y, beta):.3f}")
    Xb = np.column_stack([Xf, beta])
    print(f"   plateau + β_v1(hist) (logreg)    = {cv_auc(Xb, y, groups, lr):.3f}  ← H-A (≥0.63 ?)")
    Xbg = np.column_stack([Xf, beta, g])
    print(f"   plateau + β + g (logreg)         = {cv_auc(Xbg, y, groups, lr):.3f}")
    print(f"   plateau + β + g (GBM, plafond)   = {cv_auc(Xbg, y, groups, gb):.3f}")

    print("\n[par phase] β_v1(hist) → gaffe :")
    for ph in ("opening", "middlegame", "endgame"):
        m = phase == ph
        if m.sum() > 50:
            print(f"   {ph:11s} AUC={auc(blunder[m], beta[m]):.3f} (n={m.sum()})")


if __name__ == "__main__":
    main()
