"""playability.py — courbe de JOUABILITÉ humain vs machine (DESIGN §10, la "fidélité").

Pour chaque position : score espéré de la MACHINE (Nano WDL, g = pW + ½pD, POV trait)
confronté à l'ISSUE HUMAINE RÉELLE de la partie (result_stm, déjà dans le dataset, 1800-2000).
On range par score-machine → ce que les humains font vraiment. L'ÉCART (contraction vers 0.5)
= "à quel point c'est jouable par un humain par rapport à la machine".

⚠️ Honnêteté : la contraction mesurée = (sur-confiance éventuelle du value head Nano) + (difficulté
humaine réelle). Pour ISOLER la difficulté pure il faudra recaler contre Stockfish (best_cp, dispo
quand le run multipv finit). Ici on affiche la courbe Nano↔humain telle qu'on la MONTRERAIT.

Sortie : table console + PNG. Run : ~/.gatevenv/bin/python playability.py [--csv ... --onnx ... --out ...]
"""

from __future__ import annotations

import argparse
import csv, os
import sys

import chess
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from beta_engine import BetaEngine, board_with_history  # noqa: E402

OUTCOME = {"W": 1.0, "D": 0.5, "L": 0.0}
NBINS = 10


def machine_scores(rows, onnx):
    """Score espéré g (POV trait) via le meilleur coup (root_expand) — POV-correct 2 couleurs.

    g = (V+1)/2 où V = pW−pL (car g = pW+½pD = ½ + ½V). On lit V comme la valeur du
    meilleur coup (max valmover) plutôt que la value head directe : cette dernière est
    POV-fiable seulement pour le trait aux Blancs (cf. debug 2026-06-30) ; root_expand
    (valeurs des enfants négées) est validé pour les deux couleurs.
    """
    eng = BetaEngine(onnx, intra_op=1)
    g = np.empty(len(rows))
    for i, r in enumerate(rows):
        _moves, _P, vm = eng.root_expand(board_with_history(r["fen"], r.get("line")))
        v = float(np.max(vm)) if vm.size else 0.0
        g[i] = (v + 1.0) / 2.0
        if (i + 1) % 2000 == 0:
            print(f"  machine {i + 1}/{len(rows)}", file=sys.stderr)
    return g


def calib_table(mg, hg, mask=None):
    """Table par tranche de score-machine : n, score-machine moyen, score-humain réel, contraction."""
    if mask is not None:
        mg, hg = mg[mask], hg[mask]
    edges = np.linspace(0, 1, NBINS + 1)
    b = np.clip(np.digitize(mg, edges) - 1, 0, NBINS - 1)
    rows, ece = [], 0.0
    for k in range(NBINS):
        m = b == k
        n = int(m.sum())
        if n == 0:
            continue
        mm, hh = float(mg[m].mean()), float(hg[m].mean())
        ece += n * abs(mm - hh)
        rows.append((n, mm, hh, mm - hh))
    return rows, (ece / max(1, len(mg)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="../data/lichess_sample/lichess_2024-12_15k.csv")
    ap.add_argument("--onnx", default=os.path.expanduser("~/nanozero-night/models/gen-031-promoted.onnx"))
    ap.add_argument("--out", default="playability-1800-2000.png")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    hg = np.array([OUTCOME[r["result_stm"]] for r in rows])
    phase = np.array([r["phase"] for r in rows])
    print(f"{len(rows)} positions (Elo 1800-2000). Calcul des valeurs-machine Nano…")
    mg = machine_scores(rows, args.onnx)

    print(f"\n=== JOUABILITÉ humain vs machine — {len(rows)} positions ===")
    tbl, ece = calib_table(mg, hg)
    print(f"{'machine':>8} {'humain':>8} {'contraction':>12} {'n':>6}")
    for n, mm, hh, c in tbl:
        print(f"{mm:8.2f} {hh:8.2f} {c:+12.2f} {n:6d}")
    print(f"\nECE (écart moyen machine↔humain) = {ece:.3f}")
    # corrélation de rang : la valeur machine ORDONNE-t-elle encore les issues ? (doit rester forte)
    from scipy.stats import spearmanr
    print(f"Spearman(score-machine, issue-humaine) = {spearmanr(mg, hg).correlation:+.3f} (le classement survit, l'échelle se contracte)")

    # PNG : courbe globale + par phase
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot([0, 1], [0, 1], "k--", lw=1, label="machine = humain (jeu parfait)")
    xs = [r[1] for r in tbl]; ys = [r[2] for r in tbl]
    ax[0].plot(xs, ys, "o-", color="#d62728", lw=2.5, ms=7, label="humain réel 1800-2000")
    ax[0].fill_between(xs, xs, ys, color="#d62728", alpha=0.15, label="contraction (jouabilité)")
    ax[0].set_xlabel("score espéré MACHINE (Nano, pW+½pD)")
    ax[0].set_ylabel("score RÉEL des humains")
    ax[0].set_title(f"Jouabilité humain vs machine  (ECE={ece:.3f})", fontweight="bold")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=0.3); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)

    ax[1].plot([0, 1], [0, 1], "k--", lw=1)
    for ph, col in (("opening", "#1f77b4"), ("middlegame", "#2ca02c"), ("endgame", "#ff7f0e")):
        t, _ = calib_table(mg, hg, mask=(phase == ph))
        if t:
            ax[1].plot([r[1] for r in t], [r[2] for r in t], "o-", color=col, lw=2, ms=5,
                       label=f"{ph} (n={sum(r[0] for r in t)})")
    ax[1].set_xlabel("score espéré MACHINE"); ax[1].set_ylabel("score RÉEL humains")
    ax[1].set_title("Par phase — où l'humain décroche le plus", fontweight="bold")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3); ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)

    fig.suptitle("NanoZero — jouabilité d'une position : ce que la MACHINE obtient vs ce qu'un HUMAIN (1800-2000) obtient réellement\n"
                 "Lichess rapide 2024-12 · l'écart sous la diagonale = la position est moins gagnable pour un humain qu'elle n'en a l'air",
                 fontsize=11)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(args.out, dpi=110)
    print(f"\nPNG → {args.out}")


if __name__ == "__main__":
    main()
