"""label_blunders.py — étiquette « gaffe humaine » par réf INDÉPENDANTE (Stockfish).

☠️ Anti-circularité (DESIGN §6) : la cible n'est JAMAIS l'éval NanoZero, sinon β
prédit β. On mesure la chute d'éval du coup HUMAIN réel vs le meilleur coup, avec
**Stockfish à profondeur FIXE** (déterministe, hardware-indépendant). Tablebases
Syzygy absentes ici → finales ≤7 approchées par la profondeur (limitation notée).

drop_cp = eval(meilleur coup) − eval(après coup humain), des deux du POV du trait.
blunder = drop_cp > τ_cp. On stocke drop_cp pour que validate sweepe le seuil.

Entrée : CSV de extract_positions.py. Sortie : même CSV + colonnes best_cp, drop_cp, blunder.
Shardable : --start/--count sur les lignes (un Stockfish par process).

Run : ~/.gatevenv/bin/python label_blunders.py IN.csv OUT.csv \
        [--depth 16 --tau-cp 100 --engine /usr/games/stockfish --threads 1 --hash 128 --start 0 --count 0]
"""

from __future__ import annotations

import argparse
import csv
import sys

import chess
import chess.engine

MATE_CP = 100000


def _eval_after_human(engine, board: chess.Board, depth: int) -> int:
    """Éval (cp, POV du trait à la racine) de la position APRÈS le coup humain."""
    if board.is_checkmate():
        return MATE_CP  # le trait vient de mater → excellent pour lui
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    child_rel = info["score"].relative.score(mate_score=MATE_CP)  # POV adversaire (trait à l'enfant)
    return -child_rel  # POV du trait à la racine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--tau-cp", type=int, default=100)
    ap.add_argument("--engine", default="/usr/games/stockfish")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=128)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0, help="0 = jusqu'à la fin")
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    end = len(rows) if args.count == 0 else min(len(rows), args.start + args.count)
    rows = rows[args.start : end]

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": args.threads, "Hash": args.hash})
    out_fields = list(rows[0].keys()) + ["best_cp", "drop_cp", "blunder"] if rows else []

    n_blunder = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=out_fields)
        w.writeheader()
        for i, row in enumerate(rows):
            board = chess.Board(row["fen"])
            human = chess.Move.from_uci(row["human_move"])
            # Meilleur coup (POV trait).
            best_info = engine.analyse(board, chess.engine.Limit(depth=args.depth))
            best_cp = best_info["score"].relative.score(mate_score=MATE_CP)
            # Éval après le coup humain (POV trait).
            board.push(human)
            after_cp = _eval_after_human(engine, board, args.depth)
            drop = max(0, best_cp - after_cp)
            blunder = 1 if drop > args.tau_cp else 0
            n_blunder += blunder
            row.update(best_cp=best_cp, drop_cp=drop, blunder=blunder)
            w.writerow(row)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(rows)} (blunders={n_blunder})", file=sys.stderr)
    engine.quit()
    print(
        f"positions={len(rows)} blunders={n_blunder} ({100 * n_blunder / max(1, len(rows)):.1f}%) → {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
