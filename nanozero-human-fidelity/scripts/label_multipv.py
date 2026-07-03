"""label_multipv.py — réf Stockfish ENRICHIE pour l'expérience β-v2 (cf. docs/EXP-beta-v2-preregistration.md).

Étend label_blunders.py : en plus du label gaffe (drop du coup humain), dump par position :
  - multipv top-k (uci:cp ...) → sert β_strong (trap_mass) ET la criticité objective T_obj
    (n_within_50, gap2) ;
  - courbe d'éval de la PV par profondeur (cp@{8,12,18}) → sert β_depth (volatilité = |cp@18−cp@8|).
Réf INDÉPENDANTE de Nano (anti-circularité). Shardable (--start/--count), comme run_gate.sh.

Run : ~/.gatevenv/bin/python label_multipv.py IN.csv OUT.csv \
        [--depth 18 --multipv 8 --tau-cp 100 --engine /usr/games/stockfish --threads 1 --hash 256 --start 0 --count 0]
"""

from __future__ import annotations

import argparse
import csv
import sys

import chess
import chess.engine

MATE_CP = 100000
CURVE_DEPTHS = (8, 12, 18)


def _mover_cp(info: dict, mover: bool) -> int | None:
    sc = info.get("score")
    if sc is None:
        return None
    return sc.pov(mover).score(mate_score=MATE_CP)


def _cp_at(curve: dict[int, int], target: int) -> int | None:
    """cp de la PV à la profondeur target (exacte, sinon la plus proche disponible ≤ target, sinon min dispo)."""
    if not curve:
        return None
    if target in curve:
        return curve[target]
    below = [d for d in curve if d <= target]
    return curve[max(below)] if below else curve[min(curve)]


def _analyse_position(engine, board: chess.Board, depth: int, multipv: int):
    """Retourne (topk[(uci,cp)], curve{depth:cp_pv1}) du POV du trait."""
    mover = board.turn
    curve: dict[int, int] = {}
    with engine.analysis(board, chess.engine.Limit(depth=depth), multipv=multipv) as analysis:
        for info in analysis:
            if info.get("multipv", 1) == 1 and "depth" in info and "score" in info:
                cp = _mover_cp(info, mover)
                if cp is not None:
                    curve[int(info["depth"])] = cp
        final = analysis.multipv  # liste InfoDict, résultat final par pv
    topk: list[tuple[str, int]] = []
    for info in final:
        pv = info.get("pv")
        cp = _mover_cp(info, mover)
        if pv and cp is not None:
            topk.append((pv[0].uci(), cp))
    return topk, curve


def _after_human_cp(engine, board: chess.Board, depth: int) -> int:
    """Éval (POV trait racine) APRÈS le coup humain déjà poussé sur board."""
    if board.is_checkmate():
        return MATE_CP
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    return -info["score"].relative.score(mate_score=MATE_CP)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--depth", type=int, default=18)
    ap.add_argument("--multipv", type=int, default=8)
    ap.add_argument("--tau-cp", type=int, default=100)
    ap.add_argument("--engine", default="/usr/games/stockfish")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--hash", type=int, default=256)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=0)
    args = ap.parse_args()

    with open(args.inp, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    end = len(rows) if args.count == 0 else min(len(rows), args.start + args.count)
    rows = rows[args.start : end]

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": args.threads, "Hash": args.hash})
    extra = ["best_cp", "drop_cp", "blunder", "n_within_50", "gap2", "cp_d8", "cp_d12", "cp_d18",
             "depth_swing", "sf_topk"]
    out_fields = (list(rows[0].keys()) + extra) if rows else []

    n_blunder = 0
    with open(args.out, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=out_fields)
        w.writeheader()
        for i, row in enumerate(rows):
            board = chess.Board(row["fen"])
            topk, curve = _analyse_position(engine, board, args.depth, args.multipv)
            best_cp = topk[0][1] if topk else 0
            second_cp = topk[1][1] if len(topk) > 1 else (best_cp - MATE_CP)
            n_within_50 = sum(1 for _, cp in topk if best_cp - cp <= 50)
            cp8, cp12, cp18 = _cp_at(curve, 8), _cp_at(curve, 12), _cp_at(curve, 18)
            swing = abs((cp18 if cp18 is not None else best_cp) - (cp8 if cp8 is not None else best_cp))
            # drop du coup humain (label gaffe).
            human = chess.Move.from_uci(row["human_move"])
            board.push(human)
            after_cp = _after_human_cp(engine, board, args.depth)
            drop = max(0, best_cp - after_cp)
            blunder = 1 if drop > args.tau_cp else 0
            n_blunder += blunder
            row.update(
                best_cp=best_cp, drop_cp=drop, blunder=blunder,
                n_within_50=n_within_50, gap2=best_cp - second_cp,
                cp_d8=cp8 if cp8 is not None else "", cp_d12=cp12 if cp12 is not None else "",
                cp_d18=cp18 if cp18 is not None else "", depth_swing=swing,
                sf_topk="|".join(f"{u}:{cp}" for u, cp in topk),
            )
            w.writerow(row)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(rows)} (blunders={n_blunder})", file=sys.stderr)
    engine.quit()
    print(f"positions={len(rows)} blunders={n_blunder} → {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
