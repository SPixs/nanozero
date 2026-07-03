"""extract_positions_stratified.py — échantillonnage STRATIFIÉ de positions Lichess (piste A jouabilité).

Pourquoi : le corpus 15k (3 positions au hasard/partie) est dominé par le milieu de jeu positionnel
équilibré → le modèle de jouabilité n'a jamais vu « +une tour », finales techniques, etc. (test curated
2026-07-01). Ici : QUOTAS par case (phase × balance matérielle POV trait) pour couvrir l'espace.

Cases : phase {opening,middlegame,endgame} × mat_stm {<=-3, -2..-1, 0, +1..+2, >=+3} = 15 cases.
Même filtre que l'original (rapide, both Elo 1800-2000, standard, résultat décisif/nulle), ≤3 pos/partie,
préférence aux cases les MOINS remplies. Pré-filtre texte brut (pas de parse pour les parties rejetées).

Run :  curl -s <dump.pgn.zst> | zstdcat | python3 extract_positions_stratified.py - OUT.csv --per-cell 8000
Stdout périodique : remplissage des cases. S'arrête quand tout est plein (SIGPIPE coupe curl) ou fin de flux.
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import sys

import chess
import chess.pgn

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}
PHASES = ("opening", "middlegame", "endgame")
BUCKETS = ("m3", "m1", "eq", "p1", "p3")  # <=-3, -2..-1, 0, +1..+2, >=+3 (POV trait)


def bucket(mat_stm: int) -> str:
    if mat_stm <= -3:
        return "m3"
    if mat_stm <= -1:
        return "m1"
    if mat_stm == 0:
        return "eq"
    if mat_stm <= 2:
        return "p1"
    return "p3"


def phase_of(board: chess.Board) -> str:
    n = chess.popcount(board.occupied)
    return "opening" if n >= 26 else ("middlegame" if n >= 12 else "endgame")


def mat_stm_of(board: chess.Board) -> int:
    s = 0
    for pt, v in VAL.items():
        s += v * (len(board.pieces(pt, chess.WHITE)) - len(board.pieces(pt, chess.BLACK)))
    return s if board.turn == chess.WHITE else -s


def result_stm(result: str, stm_white: bool) -> str:
    if result == "1/2-1/2":
        return "D"
    won = (result == "1-0") if stm_white else (result == "0-1")
    return "W" if won else "L"


def fast_ok(headers: dict) -> bool:
    """Pré-filtre sur les headers bruts (aucun parse d'échecs)."""
    if headers.get("Variant", "Standard") != "Standard":
        return False
    tc = headers.get("TimeControl", "")
    if "+" not in tc:
        return False
    try:
        base = int(tc.split("+")[0])
    except ValueError:
        return False
    if not (480 <= base <= 1500):  # rapide Lichess
        return False
    try:
        we, be = int(headers.get("WhiteElo", "0")), int(headers.get("BlackElo", "0"))
    except ValueError:
        return False
    if not (1800 <= we <= 2000 and 1800 <= be <= 2000):
        return False
    return headers.get("Result", "*") in ("1-0", "0-1", "1/2-1/2")


def games_raw(stream):
    """Découpe le flux PGN en (headers dict, texte complet) sans parser les coups."""
    headers: dict = {}
    lines: list[str] = []
    in_moves = False
    for line in stream:
        if line.startswith("[Event ") and lines and in_moves:
            yield headers, "".join(lines)
            headers, lines, in_moves = {}, [], False
        lines.append(line)
        if line.startswith("["):
            k = line[1 : line.find(" ")]
            v = line[line.find('"') + 1 : line.rfind('"')]
            headers[k] = v
        elif line.strip():
            in_moves = True
    if lines and in_moves:
        yield headers, "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("out")
    ap.add_argument("--per-cell", type=int, default=8000)
    ap.add_argument("--per-game", type=int, default=3)
    ap.add_argument("--skip-plies", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--report-every", type=int, default=200000)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stream = sys.stdin if args.pgn == "-" else open(args.pgn, encoding="utf-8", errors="replace")
    quota = {(p, b): args.per_cell for p in PHASES for b in BUCKETS}
    fields = ["fen", "human_move", "stm", "result_stm", "mover_elo", "tc", "phase",
              "mat_stm", "cell", "game_id", "ply", "line"]
    n_games = n_ok = n_pos = 0

    def report():
        full = sum(1 for v in quota.values() if v <= 0)
        print(f"[strat] games={n_games} qualif={n_ok} positions={n_pos} cases_pleines={full}/15", file=sys.stderr)
        grid = " | ".join(f"{p[:3]}:" + ",".join(f"{b}={args.per_cell - quota[(p, b)]}" for b in BUCKETS) for p in PHASES)
        print(f"[strat] {grid}", file=sys.stderr)

    with open(args.out, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=fields)
        w.writeheader()
        for headers, raw in games_raw(stream):
            n_games += 1
            if n_games % args.report_every == 0:
                report()
            if not fast_ok(headers):
                continue
            game = chess.pgn.read_game(io.StringIO(raw))
            if game is None:
                continue
            n_ok += 1
            h = game.headers
            we, be = int(h["WhiteElo"]), int(h["BlackElo"])
            result = h["Result"]

            board = game.board()
            cands: list[tuple] = []
            played: list[str] = []
            ply = 0
            for mv in game.mainline_moves():
                if ply >= args.skip_plies and board.legal_moves.count() >= 2:
                    p, b = phase_of(board), bucket(mat_stm_of(board))
                    if quota[(p, b)] > 0:
                        stm_white = board.turn == chess.WHITE
                        cands.append((ply, board.fen(), mv.uci(), stm_white,
                                      we if stm_white else be, " ".join(played),
                                      p, mat_stm_of(board), (p, b)))
                board.push(mv)
                played.append(mv.uci())
                ply += 1
            if not cands:
                continue
            # préférence aux cases les MOINS remplies : tri par quota restant décroissant, tirage dans le peloton
            rng.shuffle(cands)
            cands.sort(key=lambda c: -quota[c[8]])
            game_id = h.get("Site", f"game{n_games}").rsplit("/", 1)[-1]
            taken = 0
            seen_cells: set = set()
            for ply_i, fen, uci, stm_white, elo, line, p, mat, cell in cands:
                if taken >= args.per_game or quota[cell] <= 0 or cell in seen_cells:
                    continue
                seen_cells.add(cell)  # diversité intra-partie
                quota[cell] -= 1
                taken += 1
                n_pos += 1
                w.writerow({"fen": fen, "human_move": uci, "stm": "w" if stm_white else "b",
                            "result_stm": result_stm(result, stm_white), "mover_elo": elo,
                            "tc": h.get("TimeControl", ""), "phase": p, "mat_stm": mat,
                            "cell": f"{p}:{cell[1]}", "game_id": game_id, "ply": ply_i, "line": line})
            if all(v <= 0 for v in quota.values()):
                print("[strat] TOUTES les cases pleines — stop.", file=sys.stderr)
                break
    report()


if __name__ == "__main__":
    main()
