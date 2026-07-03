"""extract_positions.py — échantillonne des positions de parties Lichess humaines.

Entrée : un flux PGN Lichess (fichier .pgn, .pgn.zst, ou stdin).
Filtre (DESIGN §6) : variante standard, **rapide**, **WhiteElo & BlackElo ∈ [lo,hi]**
(défaut 1800-2000 ≈ 1600 FIDE), terminaison normale.
Échantillonne K positions/partie (défaut 3) hors ouverture (skip plies), en
enregistrant le COUP HUMAIN réellement joué + l'issue de la partie (POV trait).

Sortie : CSV (fen, human_move, stm, result_stm, mover_elo, tc, phase, game_id, ply).
Anti-fuite : ≤3 positions/partie ; le hold-out (validate) splitte par partie+joueur+mois.

Run : ~/.gatevenv/bin/python extract_positions.py IN.pgn[.zst] OUT.csv \
        [--elo-lo 1800 --elo-hi 2000 --per-game 3 --skip-plies 16 --max-games N --seed 7]
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import sys

import chess
import chess.pgn


def _open_pgn(path: str):
    if path == "-":
        return sys.stdin
    if path.endswith(".zst"):
        import zstandard

        fh = open(path, "rb")
        dctx = zstandard.ZstdDecompressor()
        return io.TextIOWrapper(dctx.stream_reader(fh), encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _is_rapid(tc: str) -> bool:
    """Lichess rapide = temps de base 480-1500 s (8-25 min). TC = 'base+inc'."""
    if not tc or "+" not in tc:
        return False
    try:
        base = int(tc.split("+")[0])
    except ValueError:
        return False
    return 480 <= base <= 1500


def _phase(board: chess.Board) -> str:
    n = chess.popcount(board.occupied)
    if n >= 26:
        return "opening"
    if n >= 12:
        return "middlegame"
    return "endgame"


def _result_stm(result: str, stm_white: bool) -> str:
    """Issue de la partie du POV du trait à la position échantillonnée (W/D/L)."""
    if result == "1/2-1/2":
        return "D"
    white_won = result == "1-0"
    won = white_won if stm_white else (not white_won)
    return "W" if won else "L"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pgn")
    ap.add_argument("out")
    ap.add_argument("--elo-lo", type=int, default=1800)
    ap.add_argument("--elo-hi", type=int, default=2000)
    ap.add_argument("--per-game", type=int, default=3)
    ap.add_argument("--skip-plies", type=int, default=16)
    ap.add_argument("--max-games", type=int, default=0, help="0 = illimité")
    ap.add_argument("--target", type=int, default=0, help="arrêt dès N positions (0=illimité)")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    stream = _open_pgn(args.pgn)
    n_games = n_kept = n_pos = 0
    # "line" = UCI des coups MENANT à la position (pour reconstruire l'historique NN 8-plies :
    # Nano est entraîné avec historique → un board FEN-nu est hors-distribution, cf. bug 2026-06-30).
    fields = ["fen", "human_move", "stm", "result_stm", "mover_elo", "tc", "phase", "game_id", "ply", "line"]

    with open(args.out, "w", newline="", encoding="utf-8") as fout:
        w = csv.DictWriter(fout, fieldnames=fields)
        w.writeheader()
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            n_games += 1
            if args.max_games and n_games > args.max_games:
                break
            h = game.headers
            try:
                we, be = int(h.get("WhiteElo", "0")), int(h.get("BlackElo", "0"))
            except ValueError:
                continue
            if h.get("Variant", "Standard") != "Standard":
                continue
            if not _is_rapid(h.get("TimeControl", "")):
                continue
            if not (args.elo_lo <= we <= args.elo_hi and args.elo_lo <= be <= args.elo_hi):
                continue
            result = h.get("Result", "*")
            if result not in ("1-0", "0-1", "1/2-1/2"):
                continue

            # Positions candidates : hors ouverture, ≥2 coups légaux, non terminales.
            board = game.board()
            cands: list[tuple] = []  # (ply, fen, move_uci, stm_white, mover_elo, line)
            played: list[str] = []
            ply = 0
            for mv in game.mainline_moves():
                if ply >= args.skip_plies and board.legal_moves.count() >= 2:
                    stm_white = board.turn == chess.WHITE
                    cands.append(
                        (ply, board.fen(), mv.uci(), stm_white, we if stm_white else be, " ".join(played))
                    )
                board.push(mv)
                played.append(mv.uci())
                ply += 1

            if not cands:
                continue
            n_kept += 1
            chosen = cands if len(cands) <= args.per_game else rng.sample(cands, args.per_game)
            game_id = h.get("Site", f"game{n_games}").rsplit("/", 1)[-1]
            for ply_i, fen, move_uci, stm_white, mover_elo, line in chosen:
                b = chess.Board(fen)
                w.writerow(
                    {
                        "fen": fen,
                        "human_move": move_uci,
                        "stm": "w" if stm_white else "b",
                        "result_stm": _result_stm(result, stm_white),
                        "mover_elo": mover_elo,
                        "tc": h.get("TimeControl", ""),
                        "phase": _phase(b),
                        "game_id": game_id,
                        "ply": ply_i,
                        "line": line,
                    }
                )
                n_pos += 1
            if args.target and n_pos >= args.target:
                break

    print(
        f"games_lus={n_games} games_gardes={n_kept} positions={n_pos} → {args.out}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
