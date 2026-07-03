"""Parity test: training.encode_move ≡ Java MoveEncoding.encode (slow opt-in).

Vérifie que 1000 fixtures Java (générées via nanozero-nn fixture generator)
matchent bit-pour-bit l'encoding Python. Phase 1.0.0-4-a critère critique.
"""

from __future__ import annotations

import csv
from pathlib import Path

import chess
import pytest
from nanozero_training.network.move_encoding import encode_move

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "move_encoding_parity.csv"


@pytest.mark.slow()
def test_move_encoding_parity_vs_java() -> None:
    """All 1000 fixtures must match bit-perfect (Python encode_move ≡ Java)."""
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Fixture {FIXTURE_PATH} not found. Generate via "
            "mvn -pl nanozero-nn exec:java "
            "-Dexec.mainClass=org.nanozero.nn.fixtures.MoveEncodingFixtureGenerator"
        )

    mismatches: list[tuple[str, str, int, int]] = []
    total = 0

    with FIXTURE_PATH.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            fen = row["fen"]
            uci = row["uci"]
            expected = int(row["index"])

            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            got = encode_move(move, board)

            if got != expected:
                mismatches.append((fen, uci, expected, got))

    if mismatches:
        msg_lines = [f"\n{len(mismatches)}/{total} parity mismatches (showing first 10):"]
        for fen, uci, expected, got in mismatches[:10]:
            msg_lines.append(f"  fen={fen}, uci={uci}, expected={expected}, got={got}")
        pytest.fail("\n".join(msg_lines))

    assert total >= 100, f"Fixture only has {total} entries — expected ≥ 100"
