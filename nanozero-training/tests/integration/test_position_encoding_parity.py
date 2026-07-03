"""Parity test: training.encode_position ≡ Java GameState.toPlanes (slow opt-in).

Vérifie que 100 fixtures Java (générées via nanozero-nn fixture generator)
matchent bit-pour-bit l'encoding Python sur la totalité des 119 plans
x 64 cases = 7616 floats par fixture. Phase 1.0.0-4-a critère critique.

Format binaire (cf. PositionEncodingFixtureGenerator.java) :
  [int32 BE: n_fixtures]
  Pour chaque fixture :
    [int32 BE: depth] [int32 BE: fen_len] [bytes: FEN]
    [int32 BE: n_uci_moves] (uci_len, uci_bytes) * n_uci_moves
    [int32 BE: 119] [int32 BE: 8] [int32 BE: 8]
    [float32 BE x 7616: planes row-major (plane, rank, file)]
"""

from __future__ import annotations

import struct
from pathlib import Path

import chess
import numpy as np
import numpy.typing as npt
import pytest
from nanozero_training.network.position_encoding import encode_position

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "position_encoding_parity.bin"
N_PLANES = 119
ROWS = 8
COLS = 8


def _load_fixtures(
    path: Path,
) -> list[tuple[int, str, list[str], npt.NDArray[np.float32]]]:
    """Load all fixtures from binary file. Returns list of (depth, fen, uci_moves, planes)."""
    fixtures: list[tuple[int, str, list[str], npt.NDArray[np.float32]]] = []
    with path.open("rb") as f:
        n_fixtures = struct.unpack(">i", f.read(4))[0]
        for _ in range(n_fixtures):
            depth = struct.unpack(">i", f.read(4))[0]
            fen_len = struct.unpack(">i", f.read(4))[0]
            fen = f.read(fen_len).decode("utf-8")
            n_uci_moves = struct.unpack(">i", f.read(4))[0]
            uci_moves: list[str] = []
            for _m in range(n_uci_moves):
                ucilen = struct.unpack(">i", f.read(4))[0]
                uci_moves.append(f.read(ucilen).decode("utf-8"))
            planes_meta = struct.unpack(">iii", f.read(12))
            assert planes_meta == (N_PLANES, ROWS, COLS)
            n_floats = N_PLANES * ROWS * COLS
            raw = f.read(n_floats * 4)
            planes = (
                np.frombuffer(raw, dtype=">f4").reshape(N_PLANES, ROWS, COLS).astype(np.float32)
            )
            fixtures.append((depth, fen, uci_moves, planes))
    return fixtures


@pytest.mark.slow()
def test_position_encoding_parity_vs_java() -> None:
    """All 100 fixtures must match bit-perfect (Python encode_position ≡ Java toPlanes)."""
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Fixture {FIXTURE_PATH} not found. Generate via "
            "mvn -pl nanozero-nn exec:java "
            "-Dexec.mainClass=org.nanozero.nn.fixtures.PositionEncodingFixtureGenerator"
        )

    fixtures = _load_fixtures(FIXTURE_PATH)
    assert len(fixtures) >= 50, f"Fixture only has {len(fixtures)} entries — expected ≥ 50"

    mismatches: list[tuple[int, str, list[int]]] = []

    for idx, (_depth, fen, uci_moves, expected_planes) in enumerate(fixtures):
        # Reconstitute board with full history (critique pour les plans temporels).
        board = chess.Board()
        for uci in uci_moves:
            board.push(chess.Move.from_uci(uci))

        # Note : on ne compare PAS board.fen() vs fixture fen.
        # python-chess n'écrit le champ EP que si la prise est légalement
        # disponible, alors que Java Fen.write écrit le pseudo-EP square
        # dès qu'un pion a fait un double-push. Différence cosmétique sans
        # impact sur les 119 plans (EP n'est PAS encodé dans toPlanes).

        got_planes = encode_position(board)

        if not np.array_equal(got_planes, expected_planes):
            diff_mask = got_planes != expected_planes
            plane_diffs = diff_mask.any(axis=(1, 2))
            diverging_planes = [int(p) for p in np.where(plane_diffs)[0]]
            mismatches.append((idx, fen, diverging_planes))

    if mismatches:
        msg_lines = [f"\n{len(mismatches)}/{len(fixtures)} parity mismatches (first 5):"]
        for idx, fen, planes in mismatches[:5]:
            msg_lines.append(f"  fixture {idx} (fen={fen}): diverging planes={planes}")
        pytest.fail("\n".join(msg_lines))
