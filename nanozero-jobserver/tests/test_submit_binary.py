"""Tests du décodeur binaire de soumission (Story A.1) : round-trip + fuzz/adversarial."""

import struct

import numpy as np
import pytest
from nanozero_jobserver.submit_codec import (
    MAX_NNZ,
    MAX_POSITIONS,
    PLANES_LEN,
    POLICY_LEN,
    SubmitDecodeError,
    decode,
)


def _planes(seed: int = 0) -> np.ndarray:
    p = np.zeros(PLANES_LEN, dtype="<f4")
    p[(np.arange(PLANES_LEN) + seed) % 7 == 0] = 1.0
    return p


def _encode(positions) -> bytes:
    """Encode minimal (miroir du layout JS) pour produire des binaires valides de test."""
    out = bytearray(struct.pack("<I", len(positions)))
    for planes, idx, val, outcome in positions:
        out += struct.pack("<f", outcome)
        out += planes.astype("<f4").tobytes()
        out += struct.pack("<H", len(idx))
        for i, v in zip(idx, val, strict=False):
            out += struct.pack("<H", i) + struct.pack("<f", v)
    return bytes(out)


# --- round-trip / densification ---


def test_round_trip_dense():
    planes = _planes(1)
    idx, val = [3, 17, 4671], [0.5, 0.25, 0.2]
    out = decode(_encode([(planes, idx, val, 1.0)]))
    assert len(out) == 1
    planes_bytes, policy_bytes, outcome = out[0]
    assert planes_bytes == planes.tobytes()  # planes pass-through
    ref = np.zeros(POLICY_LEN, dtype="<f4")
    for i, v in zip(idx, val, strict=False):
        ref[i] = v
    assert policy_bytes == ref.tobytes()  # policy densifiée == dense de référence
    assert outcome == 1.0


def test_multi_positions():
    data = _encode([(_planes(0), [1], [1.0], 0.0), (_planes(2), [2, 5], [0.6, 0.4], -1.0)])
    assert len(decode(data)) == 2


def test_duplicate_index_last_wins():
    # Comportement spécifié : un index répété dans le creux → la dernière valeur gagne
    # (cohérent JS↔Py, parité préservée même sur un payload bizarre). Documenté par ce test.
    out = decode(_encode([(_planes(0), [3, 3], [0.4, 0.9], 0.0)]))
    policy = np.frombuffer(out[0][1], dtype="<f4")
    assert policy[3] == np.float32(0.9)


def test_max_positions_is_game_sized():
    # Un submit = une partie (≤ max_game_plies) : le cap borne l'amplification gzip→RAM.
    assert MAX_POSITIONS <= 1024
    with pytest.raises(SubmitDecodeError):  # juste au-dessus du cap → rejeté avant alloc
        decode(struct.pack("<I", MAX_POSITIONS + 1))


# --- fuzz / adversarial : SubmitDecodeError attendue, jamais de crash ---


def test_reject_too_many_positions():
    with pytest.raises(SubmitDecodeError):
        decode(struct.pack("<I", 10_000_000))  # > MAX_POSITIONS, buffer minuscule


def test_reject_truncated_header():
    with pytest.raises(SubmitDecodeError):
        decode(b"\x01")


def test_reject_truncated_position():
    with pytest.raises(SubmitDecodeError):
        decode(struct.pack("<I", 1) + struct.pack("<f", 0.0))  # annonce 1, tronqué


def test_reject_nnz_too_large():
    out = bytearray(struct.pack("<I", 1)) + struct.pack("<f", 0.0)
    out += _planes(0).tobytes() + struct.pack("<H", MAX_NNZ + 1)
    with pytest.raises(SubmitDecodeError):
        decode(bytes(out))


def test_reject_index_out_of_range():
    with pytest.raises(SubmitDecodeError):
        decode(_encode([(_planes(0), [POLICY_LEN], [0.5], 0.0)]))  # index == 4672


def test_reject_non_finite_outcome():
    out = bytearray(struct.pack("<I", 1)) + struct.pack("<f", float("nan"))
    out += _planes(0).tobytes() + struct.pack("<H", 0)
    with pytest.raises(SubmitDecodeError):
        decode(bytes(out))


def test_reject_non_finite_policy_value():
    with pytest.raises(SubmitDecodeError):
        decode(_encode([(_planes(0), [3], [float("inf")], 0.0)]))


def test_reject_trailing_bytes():
    with pytest.raises(SubmitDecodeError):
        decode(_encode([(_planes(0), [1], [1.0], 0.0)]) + b"\x00\x00")


def test_fuzz_random_garbage_no_crash():
    rng = np.random.default_rng(42)
    for _ in range(300):
        n = int(rng.integers(0, 256))
        data = rng.integers(0, 256, size=n, dtype=np.uint8).tobytes()
        try:
            decode(data)
        except SubmitDecodeError:
            pass
        except Exception as e:  # noqa: BLE001 — tout sauf SubmitDecodeError = défaut
            pytest.fail(f"exception non-maîtrisée sur garbage: {type(e).__name__}: {e}")
