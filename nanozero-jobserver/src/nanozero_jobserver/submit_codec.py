"""submit_codec.py — décodeur binaire des soumissions self-play (Story A.1).

Format compact pour le submit navigateur (alternative au JSON base64 du worker Java).
Layout LITTLE-ENDIAN (cf. ``docs/architecture-jobserver-connection.md``)::

    uint32 num_positions
    par position :
        float32       outcome
        float32[7616] planes
        uint16        nnz
        nnz × (uint16 index<4672, float32 value)   # policy CREUSE

``decode`` reconstruit des bytes denses IDENTIQUES au chemin JSON base64 :

    input_planes  = 7616 × f32 LE (pass-through)
    policy_target = 4672 × f32 LE (policy creuse densifiée)

Décodeur DURCI : le payload est NON-FIABLE (navigateur public). Toutes les bornes sont
validées au plus tôt — ``num_positions`` est plafonné AVANT la boucle, et chaque position
est bornée avant lecture — pour qu'aucun payload hostile ne provoque crash ni OOM.
"""

from __future__ import annotations

import struct

import numpy as np

PLANES_LEN = 7616
POLICY_LEN = 4672
PLANES_BYTES = PLANES_LEN * 4
# Un submit = UNE partie (positions = une par ply). Une partie d'échecs self-play est bornée par
# max_game_plies (≤ 512). Plafonner ici borne l'amplification gzip→RAM : 512 × ~30,5 Ko ≈ 15,6 Mo
# max par requête (vs ~125 Mo à 4096). Le cap du body DÉ-gzippé reste à poser côté middleware/gateway
# (A.2/B.2) — le décodeur ne voit que des bytes déjà inflatés.
MAX_POSITIONS = 512
MAX_NNZ = 218  # nb maximal de coups légaux dans une position d'échecs


class SubmitDecodeError(ValueError):
    """Payload binaire invalide ou hostile — à mapper en 4xx par le endpoint (Story A.2)."""


def decode(data: bytes) -> list[tuple[bytes, bytes, float]]:
    """Décode un body binaire en positions denses.

    Args:
        data: corps binaire (déjà dé-gzippé par le middleware).

    Returns:
        Liste de ``(input_planes, policy_target, outcome)`` où ``input_planes`` et
        ``policy_target`` sont des ``bytes`` f32 little-endian (7616 et 4672 floats),
        identiques au chemin JSON base64.

    Raises:
        SubmitDecodeError: sur tout payload malformé. ``num_positions`` est validé
            avant toute allocation de position.
    """
    mv = memoryview(data)
    total = len(mv)
    if total < 4:
        raise SubmitDecodeError("buffer trop court pour l'en-tête")
    (num_positions,) = struct.unpack_from("<I", mv, 0)
    if num_positions > MAX_POSITIONS:
        raise SubmitDecodeError(f"num_positions={num_positions} > {MAX_POSITIONS}")

    off = 4
    out: list[tuple[bytes, bytes, float]] = []
    for _ in range(num_positions):
        # En-tête de position : outcome(4) + planes + nnz(2) — borné avant lecture.
        if off + 4 + PLANES_BYTES + 2 > total:
            raise SubmitDecodeError("buffer tronqué (position)")

        (outcome,) = struct.unpack_from("<f", mv, off)
        off += 4
        if not np.isfinite(outcome) or outcome < -1.0 or outcome > 1.0:
            raise SubmitDecodeError("outcome hors bornes [-1,1] ou non-fini")

        planes_bytes = bytes(mv[off : off + PLANES_BYTES])
        off += PLANES_BYTES
        if not np.all(np.isfinite(np.frombuffer(planes_bytes, dtype="<f4"))):
            raise SubmitDecodeError("planes non-finies")

        (nnz,) = struct.unpack_from("<H", mv, off)
        off += 2
        if nnz > MAX_NNZ:
            raise SubmitDecodeError(f"nnz={nnz} > {MAX_NNZ}")
        if off + nnz * 6 > total:
            raise SubmitDecodeError("buffer tronqué (policy creuse)")

        policy = np.zeros(POLICY_LEN, dtype="<f4")
        for _j in range(nnz):
            (index,) = struct.unpack_from("<H", mv, off)
            off += 2
            (value,) = struct.unpack_from("<f", mv, off)
            off += 4
            if index >= POLICY_LEN:
                raise SubmitDecodeError(f"index policy {index} >= {POLICY_LEN}")
            if not np.isfinite(value):
                raise SubmitDecodeError("valeur policy non-finie")
            policy[index] = value

        out.append((planes_bytes, policy.tobytes(), float(outcome)))

    if off != total:
        raise SubmitDecodeError(f"longueur incorrecte : {off} octets consommés sur {total}")
    return out
