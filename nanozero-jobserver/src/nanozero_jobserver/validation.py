"""validation.py — validation structurelle des soumissions browser (Story B.3).

Couche de défense légère APRÈS le décodage A.1 (``submit_codec``) : ce dernier garantit déjà
la FORME (longueur exacte, finitude, bornes ``index<4672`` / ``nnz≤218`` / ``outcome∈[−1,1]`` /
``num_positions``). B.3 ajoute la validité que le décodeur ne vérifie pas :

  - la **policy** doit être une distribution de probabilité : valeurs ≥ 0, somme ≈ 1
    (en self-play ``π = N_child / total`` → ``Σ = 1.0`` exact, ±1e-5 en f32) ;
  - les **planes** doivent être dans la plage encodable : tous les 119 plans NanoZero
    sont ≥ 0 (bitplans pièces/rép/color/castling = {0,1}, ``fullmove/99`` ∈ [0,1],
    ``halfmove/100`` ≈ [0,1.5]) — borne sup généreuse pour rejeter le garbage fini sans
    jamais rejeter de la data légitime.

STRUCTUREL UNIQUEMENT : pas de rejeu, pas de vérif de légalité des coups (posture light
assumée ; la vraie défense anti-poison = gate Epic D). Appliqué AU SEUL chemin browser —
le JSON fleet (workers Java de confiance) reste inchangé.
"""

from __future__ import annotations

import numpy as np

POLICY_LEN = 4672
PLANES_LEN = 7616

# Tolérance sur Σπ. En pratique Σ = 1.0 ± 1e-5 (somme de ≤218 ratios f32 N_child/total) ;
# ±0.02 est ultra-généreux → zéro faux rejet, rejette une policy non-normalisée (Σ=0.9, 50…).
POLICY_SUM_TOL = 0.02
# Plage des planes. Min = 0 (tous les plans NanoZero sont ≥ 0). Max légitime ≈ 1.5
# (halfmove 75-move) ; 10.0 = borne large anti-garbage (rejette 1e6 fini) sans faux rejet.
PLANE_MIN = 0.0
PLANE_MAX = 10.0


class BrowserValidationError(ValueError):
    """Soumission browser structurellement invalide — à mapper en 422 (Story B.3)."""


def validate_browser_positions(decoded: list[tuple[bytes, bytes, float]]) -> None:
    """Valide chaque position décodée d'un submit browser. Fail-closed au 1er échec.

    Args:
        decoded: sortie de ``submit_codec.decode`` — liste de
            ``(input_planes_bytes, policy_target_bytes, outcome)`` (f32 LE denses).
            La FORME (longueurs, finitude, bornes outcome/index) est déjà garantie par
            le décodeur ; on valide ici la SÉMANTIQUE structurelle.

    Raises:
        BrowserValidationError: policy non-distributionnelle (valeur < 0 ou Σ∉[1±tol])
            ou plane hors plage [PLANE_MIN, PLANE_MAX].
    """
    for i, (planes_bytes, policy_bytes, _outcome) in enumerate(decoded):
        planes = np.frombuffer(planes_bytes, dtype="<f4")
        # Plage des planes (longueur + finitude déjà garanties par submit_codec).
        if float(planes.min()) < PLANE_MIN or float(planes.max()) > PLANE_MAX:
            raise BrowserValidationError(
                f"position {i}: plane hors plage [{PLANE_MIN}, {PLANE_MAX}]"
                f" (min={float(planes.min()):.4g}, max={float(planes.max()):.4g})"
            )

        policy = np.frombuffer(policy_bytes, dtype="<f4")
        if float(policy.min()) < 0.0:
            raise BrowserValidationError(
                f"position {i}: policy avec valeur négative ({float(policy.min()):.4g})"
            )
        s = float(policy.sum())
        if not (1.0 - POLICY_SUM_TOL <= s <= 1.0 + POLICY_SUM_TOL):
            raise BrowserValidationError(
                f"position {i}: policy non normalisée (somme={s:.4f}, attendu 1±{POLICY_SUM_TOL})"
            )
    # outcome∈[−1,1] : déjà garanti au décodage (submit_codec). nb positions == nb plies :
    # structurel (ply = index 0-based via enumerate dans _positions_from_binary).
