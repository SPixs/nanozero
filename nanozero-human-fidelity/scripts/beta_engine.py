"""Moteur β du gate de validation (human-fidelity).

Calcule l'indice de complexité β d'une position via une **expansion 1-ply du
réseau** (proxy un-ply, DESIGN §4) — déterministe, batché, sans MCTS :
  - ``P(b)``      = policy prior du réseau à la racine (softmax sur coups légaux)
  - ``val(b)``    = valeur du coup b vue du TRAIT = −V(enfant) (1 forward/enfant)
  - ``β``         = regret pondéré policy vs une référence (cf. variantes §6.2.1)

Implémente TOUTES les variantes que le gate compare empiriquement
(``nanozero-web/docs/architecture.md §6.2.1``) :
  - **forme**       : ``'swing'`` (E[regret]) | ``'threshold'`` (masse-au-seuil Σ_{δ≥τ}P)
  - **tau**         : seuil de regret (forme threshold ; ignoré en swing)
  - **ref**         : ``'maxq'`` (meilleur coup par valeur réseau) | ``'policy'`` (argmax policy)
  - **naturalness** : pénalise quand le best move a une policy faible (quadrant traître §6)

Parité de FORMULE avec ``nanozero-web/core/complexity.mjs`` : ``beta_from_children``
reproduit exactement la sémantique du navigateur (regret moyen pondéré policy vs
la PV = coup le plus visité), via le MÊME ``compute_beta``. Le test vit dans
``beta_parity_test.py``. La seule différence en mode gate = l'INPUT (valeurs
1-ply du réseau ici, vs valeurs MCTS dans le navigateur) — un choix de modèle
documenté, pas une divergence de formule.

Réutilise l'encodage bit-perfect de ``nanozero_training`` (chess + numpy, sans torch).
"""

from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np
import numpy.typing as npt
import onnxruntime as ort

# Encodage partagé (parité Java, ADR-010) — ne dépend que de chess + numpy.
_TRAIN_SRC = Path(__file__).resolve().parents[2] / "nanozero-training" / "src"
if str(_TRAIN_SRC) not in sys.path:
    sys.path.insert(0, str(_TRAIN_SRC))
from nanozero_training.network.move_encoding import encode_move  # noqa: E402
from nanozero_training.network.position_encoding import encode_position  # noqa: E402


def board_with_history(fen: str, line: str | None) -> chess.Board:
    """Reconstruit un board AVEC son move_stack (historique NN 8-plies) depuis la ligne UCI.

    ⚠️ CRUCIAL : Nano est entraîné avec 8 demi-coups d'historique (plans temporels).
    Un `chess.Board(fen)` a un move_stack VIDE → input hors-distribution → la value head
    sous-évalue gravement (mesuré 2026-06-30 : −0.30 en score espéré). On rejoue donc la
    ligne pour peupler l'historique. Fallback sur FEN-nu si la ligne est absente/incohérente.
    """
    if not line:
        return chess.Board(fen)
    b = chess.Board()
    try:
        for u in line.split():
            b.push(chess.Move.from_uci(u))
    except (ValueError, AssertionError):
        return chess.Board(fen)
    return b if b.fen() == fen else chess.Board(fen)


def _softmax(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def wdl_value(value_logits: npt.NDArray[np.float64]) -> float:
    """V = P(W) − P(L) depuis les 3 logits WDL (ordre W/D/L, cf. export_onnx)."""
    p = _softmax(np.asarray(value_logits, dtype=np.float64))
    return float(p[0] - p[2])


class BetaEngine:
    """Expansion 1-ply du réseau ONNX pour calculer (P, val) à la racine."""

    def __init__(self, onnx_path: str | Path, intra_op: int = 1) -> None:
        so = ort.SessionOptions()
        so.intra_op_num_threads = intra_op
        so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(
            str(onnx_path), so, providers=["CPUExecutionProvider"]
        )
        self.in_name = self.sess.get_inputs()[0].name
        # Mappe les sorties par forme (robuste à l'ordre du graphe) : value = dim 3, policy = 4672.
        self.pol_name = None
        self.val_name = None
        for o in self.sess.get_outputs():
            last = o.shape[-1] if o.shape else None
            if last == 3:
                self.val_name = o.name
            elif last == 4672:
                self.pol_name = o.name
        if self.pol_name is None or self.val_name is None:  # fallback : ordre export [policy, value]
            outs = [o.name for o in self.sess.get_outputs()]
            self.pol_name, self.val_name = outs[0], outs[1]

    def _run(
        self, batch: npt.NDArray[np.float32]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        pol, val = self.sess.run([self.pol_name, self.val_name], {self.in_name: batch})
        return np.asarray(pol, dtype=np.float64), np.asarray(val, dtype=np.float64)

    def root_expand(
        self, board: chess.Board
    ) -> tuple[list[chess.Move], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """1-ply expansion. Retourne (moves, P, valmover).

        P[i]        = policy prior (softmax sur coups légaux),
        valmover[i] = valeur du coup i vue du TRAIT (−V de l'enfant ; +1 = gagne).
        Les enfants terminaux (mat/pat) sont résolus exactement, pas estimés.
        """
        legal = list(board.legal_moves)
        if len(legal) < 2:
            return legal, np.ones(len(legal)), np.zeros(len(legal))

        # Forward racine → policy sur les coups légaux.
        root_planes = encode_position(board)[None].astype(np.float32)
        pol, _ = self._run(root_planes)
        logits = pol[0]
        idxs = np.array([encode_move(m, board) for m in legal])
        P = _softmax(logits[idxs])

        # Valeurs des enfants : terminaux résolus exactement, le reste batché au réseau.
        v_child = np.zeros(len(legal))  # POV de l'enfant (= adversaire au trait à l'enfant)
        nonterminal_idx: list[int] = []
        planes_buf: list[npt.NDArray[np.float32]] = []
        for i, m in enumerate(legal):
            board.push(m)
            if board.is_checkmate():
                v_child[i] = -1.0  # le trait à l'enfant est maté → mauvais pour lui
            elif board.is_stalemate() or board.is_insufficient_material():
                v_child[i] = 0.0
            else:
                nonterminal_idx.append(i)
                planes_buf.append(encode_position(board))
            board.pop()
        if planes_buf:
            batch = np.stack(planes_buf).astype(np.float32)
            _, cval = self._run(batch)
            for j, i in enumerate(nonterminal_idx):
                v_child[i] = wdl_value(cval[j])

        valmover = -v_child  # POV trait à la racine
        return legal, P, valmover


def compute_beta(
    P: npt.NDArray[np.float64],
    valmover: npt.NDArray[np.float64],
    *,
    form: str = "swing",
    tau: float = 0.0,
    ref: str = "maxq",
    ref_idx: int | None = None,
    naturalness: float = 0.0,
) -> float:
    """β depuis (P, valmover). Voir le docstring du module pour les variantes.

    - ``ref_idx`` explicite (ex. l'index de la PV/coup le plus visité, pour la
      parité avec complexity.mjs) prime sur ``ref``.
    - ``form='swing'`` : β = Σ P·max(0, ref−val) / ΣP  (regret moyen pondéré policy).
    - ``form='threshold'`` : β = Σ_{ref−val ≥ τ} P / ΣP  (masse de policy perdante).
    """
    P = np.asarray(P, dtype=np.float64)
    valmover = np.asarray(valmover, dtype=np.float64)
    if P.size < 2:
        return 0.0
    if ref_idx is None:
        ref_idx = int(np.argmax(valmover)) if ref == "maxq" else int(np.argmax(P))
    refval = valmover[ref_idx]
    regret = np.maximum(0.0, refval - valmover)
    psum = float(P.sum())
    if psum <= 0:
        return 0.0
    if form == "threshold":
        beta = float(P[regret >= tau].sum() / psum)
    else:  # swing
        beta = float(np.sum(P * regret) / psum)
    if naturalness > 0.0:
        # Quadrant traître : si le meilleur coup est "peu naturel" (policy faible),
        # un humain ne le verra pas → augmenter β. Borné par la part de policy manquante.
        beta += naturalness * (1.0 - float(P[ref_idx]))
    return beta


def level_of(beta: float, calm: float = 0.10, tense: float = 0.25) -> str:
    """Mappe β → niveau (mêmes seuils >= que complexity.mjs)."""
    return "sharp" if beta >= tense else "tense" if beta >= calm else "calm"


def beta_from_children(
    children: list[dict[str, float]], calm: float = 0.10, tense: float = 0.25
) -> dict[str, float | str]:
    """Réplique EXACTE de nanozero-web/core/complexity.mjs (input = children MCTS).

    children : liste de {'P': prior, 'N': visites, 'Q': valeur POV enfant}.
    Sert le test de parité de formule. Utilise le MÊME ``compute_beta`` que le gate.
    """
    visited = [c for c in children if c.get("N", 0) > 0]
    if len(visited) < 2:
        return {"beta": 0.0, "level": "calm"}
    P = np.array([c["P"] for c in visited], dtype=np.float64)
    valmover = np.array([-c["Q"] for c in visited], dtype=np.float64)  # POV trait
    pv_idx = int(np.argmax([c["N"] for c in visited]))  # PV = coup le plus visité
    beta = compute_beta(P, valmover, form="swing", tau=0.0, ref_idx=pv_idx)
    return {"beta": beta, "level": level_of(beta, calm, tense)}
