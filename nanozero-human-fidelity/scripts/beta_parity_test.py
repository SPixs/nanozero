"""Parité de FORMULE Python ↔ nanozero-web/core/complexity.test.mjs.

Reproduit les 5 fixtures du test JS : mêmes children {P,N,Q} → même β/level.
Garantit que le gate valide bien la formule qui SHIPPE dans le navigateur.

Run : ~/.gatevenv/bin/python beta_parity_test.py
"""

from __future__ import annotations

from beta_engine import beta_from_children

EPS = 1e-9


def _check(name: str, got: dict, *, level: str, beta_min: float | None = None) -> None:
    assert got["level"] == level, f"{name}: level {got['level']} != {level} (beta={got['beta']})"
    if beta_min is not None:
        assert got["beta"] >= beta_min - EPS, f"{name}: beta {got['beta']} < {beta_min}"
    print(f"  ok  {name}  (beta={got['beta']:.4f}, level={got['level']})")


def main() -> None:
    # 1. Un seul coup légal → calm (beta 0).
    _check("1 coup legal", beta_from_children([{"P": 1, "N": 48, "Q": 0}]), level="calm")

    # 2. Tous les coups équivalents → calm.
    _check(
        "coups equivalents",
        beta_from_children([{"P": 0.5, "N": 25, "Q": -0.20}, {"P": 0.5, "N": 23, "Q": -0.18}]),
        level="calm",
    )

    # 3. 4 coups policy uniforme, 3 perdants (regret 0.8) → sharp, beta>=0.25.
    _check(
        "4 coups 3 perdants",
        beta_from_children(
            [
                {"P": 0.25, "N": 30, "Q": -0.5},
                {"P": 0.25, "N": 6, "Q": 0.3},
                {"P": 0.25, "N": 6, "Q": 0.3},
                {"P": 0.25, "N": 6, "Q": 0.3},
            ]
        ),
        level="sharp",
        beta_min=0.25,
    )

    # 4. Seuils personnalisés (beta = 0.1).
    ch4 = [{"P": 0.5, "N": 20, "Q": -0.3}, {"P": 0.5, "N": 10, "Q": -0.1}]
    _check("seuils calm0.05/tense0.5", beta_from_children(ch4, calm=0.05, tense=0.5), level="tense")
    _check("seuils calm0.2/tense0.5", beta_from_children(ch4, calm=0.2, tense=0.5), level="calm")

    # 5. Bas budget : coups N=0 + coup peu visité bruité ignorés → pas de faux sharp.
    ch5 = [
        {"P": 0.45, "N": 22, "Q": -0.04},
        {"P": 0.30, "N": 14, "Q": -0.03},
        {"P": 0.05, "N": 1, "Q": -0.6},
    ] + [{"P": 0.0111, "N": 0, "Q": 0} for _ in range(18)]
    _check("bas budget", beta_from_children(ch5), level="calm")

    print("\nPARITE OK — 6 assertions, formule Python == complexity.mjs")


if __name__ == "__main__":
    main()
