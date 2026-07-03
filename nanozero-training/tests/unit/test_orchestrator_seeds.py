"""Unit tests for orchestrator seed helpers."""

from __future__ import annotations

import numpy as np
from nanozero_training.selfplay.orchestrator import derive_dirichlet_seed, make_game_rngs


def test_make_game_rngs_returns_n_generators() -> None:
    rngs = make_game_rngs(worker_seed=42, n_games=10)
    assert len(rngs) == 10
    for r in rngs:
        assert isinstance(r, np.random.Generator)


def test_make_game_rngs_reproducible_same_seed() -> None:
    """Same worker_seed -> equivalent generators bit-perfect."""
    rngs_a = make_game_rngs(worker_seed=42, n_games=5)
    rngs_b = make_game_rngs(worker_seed=42, n_games=5)
    # Pour chaque paire (a, b) idx i : premiers draws identiques.
    for ra, rb in zip(rngs_a, rngs_b, strict=True):
        assert ra.random() == rb.random()


def test_make_game_rngs_different_seeds_diverge() -> None:
    """Different worker_seed -> different first draws."""
    rng_a = make_game_rngs(worker_seed=42, n_games=1)[0]
    rng_b = make_game_rngs(worker_seed=43, n_games=1)[0]
    assert rng_a.random() != rng_b.random()


def test_make_game_rngs_decorrelated() -> None:
    """100 child RNGs -> tous distincts (sanity decorrelation)."""
    rngs = make_game_rngs(worker_seed=42, n_games=100)
    first_draws = [r.random() for r in rngs]
    assert len(set(first_draws)) == 100


def test_make_game_rngs_n_games_zero() -> None:
    """Edge case : n_games=0 -> liste vide."""
    rngs = make_game_rngs(worker_seed=42, n_games=0)
    assert rngs == []


def test_derive_dirichlet_seed_in_int32_range() -> None:
    """Pour 1000 RNGs, derived seeds ∈ [0, 2^31 - 1]."""
    rngs = make_game_rngs(worker_seed=42, n_games=1000)
    seeds = [derive_dirichlet_seed(r) for r in rngs]
    for s in seeds:
        assert 0 <= s < 2**31


def test_derive_dirichlet_seed_reproducible() -> None:
    """Même RNG (même seed initial) -> même seed dérivé."""
    rngs_a = make_game_rngs(worker_seed=42, n_games=1)
    rngs_b = make_game_rngs(worker_seed=42, n_games=1)
    s_a = derive_dirichlet_seed(rngs_a[0])
    s_b = derive_dirichlet_seed(rngs_b[0])
    assert s_a == s_b


def test_make_game_rngs_resume_consistency() -> None:
    """Critique pour resume : make_game_rngs(42, 500)[300] == make_game_rngs(42, 500)[300].

    Garantit que peu importe quand on génère les RNGs, le RNG pour la partie
    k est toujours le même (bit-perfect reproducibility au resume).
    """
    rngs_initial = make_game_rngs(worker_seed=42, n_games=500)
    rngs_resume = make_game_rngs(worker_seed=42, n_games=500)
    # Pour la 300e partie : RNG identique.
    assert rngs_initial[300].random() == rngs_resume[300].random()
