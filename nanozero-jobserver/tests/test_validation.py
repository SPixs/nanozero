"""B.3 — unit tests de validation.validate_browser_positions (validation structurelle)."""

from __future__ import annotations

import numpy as np
import pytest
from nanozero_jobserver.validation import (
    PLANE_MAX,
    POLICY_LEN,
    POLICY_SUM_TOL,
    BrowserValidationError,
    validate_browser_positions,
)

PLANES_LEN = 7616


def _planes(value: float = 0.0) -> bytes:
    """Planes denses : tout à `value` (0.0 = valide, dans [0, PLANE_MAX])."""
    return np.full(PLANES_LEN, value, dtype="<f4").tobytes()


def _policy(pairs: list[tuple[int, float]]) -> bytes:
    """Policy dense f32 LE à partir de (index, value)."""
    p = np.zeros(POLICY_LEN, dtype="<f4")
    for idx, val in pairs:
        p[idx] = val
    return p.tobytes()


def _decoded(planes: bytes, policy: bytes, outcome: float = 0.0):
    return [(planes, policy, outcome)]


def test_valid_passes() -> None:
    # Σπ = 1.0, planes à 0/1 → OK
    validate_browser_positions(_decoded(_planes(1.0), _policy([(3, 0.6), (17, 0.4)])))


def test_policy_sum_within_tolerance_passes() -> None:
    validate_browser_positions(_decoded(_planes(), _policy([(1, 0.99)])))  # Σ=0.99 ∈ [1±0.02]


def test_policy_sum_too_low_rejected() -> None:
    with pytest.raises(BrowserValidationError, match="non normalisée"):
        validate_browser_positions(_decoded(_planes(), _policy([(1, 0.6), (2, 0.3)])))  # Σ=0.9


def test_policy_sum_too_high_rejected() -> None:
    with pytest.raises(BrowserValidationError, match="non normalisée"):
        validate_browser_positions(_decoded(_planes(), _policy([(1, 50.0)])))


def test_policy_negative_value_rejected() -> None:
    # Σ = 1.0 mais une valeur négative → rejet (négativité testée avant la somme)
    with pytest.raises(BrowserValidationError, match="négative"):
        validate_browser_positions(_decoded(_planes(), _policy([(1, 1.2), (2, -0.2)])))


def test_plane_above_max_rejected() -> None:
    with pytest.raises(BrowserValidationError, match="plane hors plage"):
        validate_browser_positions(_decoded(_planes(PLANE_MAX + 1.0), _policy([(1, 1.0)])))


def test_plane_negative_rejected() -> None:
    with pytest.raises(BrowserValidationError, match="plane hors plage"):
        validate_browser_positions(_decoded(_planes(-1.0), _policy([(1, 1.0)])))


def test_plane_at_max_boundary_passes() -> None:
    validate_browser_positions(_decoded(_planes(PLANE_MAX), _policy([(1, 1.0)])))


def test_first_bad_position_reported() -> None:
    good = (_planes(), _policy([(1, 1.0)]), 0.0)
    bad = (_planes(), _policy([(1, 0.5)]), 0.0)  # Σ=0.5
    with pytest.raises(BrowserValidationError, match="position 1"):
        validate_browser_positions([good, bad])


def test_tolerance_constant_sane() -> None:
    assert 0.0 < POLICY_SUM_TOL < 0.1  # garde-fou : tolérance raisonnable
