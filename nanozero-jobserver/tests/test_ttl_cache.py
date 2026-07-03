"""Tests for ttl_cache.TTLCache (#6 BMAD)."""

from __future__ import annotations

from nanozero_jobserver.ttl_cache import TTLCache


class _Clock:
    """Horloge contrôlable pour tester l'expiration sans sleep."""

    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_get_miss_returns_none() -> None:
    cache: TTLCache[str, int] = TTLCache(10.0)
    assert cache.get("absent") is None


def test_set_then_get_hit() -> None:
    cache: TTLCache[str, int] = TTLCache(10.0)
    cache.set("x", 42)
    assert cache.get("x") == 42


def test_entry_expires_after_ttl() -> None:
    clock = _Clock()
    cache: TTLCache[str, int] = TTLCache(10.0, clock=clock)
    cache.set("x", 42)
    clock.t = 9.9
    assert cache.get("x") == 42  # encore valide
    clock.t = 10.1
    assert cache.get("x") is None  # expiré → purgé


def test_clear_empties_cache() -> None:
    cache: TTLCache[str, int] = TTLCache(10.0)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.clear()
    assert cache.get("a") is None
    assert cache.get("b") is None
