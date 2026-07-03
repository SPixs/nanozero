"""Cache TTL en mémoire pour amortir les agrégats lourds en lecture seule (#6 revue BMAD).

``/stats/drift`` recompute ``drift_report`` (scan ~34M lignes → minutes) à chaque appel ;
interrogé en polling par le monitoring, c'est intenable. Un cache par clé (model_version)
à TTL court sert les appels répétés en O(1) sans recalcul, la donnée bougeant lentement.

Instance PAR APP (``app.state``) → isolation entre tests. Horloge injectable pour tester
l'expiration sans ``sleep``. Non thread-safe au-delà de l'atomicité des dict CPython —
suffisant pour des lectures depuis le threadpool FastAPI (au pire, deux recalculs en parallèle).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    """Cache clé→valeur avec expiration par entrée (TTL en secondes)."""

    def __init__(self, ttl_seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self._ttl = ttl_seconds
        self._clock = clock
        self._store: dict[K, tuple[float, V]] = {}

    def get(self, key: K) -> V | None:
        """Valeur si présente ET non expirée, sinon None (et purge l'entrée expirée)."""
        entry = self._store.get(key)
        if entry is None:
            return None
        ts, value = entry
        if self._clock() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: K, value: V) -> None:
        self._store[key] = (self._clock(), value)

    def clear(self) -> None:
        self._store.clear()
