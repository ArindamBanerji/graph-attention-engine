"""Thread-safe in-memory reference implementation of :mod:`gae.state`."""

from __future__ import annotations

from copy import deepcopy
from threading import Lock
from typing import Any

from gae.state.protocol import VersionConflictError


class InMemoryStateBackend:
    """Dict-backed StateBackend implementation for tests and local use.

    Each ``(scorer_id, category)`` record has its own version token.  The
    single lock keeps the reference implementation safe for concurrent test
    threads while the version check makes writes optimistic and category-local.
    """

    def __init__(self) -> None:
        self._states: dict[str, dict[str, tuple[dict[str, Any], str]]] = {}
        self._lock = Lock()

    def load_category(self, scorer_id: str, category: str) -> tuple[dict, str]:
        """Return a defensive copy of one category state and its version."""
        with self._lock:
            record = self._states.get(scorer_id, {}).get(category)
            if record is None:
                return {}, "0"
            state, version = record
            return deepcopy(state), version

    def save_category(
        self, scorer_id: str, category: str, state: dict, version: str
    ) -> str:
        """CAS-upsert a category state and return its incremented version."""
        with self._lock:
            categories = self._states.setdefault(scorer_id, {})
            current = categories.get(category)
            current_version = "0" if current is None else current[1]
            if version != current_version:
                raise VersionConflictError(
                    f"stale version for scorer={scorer_id!r}, category={category!r}: "
                    f"expected {current_version!r}, received {version!r}"
                )
            new_version = str(int(current_version) + 1)
            categories[category] = (deepcopy(state), new_version)
            return new_version

    def load_all(self, scorer_id: str) -> dict[str, tuple[dict, str]]:
        """Return defensive copies of all states for *scorer_id*."""
        with self._lock:
            return {
                category: (deepcopy(state), version)
                for category, (state, version) in self._states.get(scorer_id, {}).items()
            }

    def list_scorers(self) -> list[str]:
        """Return scorer IDs in deterministic order."""
        with self._lock:
            return sorted(self._states)
