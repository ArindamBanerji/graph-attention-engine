"""Protocol for category-scoped, compare-and-swap scorer persistence."""

from __future__ import annotations

from typing import Protocol


class VersionConflictError(Exception):
    """Raised when a category CAS version does not match the current version."""


class StateBackend(Protocol):
    """Persist scorer state with independent optimistic concurrency per category.

    Version tokens are backend-defined opaque strings.  Callers must use the
    token returned by a load operation for the corresponding category and
    reload the category before retrying after :class:`VersionConflictError`.
    """

    def load_category(self, scorer_id: str, category: str) -> tuple[dict, str]:
        """Return a category state dictionary and its opaque version token.

        A missing category is represented by ``({}, "0")``.
        """
        ...

    def save_category(
        self, scorer_id: str, category: str, state: dict, version: str
    ) -> str:
        """CAS-upsert category state and return its new opaque version token.

        Raises :class:`VersionConflictError` when *version* is stale.
        """
        ...

    def load_all(self, scorer_id: str) -> dict[str, tuple[dict, str]]:
        """Return every persisted category state and version for a scorer."""
        ...

    def list_scorers(self) -> list[str]:
        """Return all scorer identifiers that have persisted categories."""
        ...
