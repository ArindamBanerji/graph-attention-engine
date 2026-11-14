"""Backend-agnostic StateBackend conformance tests."""

from __future__ import annotations

from threading import Barrier, Thread

import pytest

from gae.state import InMemoryStateBackend, VersionConflictError


@pytest.fixture
def backend() -> InMemoryStateBackend:
    """Return the reference backend under the shared protocol contract."""
    return InMemoryStateBackend()


def test_save_load_roundtrip(backend: InMemoryStateBackend) -> None:
    """A saved state is available with the version returned by save."""
    state = {"centroids": [[0.1, 0.2]], "counts": [3]}
    version = backend.save_category("scorer", "routing", state, "0")

    assert backend.load_category("scorer", "routing") == (state, version)


def test_cas_rejects_stale_version(backend: InMemoryStateBackend) -> None:
    """A stale CAS token cannot overwrite a newer category update."""
    version = backend.save_category("scorer", "routing", {"value": 1}, "0")
    backend.save_category("scorer", "routing", {"value": 2}, version)

    with pytest.raises(VersionConflictError):
        backend.save_category("scorer", "routing", {"value": 3}, version)


def test_cas_accepts_current_version(backend: InMemoryStateBackend) -> None:
    """The token read from a category authorizes its next update."""
    backend.save_category("scorer", "routing", {"value": 1}, "0")
    _, version = backend.load_category("scorer", "routing")

    new_version = backend.save_category("scorer", "routing", {"value": 2}, version)

    assert new_version != version
    assert backend.load_category("scorer", "routing")[0] == {"value": 2}


def test_concurrent_different_categories(backend: InMemoryStateBackend) -> None:
    """Independent categories can be saved concurrently from version zero."""
    barrier = Barrier(2)
    errors: list[Exception] = []

    def save(category: str) -> None:
        try:
            _, version = backend.load_category("scorer", category)
            barrier.wait()
            backend.save_category("scorer", category, {"category": category}, version)
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [Thread(target=save, args=(category,)) for category in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert set(backend.load_all("scorer")) == {"a", "b"}


def test_concurrent_same_category(backend: InMemoryStateBackend) -> None:
    """Only one concurrent CAS writer may commit a shared category version."""
    barrier = Barrier(2)
    successes: list[str] = []
    conflicts: list[VersionConflictError] = []

    def save(value: str) -> None:
        _, version = backend.load_category("scorer", "shared")
        barrier.wait()
        try:
            successes.append(backend.save_category("scorer", "shared", {"value": value}, version))
        except VersionConflictError as error:
            conflicts.append(error)

    threads = [Thread(target=save, args=(value,)) for value in ("one", "two")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(successes) == 1
    assert len(conflicts) == 1


def test_load_nonexistent_returns_empty(backend: InMemoryStateBackend) -> None:
    """Missing categories use the documented empty initial record."""
    assert backend.load_category("missing", "routing") == ({}, "0")


def test_save_creates_category(backend: InMemoryStateBackend) -> None:
    """Saving at initial version upserts a new category."""
    backend.save_category("scorer", "new", {"value": True}, "0")

    assert backend.load_category("scorer", "new")[0] == {"value": True}


def test_load_all(backend: InMemoryStateBackend) -> None:
    """load_all returns every category for the requested scorer only."""
    backend.save_category("scorer", "a", {"value": 1}, "0")
    backend.save_category("scorer", "b", {"value": 2}, "0")
    backend.save_category("other", "c", {"value": 3}, "0")

    states = backend.load_all("scorer")

    assert {category: state for category, (state, _) in states.items()} == {
        "a": {"value": 1},
        "b": {"value": 2},
    }


def test_list_scorers(backend: InMemoryStateBackend) -> None:
    """list_scorers includes each scorer that has persisted state."""
    backend.save_category("zeta", "a", {}, "0")
    backend.save_category("alpha", "a", {}, "0")

    assert backend.list_scorers() == ["alpha", "zeta"]


def test_version_increments(backend: InMemoryStateBackend) -> None:
    """Every successful save produces a distinct opaque version token."""
    version = "0"
    versions: list[str] = []
    for value in range(5):
        version = backend.save_category("scorer", "routing", {"value": value}, version)
        versions.append(version)

    assert len(set(versions)) == 5


def test_large_state(backend: InMemoryStateBackend) -> None:
    """A moderately sized category payload round-trips intact."""
    state = {f"key_{index}": {"value": index} for index in range(100)}
    backend.save_category("scorer", "routing", state, "0")

    assert backend.load_category("scorer", "routing")[0] == state
