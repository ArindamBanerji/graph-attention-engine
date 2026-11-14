"""Redis StateBackend conformance tests using fakeredis when available."""

from __future__ import annotations

from threading import Barrier, Thread
from typing import Any

import pytest

fakeredis = pytest.importorskip("fakeredis")

from gae.state.redis_backend import RedisStateBackend
from gae.state.protocol import VersionConflictError


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> RedisStateBackend:
    """Construct a Redis backend against an isolated fakeredis instance."""
    client = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(
        "gae.state.redis_backend.redis.Redis.from_url",
        lambda _url, **_kwargs: client,
    )
    return RedisStateBackend()


def test_save_load_roundtrip(backend: RedisStateBackend) -> None:
    """A JSON state round-trips with the version returned by save."""
    state = {"centroids": [[0.1, 0.2]], "counts": [3]}
    version = backend.save_category("scorer", "routing", state, "0")

    assert backend.load_category("scorer", "routing") == (state, version)


def test_cas_rejects_stale_version(backend: RedisStateBackend) -> None:
    """A stale category token cannot overwrite a newer value."""
    version = backend.save_category("scorer", "routing", {"value": 1}, "0")
    backend.save_category("scorer", "routing", {"value": 2}, version)

    with pytest.raises(VersionConflictError):
        backend.save_category("scorer", "routing", {"value": 3}, version)


def test_cas_accepts_current_version(backend: RedisStateBackend) -> None:
    """The loaded token authorizes the next category update."""
    backend.save_category("scorer", "routing", {"value": 1}, "0")
    _, version = backend.load_category("scorer", "routing")

    new_version = backend.save_category("scorer", "routing", {"value": 2}, version)

    assert new_version != version
    assert backend.load_category("scorer", "routing")[0] == {"value": 2}


def test_concurrent_different_categories(backend: RedisStateBackend) -> None:
    """Separate category writes both succeed when started together."""
    barrier = Barrier(2)
    errors: list[Exception] = []

    def save(category: str) -> None:
        try:
            _, version = backend.load_category("scorer", category)
            barrier.wait(timeout=10)
            backend.save_category("scorer", category, {"category": category}, version)
        except Exception as error:  # pragma: no cover - asserted below
            errors.append(error)

    threads = [Thread(target=save, args=(category,)) for category in ("a", "b")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert errors == []
    assert set(backend.load_all("scorer")) == {"a", "b"}


def test_concurrent_same_category(backend: RedisStateBackend) -> None:
    """Exactly one concurrent same-category CAS write commits."""
    barrier = Barrier(2)
    successes: list[str] = []
    conflicts: list[VersionConflictError] = []

    def save(value: str) -> None:
        _, version = backend.load_category("scorer", "shared")
        barrier.wait(timeout=10)
        try:
            successes.append(backend.save_category("scorer", "shared", {"value": value}, version))
        except VersionConflictError as error:
            conflicts.append(error)

    threads = [Thread(target=save, args=(value,)) for value in ("one", "two")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    assert all(not thread.is_alive() for thread in threads)
    assert len(successes) == 1
    assert len(conflicts) == 1


def test_load_nonexistent_returns_empty(backend: RedisStateBackend) -> None:
    """A missing Redis field uses the protocol initial record."""
    assert backend.load_category("missing", "routing") == ({}, "0")


def test_save_creates_category(backend: RedisStateBackend) -> None:
    """An initial-version save creates the containing hash and its field."""
    backend.save_category("scorer", "new", {"value": True}, "0")

    assert backend.load_category("scorer", "new")[0] == {"value": True}


def test_load_all(backend: RedisStateBackend) -> None:
    """load_all returns only fields for the requested scorer hash."""
    backend.save_category("scorer", "a", {"value": 1}, "0")
    backend.save_category("scorer", "b", {"value": 2}, "0")
    backend.save_category("other", "c", {"value": 3}, "0")

    states = backend.load_all("scorer")

    assert {category: state for category, (state, _) in states.items()} == {
        "a": {"value": 1},
        "b": {"value": 2},
    }


def test_list_scorers(backend: RedisStateBackend) -> None:
    """list_scorers strips the Redis namespace prefix."""
    backend.save_category("zeta", "a", {}, "0")
    backend.save_category("alpha", "a", {}, "0")

    assert backend.list_scorers() == ["alpha", "zeta"]


def test_version_increments(backend: RedisStateBackend) -> None:
    """Each successful write receives a distinct token."""
    version = "0"
    versions: list[str] = []
    for value in range(5):
        version = backend.save_category("scorer", "routing", {"value": value}, version)
        versions.append(version)

    assert len(set(versions)) == 5


def test_large_state(backend: RedisStateBackend) -> None:
    """A 100-key JSON payload round-trips without data loss."""
    state = {f"key_{index}": {"value": index} for index in range(100)}
    backend.save_category("scorer", "routing", state, "0")

    assert backend.load_category("scorer", "routing")[0] == state


def test_json_serialization(backend: RedisStateBackend) -> None:
    """Nested JSON-compatible scorer state survives Redis serialization."""
    state: dict[str, Any] = {
        "centroids": [[0.1, 0.2], [0.3, 0.4]],
        "counts": [3, 7],
        "metadata": {"phase": "learning", "history": [True, False]},
    }
    backend.save_category("scorer", "routing", state, "0")

    assert backend.load_category("scorer", "routing")[0] == state
