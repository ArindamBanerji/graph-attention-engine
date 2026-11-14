"""PostgreSQL StateBackend conformance tests with a deterministic DB-API fake."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

pytest.importorskip("psycopg2")

from gae.state import pg_backend
from gae.state.pg_backend import PostgresStateBackend
from gae.state.protocol import VersionConflictError


class _FakeCursor:
    def __init__(self, connection: _FakeConnection) -> None:
        self._connection = connection
        self._rows: list[tuple[Any, ...]] = []
        self.rowcount = -1

    def execute(self, query: str, params: tuple[Any, ...] | None = None) -> None:
        normalized = " ".join(query.split()).upper()
        values = params or ()
        store = self._connection.store
        if normalized.startswith("CREATE TABLE"):
            self._connection.table_created = True
        elif normalized.startswith("SELECT STATE, VERSION"):
            record = store.get((str(values[0]), str(values[1])))
            self._rows = [] if record is None else [(deepcopy(record[0]), record[1])]
        elif normalized.startswith("UPDATE"):
            payload, scorer_id, category, expected = values
            record = store.get((str(scorer_id), str(category)))
            if record is None or record[1] != int(expected):
                self._rows = []
                self.rowcount = 0
            else:
                next_version = record[1] + 1
                store[(str(scorer_id), str(category))] = (
                    deepcopy(_unwrap_json(payload)),
                    next_version,
                )
                self._rows = [(next_version,)]
                self.rowcount = 1
        elif normalized.startswith("SELECT 1"):
            self._rows = [
                (1,)
            ] if (str(values[0]), str(values[1])) in store else []
        elif normalized.startswith("INSERT INTO"):
            scorer_id, category, payload = values
            key = (str(scorer_id), str(category))
            if key in store:
                raise pg_backend.psycopg2.IntegrityError("duplicate category")
            store[key] = (deepcopy(_unwrap_json(payload)), 1)
            self._rows = [(1,)]
            self.rowcount = 1
        elif normalized.startswith("SELECT CATEGORY, STATE, VERSION"):
            self._rows = [
                (category, deepcopy(state), version)
                for (scorer_id, category), (state, version) in store.items()
                if scorer_id == str(values[0])
            ]
        elif normalized.startswith("SELECT DISTINCT SCORER_ID"):
            self._rows = [(scorer_id,) for scorer_id in sorted({key[0] for key in store})]
        else:  # pragma: no cover - detects accidental query changes
            raise AssertionError(f"Unexpected SQL: {query}")

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._rows[0] if self._rows else None

    def fetchall(self) -> list[tuple[Any, ...]]:
        return self._rows

    def close(self) -> None:
        pass


class _FakeConnection:
    def __init__(self) -> None:
        self.store: dict[tuple[str, str], tuple[dict[str, Any], int]] = {}
        self.autocommit = True
        self.table_created = False
        self.commit_count = 0
        self.rollback_count = 0
        self.closed = False

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.closed = True


def _unwrap_json(value: Any) -> Any:
    return getattr(value, "adapted", value)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> tuple[PostgresStateBackend, _FakeConnection]:
    """Create an isolated backend while exercising the DB-API SQL path."""
    connection = _FakeConnection()
    monkeypatch.setattr(pg_backend.psycopg2, "connect", lambda _dsn: connection)
    return PostgresStateBackend("postgresql://test/rgi"), connection


def test_save_load_roundtrip(backend: tuple[PostgresStateBackend, _FakeConnection]) -> None:
    store, _ = backend
    state = {"centroids": [[0.1, 0.2]], "counts": [3]}
    version = store.save_category("scorer", "routing", state, "0")
    assert store.load_category("scorer", "routing") == (state, version)


def test_cas_rejects_stale_version(backend: tuple[PostgresStateBackend, _FakeConnection]) -> None:
    store, _ = backend
    version = store.save_category("scorer", "routing", {"value": 1}, "0")
    store.save_category("scorer", "routing", {"value": 2}, version)
    with pytest.raises(VersionConflictError):
        store.save_category("scorer", "routing", {"value": 3}, version)


def test_cas_accepts_current_version(backend: tuple[PostgresStateBackend, _FakeConnection]) -> None:
    store, _ = backend
    store.save_category("scorer", "routing", {"value": 1}, "0")
    _, version = store.load_category("scorer", "routing")
    next_version = store.save_category("scorer", "routing", {"value": 2}, version)
    assert next_version == "2"
    assert store.load_category("scorer", "routing")[0] == {"value": 2}


def test_load_nonexistent_returns_empty(backend: tuple[PostgresStateBackend, _FakeConnection]) -> None:
    assert backend[0].load_category("missing", "routing") == ({}, "0")


def test_save_creates_category(backend: tuple[PostgresStateBackend, _FakeConnection]) -> None:
    store, _ = backend
    store.save_category("scorer", "new", {"value": True}, "0")
    assert store.load_category("scorer", "new")[0] == {"value": True}


def test_load_all(backend: tuple[PostgresStateBackend, _FakeConnection]) -> None:
    store, _ = backend
    store.save_category("scorer", "a", {"value": 1}, "0")
    store.save_category("scorer", "b", {"value": 2}, "0")
    store.save_category("other", "c", {"value": 3}, "0")
    assert {category: state for category, (state, _) in store.load_all("scorer").items()} == {
        "a": {"value": 1},
        "b": {"value": 2},
    }


def test_list_scorers(backend: tuple[PostgresStateBackend, _FakeConnection]) -> None:
    store, _ = backend
    store.save_category("zeta", "a", {}, "0")
    store.save_category("alpha", "a", {}, "0")
    assert store.list_scorers() == ["alpha", "zeta"]


def test_version_increments(backend: tuple[PostgresStateBackend, _FakeConnection]) -> None:
    store, _ = backend
    version = "0"
    versions: list[str] = []
    for value in range(5):
        version = store.save_category("scorer", "routing", {"value": value}, version)
        versions.append(version)
    assert versions == ["1", "2", "3", "4", "5"]


def test_large_state(backend: tuple[PostgresStateBackend, _FakeConnection]) -> None:
    store, _ = backend
    state = {f"key_{index}": {"value": index} for index in range(100)}
    store.save_category("scorer", "routing", state, "0")
    assert store.load_category("scorer", "routing")[0] == state


def test_table_auto_creation(backend: tuple[PostgresStateBackend, _FakeConnection]) -> None:
    _, connection = backend
    assert connection.table_created is True


def test_table_name_must_be_a_safe_identifier() -> None:
    with pytest.raises(ValueError, match="simple PostgreSQL identifier"):
        PostgresStateBackend(table="rgi_state; DROP TABLE rgi_state")
