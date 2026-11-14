"""PostgreSQL-backed category persistence with optimistic CAS writes."""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

import psycopg2  # type: ignore[import]
import psycopg2.extras  # type: ignore[import]

from gae.state.protocol import VersionConflictError


class PostgresStateBackend:
    """Persist category state in PostgreSQL using a versioned JSONB record.

    Each ``(scorer_id, category)`` has an independent ``BIGINT`` version.
    Updates use ``UPDATE ... WHERE version = %s`` so a stale writer cannot
    overwrite a newer category state.
    """

    def __init__(
        self,
        dsn: str = "postgresql://localhost:5432/rgi",
        table: str = "rgi_state",
    ) -> None:
        """Connect to PostgreSQL and create the state table if needed."""
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table):
            raise ValueError("table must be a simple PostgreSQL identifier")
        self._table = table
        self._conn: Any = psycopg2.connect(dsn)
        self._conn.autocommit = False
        self._ensure_table()

    def load_category(self, scorer_id: str, category: str) -> tuple[dict, str]:
        """Load one category state or return the protocol's empty record."""
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                f"SELECT state, version FROM {self._table} "
                "WHERE scorer_id = %s AND category = %s",
                (scorer_id, category),
            )
            row = cursor.fetchone()
            if row is None:
                self._conn.commit()
                return {}, "0"
            result = deepcopy(dict(row[0])), str(row[1])
            self._conn.commit()
            return result
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def save_category(
        self, scorer_id: str, category: str, state: dict, version: str
    ) -> str:
        """CAS-upsert *state* and return the next PostgreSQL version token."""
        expected_version = self._parse_version(version)
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                f"UPDATE {self._table} "
                "SET state = %s, version = version + 1, updated_at = now() "
                "WHERE scorer_id = %s AND category = %s AND version = %s "
                "RETURNING version",
                (
                    psycopg2.extras.Json(deepcopy(state)),
                    scorer_id,
                    category,
                    expected_version,
                ),
            )
            row = cursor.fetchone()
            if row is not None:
                self._conn.commit()
                return str(row[0])

            cursor.execute(
                f"SELECT 1 FROM {self._table} "
                "WHERE scorer_id = %s AND category = %s",
                (scorer_id, category),
            )
            if cursor.fetchone() is not None or expected_version != 0:
                self._conn.commit()
                raise self._conflict(scorer_id, category, version)

            try:
                cursor.execute(
                    f"INSERT INTO {self._table} "
                    "(scorer_id, category, state, version) "
                    "VALUES (%s, %s, %s, 1) RETURNING version",
                    (scorer_id, category, psycopg2.extras.Json(deepcopy(state))),
                )
                row = cursor.fetchone()
                self._conn.commit()
                return str(row[0])
            except psycopg2.IntegrityError as error:
                # A concurrent writer created the record after the existence
                # check. Its version is no longer the caller's initial token.
                self._conn.rollback()
                raise self._conflict(scorer_id, category, version) from error
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def load_all(self, scorer_id: str) -> dict[str, tuple[dict, str]]:
        """Load every category state persisted for *scorer_id*."""
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                f"SELECT category, state, version FROM {self._table} "
                "WHERE scorer_id = %s",
                (scorer_id,),
            )
            result = {
                str(category): (deepcopy(dict(state)), str(version))
                for category, state, version in cursor.fetchall()
            }
            self._conn.commit()
            return result
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def list_scorers(self) -> list[str]:
        """Return every scorer identifier with persisted category state."""
        cursor = self._conn.cursor()
        try:
            cursor.execute(f"SELECT DISTINCT scorer_id FROM {self._table} ORDER BY scorer_id")
            result = [str(row[0]) for row in cursor.fetchall()]
            self._conn.commit()
            return result
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    def close(self) -> None:
        """Close the owned database connection."""
        self._conn.close()

    def _ensure_table(self) -> None:
        cursor = self._conn.cursor()
        try:
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {self._table} ("
                "scorer_id TEXT NOT NULL, "
                "category TEXT NOT NULL, "
                "state JSONB NOT NULL, "
                "version BIGINT NOT NULL DEFAULT 1, "
                "updated_at TIMESTAMPTZ DEFAULT now(), "
                "PRIMARY KEY (scorer_id, category)"
                ")"
            )
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cursor.close()

    @staticmethod
    def _parse_version(version: str) -> int:
        """Convert an opaque protocol token to this backend's integer form."""
        try:
            parsed = int(version)
        except (TypeError, ValueError) as error:
            raise VersionConflictError(f"invalid PostgreSQL version token {version!r}") from error
        if parsed < 0:
            raise VersionConflictError(f"invalid PostgreSQL version token {version!r}")
        return parsed

    @staticmethod
    def _conflict(scorer_id: str, category: str, version: str) -> VersionConflictError:
        return VersionConflictError(
            f"stale version for scorer={scorer_id!r}, category={category!r}: "
            f"received {version!r}"
        )
