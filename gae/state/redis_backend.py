"""Redis-backed category persistence with optimistic compare-and-swap."""

from __future__ import annotations

import json
from typing import Any

import redis

from gae.state.protocol import VersionConflictError


class RedisStateBackend:
    """Store scorer category state as JSON fields in Redis hashes.

    A hash key is shared by a scorer and a field is shared by one category.
    Redis watches hashes rather than individual fields, so a transaction that
    loses a watch to another category retries. A changed version in the target
    field is the only condition exposed as a category CAS conflict.

    Reference: persistence for blog Eq. 4b-final learning state.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "rgi",
    ) -> None:
        """Connect to Redis using *url* and namespace keys with *prefix*.

        Reference: persistence for blog Eq. 4b-final learning state.
        """
        self._prefix = prefix
        self._client: Any = redis.Redis.from_url(url, decode_responses=True)

    def load_category(self, scorer_id: str, category: str) -> tuple[dict, str]:
        """Load one JSON category field or return the initial empty record.

        Reference: persistence for blog Eq. 4b-final learning state.
        """
        raw = self._client.hget(self._key(scorer_id), category)
        if raw is None:
            return {}, "0"
        return self._decode_record(raw)

    def save_category(
        self, scorer_id: str, category: str, state: dict, version: str
    ) -> str:
        """CAS-upsert one category and return its next integer version token.

        A watch invalidated by a different category is retried. A target
        category version change raises :class:`VersionConflictError`.

        Reference: persistence for blog Eq. 4b-final learning state.
        """
        key = self._key(scorer_id)
        while True:
            with self._client.pipeline() as pipeline:
                try:
                    pipeline.watch(key)
                    raw = pipeline.hget(key, category)
                    _state, current_version = (
                        ({}, "0") if raw is None else self._decode_record(raw)
                    )
                    if version != current_version:
                        pipeline.unwatch()
                        raise VersionConflictError(
                            f"stale version for scorer={scorer_id!r}, "
                            f"category={category!r}: expected {current_version!r}, "
                            f"received {version!r}"
                        )

                    new_version = str(int(current_version) + 1)
                    payload = json.dumps(
                        {"state": state, "version": new_version},
                        separators=(",", ":"),
                    )
                    pipeline.multi()
                    pipeline.hset(key, category, payload)
                    pipeline.execute()
                    return new_version
                except redis.WatchError:
                    # Another field may have changed. Re-read the target field:
                    # a target-token change becomes VersionConflictError above;
                    # otherwise retry so independent categories do not conflict.
                    continue

    def load_all(self, scorer_id: str) -> dict[str, tuple[dict, str]]:
        """Load every JSON category field stored for *scorer_id*.

        Reference: persistence for blog Eq. 4b-final learning state.
        """
        return {
            self._as_text(category): self._decode_record(raw)
            for category, raw in self._client.hgetall(self._key(scorer_id)).items()
        }

    def list_scorers(self) -> list[str]:
        """List scorer IDs currently represented by Redis hash keys.

        ``KEYS`` is intentional for this initial implementation. Production
        deployments with large keyspaces should replace it with cursor-based
        ``SCAN`` iteration.

        Reference: persistence for blog Eq. 4b-final learning state.
        """
        key_prefix = f"{self._prefix}:"
        return sorted(
            self._as_text(key)[len(key_prefix):]
            for key in self._client.keys(f"{key_prefix}*")
            if self._as_text(key).startswith(key_prefix)
        )

    def _key(self, scorer_id: str) -> str:
        """Return the Redis hash key for one scorer identifier.

        Reference: persistence for blog Eq. 4b-final learning state.
        """
        return f"{self._prefix}:{scorer_id}"

    @staticmethod
    def _decode_record(raw: str | bytes) -> tuple[dict, str]:
        """Decode one stored state record into its state and opaque token.

        Reference: persistence for blog Eq. 4b-final learning state.
        """
        payload = json.loads(RedisStateBackend._as_text(raw))
        return dict(payload["state"]), str(payload["version"])

    @staticmethod
    def _as_text(value: str | bytes) -> str:
        """Normalize redis-py text or bytes responses to text.

        Reference: persistence for blog Eq. 4b-final learning state.
        """
        return value.decode("utf-8") if isinstance(value, bytes) else value
