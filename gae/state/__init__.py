"""Pluggable, category-scoped persistence primitives for ProfileScorer."""

from gae.state.memory_backend import InMemoryStateBackend
from gae.state.protocol import StateBackend, VersionConflictError

__all__ = ["InMemoryStateBackend", "StateBackend", "VersionConflictError"]
_optional_errors: dict[str, ImportError] = {}

try:
    from gae.state.redis_backend import RedisStateBackend
except ImportError:
    _optional_errors["RedisStateBackend"] = ImportError(
        "RedisStateBackend requires the optional dependency. Install with "
        "`pip install graph-attention-engine[redis]`."
    )
else:
    __all__.append("RedisStateBackend")

try:
    from gae.state.pg_backend import PostgresStateBackend
except ImportError:
    _optional_errors["PostgresStateBackend"] = ImportError(
        "PostgresStateBackend requires the optional dependency. Install with "
        "`pip install graph-attention-engine[postgres]`."
    )
else:
    __all__.append("PostgresStateBackend")


def __getattr__(name: str) -> object:
    """Explain how to install an optional backend when it is requested."""
    if name in _optional_errors:
        raise _optional_errors[name] from None
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
