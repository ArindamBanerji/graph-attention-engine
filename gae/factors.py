"""
GAE factor assembly — FactorComputer Protocol and assemble_factor_vector().

Factor assembly packs a node's raw scalar properties into a dense numpy
vector in the order declared by a SchemaContract:

    f = [ raw[prop_0], raw[prop_1], ..., raw[prop_{d-1}] ]     [Eq. 2, blog]

where each raw[prop_i] is resolved through the SchemaContract (required /
optional / default logic).

Reference: docs/gae_design_v5.md §5; blog Eq. 2 (factor vector assembly).

Note on async
-------------
FactorComputer.compute() is declared async so that *implementors* (external
code, e.g. a SOC adapter) can fetch properties from async data sources.
The library itself never calls asyncio.  assemble_factor_vector() is fully
synchronous and takes already-resolved raw dicts.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from gae.contracts import SchemaContract


# ---------------------------------------------------------------------------
# FactorComputer Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class FactorComputer(Protocol):
    """
    Protocol for objects that compute raw property dicts for a node.

    Implementors live outside this library (e.g. SOC adapters).
    They may use async I/O — hence the async signature — but the library
    never drives the event loop itself.

    Reference: docs/gae_design_v5.md §5.1.
    """

    async def compute(self, entity_id: str, context: Any = None) -> float:
        """
        Compute a normalized score ∈ [0, 1] for *entity_id*.

        Each FactorComputer returns a single scalar — one element of the
        factor vector f that is assembled by assemble_factor_vector().

        Parameters
        ----------
        entity_id : str
            Opaque identifier for the graph entity being scored.
        context : Any, default None
            Optional caller-provided context (e.g. alert metadata).
            Implementations may ignore this argument.

        Returns
        -------
        float
            Normalized factor score ∈ [0, 1].
        """
        ...


# ---------------------------------------------------------------------------
# assemble_factor_vector
# ---------------------------------------------------------------------------

def assemble_factor_vector(
    raw: dict[str, float],
    schema: SchemaContract,
) -> np.ndarray:
    """
    Pack raw property values into a dense 1-D factor vector.

    Implements Eq. 2 from the blog:

        f[i] = resolve(schema.properties[i], raw)               [Eq. 2]

    where resolve() applies required/optional/default logic from the
    SchemaContract (see contracts.SchemaContract.resolve_value).

    Reference: docs/gae_design_v5.md §5.2; blog Eq. 2.

    Parameters
    ----------
    raw : dict[str, float]
        Property name → scalar value mapping for one node.
    schema : SchemaContract
        Declares the property order and optional/required semantics.

    Returns
    -------
    np.ndarray, shape (d_f,)
        Dense factor vector where d_f = schema.factor_dim.

    Raises
    ------
    KeyError
        If a required property is missing from *raw*.
    AssertionError
        On internal shape invariant violations.
    """
    assert isinstance(raw, dict), (
        f"assemble_factor_vector: raw must be dict, got {type(raw)}"
    )
    assert isinstance(schema, SchemaContract), (
        "assemble_factor_vector: schema must be a SchemaContract"
    )

    d_f = schema.factor_dim
    values = [
        schema.resolve_value(name, raw)
        for name in schema.property_names()
    ]
    vector = np.array(values, dtype=np.float64)

    assert vector.shape == (d_f,), (
        f"assemble_factor_vector: output shape {vector.shape} != expected ({d_f},)"
    )
    return vector
