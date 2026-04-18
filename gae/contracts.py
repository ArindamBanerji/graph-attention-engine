"""
GAE schema contracts — declarative descriptions of node property schemas
and embedding requirements.

These are pure value objects; no I/O, no domain logic.
Callers use them to validate factor vector assembly and embedding shapes
before passing data into scoring or learning pipelines.

Reference: docs/gae_design_v10_6.md §4 (Schema contracts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# PropertySpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PropertySpec:
    """
    Declares one scalar property that a node type exposes.

    Reference: docs/gae_design_v10_6.md §4.1.

    Attributes
    ----------
    name : str
        Property name, used as a key in factor dictionaries.
    dtype : Literal["float", "int", "bool"]
        Expected numeric dtype family.
    min_value : float or None
        Optional lower bound for validation.
    max_value : float or None
        Optional upper bound for validation.
    required : bool
        If True, the property must be present in every factor dict.
        If False, a missing value is replaced by *default_value*.
    default_value : float
        Substitute when *required* is False and the property is absent.
    """

    name: str
    dtype: Literal["float", "int", "bool"] = "float"
    min_value: float | None = None
    max_value: float | None = None
    required: bool = True
    default_value: float = 0.0

    def __post_init__(self) -> None:
        assert self.name, "PropertySpec.name must be a non-empty string"
        assert self.dtype in ("float", "int", "bool"), (
            f"PropertySpec.dtype must be 'float', 'int', or 'bool', got '{self.dtype}'"
        )
        if self.min_value is not None and self.max_value is not None:
            assert self.min_value <= self.max_value, (
                f"PropertySpec.min_value ({self.min_value}) must be <= "
                f"max_value ({self.max_value})"
            )

    def validate_value(self, value: float) -> bool:
        """
        Return True if *value* satisfies the bounds declared in this spec.

        Reference: docs/gae_design_v10_6.md §4.1 (validation rule).

        Parameters
        ----------
        value : float
            Scalar value to check.

        Returns
        -------
        bool
            True when *value* is within [min_value, max_value] (bounds that
            are None are treated as unbounded).
        """
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


# ---------------------------------------------------------------------------
# EmbeddingContract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class EmbeddingContract:
    """
    Declares the expected shape and numeric properties of an embedding vector.

    Reference: docs/gae_design_v10_6.md §4.2.

    Attributes
    ----------
    dim : int
        Expected embedding dimensionality (d_e > 0).
    normalized : bool
        If True, embeddings are expected to be L2-unit-norm vectors.
    dtype_name : str
        NumPy dtype name, e.g. "float32" or "float64".
    """

    dim: int
    normalized: bool = False
    dtype_name: str = "float32"

    def __post_init__(self) -> None:
        assert self.dim > 0, f"EmbeddingContract.dim must be > 0, got {self.dim}"
        assert self.dtype_name, "EmbeddingContract.dtype_name must be non-empty"


# ---------------------------------------------------------------------------
# SchemaContract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SchemaContract:
    """
    Full schema declaration for one node type: its scalar properties and
    optional embedding contract.

    Reference: docs/gae_design_v10_6.md §4.3.

    Attributes
    ----------
    node_type : str
        Opaque label for the node type (e.g. "host", "process").
    properties : tuple[PropertySpec, ...]
        Ordered property specs.  The order defines the packed factor-vector
        layout: factor_vector[i] corresponds to properties[i].
    embedding : EmbeddingContract or None
        When set, this node type also carries an embedding vector.
    """

    node_type: str
    properties: tuple[PropertySpec, ...]
    embedding: EmbeddingContract | None = None

    def __post_init__(self) -> None:
        assert self.node_type, "SchemaContract.node_type must be non-empty"
        assert isinstance(self.properties, tuple), (
            "SchemaContract.properties must be a tuple of PropertySpec"
        )
        names = [p.name for p in self.properties]
        assert len(names) == len(set(names)), (
            f"SchemaContract.properties contains duplicate names: {names}"
        )

    @property
    def factor_dim(self) -> int:
        """
        Number of scalar factors in the packed factor vector.

        Reference: docs/gae_design_v10_6.md §4.3.

        Returns
        -------
        int
            len(properties).
        """
        return len(self.properties)

    def property_names(self) -> tuple[str, ...]:
        """
        Ordered property names, matching the factor_vector layout.

        Reference: docs/gae_design_v10_6.md §4.3.

        Returns
        -------
        tuple[str, ...]
        """
        return tuple(p.name for p in self.properties)

    def resolve_value(self, name: str, raw: dict[str, float]) -> float:
        """
        Look up *name* in *raw*, falling back to the spec default when the
        property is optional and absent.

        Reference: docs/gae_design_v10_6.md §4.3 (resolution rule).

        Parameters
        ----------
        name : str
            Property name to resolve.
        raw : dict[str, float]
            Mapping of property name → raw scalar value.

        Returns
        -------
        float
            Resolved scalar value.

        Raises
        ------
        KeyError
            If the property is required and absent from *raw*.
        KeyError
            If *name* is not declared in this schema.
        """
        spec_map = {p.name: p for p in self.properties}
        if name not in spec_map:
            raise KeyError(f"Property '{name}' not declared in schema '{self.node_type}'")
        spec = spec_map[name]
        if name in raw:
            return float(raw[name])
        if not spec.required:
            return spec.default_value
        raise KeyError(
            f"Required property '{name}' missing from raw dict for schema '{self.node_type}'"
        )
