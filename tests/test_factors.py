"""Tests for gae.factors — FactorComputer Protocol and assemble_factor_vector."""

import numpy as np
import pytest

from gae.contracts import PropertySpec, SchemaContract
from gae.factors import FactorComputer, assemble_factor_vector


def _make_schema(*names, **optional_defaults):
    """Helper: build a SchemaContract from property names.

    Positional *names* become required properties.
    Keyword *optional_defaults* become optional properties (appended after
    required ones, in declaration order).
    """
    props = []
    for name in names:
        props.append(PropertySpec(name))
    for name, default in optional_defaults.items():
        props.append(PropertySpec(name, required=False, default_value=default))
    return SchemaContract(node_type="test", properties=tuple(props))


class TestAssembleFactorVector:
    def test_basic_assembly(self):
        schema = _make_schema("a", "b", "c")
        raw = {"a": 1.0, "b": 2.0, "c": 3.0}
        v = assemble_factor_vector(raw, schema)
        assert v.shape == (3,)
        np.testing.assert_array_equal(v, [1.0, 2.0, 3.0])

    def test_order_matches_schema(self):
        # Properties declared b, a — packed in that order
        schema = SchemaContract(
            node_type="t",
            properties=(PropertySpec("b"), PropertySpec("a")),
        )
        raw = {"a": 10.0, "b": 20.0}
        v = assemble_factor_vector(raw, schema)
        assert v[0] == pytest.approx(20.0)  # b first
        assert v[1] == pytest.approx(10.0)  # a second

    def test_optional_default_fills_in(self):
        schema = _make_schema("x", y=0.7)
        raw = {"x": 5.0}   # y omitted
        v = assemble_factor_vector(raw, schema)
        assert v[1] == pytest.approx(0.7)

    def test_required_missing_raises(self):
        schema = _make_schema("x", "y")
        with pytest.raises(KeyError):
            assemble_factor_vector({"x": 1.0}, schema)

    def test_output_dtype_is_float64(self):
        schema = _make_schema("a", "b")
        v = assemble_factor_vector({"a": 1, "b": 2}, schema)
        assert v.dtype == np.float64

    def test_non_dict_raw_raises(self):
        schema = _make_schema("a")
        with pytest.raises(AssertionError):
            assemble_factor_vector([1.0], schema)  # type: ignore[arg-type]

    def test_non_schema_raises(self):
        with pytest.raises(AssertionError):
            assemble_factor_vector({"a": 1.0}, {"a": 1.0})  # type: ignore[arg-type]

    def test_empty_schema(self):
        schema = SchemaContract(node_type="empty", properties=())
        v = assemble_factor_vector({}, schema)
        assert v.shape == (0,)


class TestFactorComputerProtocol:
    def test_protocol_is_runtime_checkable(self):
        """FactorComputer is @runtime_checkable — we can isinstance-check it."""

        class MyComputer:
            async def compute(self, entity_id: str, context=None) -> float:
                return 1.0

        assert isinstance(MyComputer(), FactorComputer)

    def test_missing_compute_not_checkable(self):
        class NotAComputer:
            def other_method(self):
                pass

        assert not isinstance(NotAComputer(), FactorComputer)
