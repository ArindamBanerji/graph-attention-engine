"""Tests for gae.contracts — PropertySpec, EmbeddingContract, SchemaContract."""

import pytest

from gae.contracts import PropertySpec, EmbeddingContract, SchemaContract


class TestPropertySpec:
    def test_basic_construction(self):
        p = PropertySpec(name="score", dtype="float", min_value=0.0, max_value=1.0)
        assert p.name == "score"
        assert p.required is True

    def test_invalid_dtype_raises(self):
        with pytest.raises(AssertionError):
            PropertySpec(name="x", dtype="complex")  # type: ignore[arg-type]

    def test_min_gt_max_raises(self):
        with pytest.raises(AssertionError):
            PropertySpec(name="x", min_value=10.0, max_value=1.0)

    def test_empty_name_raises(self):
        with pytest.raises(AssertionError):
            PropertySpec(name="")

    def test_validate_value_in_bounds(self):
        p = PropertySpec(name="v", min_value=0.0, max_value=1.0)
        assert p.validate_value(0.5) is True
        assert p.validate_value(0.0) is True
        assert p.validate_value(1.0) is True

    def test_validate_value_out_of_bounds(self):
        p = PropertySpec(name="v", min_value=0.0, max_value=1.0)
        assert p.validate_value(-0.1) is False
        assert p.validate_value(1.1) is False

    def test_validate_value_unbounded(self):
        p = PropertySpec(name="v")
        assert p.validate_value(-1e9) is True
        assert p.validate_value(1e9) is True


class TestEmbeddingContract:
    def test_basic_construction(self):
        ec = EmbeddingContract(dim=128, normalized=True, dtype_name="float32")
        assert ec.dim == 128

    def test_zero_dim_raises(self):
        with pytest.raises(AssertionError):
            EmbeddingContract(dim=0)

    def test_negative_dim_raises(self):
        with pytest.raises(AssertionError):
            EmbeddingContract(dim=-1)


class TestSchemaContract:
    def _make_schema(self, props=None, embedding=None):
        if props is None:
            props = (
                PropertySpec("alpha"),
                PropertySpec("beta", required=False, default_value=0.5),
            )
        return SchemaContract(
            node_type="test_node",
            properties=props,
            embedding=embedding,
        )

    def test_factor_dim(self):
        sc = self._make_schema()
        assert sc.factor_dim == 2

    def test_property_names(self):
        sc = self._make_schema()
        assert sc.property_names() == ("alpha", "beta")

    def test_empty_node_type_raises(self):
        with pytest.raises(AssertionError):
            SchemaContract(node_type="", properties=())

    def test_duplicate_property_names_raises(self):
        with pytest.raises(AssertionError):
            SchemaContract(
                node_type="dup",
                properties=(
                    PropertySpec("x"),
                    PropertySpec("x"),
                ),
            )

    def test_resolve_value_present(self):
        sc = self._make_schema()
        assert sc.resolve_value("alpha", {"alpha": 3.0, "beta": 4.0}) == pytest.approx(3.0)

    def test_resolve_value_optional_missing_uses_default(self):
        sc = self._make_schema()
        assert sc.resolve_value("beta", {"alpha": 1.0}) == pytest.approx(0.5)

    def test_resolve_value_required_missing_raises(self):
        sc = self._make_schema()
        with pytest.raises(KeyError):
            sc.resolve_value("alpha", {})

    def test_resolve_value_unknown_property_raises(self):
        sc = self._make_schema()
        with pytest.raises(KeyError):
            sc.resolve_value("nonexistent", {"alpha": 1.0})

    def test_with_embedding(self):
        ec = EmbeddingContract(dim=64)
        sc = self._make_schema(embedding=ec)
        assert sc.embedding is not None
        assert sc.embedding.dim == 64
