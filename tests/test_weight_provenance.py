import json

import numpy as np
import pytest

from gae.kernels import DiagonalKernel, WeightProvenance
from gae.profile_scorer import ProfileScorer


def test_weight_provenance_enum_has_expected_values_and_serializes():
    expected = {
        "sigma_derived",
        "inverse_variance",
        "discriminative",
        "effective",
    }

    values = {item.value for item in WeightProvenance}

    assert values == expected
    assert len(WeightProvenance) == 4
    assert all(isinstance(item, str) for item in WeightProvenance)
    assert json.loads(json.dumps({"p": WeightProvenance.DISCRIMINATIVE})) == {
        "p": "discriminative"
    }


def test_from_sigma_preserves_legacy_weights_and_raw_inverse_variance():
    sigma = np.array([1.0, 2.0, 0.5], dtype=np.float64)

    legacy = DiagonalKernel(sigma=sigma)
    kernel = DiagonalKernel.from_sigma(sigma)

    raw = 1.0 / sigma ** 2
    expected_weights = raw / raw.max()
    np.testing.assert_allclose(kernel.weights, expected_weights)
    np.testing.assert_allclose(kernel.weights, legacy.weights)
    np.testing.assert_allclose(kernel.raw_weights, raw)
    assert kernel.provenance == WeightProvenance.SIGMA_DERIVED


def test_from_learned_preserves_above_one_values_for_scoring():
    learned = np.array([4.0, 0.2, 1.5], dtype=np.float64)

    kernel = DiagonalKernel.from_learned(learned)

    np.testing.assert_allclose(kernel.weights, learned)
    np.testing.assert_allclose(kernel.raw_weights, learned)
    assert kernel.weights.max() > 1.0
    assert kernel.provenance == WeightProvenance.DISCRIMINATIVE


def test_from_effective_preserves_shrinkage_blend_values():
    effective = np.array([2.5, 0.7, 1.2], dtype=np.float64)

    kernel = DiagonalKernel.from_effective(effective)

    np.testing.assert_allclose(kernel.weights, effective)
    np.testing.assert_allclose(kernel.raw_weights, effective)
    assert kernel.provenance == WeightProvenance.EFFECTIVE


def test_normalized_returns_display_view_without_mutating_weights():
    weights = np.array([4.0, 0.2, 2.0], dtype=np.float64)
    kernel = DiagonalKernel.from_learned(weights)
    before = kernel.weights.copy()

    normalized = kernel.normalized()

    np.testing.assert_allclose(normalized, np.array([1.0, 0.05, 0.5]))
    np.testing.assert_allclose(kernel.weights, before)
    assert normalized is not kernel.weights


def test_normalized_handles_all_zero_weights_view():
    kernel = DiagonalKernel.from_learned(np.ones(3, dtype=np.float64))
    kernel.weights = np.zeros(3, dtype=np.float64)

    normalized = kernel.normalized()

    np.testing.assert_array_equal(normalized, np.zeros(3, dtype=np.float64))
    assert normalized is not kernel.weights


def test_sigma_derived_normalized_view_equals_scoring_weights():
    sigma = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    kernel = DiagonalKernel.from_sigma(sigma)

    np.testing.assert_allclose(kernel.normalized(), kernel.weights)


def test_legacy_sigma_and_from_sigma_compute_distance_identically():
    sigma = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    f = np.array([0.7, 0.2, 0.4], dtype=np.float64)
    mu = np.array(
        [[0.2, 0.2, 0.2], [0.9, 0.1, 0.3]],
        dtype=np.float64,
    )

    legacy = DiagonalKernel(sigma=sigma)
    kernel = DiagonalKernel.from_sigma(sigma)

    np.testing.assert_allclose(
        kernel.compute_distance(f, mu),
        legacy.compute_distance(f, mu),
    )


def test_legacy_weights_and_from_learned_compute_distance_identically():
    weights = np.array([4.0, 0.2, 1.5], dtype=np.float64)
    f = np.array([0.7, 0.2, 0.4], dtype=np.float64)
    mu = np.array(
        [[0.2, 0.2, 0.2], [0.9, 0.1, 0.3]],
        dtype=np.float64,
    )

    legacy = DiagonalKernel(weights=weights)
    kernel = DiagonalKernel.from_learned(weights)

    np.testing.assert_allclose(
        kernel.compute_distance(f, mu),
        legacy.compute_distance(f, mu),
    )


def test_legacy_constructors_set_backward_compatible_provenance():
    sigma_kernel = DiagonalKernel(sigma=np.array([1.0, 2.0], dtype=np.float64))
    weight_kernel = DiagonalKernel(weights=np.array([2.0, 0.5], dtype=np.float64))

    assert sigma_kernel.provenance == WeightProvenance.SIGMA_DERIVED
    assert weight_kernel.provenance == WeightProvenance.DISCRIMINATIVE
    np.testing.assert_allclose(weight_kernel.raw_weights, np.array([2.0, 0.5]))


def test_raw_weights_returns_copy():
    kernel = DiagonalKernel.from_learned(np.array([4.0, 0.2], dtype=np.float64))

    raw = kernel.raw_weights
    raw[0] = 999.0

    np.testing.assert_allclose(kernel.raw_weights, np.array([4.0, 0.2]))


def test_profile_scorer_dk_weights_raw_and_normalized_accessors():
    mu = np.full((1, 2, 3), 0.5, dtype=np.float64)
    scorer = ProfileScorer(mu=mu, actions=["a", "b"])
    scorer._dk_weights = np.array([[4.0, 0.2, 2.0]], dtype=np.float64)

    raw = scorer.get_dk_weights(0)
    normalized = scorer.get_dk_weights_normalized(0)

    np.testing.assert_allclose(raw, np.array([4.0, 0.2, 2.0]))
    np.testing.assert_allclose(normalized, np.array([1.0, 0.05, 0.5]))
    np.testing.assert_allclose(scorer._dk_weights[0], np.array([4.0, 0.2, 2.0]))


def test_profile_scorer_normalized_accessor_preserves_none_behavior():
    mu = np.full((1, 2, 3), 0.5, dtype=np.float64)
    scorer = ProfileScorer(mu=mu, actions=["a", "b"])

    assert scorer.get_dk_weights(0) is None
    assert scorer.get_dk_weights_normalized(0) is None
