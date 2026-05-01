"""Tests for gae.snr."""

import numpy as np

from gae.calibration import compute_dominant_axis
from gae.snr import CategorySNR, SNRReport, _phi, compute_snr_report


def test_compute_snr_report_preserves_expected_shapes_and_names():
    mu = np.array(
        [
            [
                [0.1, 0.2, 0.3],
                [0.9, 0.2, 0.3],
            ],
            [
                [0.3, 0.4, 0.2],
                [0.7, 0.6, 0.8],
            ],
        ],
        dtype=np.float64,
    )
    sigma = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    report = compute_snr_report(
        mu,
        sigma,
        categories=["cat_a", "cat_b"],
        actions=["allow", "block"],
        factor_names=["f0", "f1", "f2"],
    )

    assert isinstance(report, SNRReport)
    assert len(report.categories) == 2
    assert isinstance(report.categories[0], CategorySNR)
    assert report.categories[0].category_name == "cat_a"
    assert report.categories[1].category_name == "cat_b"
    assert set(report.factor_importance.keys()) == {"f0", "f1", "f2"}


def test_snr_uses_l2_diagnostics_when_kernel_weights_omitted():
    mu = np.array(
        [
            [
                [0.0, 0.0],
                [3.0, 4.0],
            ]
        ],
        dtype=np.float64,
    )
    sigma = np.array([3.0, 4.0], dtype=np.float64)

    report = compute_snr_report(
        mu,
        sigma,
        categories=["only"],
        actions=["left", "right"],
        factor_names=["x", "y"],
    )

    category = report.categories[0]
    assert category.action_separation == 5.0
    assert category.weighted_noise == 5.0
    assert category.snr == 1.0


def test_weighted_noise_and_weighted_distance_use_raw_kernel_weights():
    mu = np.array(
        [
            [
                [0.0, 0.0],
                [2.0, 1.0],
            ]
        ],
        dtype=np.float64,
    )
    sigma = np.array([3.0, 4.0], dtype=np.float64)
    kernel_weights = np.array([2.0, 0.5], dtype=np.float64)

    report = compute_snr_report(
        mu,
        sigma,
        kernel_weights=kernel_weights,
        categories=["only"],
        actions=["left", "right"],
        factor_names=["x", "y"],
    )

    expected_noise = float(np.sqrt((3.0 * 2.0) ** 2 + (4.0 * 0.5) ** 2))
    expected_separation = float(np.sqrt((2.0 * 2.0) ** 2 + (1.0 * 0.5) ** 2))
    category = report.categories[0]
    assert abs(report.weighted_noise - expected_noise) < 1e-12
    assert abs(category.action_separation - expected_separation) < 1e-12
    assert abs(category.weakest_pair_distance - expected_separation) < 1e-12


def test_ceiling_formula_uses_phi_of_half_snr():
    mu = np.array(
        [
            [
                [0.0],
                [2.0],
            ]
        ],
        dtype=np.float64,
    )
    sigma = np.array([1.0], dtype=np.float64)

    report = compute_snr_report(
        mu,
        sigma,
        categories=["only"],
        actions=["a0", "a1"],
        factor_names=["f0"],
    )

    expected = float(_phi(1.0))
    assert abs(report.categories[0].ceiling_estimate - expected) < 1e-12
    assert report.categories[0].status == "near_ceiling"


def test_proposed_improvement_targets_weakest_category_and_action_pair():
    mu = np.array(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [3.0, 0.0],
            ],
            [
                [0.0, 0.0],
                [0.2, 0.0],
                [5.0, 0.0],
            ],
        ],
        dtype=np.float64,
    )
    sigma = np.array([1.0, 1.0], dtype=np.float64)

    report = compute_snr_report(
        mu,
        sigma,
        categories=["strong", "weak"],
        actions=["act0", "act1", "act2"],
        factor_names=["signal", "flat"],
    )

    assert "weak" in report.proposed_improvement
    assert "act0" in report.proposed_improvement
    assert "act1" in report.proposed_improvement


def test_factor_importance_reuses_compute_dominant_axis_scores():
    mu = np.array(
        [
            [
                [0.0, 0.5, 0.5],
                [1.0, 0.5, 0.5],
            ],
            [
                [0.2, 0.5, 0.5],
                [0.8, 0.5, 0.5],
            ],
        ],
        dtype=np.float64,
    )
    sigma = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    report = compute_snr_report(
        mu,
        sigma,
        factor_names=["dominant", "flat_a", "flat_b"],
    )

    expected = compute_dominant_axis(mu)
    assert abs(report.factor_importance["dominant"] - expected[0]) < 1e-12
    assert abs(report.factor_importance["flat_a"] - expected[1]) < 1e-12
    assert abs(report.factor_importance["flat_b"] - expected[2]) < 1e-12
    assert report.factor_importance["dominant"] == 1.0
    assert report.factor_importance["flat_a"] == 0.0
    assert report.factor_importance["flat_b"] == 0.0
