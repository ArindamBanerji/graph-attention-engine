"""
Unit tests for gae/covariance.py — CovarianceEstimator + CovarianceSnapshot.

Reference: docs/gae_design_v5.md §9; v6.0 kernel roadmap.
"""

import numpy as np
import pytest

from gae.covariance import CovarianceEstimator, CovarianceSnapshot


# ------------------------------------------------------------------ #
# CovarianceEstimator construction                                    #
# ------------------------------------------------------------------ #

class TestCovarianceEstimatorInit:
    def test_basic_construction(self):
        est = CovarianceEstimator(d=4)
        assert est.d == 4
        assert est.n_samples == 0

    def test_decay_value(self):
        est = CovarianceEstimator(d=3, half_life_decisions=300)
        assert est.decay == pytest.approx(0.5 ** (1.0 / 300), rel=1e-6)

    def test_zero_half_life_no_decay(self):
        est = CovarianceEstimator(d=3, half_life_decisions=0)
        assert est.decay == pytest.approx(1.0)

    def test_invalid_d_raises(self):
        with pytest.raises(AssertionError):
            CovarianceEstimator(d=0)

    def test_negative_half_life_raises(self):
        with pytest.raises(AssertionError):
            CovarianceEstimator(d=3, half_life_decisions=-1)

    def test_initial_statistics_are_zero(self):
        est = CovarianceEstimator(d=5)
        np.testing.assert_array_equal(est.weighted_sum, np.zeros(5))
        np.testing.assert_array_equal(est.weighted_outer, np.zeros((5, 5)))
        assert est.total_weight == 0.0


# ------------------------------------------------------------------ #
# CovarianceEstimator.update                                          #
# ------------------------------------------------------------------ #

class TestCovarianceEstimatorUpdate:
    def test_n_samples_increments(self):
        est = CovarianceEstimator(d=3)
        est.update(np.array([0.1, 0.2, 0.3]))
        assert est.n_samples == 1
        est.update(np.array([0.4, 0.5, 0.6]))
        assert est.n_samples == 2

    def test_wrong_shape_raises(self):
        est = CovarianceEstimator(d=4)
        with pytest.raises(AssertionError):
            est.update(np.array([0.1, 0.2]))

    def test_total_weight_accumulates(self):
        est = CovarianceEstimator(d=2, half_life_decisions=0)  # decay=1.0 (no decay)
        est.update(np.array([1.0, 0.0]))
        assert est.total_weight == pytest.approx(1.0)
        est.update(np.array([0.0, 1.0]))
        assert est.total_weight == pytest.approx(2.0)

    def test_decay_reduces_old_weight(self):
        est = CovarianceEstimator(d=2, half_life_decisions=1)
        # decay = 0.5^(1/1) = 0.5
        est.update(np.array([1.0, 0.0]))
        w_before = est.total_weight
        est.update(np.array([0.0, 1.0]))
        # After second update: old weight 1.0 * 0.5 = 0.5, new weight 1.0 → total 1.5
        assert est.total_weight == pytest.approx(w_before * 0.5 + 1.0)


# ------------------------------------------------------------------ #
# CovarianceEstimator.get_snapshot — cold start                       #
# ------------------------------------------------------------------ #

class TestCovarianceSnapshotColdStart:
    def test_identity_at_zero_samples(self):
        est = CovarianceEstimator(d=3)
        snap = est.get_snapshot()
        np.testing.assert_array_almost_equal(snap.sigma, np.eye(3))
        np.testing.assert_array_almost_equal(snap.sigma_inv, np.eye(3))
        assert snap.shrinkage_lambda == pytest.approx(1.0)
        assert snap.condition_number == pytest.approx(1.0)
        assert snap.n_samples == 0

    def test_identity_at_one_sample(self):
        est = CovarianceEstimator(d=4)
        est.update(np.array([0.1, 0.5, 0.9, 0.3]))
        snap = est.get_snapshot()
        np.testing.assert_array_almost_equal(snap.sigma, np.eye(4))
        assert snap.n_samples == 1

    def test_snapshot_type(self):
        est = CovarianceEstimator(d=2)
        snap = est.get_snapshot()
        assert isinstance(snap, CovarianceSnapshot)


# ------------------------------------------------------------------ #
# CovarianceEstimator.get_snapshot — warm state                       #
# ------------------------------------------------------------------ #

class TestCovarianceSnapshotWarm:
    def _feed(self, est, n, seed=0):
        rng = np.random.default_rng(seed)
        for _ in range(n):
            est.update(rng.random(est.d))

    def test_sigma_is_symmetric(self):
        est = CovarianceEstimator(d=4)
        self._feed(est, 50)
        snap = est.get_snapshot()
        np.testing.assert_array_almost_equal(snap.sigma, snap.sigma.T)

    def test_sigma_is_positive_definite(self):
        est = CovarianceEstimator(d=4)
        self._feed(est, 50)
        snap = est.get_snapshot()
        eigvals = np.linalg.eigvalsh(snap.sigma)
        assert np.all(eigvals > 0)

    def test_sigma_inv_is_real_inverse(self):
        est = CovarianceEstimator(d=3)
        self._feed(est, 30)
        snap = est.get_snapshot()
        assert snap.sigma_inv is not None
        product = snap.sigma @ snap.sigma_inv
        np.testing.assert_array_almost_equal(product, np.eye(3), decimal=5)

    def test_correlation_diagonal_is_one(self):
        est = CovarianceEstimator(d=5)
        self._feed(est, 40)
        snap = est.get_snapshot()
        np.testing.assert_array_almost_equal(np.diag(snap.correlation), np.ones(5))

    def test_shrinkage_lambda_in_range(self):
        est = CovarianceEstimator(d=4)
        self._feed(est, 20)
        snap = est.get_snapshot()
        assert 0.0 <= snap.shrinkage_lambda <= 1.0

    def test_shrinkage_decreases_with_more_samples(self):
        est_few = CovarianceEstimator(d=6, half_life_decisions=0)
        est_many = CovarianceEstimator(d=6, half_life_decisions=0)
        rng = np.random.default_rng(99)
        vectors = [rng.random(6) for _ in range(200)]
        for v in vectors[:5]:
            est_few.update(v)
        for v in vectors:
            est_many.update(v)
        snap_few = est_few.get_snapshot()
        snap_many = est_many.get_snapshot()
        assert snap_many.shrinkage_lambda <= snap_few.shrinkage_lambda

    def test_per_factor_sigma_shape(self):
        est = CovarianceEstimator(d=6)
        self._feed(est, 30)
        snap = est.get_snapshot()
        assert snap.per_factor_sigma.shape == (6,)

    def test_per_factor_sigma_equals_diag(self):
        est = CovarianceEstimator(d=4)
        self._feed(est, 30)
        snap = est.get_snapshot()
        np.testing.assert_array_almost_equal(snap.per_factor_sigma, np.diag(snap.sigma))

    def test_condition_number_positive(self):
        est = CovarianceEstimator(d=3)
        self._feed(est, 30)
        snap = est.get_snapshot()
        assert snap.condition_number > 0.0

    def test_n_samples_matches(self):
        est = CovarianceEstimator(d=3)
        self._feed(est, 17)
        snap = est.get_snapshot()
        assert snap.n_samples == 17


# ------------------------------------------------------------------ #
# CovarianceEstimator.get_change_rate                                  #
# ------------------------------------------------------------------ #

class TestCovarianceChangeRate:
    def test_zero_change_same_state(self):
        est = CovarianceEstimator(d=3)
        rng = np.random.default_rng(5)
        for _ in range(20):
            est.update(rng.random(3))
        snap = est.get_snapshot()
        rate = est.get_change_rate(snap)
        assert rate == pytest.approx(0.0, abs=1e-10)

    def test_nonzero_after_updates(self):
        est = CovarianceEstimator(d=4)
        rng = np.random.default_rng(6)
        for _ in range(20):
            est.update(rng.random(4))
        snap = est.get_snapshot()
        for _ in range(20):
            est.update(rng.random(4) * 5)  # different distribution
        rate = est.get_change_rate(snap)
        assert rate > 0.0

    def test_change_rate_nonnegative(self):
        est = CovarianceEstimator(d=3)
        rng = np.random.default_rng(7)
        for _ in range(10):
            est.update(rng.random(3))
        snap = est.get_snapshot()
        est.update(rng.random(3))
        rate = est.get_change_rate(snap)
        assert rate >= 0.0
