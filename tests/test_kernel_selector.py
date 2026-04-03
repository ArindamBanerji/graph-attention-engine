"""
Tests for gae/kernel_selector.py — KernelSelector, KernelScore, KernelRecommendation.

Reference: docs/gae_design_v5.md §9; v6.0 kernel roadmap.
"""

import numpy as np
import pytest

from gae.kernel_selector import KernelSelector, KernelRecommendation, KernelScore


# ------------------------------------------------------------------ #
# Fixtures                                                            #
# ------------------------------------------------------------------ #

@pytest.fixture
def uniform_sigma():
    return np.array([0.10, 0.10, 0.10, 0.10, 0.10, 0.10])


@pytest.fixture
def hetero_sigma():
    return np.array([0.08, 0.09, 0.07, 0.25, 0.12, 0.28])


@pytest.fixture
def mu():
    rng = np.random.default_rng(42)
    return rng.random((6, 4, 6)) * 0.5 + 0.25


# ------------------------------------------------------------------ #
# KernelScore                                                         #
# ------------------------------------------------------------------ #

class TestKernelScore:
    def test_agreement_rate_zero_when_no_decisions(self):
        ks = KernelScore(kernel_name="l2")
        assert ks.agreement_rate == 0.0

    def test_mean_confidence_zero_when_no_decisions(self):
        ks = KernelScore(kernel_name="l2")
        assert ks.mean_confidence == 0.0

    def test_agreement_rate_computed_correctly(self):
        ks = KernelScore(kernel_name="l2", total_decisions=10, agreements=7)
        assert ks.agreement_rate == pytest.approx(0.7)

    def test_mean_confidence_computed_correctly(self):
        ks = KernelScore(kernel_name="l2", total_decisions=4,
                         cumulative_confidence=3.0)
        assert ks.mean_confidence == pytest.approx(0.75)


# ------------------------------------------------------------------ #
# KernelSelector construction                                         #
# ------------------------------------------------------------------ #

class TestKernelSelectorInit:
    def test_builds_three_kernels(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        assert set(sel.kernels.keys()) == {"l2", "diagonal", "shrinkage"}

    def test_scores_initialised_for_each_kernel(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        assert set(sel.scores.keys()) == {"l2", "diagonal", "shrinkage"}
        assert all(s.total_decisions == 0 for s in sel.scores.values())

    def test_invalid_d_raises(self, hetero_sigma):
        with pytest.raises(AssertionError):
            KernelSelector(d=0, sigma_per_factor=hetero_sigma[:1])

    def test_sigma_shape_mismatch_raises(self, hetero_sigma):
        with pytest.raises(AssertionError):
            KernelSelector(d=6, sigma_per_factor=hetero_sigma[:4])

    def test_sigma_stored_as_float64(self, uniform_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=uniform_sigma.astype(np.float32))
        assert sel.sigma.dtype == np.float64

    def test_correlation_max_stored(self, uniform_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=uniform_sigma, correlation_max=0.35)
        assert sel.correlation_max == pytest.approx(0.35)


# ------------------------------------------------------------------ #
# Preliminary recommendation (Phase 2 rules)                         #
# ------------------------------------------------------------------ #

class TestPreliminaryRecommendation:
    def test_uniform_low_rho_gives_l2(self, uniform_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=uniform_sigma, correlation_max=0.1)
        rec = sel.preliminary_recommendation()
        assert rec.recommended_kernel == "l2"
        assert rec.method == "rule"

    def test_hetero_low_rho_gives_diagonal(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma, correlation_max=0.15)
        rec = sel.preliminary_recommendation()
        assert rec.recommended_kernel == "diagonal"

    def test_high_rho_gives_shrinkage(self, uniform_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=uniform_sigma, correlation_max=0.45)
        rec = sel.preliminary_recommendation()
        assert rec.recommended_kernel == "shrinkage"

    def test_hetero_high_rho_gives_shrinkage(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma, correlation_max=0.50)
        rec = sel.preliminary_recommendation()
        assert rec.recommended_kernel == "shrinkage"

    def test_zero_sigma_defaults_to_l2(self):
        sel = KernelSelector(d=3, sigma_per_factor=np.zeros(3))
        rec = sel.preliminary_recommendation()
        assert rec.recommended_kernel == "l2"

    def test_rec_is_KernelRecommendation_instance(self, uniform_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=uniform_sigma)
        rec = sel.preliminary_recommendation()
        assert isinstance(rec, KernelRecommendation)

    def test_reason_is_nonempty_string(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma, correlation_max=0.1)
        rec = sel.preliminary_recommendation()
        assert isinstance(rec.reason, str) and len(rec.reason) > 0

    def test_rho_boundary_0_3_gives_shrinkage(self, uniform_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=uniform_sigma, correlation_max=0.30)
        rec = sel.preliminary_recommendation()
        assert rec.recommended_kernel == "shrinkage"


# ------------------------------------------------------------------ #
# record_comparison (Phase 3)                                         #
# ------------------------------------------------------------------ #

class TestRecordComparison:
    def test_returns_predictions_for_all_kernels(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        f = np.random.default_rng(1).random(6)
        preds = sel.record_comparison(f, 0, mu, 1, ["a", "b", "c", "d"])
        assert set(preds.keys()) == {"l2", "diagonal", "shrinkage"}
        assert all(0 <= v < 4 for v in preds.values())

    def test_total_decisions_increments(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        f = np.random.default_rng(2).random(6)
        sel.record_comparison(f, 0, mu, 0, ["a", "b", "c", "d"])
        assert all(s.total_decisions == 1 for s in sel.scores.values())

    def test_agreement_tracked_correctly(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        f = np.random.default_rng(3).random(6)
        preds = sel.record_comparison(f, 0, mu, 1, ["a", "b", "c", "d"])
        for name, action in preds.items():
            if action == 1:
                assert sel.scores[name].agreements == 1
                assert sel.scores[name].disagreements == 0
            else:
                assert sel.scores[name].agreements == 0
                assert sel.scores[name].disagreements == 1

    def test_multiple_calls_accumulate(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        rng = np.random.default_rng(42)
        for _ in range(50):
            f = rng.random(6)
            sel.record_comparison(f, rng.integers(6), mu, rng.integers(4),
                                  ["a", "b", "c", "d"])
        assert all(s.total_decisions == 50 for s in sel.scores.values())

    def test_confidence_accumulates(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        f = np.random.default_rng(4).random(6)
        sel.record_comparison(f, 0, mu, 0, ["a", "b", "c", "d"])
        assert all(s.cumulative_confidence > 0.0 for s in sel.scores.values())

    def test_wrong_factor_shape_raises(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        with pytest.raises(AssertionError):
            sel.record_comparison(np.zeros(4), 0, mu, 0, ["a", "b", "c", "d"])


# ------------------------------------------------------------------ #
# recommend (Phase 4)                                                 #
# ------------------------------------------------------------------ #

class TestRecommend:
    def test_insufficient_data_returns_rule_based(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        rec = sel.recommend()
        assert rec.sufficient_data is False
        assert rec.method == "rule"

    def test_insufficient_data_reason_mentions_threshold(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        rec = sel.recommend()
        assert str(KernelSelector.MIN_DECISIONS_FOR_RECOMMENDATION) in rec.reason

    def test_after_enough_data_method_is_empirical(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        rng = np.random.default_rng(42)
        for _ in range(150):
            f = rng.random(6)
            sel.record_comparison(f, rng.integers(6), mu, rng.integers(4),
                                  ["a", "b", "c", "d"])
        rec = sel.recommend()
        assert rec.sufficient_data is True
        assert rec.method == "empirical"

    def test_recommended_kernel_is_valid(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        rng = np.random.default_rng(42)
        for _ in range(150):
            f = rng.random(6)
            sel.record_comparison(f, rng.integers(6), mu, rng.integers(4),
                                  ["a", "b", "c", "d"])
        rec = sel.recommend()
        assert rec.recommended_kernel in {"l2", "diagonal", "shrinkage"}

    def test_picks_highest_agreement(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        sel.scores["l2"].total_decisions = 100
        sel.scores["l2"].agreements = 60
        sel.scores["diagonal"].total_decisions = 100
        sel.scores["diagonal"].agreements = 80
        sel.scores["shrinkage"].total_decisions = 100
        sel.scores["shrinkage"].agreements = 75
        # Rolling buffers must match so recommend() uses rolling rates
        sel._buffers["l2"] = [True] * 60 + [False] * 40
        sel._buffers["diagonal"] = [True] * 80 + [False] * 20
        sel._buffers["shrinkage"] = [True] * 75 + [False] * 25
        rec = sel.recommend()
        assert rec.recommended_kernel == "diagonal"
        assert rec.confidence > 0.0

    def test_confidence_is_margin_over_runner_up(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        sel.scores["l2"].total_decisions = 100
        sel.scores["l2"].agreements = 60
        sel.scores["diagonal"].total_decisions = 100
        sel.scores["diagonal"].agreements = 80
        sel.scores["shrinkage"].total_decisions = 100
        sel.scores["shrinkage"].agreements = 75
        # Rolling buffers must match so recommend() uses rolling rates
        sel._buffers["l2"] = [True] * 60 + [False] * 40
        sel._buffers["diagonal"] = [True] * 80 + [False] * 20
        sel._buffers["shrinkage"] = [True] * 75 + [False] * 25
        rec = sel.recommend()
        # diagonal rolling(0.80) - shrinkage rolling(0.75) = 0.05
        assert rec.confidence == pytest.approx(0.05, abs=1e-9)

    def test_reason_contains_winner_name(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        sel.scores["l2"].total_decisions = 100
        sel.scores["l2"].agreements = 55
        sel.scores["diagonal"].total_decisions = 100
        sel.scores["diagonal"].agreements = 70
        sel.scores["shrinkage"].total_decisions = 100
        sel.scores["shrinkage"].agreements = 60
        sel._buffers["l2"] = [True] * 55 + [False] * 45
        sel._buffers["diagonal"] = [True] * 70 + [False] * 30
        sel._buffers["shrinkage"] = [True] * 60 + [False] * 40
        rec = sel.recommend()
        assert "diagonal" in rec.reason

    def test_scores_dict_in_recommendation(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        rec = sel.recommend()
        assert set(rec.scores.keys()) == {"l2", "diagonal", "shrinkage"}
        for v in rec.scores.values():
            assert "agreement_rate" in v
            assert "total" in v

    def test_tiebreaker_uses_noise_ratio_when_margin_small(self):
        # noise_ratio = 0.2/0.1 = 2.0 > 1.5 → tiebreaker picks diagonal
        sigma = np.array([0.1, 0.2])
        sel = KernelSelector(d=2, sigma_per_factor=sigma)
        # margin = 0.005 (< 1pp threshold): diagonal 101/200, l2 100/200
        sel.scores["diagonal"].total_decisions = 200
        sel.scores["l2"].total_decisions = 200
        sel.scores["shrinkage"].total_decisions = 200
        sel._buffers["diagonal"] = [True] * 101 + [False] * 99
        sel._buffers["l2"]       = [True] * 100 + [False] * 100
        sel._buffers["shrinkage"] = [True] * 99  + [False] * 101
        rec = sel.recommend()
        assert rec.recommended_kernel == "diagonal", (
            f"Tiebreaker with noise_ratio=2.0 > 1.5 should pick diagonal, "
            f"got {rec.recommended_kernel!r}"
        )

    def test_tiebreaker_l2_when_low_noise_ratio(self):
        # noise_ratio = 0.12/0.1 = 1.2 < 1.5 → tiebreaker picks l2
        sigma = np.array([0.1, 0.12])
        sel = KernelSelector(d=2, sigma_per_factor=sigma)
        # margin = 0.005 (< 1pp threshold): diagonal 101/200, l2 100/200
        sel.scores["diagonal"].total_decisions = 200
        sel.scores["l2"].total_decisions = 200
        sel.scores["shrinkage"].total_decisions = 200
        sel._buffers["diagonal"] = [True] * 101 + [False] * 99
        sel._buffers["l2"]       = [True] * 100 + [False] * 100
        sel._buffers["shrinkage"] = [True] * 99  + [False] * 101
        rec = sel.recommend()
        assert rec.recommended_kernel == "l2", (
            f"Tiebreaker with noise_ratio=1.2 < 1.5 should pick l2, "
            f"got {rec.recommended_kernel!r}"
        )


# ------------------------------------------------------------------ #
# get_comparison_summary                                              #
# ------------------------------------------------------------------ #

class TestComparisonSummary:
    def test_returns_all_kernels(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        summary = sel.get_comparison_summary()
        assert set(summary.keys()) == {"l2", "diagonal", "shrinkage"}

    def test_summary_fields_present(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        rng = np.random.default_rng(7)
        for _ in range(10):
            sel.record_comparison(rng.random(6), 0, mu, rng.integers(4),
                                  ["a", "b", "c", "d"])
        summary = sel.get_comparison_summary()
        for v in summary.values():
            assert "agreement_rate" in v
            assert "mean_confidence" in v
            assert "total_decisions" in v
            assert "agreements" in v

    def test_agreement_rate_in_summary_matches_score(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        rng = np.random.default_rng(8)
        for _ in range(20):
            sel.record_comparison(rng.random(6), 0, mu, rng.integers(4),
                                  ["a", "b", "c", "d"])
        summary = sel.get_comparison_summary()
        for name in sel.scores:
            assert summary[name]["agreement_rate"] == pytest.approx(
                sel.scores[name].agreement_rate
            )


# ------------------------------------------------------------------ #
# should_reconsider                                                   #
# ------------------------------------------------------------------ #

class TestShouldReconsider:
    def test_noise_ratio_change_triggers(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        flat = np.full(6, 0.05)
        reason = sel.should_reconsider(new_sigma=flat)
        assert reason is not None
        assert "Noise ratio" in reason

    def test_rho_change_triggers(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma, correlation_max=0.1)
        reason = sel.should_reconsider(new_rho_max=0.55)
        assert reason is not None
        assert "correlation" in reason

    def test_no_change_returns_none(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma, correlation_max=0.2)
        reason = sel.should_reconsider(new_sigma=hetero_sigma, new_rho_max=0.22)
        assert reason is None

    def test_lambda_drop_triggers(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma, correlation_max=0.1)
        reason = sel.should_reconsider(covariance_lambda=0.2)
        assert reason is not None
        assert "shrinkage" in reason.lower() or "stabiliz" in reason.lower()

    def test_lambda_drop_ignored_when_high_rho(self, hetero_sigma):
        # λ < 0.3 but ρ_max >= 0.3 → condition not met
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma, correlation_max=0.35)
        reason = sel.should_reconsider(covariance_lambda=0.2)
        assert reason is None

    def test_all_none_returns_none(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        assert sel.should_reconsider() is None

    def test_multiple_triggers_joined(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma, correlation_max=0.1)
        flat = np.full(6, 0.05)
        reason = sel.should_reconsider(new_sigma=flat, new_rho_max=0.6)
        assert reason is not None
        assert ";" in reason   # two reasons joined

    def test_small_rho_change_returns_none(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma, correlation_max=0.20)
        reason = sel.should_reconsider(new_rho_max=0.25)  # delta=0.05 < 0.15
        assert reason is None


# ------------------------------------------------------------------ #
# reset_comparison                                                    #
# ------------------------------------------------------------------ #

class TestResetComparison:
    def test_clears_all_counts(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        rng = np.random.default_rng(42)
        for _ in range(50):
            sel.record_comparison(rng.random(6), 0, mu, rng.integers(4),
                                  ["a", "b", "c", "d"])
        assert all(s.total_decisions == 50 for s in sel.scores.values())
        sel.reset_comparison()
        assert all(s.total_decisions == 0 for s in sel.scores.values())

    def test_clears_agreements(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        rng = np.random.default_rng(42)
        for _ in range(20):
            sel.record_comparison(rng.random(6), 0, mu, 0, ["a", "b", "c", "d"])
        sel.reset_comparison()
        assert all(s.agreements == 0 for s in sel.scores.values())

    def test_kernels_still_usable_after_reset(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        rng = np.random.default_rng(42)
        for _ in range(10):
            sel.record_comparison(rng.random(6), 0, mu, 0, ["a", "b", "c", "d"])
        sel.reset_comparison()
        sel.record_comparison(rng.random(6), 0, mu, 0, ["a", "b", "c", "d"])
        assert all(s.total_decisions == 1 for s in sel.scores.values())

    def test_reset_clears_buffers(self, hetero_sigma, mu):
        """reset_comparison() must also clear rolling buffers."""
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        rng = np.random.default_rng(42)
        for _ in range(30):
            sel.record_comparison(rng.random(6), 0, mu, 0, ["a", "b", "c", "d"])
        sel.reset_comparison()
        assert all(len(buf) == 0 for buf in sel._buffers.values())


# ------------------------------------------------------------------ #
# Rolling window tests                                                #
# ------------------------------------------------------------------ #

class TestRollingWindow:
    def test_window_size_configurable(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma, window_size=50)
        assert sel.window_size == 50

    def test_default_window_size(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        assert sel.window_size == 100

    def test_buffer_capped_at_window_size(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma, window_size=20)
        rng = np.random.default_rng(1)
        for _ in range(50):
            sel.record_comparison(rng.random(6), 0, mu, 0, ["a", "b", "c", "d"])
        assert all(len(buf) == 20 for buf in sel._buffers.values())

    def test_buffer_grows_before_cap(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma, window_size=50)
        rng = np.random.default_rng(2)
        for _ in range(30):
            sel.record_comparison(rng.random(6), 0, mu, 0, ["a", "b", "c", "d"])
        assert all(len(buf) == 30 for buf in sel._buffers.values())

    def test_rolling_beats_cumulative_on_improving_kernel(self, hetero_sigma):
        """Kernel B wins recently but A leads cumulatively. Rolling picks B."""
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        # Cumulative: l2 wins (110 vs 100 agreements)
        sel.scores["l2"].total_decisions = 200
        sel.scores["l2"].agreements = 110
        sel.scores["diagonal"].total_decisions = 200
        sel.scores["diagonal"].agreements = 100
        sel.scores["shrinkage"].total_decisions = 200
        sel.scores["shrinkage"].agreements = 95
        # Rolling: diagonal wins recently (70% vs l2 40%)
        sel._buffers["l2"] = [True] * 40 + [False] * 60
        sel._buffers["diagonal"] = [True] * 70 + [False] * 30
        sel._buffers["shrinkage"] = [True] * 65 + [False] * 35
        rec = sel.recommend()
        assert rec.recommended_kernel == "diagonal"

    def test_rolling_summary_includes_both_rates(self, hetero_sigma, mu):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        rng = np.random.default_rng(42)
        for _ in range(50):
            sel.record_comparison(rng.random(6), 0, mu, rng.integers(4),
                                  ["a", "b", "c", "d"])
        summary = sel.get_comparison_summary()
        for v in summary.values():
            assert "rolling_agreement_rate" in v
            assert "agreement_rate" in v

    def test_rolling_rate_reflects_recent_decisions(self, hetero_sigma):
        """Rolling rate matches manual sum of buffer."""
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma, window_size=10)
        sel._buffers["l2"] = [True, False, True, True, False, True, True, True, False, True]
        summary = sel.get_comparison_summary()
        expected = 7 / 10
        assert summary["l2"]["rolling_agreement_rate"] == pytest.approx(expected)

    def test_rolling_rate_zero_when_buffer_empty(self, hetero_sigma):
        sel = KernelSelector(d=6, sigma_per_factor=hetero_sigma)
        summary = sel.get_comparison_summary()
        assert all(v["rolling_agreement_rate"] == 0.0 for v in summary.values())

    def test_invalid_window_size_raises(self, hetero_sigma):
        with pytest.raises(AssertionError):
            KernelSelector(d=6, sigma_per_factor=hetero_sigma, window_size=0)
