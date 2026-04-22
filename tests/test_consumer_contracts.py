"""
Consumer contract tests — validate the API surface that external
consumers (SOC backend, Colab notebooks, copilot-sdk) depend on.

These tests exercise the PATTERNS consumers write, not internal
engine behavior. Each test represents a real consumer failure
mode discovered during integration testing.
"""
import numpy as np
import pytest
import warnings


class TestConstructorContracts:
    """Consumers must get clear errors for wrong constructor args."""

    def test_constructor_rejects_eta_kwarg(self):
        """ProfileScorer(eta=0.05) is not valid — eta comes via CalibrationProfile."""
        from gae.profile_scorer import ProfileScorer
        with pytest.raises(TypeError):
            ProfileScorer(
                mu=np.random.rand(6, 4, 6),
                actions=["escalate", "investigate", "suppress", "monitor"],
                eta=0.05,
            )

    def test_constructor_rejects_eta_neg_kwarg(self):
        """ProfileScorer(eta_neg=0.05) is not valid."""
        from gae.profile_scorer import ProfileScorer
        with pytest.raises(TypeError):
            ProfileScorer(
                mu=np.random.rand(6, 4, 6),
                actions=["escalate", "investigate", "suppress", "monitor"],
                eta_neg=0.05,
            )


class TestScoringResultContract:
    """Consumers must not confuse ScoringResult with ndarray."""

    def test_score_returns_scoring_result_not_ndarray(self):
        """score() returns ScoringResult, not np.ndarray."""
        from gae.profile_scorer import ProfileScorer, ScoringResult
        scorer = ProfileScorer.for_soc(mu=np.random.rand(6, 4, 6))
        result = scorer.score(np.random.rand(6), 0)
        assert isinstance(result, ScoringResult)
        assert not isinstance(result, np.ndarray)

    def test_scoring_result_has_required_fields(self):
        """ScoringResult exposes action_index, probabilities, confidence, entropy, confidence_gap."""
        from gae.profile_scorer import ProfileScorer
        scorer = ProfileScorer.for_soc(mu=np.random.rand(6, 4, 6))
        result = scorer.score(np.random.rand(6), 0)
        assert hasattr(result, 'action_index')
        assert hasattr(result, 'action_name')
        assert hasattr(result, 'probabilities')
        assert hasattr(result, 'distances')
        assert hasattr(result, 'confidence')
        assert hasattr(result, 'entropy')
        assert hasattr(result, 'confidence_gap')

    def test_probabilities_sum_to_one(self):
        """result.probabilities is a valid probability distribution."""
        from gae.profile_scorer import ProfileScorer
        scorer = ProfileScorer.for_soc(mu=np.random.rand(6, 4, 6))
        result = scorer.score(np.random.rand(6), 0)
        assert result.probabilities.shape == (4,)
        assert abs(result.probabilities.sum() - 1.0) < 1e-6
        assert (result.probabilities >= 0).all()

    def test_action_index_matches_argmax_probabilities(self):
        """action_index is argmax of probabilities."""
        from gae.profile_scorer import ProfileScorer
        scorer = ProfileScorer.for_soc(mu=np.random.rand(6, 4, 6))
        result = scorer.score(np.random.rand(6), 0)
        assert result.action_index == int(np.argmax(result.probabilities))

    def test_top_level_import_is_profile_scorer_version(self):
        """from gae import ScoringResult gets the current one, not deprecated."""
        from gae import ScoringResult
        assert ScoringResult.__module__ == 'gae.profile_scorer'


class TestDeprecationWarnings:
    """Deprecated APIs must warn consumers."""

    def test_score_entity_emits_deprecation_warning(self):
        """score_entity() should emit DeprecationWarning."""
        from gae.scoring import score_entity
        W = np.random.rand(4, 6)   # (n_actions, n_factors)
        f = np.random.rand(1, 6)   # score_entity expects (1, n_f)
        actions = ["a", "b", "c", "d"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            score_entity(f, W, actions)
            deprecations = [x for x in w
                          if issubclass(x.category, DeprecationWarning)]
            assert len(deprecations) >= 1


class TestConsumerSequences:
    """End-to-end sequences that real consumers execute."""

    def test_sequence_a_for_soc_score_learn_loop(self):
        """for_soc() → score() → update() — the basic consumer loop."""
        from gae.profile_scorer import ProfileScorer
        mu_init = np.random.rand(6, 4, 6)
        scorer = ProfileScorer.for_soc(mu=mu_init.copy())
        assert scorer.eta_override == 0.01

        for i in range(20):
            f = np.random.rand(6)
            cat = i % 6
            result = scorer.score(f, cat)
            pred = result.action_index
            gt = np.random.randint(4)
            if pred == gt:
                scorer.update(f, cat, pred, correct=True)
            else:
                scorer.update(f, cat, pred, correct=False,
                              gt_action_index=gt)

        # Centroids should have moved
        assert not np.allclose(scorer.centroids, mu_init)

    def test_sequence_b_diagonal_kernel_lifecycle(self):
        """DiagonalKernel construction → for_soc → per-factor N_half."""
        from gae.profile_scorer import ProfileScorer
        from gae.kernels import DiagonalKernel
        from gae.profile_scorer import KernelType
        from gae.convergence import compute_per_factor_n_half

        sigma = np.array([0.15, 0.20, 0.07, 0.10, 0.28, 0.35])
        dk = DiagonalKernel(sigma=sigma)
        scorer = ProfileScorer.for_soc(
            mu=np.random.rand(6, 4, 6),
            kernel=KernelType.DIAGONAL,
            scoring_kernel=dk,
        )
        n_half = compute_per_factor_n_half(dk.weights, eta=0.05)

        # Per-factor N_half should vary (not all equal)
        assert n_half.shape == (6,)
        assert n_half.max() / n_half.min() > 2.0

        # Score and learn should work with diagonal kernel
        f = np.random.rand(6)
        result = scorer.score(f, 0)
        assert result.probabilities.shape == (4,)
        scorer.update(f, 0, result.action_index, correct=True)

    def test_sequence_c_kernel_selection_lifecycle(self):
        """Rule → shadow accumulation → empirical recommendation."""
        from gae.kernel_selector import KernelSelector

        sigma = np.array([0.10, 0.18, 0.26, 0.34, 0.42, 0.50])
        selector = KernelSelector(d=6, sigma_per_factor=sigma)

        # Phase 2: rule-based
        rec = selector.preliminary_recommendation()
        assert rec.method == 'rule'
        assert rec.recommended_kernel in ('l2', 'diagonal', 'shrinkage')

        # Phase 3: accumulate shadow data
        mu = np.random.rand(6, 4, 6)
        actions = ["escalate", "investigate", "suppress", "monitor"]
        for i in range(200):
            f = np.random.rand(6)
            selector.record_comparison(
                f, i % 6, mu, np.random.randint(4), actions)

        # Phase 4: rule-based recommendation with monitoring note
        rec = selector.recommend()
        assert rec.sufficient_data == True
        assert rec.method == 'rule'

    def test_sequence_d_conservation_wiring(self):
        """derive_theta_min → check_conservation → set_conservation_status."""
        from gae.profile_scorer import ProfileScorer
        from gae.calibration import derive_theta_min, check_conservation

        theta = derive_theta_min()
        assert abs(theta - 0.467) < 0.001

        status = check_conservation(
            alpha=0.3, q=0.9, V=100.0, theta_min=theta)
        assert hasattr(status, 'status')
        assert status.status in ('GREEN', 'AMBER', 'RED')

        scorer = ProfileScorer.for_soc(mu=np.random.rand(6, 4, 6))
        scorer.set_conservation_status(status.status)
        assert scorer.conservation_status == status.status


class TestEntropyAndConfidenceGap:
    """ScoringResult entropy and confidence_gap field contracts."""

    def test_entropy_uniform_distribution(self):
        """Uniform probs → maximum entropy."""
        from gae.primitives import compute_entropy
        p = np.array([0.25, 0.25, 0.25, 0.25])
        e = compute_entropy(p)
        assert abs(e - np.log(4)) < 1e-6

    def test_entropy_one_hot(self):
        """One-hot probs → near-zero entropy."""
        from gae.primitives import compute_entropy
        p = np.array([1.0, 0.0, 0.0, 0.0])
        e = compute_entropy(p)
        assert e > -1e-8   # corrected formula: p*log(p+eps) yields ~-eps for p=1
        assert e < 0.01

    def test_scoring_result_has_entropy_and_gap(self):
        """ScoringResult includes entropy and confidence_gap."""
        from gae.profile_scorer import ProfileScorer
        scorer = ProfileScorer.for_soc(mu=np.random.rand(6, 4, 6))
        result = scorer.score(np.random.rand(6), 0)
        assert hasattr(result, 'entropy')
        assert hasattr(result, 'confidence_gap')
        assert result.entropy >= 0.0
        assert result.confidence_gap >= 0.0
        assert result.confidence_gap <= 1.0

    def test_confidence_gap_single_action(self):
        """Single action → confidence_gap = 0.0."""
        from gae.profile_scorer import ProfileScorer
        scorer = ProfileScorer(
            mu=np.random.rand(1, 1, 4),
            actions=["only_action"],
        )
        result = scorer.score(np.random.rand(4), 0)
        assert result.confidence_gap == 0.0
        assert result.entropy > -1e-8   # corrected formula: tiny negative possible for p=1

    def test_entropy_decreases_with_confidence(self):
        """Higher confidence → lower entropy (more decisive)."""
        from gae.profile_scorer import ProfileScorer
        mu = np.zeros((2, 4, 6))
        mu[0, 0, :] = 0.5
        mu[0, 1, :] = 0.9
        mu[0, 2, :] = 0.9
        mu[0, 3, :] = 0.9
        scorer = ProfileScorer.for_soc(mu=mu)
        f_near = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        r_near = scorer.score(f_near, 0)
        f_mid = np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        r_mid = scorer.score(f_mid, 0)
        assert r_near.entropy < r_mid.entropy


# ── GAE-05b: compute_ece public API ──────────────────────────────────────────

class TestComputeEceContract:
    """GAE-05b: compute_ece must be importable and return a non-negative float."""

    def test_compute_ece_importable(self):
        from gae import compute_ece
        assert callable(compute_ece)

    def test_compute_ece_returns_float(self):
        """ECE of random predictions is a non-negative float."""
        from gae.evaluation import compute_ece
        predictions = np.random.rand(100).tolist()
        actuals = (np.random.rand(100) > 0.5).tolist()
        result = compute_ece(predictions, actuals)
        assert isinstance(result, float)
        assert result >= 0.0


# ── GAE-07: Deeper consumer sequence tests ───────────────────────────────────

class TestConsumerSequences:
    """GAE-07: Multi-step score→update sequences."""

    def test_consumer_sequence_score_update_score_changes(self):
        """After update, scoring the same input gives different probabilities."""
        from gae.profile_scorer import ProfileScorer
        scorer = ProfileScorer.for_soc(mu=np.full((6, 4, 6), 0.5))
        f = np.random.rand(6)
        r1 = scorer.score(f, category_index=0)
        scorer.update(f, category_index=0, action_index=0, correct=True)
        r2 = scorer.score(f, category_index=0)
        assert not np.allclose(r1.probabilities, r2.probabilities)

    def test_consumer_sequence_freeze_update_no_change(self):
        """Frozen scorer ignores update calls."""
        from gae.profile_scorer import ProfileScorer
        scorer = ProfileScorer.for_soc(mu=np.full((6, 4, 6), 0.5))
        f = np.random.rand(6)
        r1 = scorer.score(f, category_index=0)
        scorer.freeze()
        scorer.update(f, category_index=0, action_index=0, correct=True)
        r2 = scorer.score(f, category_index=0)
        assert np.allclose(r1.probabilities, r2.probabilities)

    def test_consumer_sequence_multi_category_independence(self):
        """Updates to category 0 don't affect category 1."""
        from gae.profile_scorer import ProfileScorer
        scorer = ProfileScorer.for_soc(mu=np.full((6, 4, 6), 0.5))
        f = np.random.rand(6)
        r_cat1_before = scorer.score(f, category_index=1)
        scorer.update(f, category_index=0, action_index=0, correct=True)
        r_cat1_after = scorer.score(f, category_index=1)
        assert np.allclose(
            r_cat1_before.probabilities,
            r_cat1_after.probabilities)

    def test_consumer_sequence_centroids_reflect_updates(self):
        """After update, centroids tensor has changed."""
        from gae.profile_scorer import ProfileScorer
        scorer = ProfileScorer.for_soc(mu=np.full((6, 4, 6), 0.5))
        before = scorer.centroids.copy()
        f = np.random.rand(6)
        scorer.update(f, category_index=0, action_index=0, correct=True)
        assert not np.allclose(scorer.centroids, before)


# ── GAE-09: calibration.py standalone importable ─────────────────────────────

class TestCalibrationImportable:
    """GAE-09: CalibrationProfile must be importable from gae."""

    def test_calibration_importable(self):
        from gae import CalibrationProfile
        assert callable(CalibrationProfile)
