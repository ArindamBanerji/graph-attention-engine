"""
Determinism tests for GAE.

An open-source library must produce identical results given identical inputs,
regardless of call order, instance count, or prior history. Tests here
verify there is no hidden global mutable state, no random seeding, and no
order-dependent behaviour.
"""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

from gae.kernels import DiagonalKernel, L2Kernel
from gae.profile_scorer import ProfileScorer, ScoringResult


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_scorer(seed=42) -> ProfileScorer:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.1, 0.9, (3, 4, 6))
    return ProfileScorer(mu=mu, actions=["a0", "a1", "a2", "a3"])


# ── score() determinism ───────────────────────────────────────────────────────

class TestScoreDeterminism:
    def test_same_input_same_output_100_times(self):
        """Calling score() 100 times with identical input yields identical output."""
        scorer = make_scorer()
        f = np.array([0.1, 0.3, 0.5, 0.7, 0.4, 0.2])
        results = [scorer.score(f, 0) for _ in range(100)]
        ref = results[0].probabilities
        for i, r in enumerate(results[1:], start=1):
            np.testing.assert_array_equal(
                r.probabilities, ref,
                err_msg=f"Call {i} produced different probabilities",
            )

    def test_score_independent_of_previous_category_scored(self):
        """Scoring category 1 does not alter the output of scoring category 0."""
        scorer = make_scorer()
        f = np.array([0.4, 0.6, 0.3, 0.5, 0.7, 0.2])
        # Score cat 0 before and after scoring cat 1 (different category)
        r_before = scorer.score(f, 0)
        _ = scorer.score(f, 1)  # side-effect test
        _ = scorer.score(f, 2)
        r_after = scorer.score(f, 0)
        np.testing.assert_array_equal(
            r_before.probabilities, r_after.probabilities,
            err_msg="Scoring other categories must not affect category 0",
        )

    def test_score_independent_of_decision_count(self):
        """decision_count increments only via update(); score() must not increment it."""
        scorer = make_scorer()
        f = np.full(6, 0.5)
        r1 = scorer.score(f, 0)
        count_after_score = scorer.decision_count
        r2 = scorer.score(f, 0)
        assert scorer.decision_count == count_after_score, "score() must not mutate decision_count"
        np.testing.assert_array_equal(r1.probabilities, r2.probabilities)

    def test_score_result_is_not_cached_reference(self):
        """Each score() call returns a fresh ScoringResult; modifying one does not corrupt another."""
        scorer = make_scorer()
        f = np.full(6, 0.5)
        r1 = scorer.score(f, 0)
        r1_probs_copy = r1.probabilities.copy()
        r2 = scorer.score(f, 0)
        # Mutate r1's probabilities (simulating user error)
        r1.probabilities[:] = 0.0
        # r2 must be unaffected
        np.testing.assert_array_equal(r2.probabilities, r1_probs_copy)


# ── update() determinism ──────────────────────────────────────────────────────

class TestUpdateDeterminism:
    def test_same_update_sequence_same_centroids(self):
        """Two scorers with same initial centroids, same update sequence → same mu."""
        rng = np.random.default_rng(7)
        mu_init = rng.uniform(0.0, 1.0, (2, 3, 6))
        actions = ["a0", "a1", "a2"]

        s1 = ProfileScorer(mu=mu_init.copy(), actions=actions)
        s2 = ProfileScorer(mu=mu_init.copy(), actions=actions)

        seq_rng = np.random.default_rng(99)
        for _ in range(500):
            f = seq_rng.uniform(0.0, 1.0, 6)
            c = int(seq_rng.integers(2))
            a = int(seq_rng.integers(3))
            correct = bool(seq_rng.integers(2))
            s1.update(f, c, a, correct=correct)
            s2.update(f, c, a, correct=correct)

        np.testing.assert_array_equal(s1.centroids, s2.centroids)

    def test_update_does_not_mutate_input_array(self):
        """update() must not mutate the caller's f array."""
        scorer = make_scorer()
        f = np.array([0.2, 0.4, 0.6, 0.8, 0.3, 0.5])
        f_copy = f.copy()
        scorer.update(f, 0, 0, correct=True)
        np.testing.assert_array_equal(f, f_copy, err_msg="update() must not mutate input f")


# ── Instance isolation ────────────────────────────────────────────────────────

class TestInstanceIsolation:
    def test_two_instances_same_centroids_same_scores(self):
        """Two separate ProfileScorer instances with identical mu produce identical scores."""
        rng = np.random.default_rng(11)
        mu = rng.uniform(0.0, 1.0, (2, 3, 6))
        s1 = ProfileScorer(mu=mu.copy(), actions=["a", "b", "c"])
        s2 = ProfileScorer(mu=mu.copy(), actions=["a", "b", "c"])
        f = np.array([0.3, 0.5, 0.7, 0.2, 0.4, 0.6])
        r1 = s1.score(f, 0)
        r2 = s2.score(f, 0)
        np.testing.assert_array_equal(r1.probabilities, r2.probabilities)

    def test_updates_on_instance_a_dont_affect_instance_b(self):
        """Updating s1 must not affect s2 (no shared mu reference)."""
        rng = np.random.default_rng(12)
        mu = rng.uniform(0.0, 1.0, (1, 2, 4))
        s1 = ProfileScorer(mu=mu.copy(), actions=["a", "b"])
        s2 = ProfileScorer(mu=mu.copy(), actions=["a", "b"])
        mu2_before = s2.centroids.copy()

        for _ in range(100):
            s1.update(np.ones(4), 0, 0, correct=True)

        np.testing.assert_array_equal(
            s2.centroids,
            mu2_before,
            err_msg="s2 must not be affected by s1 updates",
        )

    def test_constructor_copies_mu(self):
        """ProfileScorer must copy mu at construction; external mutation must not propagate."""
        rng = np.random.default_rng(13)
        mu = rng.uniform(0.0, 1.0, (1, 2, 4))
        mu_original = mu.copy()
        scorer = ProfileScorer(mu=mu, actions=["a", "b"])
        # Mutate the external array
        mu[:] = 999.0
        # Scorer's centroids must be unaffected
        np.testing.assert_array_equal(scorer.centroids, mu_original)

    def test_centroids_property_returns_correct_shape(self):
        """scorer.centroids exposes the centroid tensor via the public API."""
        scorer = make_scorer()
        assert scorer.centroids is not None
        assert scorer.centroids.shape == (3, 4, 6)


# ── Serialize → deserialize determinism ──────────────────────────────────────

class TestSerializeDeserializeDeterminism:
    def test_score_same_after_centroid_roundtrip(self):
        """Saving and restoring centroids via numpy gives identical score output."""
        scorer = make_scorer()
        f = np.array([0.6, 0.2, 0.8, 0.4, 0.1, 0.7])
        r_before = scorer.score(f, 0)

        # Serialize/deserialize centroids
        saved = scorer.centroids.tolist()
        restored = np.array(saved)

        scorer2 = ProfileScorer(mu=restored, actions=["a0", "a1", "a2", "a3"])
        r_after = scorer2.score(f, 0)
        np.testing.assert_array_equal(
            r_before.probabilities, r_after.probabilities,
            err_msg="Score must match after centroid JSON roundtrip",
        )

    def test_score_same_after_pickle_roundtrip(self):
        """Pickling and unpickling ProfileScorer gives identical score output."""
        scorer = make_scorer()
        f = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        r_before = scorer.score(f, 1)

        buf = pickle.dumps(scorer)
        scorer2 = pickle.loads(buf)
        r_after = scorer2.score(f, 1)

        np.testing.assert_array_equal(r_before.probabilities, r_after.probabilities)

    def test_score_same_after_deepcopy(self):
        """deepcopy(scorer) gives identical score output."""
        scorer = make_scorer()
        f = np.full(6, 0.42)
        r_before = scorer.score(f, 0)

        scorer2 = copy.deepcopy(scorer)
        r_after = scorer2.score(f, 0)
        np.testing.assert_array_equal(r_before.probabilities, r_after.probabilities)


# ── Kernel determinism ────────────────────────────────────────────────────────

class TestKernelDeterminism:
    def test_l2_kernel_same_input_same_distance(self):
        k = L2Kernel()
        f = np.array([0.3, 0.7, 0.2, 0.8])
        mu = np.array([[0.5, 0.5, 0.5, 0.5], [0.1, 0.9, 0.1, 0.9]])
        d1 = k.compute_distance(f, mu)
        d2 = k.compute_distance(f, mu)
        np.testing.assert_array_equal(d1, d2)

    def test_diagonal_kernel_same_sigma_same_weights(self):
        sigma = np.array([0.1, 0.3, 0.2, 0.5])
        k1 = DiagonalKernel(sigma)
        k2 = DiagonalKernel(sigma)
        np.testing.assert_array_equal(k1.weights, k2.weights)

    def test_diagonal_kernel_same_input_same_distance(self):
        sigma = np.array([0.2, 0.4, 0.1, 0.8])
        k = DiagonalKernel(sigma)
        f = np.array([0.5, 0.3, 0.7, 0.2])
        mu = np.array([[0.1, 0.9, 0.4, 0.6]])
        d1 = k.compute_distance(f, mu)
        d2 = k.compute_distance(f, mu)
        np.testing.assert_array_equal(d1, d2)
