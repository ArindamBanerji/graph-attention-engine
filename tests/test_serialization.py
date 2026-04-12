"""
Serialization tests for GAE.

Verifies that ProfileScorer centroids and LearningState survive
all common serialization roundtrips: numpy save/load, JSON (tolist/array),
pickle, and deepcopy. Score output must be identical before and after.
"""

from __future__ import annotations

import copy
import json
import pickle

import numpy as np
import pytest

from gae.profile_scorer import ProfileScorer
from gae.store import LearningState, save_state, load_state


def make_scorer(n_cat=3, n_act=4, n_fac=6, seed=42) -> ProfileScorer:
    rng = np.random.default_rng(seed)
    mu = rng.uniform(0.1, 0.9, (n_cat, n_act, n_fac))
    return ProfileScorer(mu=mu, actions=[f"a{i}" for i in range(n_act)])


# ── numpy save / load ─────────────────────────────────────────────────────────

class TestNumpySerialization:
    def test_centroids_numpy_save_load_same_scores(self, tmp_path):
        scorer = make_scorer()
        f = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        r_before = scorer.score(f, 0)

        path = tmp_path / "centroids.npy"
        np.save(str(path), scorer.centroids)
        loaded = np.load(str(path))

        scorer2 = ProfileScorer(mu=loaded, actions=[f"a{i}" for i in range(4)])
        r_after = scorer2.score(f, 0)
        np.testing.assert_array_equal(r_before.probabilities, r_after.probabilities)

    def test_centroids_numpy_save_load_preserves_shape(self, tmp_path):
        scorer = make_scorer(n_cat=5, n_act=3, n_fac=8)
        path = tmp_path / "centroids.npy"
        np.save(str(path), scorer.centroids)
        loaded = np.load(str(path))
        assert loaded.shape == (5, 3, 8)

    def test_centroids_numpy_savez_roundtrip(self, tmp_path):
        scorer = make_scorer()
        path = tmp_path / "data.npz"
        np.savez(str(path), centroids=scorer.centroids)
        data = np.load(str(path))
        np.testing.assert_array_equal(data["centroids"], scorer.centroids)

    def test_centroids_edge_values_survive_numpy_roundtrip(self, tmp_path):
        """0.0, 0.5, and 1.0 boundary values survive numpy save/load."""
        mu = np.array([[[0.0, 0.5, 1.0, 0.5]]])
        scorer = ProfileScorer(mu=mu, actions=["a"])
        path = tmp_path / "edge.npy"
        np.save(str(path), scorer.centroids)
        loaded = np.load(str(path))
        np.testing.assert_array_equal(loaded, scorer.centroids)

    def test_centroids_numpy_roundtrip_after_updates(self, tmp_path):
        """Centroids after 100 updates survive numpy save/load."""
        scorer = make_scorer(seed=1)
        for _ in range(100):
            scorer.update(np.full(6, 0.7), 0, 0, correct=True)
        mu_after = scorer.centroids.copy()

        path = tmp_path / "updated.npy"
        np.save(str(path), scorer.centroids)
        loaded = np.load(str(path))
        np.testing.assert_array_equal(loaded, mu_after)


# ── JSON serialization ────────────────────────────────────────────────────────

class TestJSONSerialization:
    def test_centroids_json_roundtrip_same_scores(self):
        scorer = make_scorer()
        f = np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        r_before = scorer.score(f, 1)

        json_str = json.dumps(scorer.centroids.tolist())
        restored = np.array(json.loads(json_str))

        scorer2 = ProfileScorer(mu=restored, actions=[f"a{i}" for i in range(4)])
        r_after = scorer2.score(f, 1)
        np.testing.assert_array_equal(r_before.probabilities, r_after.probabilities)

    def test_centroids_json_preserves_shape(self):
        scorer = make_scorer(n_cat=2, n_act=3, n_fac=4)
        restored = np.array(json.loads(json.dumps(scorer.centroids.tolist())))
        assert restored.shape == (2, 3, 4)

    def test_centroids_json_values_close(self):
        """JSON round-trip preserves values to at least 12 decimal places."""
        scorer = make_scorer()
        restored = np.array(json.loads(json.dumps(scorer.centroids.tolist())))
        np.testing.assert_allclose(restored, scorer.centroids, atol=1e-12)

    def test_large_tensor_json_roundtrip(self):
        """Large tensor (10, 8, 20) survives JSON round-trip."""
        rng = np.random.default_rng(77)
        mu = rng.uniform(0.0, 1.0, (10, 8, 20))
        scorer = ProfileScorer(mu=mu, actions=[f"a{i}" for i in range(8)])
        f = rng.uniform(0.0, 1.0, 20)
        r_before = scorer.score(f, 0)

        restored = np.array(json.loads(json.dumps(scorer.centroids.tolist())))
        scorer2 = ProfileScorer(mu=restored, actions=[f"a{i}" for i in range(8)])
        r_after = scorer2.score(f, 0)
        np.testing.assert_allclose(r_before.probabilities, r_after.probabilities, atol=1e-12)

    def test_centroids_json_is_valid_json(self):
        """Centroid tolist() produces valid JSON without errors."""
        scorer = make_scorer()
        try:
            json.dumps(scorer.centroids.tolist())
        except (TypeError, ValueError) as e:
            pytest.fail(f"centroids.tolist() is not JSON-serializable: {e}")


# ── pickle serialization ──────────────────────────────────────────────────────

class TestPickleSerialization:
    def test_pickle_roundtrip_same_scores(self):
        scorer = make_scorer()
        f = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        r_before = scorer.score(f, 0)

        buf = pickle.dumps(scorer)
        scorer2 = pickle.loads(buf)
        r_after = scorer2.score(f, 0)
        np.testing.assert_array_equal(r_before.probabilities, r_after.probabilities)

    def test_pickle_preserves_centroids(self):
        scorer = make_scorer()
        mu_orig = scorer.centroids.copy()
        scorer2 = pickle.loads(pickle.dumps(scorer))
        np.testing.assert_array_equal(scorer2.centroids, mu_orig)

    def test_pickle_preserves_n_categories(self):
        scorer = make_scorer(n_cat=7)
        scorer2 = pickle.loads(pickle.dumps(scorer))
        assert scorer2.n_categories == 7

    def test_pickle_instance_independence(self):
        """Unpickled scorer is independent of original."""
        scorer = make_scorer()
        scorer2 = pickle.loads(pickle.dumps(scorer))
        mu_before = scorer2.centroids.copy()
        scorer.update(np.full(6, 0.5), 0, 0, correct=True)
        np.testing.assert_array_equal(scorer2.centroids, mu_before)

    def test_pickle_after_updates(self):
        """Pickle works after centroids have been modified by updates."""
        scorer = make_scorer()
        for _ in range(100):
            scorer.update(np.full(6, 0.7), 0, 0, correct=True)
        mu_after_updates = scorer.centroids.copy()
        scorer2 = pickle.loads(pickle.dumps(scorer))
        np.testing.assert_array_equal(scorer2.centroids, mu_after_updates)


# ── deepcopy ──────────────────────────────────────────────────────────────────

class TestDeepcopy:
    def test_deepcopy_same_scores(self):
        scorer = make_scorer()
        f = np.full(6, 0.42)
        r_before = scorer.score(f, 0)
        scorer2 = copy.deepcopy(scorer)
        r_after = scorer2.score(f, 0)
        np.testing.assert_array_equal(r_before.probabilities, r_after.probabilities)

    def test_deepcopy_is_independent(self):
        scorer = make_scorer()
        scorer2 = copy.deepcopy(scorer)
        mu_before = scorer2.centroids.copy()
        scorer.update(np.full(6, 0.5), 0, 0, correct=True)
        np.testing.assert_array_equal(scorer2.centroids, mu_before)

    def test_deepcopy_across_multiple_categories(self):
        scorer = make_scorer(n_cat=3)
        for c in range(3):
            f = np.full(6, float(c) * 0.3 + 0.1)
            r_orig = scorer.score(f, c)
            scorer2 = copy.deepcopy(scorer)
            r_copy = scorer2.score(f, c)
            np.testing.assert_array_equal(r_orig.probabilities, r_copy.probabilities)


# ── LearningState persistence ─────────────────────────────────────────────────

class TestLearningStateSerialization:
    def test_save_load_weights_roundtrip(self, tmp_path):
        weights = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        state = LearningState(weights=weights.copy(), step=10, converged=False)
        path = tmp_path / "state.json"
        save_state(state, path)
        loaded = load_state(path)
        np.testing.assert_array_equal(loaded.weights, weights)

    def test_save_load_step_preserved(self, tmp_path):
        weights = np.array([0.5, 0.5])
        state = LearningState(weights=weights, step=42)
        path = tmp_path / "state.json"
        save_state(state, path)
        loaded = load_state(path)
        assert loaded.step == 42

    def test_save_load_converged_true(self, tmp_path):
        weights = np.array([0.5, 0.5])
        state = LearningState(weights=weights, converged=True)
        path = tmp_path / "state.json"
        save_state(state, path)
        loaded = load_state(path)
        assert loaded.converged is True

    def test_save_load_converged_false(self, tmp_path):
        weights = np.array([0.5, 0.5])
        state = LearningState(weights=weights, converged=False)
        path = tmp_path / "state.json"
        save_state(state, path)
        loaded = load_state(path)
        assert loaded.converged is False

    def test_save_load_metadata_preserved(self, tmp_path):
        weights = np.array([0.3, 0.7])
        state = LearningState(
            weights=weights, metadata={"version": "1.0", "domain": "soc"}
        )
        path = tmp_path / "state.json"
        save_state(state, path)
        loaded = load_state(path)
        assert loaded.metadata["version"] == "1.0"
        assert loaded.metadata["domain"] == "soc"

    def test_save_load_weights_shape_preserved(self, tmp_path):
        weights = np.zeros(100)
        state = LearningState(weights=weights)
        path = tmp_path / "state.json"
        save_state(state, path)
        loaded = load_state(path)
        assert loaded.weights.shape == (100,)

    def test_save_load_weights_values_preserved(self, tmp_path):
        rng = np.random.default_rng(5)
        weights = rng.uniform(0.0, 1.0, 20)
        state = LearningState(weights=weights.copy())
        path = tmp_path / "state.json"
        save_state(state, path)
        loaded = load_state(path)
        np.testing.assert_allclose(loaded.weights, weights, atol=1e-12)

    def test_learning_state_weights_must_be_1d(self):
        with pytest.raises(AssertionError):
            LearningState(weights=np.zeros((3, 4)))

    def test_learning_state_step_must_be_nonnegative(self):
        with pytest.raises(AssertionError):
            LearningState(weights=np.array([0.5]), step=-1)

    def test_save_load_step_zero(self, tmp_path):
        weights = np.array([0.5, 0.3, 0.1])
        state = LearningState(weights=weights, step=0)
        path = tmp_path / "state.json"
        save_state(state, path)
        loaded = load_state(path)
        assert loaded.step == 0

    def test_overwrite_existing_file(self, tmp_path):
        """save_state overwrites an existing file safely."""
        weights_v1 = np.array([0.1, 0.2])
        weights_v2 = np.array([0.9, 0.8])
        path = tmp_path / "state.json"

        save_state(LearningState(weights=weights_v1, step=1), path)
        save_state(LearningState(weights=weights_v2, step=2), path)

        loaded = load_state(path)
        np.testing.assert_array_equal(loaded.weights, weights_v2)
        assert loaded.step == 2
