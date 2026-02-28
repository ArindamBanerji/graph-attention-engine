"""Tests for gae.store — LearningState serialisation and atomic file I/O."""

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from gae.store import LearningState, save_state, load_state


class TestLearningState:
    def test_construction(self):
        w = np.array([0.1, 0.2, 0.3])
        s = LearningState(weights=w, step=5, converged=False)
        assert s.step == 5
        assert s.weights.shape == (3,)

    def test_negative_step_raises(self):
        with pytest.raises(AssertionError):
            LearningState(weights=np.array([1.0]), step=-1)

    def test_2d_weights_raise(self):
        with pytest.raises(AssertionError):
            LearningState(weights=np.array([[1.0, 2.0]]))

    def test_non_array_raises(self):
        with pytest.raises(AssertionError):
            LearningState(weights=[1.0, 2.0])  # type: ignore[arg-type]

    def test_to_dict_roundtrip(self):
        w = np.array([0.5, 0.6, 0.7])
        s = LearningState(weights=w, step=3, converged=True, metadata={"v": "1"})
        d = s.to_dict()
        assert d["step"] == 3
        assert d["converged"] is True
        assert d["weights"] == pytest.approx([0.5, 0.6, 0.7])
        assert d["metadata"] == {"v": "1"}

    def test_from_dict_roundtrip(self):
        w = np.array([1.0, 2.0])
        original = LearningState(weights=w, step=7, converged=False)
        restored = LearningState.from_dict(original.to_dict())
        np.testing.assert_array_almost_equal(restored.weights, original.weights)
        assert restored.step == 7

    def test_from_dict_bad_shape_raises(self):
        with pytest.raises(ValueError):
            LearningState.from_dict({"weights": [[1.0, 2.0], [3.0, 4.0]]})

    def test_from_dict_missing_weights_raises(self):
        with pytest.raises(KeyError):
            LearningState.from_dict({"step": 0})


class TestSaveLoadState:
    def test_roundtrip(self, tmp_path):
        w = np.array([0.1, 0.9, 0.5])
        state = LearningState(weights=w, step=42, converged=True)
        path = tmp_path / "state.json"
        save_state(state, path)
        loaded = load_state(path)
        np.testing.assert_array_almost_equal(loaded.weights, w)
        assert loaded.step == 42
        assert loaded.converged is True

    def test_file_is_valid_json(self, tmp_path):
        state = LearningState(weights=np.array([1.0]))
        path = tmp_path / "state.json"
        save_state(state, path)
        with open(path) as f:
            data = json.load(f)
        assert "weights" in data

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_state(tmp_path / "does_not_exist.json")

    def test_overwrite_previous(self, tmp_path):
        path = tmp_path / "state.json"
        s1 = LearningState(weights=np.array([0.1, 0.2]), step=1)
        s2 = LearningState(weights=np.array([0.9, 0.8]), step=2)
        save_state(s1, path)
        save_state(s2, path)
        loaded = load_state(path)
        assert loaded.step == 2

    def test_non_learning_state_raises(self, tmp_path):
        with pytest.raises(AssertionError):
            save_state({"weights": [1.0]}, tmp_path / "x.json")  # type: ignore[arg-type]

    def test_metadata_roundtrip(self, tmp_path):
        meta = {"schema_version": "2", "node_type": "host"}
        state = LearningState(weights=np.array([0.3]), metadata=meta)
        path = tmp_path / "m.json"
        save_state(state, path)
        loaded = load_state(path)
        assert loaded.metadata == meta
