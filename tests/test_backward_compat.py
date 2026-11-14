"""Legacy missing-field fixture and current ProfileScorer pickle regression tests."""

import pickle
from pathlib import Path

import numpy as np

from gae import build_profile_scorer


FIXTURE = Path(__file__).parent / "fixtures" / "v0.7.27_scorer_state.pkl"


def test_load_v0727_pickle() -> None:
    """F8: simulated v0.7.27 state restores defaults and remains functional."""
    assert FIXTURE.exists(), f"Fixture missing: {FIXTURE}"
    s = build_profile_scorer(
        categories=["routing", "validation"],
        actions=["approve", "flag", "escalate"],
        factors=["freshness", "quality", "reliability"],
    )
    with FIXTURE.open("rb") as handle:
        old_state = pickle.load(handle)
    for key in (
        "_verified_count", "_correct_count", "_centroid_drift", "_backend",
        "_scorer_id", "_backend_versions", "factors", "_factors",
    ):
        assert key not in old_state
    s.__setstate__(old_state)
    assert s.verified_count(0) == s.verified_count(1) == 0
    assert s.accuracy(0) == s.accuracy(1) == 0.0
    assert s.centroid_drift(0) == s.centroid_drift(1) == 0.0
    assert s.decision_count == 35
    assert not np.allclose(s.centroids, 0.5)
    result = s.score(np.array([0.5, 0.3, 0.8]), 0)
    assert result.action_index in [0, 1, 2]
    s.update(f=np.array([0.5, 0.3, 0.8]), category_index=0,
             action_index=0, correct=True)
    assert s.verified_count(0) == 1
    assert s.accuracy(0) == 1.0
    assert s.decision_count == 36


def test_v080_pickle_roundtrip(tmp_path: Path) -> None:
    """Current pickle save/load preserves learning and observability state."""
    s = build_profile_scorer(
        categories=["a", "b"], actions=["x", "y"], factors=["f1", "f2"],
    )
    for _ in range(10):
        s.update(f=np.array([0.5, 0.3]), category_index=0,
                 action_index=0, correct=True)
    path = tmp_path / "v080_test.pkl"
    s.save(path)
    s2 = build_profile_scorer(
        categories=["a", "b"], actions=["x", "y"], factors=["f1", "f2"],
    )
    s2.load(path)
    assert s2.verified_count(0) == 10
    assert s2.accuracy(0) == 1.0
    assert s2.verified_count(1) == 0
    assert s2.decision_count == s.decision_count
    assert s2.centroid_drift(0) == s.centroid_drift(0)
    assert s2.factors == s.factors
    np.testing.assert_array_equal(s2.centroids, s.centroids)
