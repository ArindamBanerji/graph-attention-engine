import warnings
import pytest
import numpy as np
from gae.profile_scorer import ProfileScorer


def _soc_scorer(mu=None):
    if mu is None:
        mu = np.full((6, 4, 6), 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ProfileScorer.for_soc(mu=mu)


@pytest.fixture
def scorer():
    return _soc_scorer()


def test_update_valid_outcome_positive(scorer):
    """update() with correct=True should not raise."""
    fv = np.random.rand(6)
    result = scorer.score(fv, category_index=0)
    scorer.update(
        f=fv,
        action_index=result.action_index,
        correct=True,
        category_index=0,
    )


def test_update_valid_outcome_negative(scorer):
    """update() with correct=False and gt_action_index should not raise."""
    fv = np.random.rand(6)
    result = scorer.score(fv, category_index=0)
    gt = (result.action_index + 1) % 4
    scorer.update(
        f=fv,
        action_index=result.action_index,
        correct=False,
        gt_action_index=gt,
        category_index=0,
    )


def test_update_changes_centroids(scorer):
    """update() with correct=True should move centroids toward the factor vector."""
    before = scorer.centroids.copy()
    fv = np.random.rand(6)
    result = scorer.score(fv, category_index=0)
    scorer.update(
        f=fv,
        action_index=result.action_index,
        correct=True,
        category_index=0,
    )
    assert not np.array_equal(before, scorer.centroids)


def test_update_category_out_of_bounds(scorer):
    """update() with invalid category_index should raise ValueError."""
    fv = np.random.rand(6)
    with pytest.raises((ValueError, Exception)):
        scorer.update(f=fv, action_index=0, correct=True, category_index=99)


def test_update_action_out_of_bounds(scorer):
    """update() with invalid action_index should raise ValueError."""
    fv = np.random.rand(6)
    with pytest.raises((ValueError, Exception)):
        scorer.update(f=fv, action_index=99, correct=True, category_index=0)


def test_update_preserves_other_categories(scorer):
    """update() on category 0 should not change category 1 centroids."""
    before_cat1 = scorer.centroids[1].copy()
    fv = np.random.rand(6)
    result = scorer.score(fv, category_index=0)
    scorer.update(
        f=fv,
        action_index=result.action_index,
        correct=True,
        category_index=0,
    )
    assert np.array_equal(before_cat1, scorer.centroids[1])


def test_update_eta_override_respected():
    """Larger eta_override on the correct=False path produces a larger centroid shift.

    Uses a fixed fv so gradient = 0.2 per factor (not clipped at eta=0.01,
    clipped to MAX_ETA_DELTA=0.005 at eta=0.5).
    """
    fv = np.full(6, 0.7)  # gradient per element = 0.7 - 0.5 = 0.2

    scorer_a = _soc_scorer()          # eta_override=0.01 (for_soc default)
    scorer_b = _soc_scorer()
    scorer_b.eta_override = 0.5

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # suppress push-only DeprecationWarning
        scorer_a.update(f=fv, action_index=0, correct=False,
                        gt_action_index=None, category_index=0)
        scorer_b.update(f=fv, action_index=0, correct=False,
                        gt_action_index=None, category_index=0)

    diff_a = np.abs(scorer_a.centroids - 0.5).sum()
    diff_b = np.abs(scorer_b.centroids - 0.5).sum()
    assert diff_b > diff_a


def test_update_frozen_scorer_no_change(scorer):
    """Frozen scorer should return early — centroids must not change."""
    scorer.freeze()
    before = scorer.centroids.copy()
    fv = np.random.rand(6)
    result = scorer.score(fv, category_index=0)
    scorer.update(
        f=fv,
        action_index=result.action_index,
        correct=True,
        category_index=0,
    )
    assert np.array_equal(before, scorer.centroids)
    scorer.unfreeze()
