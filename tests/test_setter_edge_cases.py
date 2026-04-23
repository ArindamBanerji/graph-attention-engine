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


def test_centroids_setter_rejects_inf():
    scorer = _soc_scorer()
    with pytest.raises(ValueError):
        scorer.centroids = np.full((6, 4, 6), np.inf)


def test_centroids_setter_rejects_neg_inf():
    scorer = _soc_scorer()
    with pytest.raises(ValueError):
        scorer.centroids = np.full((6, 4, 6), -np.inf)


def test_centroids_setter_creates_copy():
    """Setter copies data — mutation of source doesn't affect internal centroids."""
    scorer = _soc_scorer()
    external = np.full((6, 4, 6), 0.7)
    scorer.centroids = external
    external[0, 0, 0] = 999.0
    assert scorer.centroids[0, 0, 0] != 999.0


def test_centroids_setter_shape_mismatch_raises():
    scorer = _soc_scorer()
    with pytest.raises((ValueError, Exception)):
        scorer.centroids = np.full((3, 3, 3), 0.5)
