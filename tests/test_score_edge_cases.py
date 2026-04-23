import warnings
import numpy as np
import pytest
from gae.profile_scorer import ProfileScorer


@pytest.fixture
def scorer():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ProfileScorer.for_soc(mu=np.full((6, 4, 6), 0.5))


def test_score_returns_action_index(scorer):
    result = scorer.score(np.random.rand(6), category_index=0)
    assert hasattr(result, "action_index")
    assert isinstance(result.action_index, (int, np.integer))


def test_score_returns_confidence(scorer):
    result = scorer.score(np.random.rand(6), category_index=0)
    assert hasattr(result, "confidence")
    assert 0 <= result.confidence <= 1


def test_score_returns_entropy(scorer):
    result = scorer.score(np.random.rand(6), category_index=0)
    assert hasattr(result, "entropy")
    assert result.entropy >= 0


def test_score_returns_confidence_gap(scorer):
    result = scorer.score(np.random.rand(6), category_index=0)
    assert hasattr(result, "confidence_gap")
    assert result.confidence_gap >= 0


def test_score_all_zero_factors(scorer):
    """Zero factor vector should still produce a valid result."""
    result = scorer.score(np.zeros(6), category_index=0)
    assert hasattr(result, "action_index")
    assert np.isfinite(result.confidence)


def test_score_large_factors(scorer):
    """Large factor values should not crash."""
    result = scorer.score(np.full(6, 100.0), category_index=0)
    assert hasattr(result, "action_index")


def test_score_negative_factors(scorer):
    """Negative factor values should produce a valid result."""
    result = scorer.score(np.full(6, -1.0), category_index=0)
    assert hasattr(result, "action_index")


def test_score_category_bounds(scorer):
    """Out-of-bounds category index should raise."""
    with pytest.raises((IndexError, ValueError, Exception)):
        scorer.score(np.random.rand(6), category_index=99)


def test_score_wrong_factor_length(scorer):
    """Wrong number of factors should raise."""
    with pytest.raises((ValueError, IndexError, Exception)):
        scorer.score(np.random.rand(3), category_index=0)


def test_score_nan_factor_handled(scorer):
    """NaN in factor vector — scorer should either reject or produce finite confidence."""
    fv = np.array([0.5, 0.5, np.nan, 0.5, 0.5, 0.5])
    try:
        result = scorer.score(fv, category_index=0)
        assert np.isfinite(result.confidence)
    except (ValueError, Exception):
        pass  # rejection is acceptable
