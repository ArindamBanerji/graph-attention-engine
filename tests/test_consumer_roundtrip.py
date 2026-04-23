"""Consumer contract: score → update → score roundtrip."""
import warnings
import numpy as np
from gae.profile_scorer import ProfileScorer


def _soc_scorer(mu=None):
    if mu is None:
        mu = np.full((6, 4, 6), 0.5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return ProfileScorer.for_soc(mu=mu)


def test_score_update_score_roundtrip():
    """Full cycle: score, update with correct=True, score again."""
    scorer = _soc_scorer()
    fv = np.random.rand(6)

    r1 = scorer.score(fv, category_index=0)
    assert hasattr(r1, "action_index")

    scorer.update(
        f=fv,
        action_index=r1.action_index,
        correct=True,
        category_index=0,
    )

    r2 = scorer.score(fv, category_index=0)
    assert hasattr(r2, "action_index")
    assert isinstance(r2.confidence, float)


def test_multi_category_independence():
    """Updates to category 0 don't affect scoring in category 1."""
    scorer = _soc_scorer()
    fv = np.random.rand(6)

    r_before = scorer.score(fv, category_index=1)

    for _ in range(10):
        r = scorer.score(fv, category_index=0)
        scorer.update(f=fv, action_index=r.action_index,
                      correct=True, category_index=0)

    r_after = scorer.score(fv, category_index=1)
    assert r_before.action_index == r_after.action_index
    assert abs(r_before.confidence - r_after.confidence) < 1e-10


def test_checkpoint_restore_preserves_scoring():
    """Save centroids then restore in a new scorer → identical scoring."""
    scorer = _soc_scorer()
    fv = np.random.rand(6)

    for _ in range(5):
        r = scorer.score(fv, category_index=0)
        scorer.update(f=fv, action_index=r.action_index,
                      correct=True, category_index=0)

    saved = scorer.centroids.copy()
    r_before = scorer.score(fv, category_index=0)

    scorer2 = _soc_scorer(mu=saved)
    r_after = scorer2.score(fv, category_index=0)

    assert r_before.action_index == r_after.action_index
    assert abs(r_before.confidence - r_after.confidence) < 1e-10


def test_learning_loop_20_decisions():
    """20 confirmed decisions should shift centroids measurably from 0.5."""
    scorer = _soc_scorer()
    fv = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3])

    for _ in range(20):
        r = scorer.score(fv, category_index=0)
        scorer.update(f=fv, action_index=r.action_index,
                      correct=True, category_index=0)

    drift = np.abs(scorer.centroids[0] - 0.5).max()
    assert drift > 0.01
