import numpy as np

from gae import KernelType, ProfileScorer, build_profile_scorer
from gae.calibration import soc_calibration_profile
from gae.dk_estimator import CoordinateDescentEstimator
from gae.shrinkage import FixedAlpha
from gae.two_phase import (
    DecisionCountPolicy,
    MEAN_CONVERGENCE,
    VARIANCE_LEARNING,
)
from gae.profile_scorer import LearningStrategy


def _soc_centroids():
    cats = [f"cat_{i}" for i in range(6)]
    acts = ["escalate", "investigate", "suppress", "monitor"]
    return cats, acts, {c: {a: [0.5] * 6 for a in acts} for c in cats}


def _s2p_centroids():
    cats = [f"s2p_cat_{i}" for i in range(5)]
    acts = [f"s2p_act_{i}" for i in range(5)]
    return cats, acts, {c: {a: [0.5] * 7 for a in acts} for c in cats}


def _make_strategy(n=200):
    return LearningStrategy(
        phase_policy=DecisionCountPolicy(n=n),
        dk_estimator=CoordinateDescentEstimator(),
        shrinkage_schedule=FixedAlpha(0.5),
    )


def _config(cats, centroids):
    return {
        "categories": cats,
        "centroids": centroids,
        "kernel": KernelType.L2.value,
    }


def test_build_profile_scorer_no_strategy():
    cats, acts, centroids = _soc_centroids()

    scorer = build_profile_scorer(cats, acts, centroids, 6)

    assert scorer.centroids.shape == (6, 4, 6)
    assert scorer._learning_strategy is None


def test_init_from_config_no_strategy():
    cats, acts, centroids = _soc_centroids()

    scorer = ProfileScorer.init_from_config(_config(cats, centroids), acts)

    assert scorer._learning_strategy is None


def test_build_explicit_none():
    cats, acts, centroids = _soc_centroids()

    scorer = build_profile_scorer(
        cats,
        acts,
        centroids,
        6,
        learning_strategy=None,
    )

    assert scorer._learning_strategy is None
    assert scorer.get_phase(0) == MEAN_CONVERGENCE
    assert scorer.get_alpha(0) == 0.0


def test_build_with_strategy_soc_shape():
    cats, acts, centroids = _soc_centroids()
    strategy = _make_strategy()

    scorer = build_profile_scorer(
        cats,
        acts,
        centroids,
        6,
        learning_strategy=strategy,
    )

    assert scorer._learning_strategy is strategy
    assert scorer._category_states is not None
    assert len(scorer._category_states) == 6
    assert scorer.get_phase(0) == MEAN_CONVERGENCE


def test_build_with_strategy_s2p_shape():
    cats, acts, centroids = _s2p_centroids()
    strategy = _make_strategy()

    scorer = build_profile_scorer(
        cats,
        acts,
        centroids,
        7,
        learning_strategy=strategy,
    )

    assert scorer._learning_strategy is strategy
    assert scorer.centroids.shape == (5, 5, 7)
    assert scorer._category_states is not None
    assert len(scorer._category_states) == 5
    assert scorer.get_phase(0) == MEAN_CONVERGENCE
    assert scorer.get_alpha(0) >= 0.0
    assert scorer.get_dk_weights(0) is None


def test_init_from_config_with_strategy():
    cats, acts, centroids = _soc_centroids()
    strategy = _make_strategy()

    scorer = ProfileScorer.init_from_config(
        _config(cats, centroids),
        acts,
        learning_strategy=strategy,
    )

    assert scorer._learning_strategy is strategy
    assert scorer._category_states is not None
    assert len(scorer._category_states) == 6


def test_s2p_phase_transition():
    cats, acts, centroids = _s2p_centroids()
    scorer = build_profile_scorer(
        cats,
        acts,
        centroids,
        7,
        learning_strategy=_make_strategy(n=5),
    )

    f = np.array([0.55] * 7, dtype=np.float64)
    for _ in range(5):
        scorer.update(f, category_index=0, action_index=0, correct=True)

    assert scorer.get_phase(0) == VARIANCE_LEARNING
    assert scorer.get_phase(1) == MEAN_CONVERGENCE


def test_init_from_config_with_profile_and_strategy():
    cats, acts, centroids = _soc_centroids()
    profile = soc_calibration_profile()
    strategy = _make_strategy()

    scorer = ProfileScorer.init_from_config(
        _config(cats, centroids),
        acts,
        profile=profile,
        learning_strategy=strategy,
    )

    assert scorer._learning_strategy is strategy
    assert scorer.tau == profile.temperature


def test_s2p_scoring_with_strategy():
    cats, acts, centroids = _s2p_centroids()
    scorer = build_profile_scorer(
        cats,
        acts,
        centroids,
        7,
        learning_strategy=_make_strategy(),
    )

    result = scorer.score(np.array([0.45] * 7, dtype=np.float64), category_index=0)

    assert result is not None
    assert len(result.probabilities) == 5
    assert np.isclose(result.probabilities.sum(), 1.0)


def test_build_profile_scorer_passes_learning_strategy_keyword_only():
    cats, acts, centroids = _soc_centroids()
    strategy = _make_strategy()

    scorer = build_profile_scorer(
        categories=cats,
        actions=acts,
        centroids=centroids,
        n_factors=6,
        learning_strategy=strategy,
    )

    assert scorer._learning_strategy is strategy
