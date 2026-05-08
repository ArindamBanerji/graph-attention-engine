"""
VAL-01: conservation law validation under coordinate-descent DK estimation.

Validates that alpha * q * V >= theta_min holds during the per-category
Phase 1 -> Phase 2 transition when DK weights are estimated through
ProfileScorer.reestimate_dk() using CoordinateDescentEstimator.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import pytest

from gae import LearningStrategy, ProfileScorer, build_profile_scorer
from gae.calibration import check_conservation, compute_theta_min
from gae.dk_estimator import CoordinateDescentEstimator
from gae.shrinkage import FixedAlpha
from gae.two_phase import DecisionCountPolicy, VARIANCE_LEARNING


FREEZE_THRESHOLDS = [10, 30]
ORACLE_ACCURACIES = [0.75, 0.90]
CATEGORY_COUNTS = [3, 6]
N_ACTIONS = 4
N_FACTORS = 6
ALPHA = 0.25
V_PER_DAY = 200.0
DECISIONS_PER_CONDITION = 210
CHECKPOINT_INTERVAL = 10
SEED = 42


@dataclass(frozen=True)
class ConservationCheckpoint:
    decision_number: int
    q: float
    signal: float
    theta_min: float
    status: str
    passed: bool
    phase: str
    paused: bool


def _build_centroids(n_cat: int):
    cats = [f"cat_{i}" for i in range(n_cat)]
    acts = [f"action_{i}" for i in range(N_ACTIONS)]
    centroids = {}
    for c, cat in enumerate(cats):
        centroids[cat] = {}
        category_offset = 0.01 * c
        for a, act in enumerate(acts):
            values = np.full(N_FACTORS, 0.15 + 0.18 * a + category_offset)
            values[a % N_FACTORS] = min(0.95, 0.85 - category_offset)
            centroids[cat][act] = np.clip(values, 0.0, 1.0).tolist()
    return cats, acts, centroids


def _oracle_outcomes(total: int, oracle_accuracy: float, seed: int) -> list[bool]:
    n_correct = int(round(total * oracle_accuracy))
    outcomes = np.array(
        [True] * n_correct + [False] * (total - n_correct),
        dtype=np.bool_,
    )
    rng = np.random.RandomState(seed)
    rng.shuffle(outcomes)
    if total >= CHECKPOINT_INTERVAL and not outcomes[:CHECKPOINT_INTERVAL].any():
        first_correct = int(np.flatnonzero(outcomes)[0])
        outcomes[0], outcomes[first_correct] = outcomes[first_correct], outcomes[0]
    return [bool(value) for value in outcomes]


def _factor_vector(category_index: int, action_index: int, decision_number: int) -> np.ndarray:
    base = np.full(N_FACTORS, 0.15 + 0.18 * action_index + 0.01 * category_index)
    base[action_index % N_FACTORS] = 0.85 - 0.001 * (decision_number % 5)
    return np.clip(base, 0.0, 1.0).astype(np.float64)


def _make_strategy(freeze_n: int) -> LearningStrategy:
    return LearningStrategy(
        phase_policy=DecisionCountPolicy(n=freeze_n),
        dk_estimator=CoordinateDescentEstimator(seed=SEED, n_rounds=3, max_per_cat=10),
        shrinkage_schedule=FixedAlpha(0.5),
    )


def _simulate_condition(
    freeze_n: int,
    oracle_accuracy: float,
    n_categories: int,
) -> tuple[list[ConservationCheckpoint], object]:
    cats, acts, centroids = _build_centroids(n_categories)
    scorer = build_profile_scorer(
        cats,
        acts,
        centroids,
        N_FACTORS,
        learning_strategy=_make_strategy(freeze_n),
    )
    total_decisions = max(DECISIONS_PER_CONDITION, freeze_n * n_categories + 60)
    outcomes = _oracle_outcomes(
        total_decisions,
        oracle_accuracy,
        seed=SEED + freeze_n + n_categories + int(oracle_accuracy * 100),
    )
    theta_min = compute_theta_min(ALPHA, V_PER_DAY)
    checkpoints: list[ConservationCheckpoint] = []
    correct_count = 0

    for index, oracle_correct in enumerate(outcomes, start=1):
        category_index = (index - 1) % n_categories
        intended_action = ((index - 1) // n_categories) % N_ACTIONS
        factors = _factor_vector(category_index, intended_action, index)
        result = scorer.score(factors, category_index=category_index)
        if oracle_correct:
            action_index = result.action_index
            gt_action_index = None
            correct_count += 1
        else:
            action_index = result.action_index
            gt_action_index = (result.action_index + 1) % N_ACTIONS

        scorer.update(
            factors,
            category_index=category_index,
            action_index=action_index,
            correct=oracle_correct,
            gt_action_index=gt_action_index,
        )

        if index % CHECKPOINT_INTERVAL == 0 or index == total_decisions:
            q = correct_count / index
            cc = check_conservation(ALPHA, q, V_PER_DAY, theta_min)
            checkpoints.append(
                ConservationCheckpoint(
                    decision_number=index,
                    q=q,
                    signal=cc.signal,
                    theta_min=cc.theta_min,
                    status=cc.status,
                    passed=cc.passed,
                    phase=scorer.get_phase(0),
                    paused=getattr(scorer, "_paused_by_conservation", False),
                )
            )

    scorer.reestimate_dk()
    return checkpoints, scorer


@pytest.mark.parametrize(
    "freeze_n,oracle_accuracy,n_categories",
    itertools.product(FREEZE_THRESHOLDS, ORACLE_ACCURACIES, CATEGORY_COUNTS),
)
def test_conservation_holds_through_phase_transition(
    freeze_n: int,
    oracle_accuracy: float,
    n_categories: int,
) -> None:
    checkpoints, scorer = _simulate_condition(freeze_n, oracle_accuracy, n_categories)
    theta_min = compute_theta_min(ALPHA, V_PER_DAY)
    q_min = theta_min / (ALPHA * V_PER_DAY)

    assert checkpoints
    assert oracle_accuracy >= q_min
    assert all(checkpoint.status != "RED" for checkpoint in checkpoints)
    assert all(checkpoint.passed for checkpoint in checkpoints)
    assert any(checkpoint.phase == VARIANCE_LEARNING for checkpoint in checkpoints)
    assert scorer.get_phase(0) == VARIANCE_LEARNING
    weights = scorer.get_dk_weights(0)
    assert weights is not None
    assert weights.shape == (N_FACTORS,)


def test_auto_pause_activates_on_amber_with_soc_twophase() -> None:
    mu = np.full((2, N_ACTIONS, N_FACTORS), 0.5, dtype=np.float64)
    scorer = ProfileScorer.for_soc_twophase(
        mu=mu,
        phase_policy=DecisionCountPolicy(n=10),
        dk_estimator=CoordinateDescentEstimator(seed=SEED, n_rounds=3, max_per_cat=10),
        shrinkage_schedule=FixedAlpha(0.5),
    )
    theta_min = compute_theta_min(ALPHA, V_PER_DAY)
    amber_q = 1.5 * theta_min / (ALPHA * V_PER_DAY)
    assert 0.0 < amber_q < 1.0

    amber = check_conservation(ALPHA, amber_q, V_PER_DAY, theta_min)
    assert amber.status == "AMBER"
    assert scorer.auto_pause_on_amber is True

    scorer.set_conservation_status(amber.status)

    assert scorer.is_paused is True
    result = scorer.update(
        np.full(N_FACTORS, 0.5, dtype=np.float64),
        category_index=0,
        action_index=0,
        correct=True,
    )
    assert result.outcome == "paused_conservation"


def test_conservation_summary() -> None:
    rows = []
    for freeze_n, oracle_accuracy, n_categories in itertools.product(
        FREEZE_THRESHOLDS,
        ORACLE_ACCURACIES,
        CATEGORY_COUNTS,
    ):
        checkpoints, _ = _simulate_condition(freeze_n, oracle_accuracy, n_categories)
        statuses = [checkpoint.status for checkpoint in checkpoints]
        phase2_reached = any(
            checkpoint.phase == VARIANCE_LEARNING for checkpoint in checkpoints
        )
        min_q = min(checkpoint.q for checkpoint in checkpoints)
        rows.append(
            (
                freeze_n,
                oracle_accuracy,
                n_categories,
                statuses.count("GREEN"),
                statuses.count("AMBER"),
                statuses.count("RED"),
                phase2_reached,
                min_q,
            )
        )

    print("freeze oracle_acc n_cats green amber red phase2_reached min_q")
    for row in rows:
        print(
            f"{row[0]:>6} {row[1]:>10.2f} {row[2]:>6} "
            f"{row[3]:>5} {row[4]:>5} {row[5]:>3} {str(row[6]):>14} {row[7]:.3f}"
        )

    assert len(rows) == 8
    assert all(row[3] + row[4] + row[5] > 0 for row in rows)
    assert all(row[6] for row in rows)
