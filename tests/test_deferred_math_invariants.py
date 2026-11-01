"""Targeted conformance tests for the seven JM v2.9 math gaps."""

from __future__ import annotations

import numpy as np
import pytest

from gae.batch_pipeline import check_batch_composition, check_holdout_non_inferiority
from gae.convergence import (
    ConservationMonitor,
    compute_n_half,
    compute_reconvergence_ratio,
    gamma_threshold,
)
from gae.shrinkage import compute_effective_weights
from gae.novelty import NearestNeighborNovelty


def test_gap_6_convergence_half_life_is_exact_discrete_update() -> None:
    eta = 0.05
    expected = np.log(2.0) / np.log(1.0 / (1.0 - eta))
    assert compute_n_half(eta) == pytest.approx(expected)


def test_gap_7_shrinkage_is_convex_and_has_safe_endpoints() -> None:
    learned = np.array([0.2, 1.4, 2.0], dtype=np.float64)
    assert np.allclose(compute_effective_weights(learned, 0.0), 1.0)
    assert np.allclose(compute_effective_weights(learned, 1.0), learned)
    blended = compute_effective_weights(learned, 0.5)
    assert np.all((blended - 1.0) * (learned - blended) >= 0.0)


def test_gap_9_novelty_is_nearest_neighbor_within_category() -> None:
    tracker = NearestNeighborNovelty(max_look=300, threshold=0.1)
    first = np.array([0.0, 0.0], dtype=np.float64)
    other_category = np.array([1.0, 1.0], dtype=np.float64)
    tracker.record(first, category_index=0)
    tracker.record(other_category, category_index=1)
    assert tracker.compute_novelty(np.array([0.0, 0.0]), 0) == pytest.approx(0.0)
    assert tracker.compute_novelty(np.array([0.0, 0.0]), 1) == pytest.approx(np.sqrt(2.0))


def test_gap_10_batch_composition_requires_all_three_conditions() -> None:
    passing = check_batch_composition(10, 50, 3, 6)
    assert passing.passed
    failing = check_batch_composition(4, 49, 2, 6)
    assert not failing.passed
    assert failing.reason == "novel_fraction_fail; coverage_fail; count_fail"


def test_gap_11_holdout_non_inferiority_requires_overall_category_and_counts() -> None:
    passing = check_holdout_non_inferiority(
        0.80,
        0.795,
        {"a": 0.80, "b": 0.80},
        {"a": 0.79, "b": 0.79},
        {"a": 20, "b": 20},
    )
    assert passing.passed
    failing = check_holdout_non_inferiority(
        0.80,
        0.78,
        {"a": 0.80},
        {"a": 0.75},
        {"a": 19},
    )
    assert not failing.passed
    assert "overall_non_inferiority_fail" in failing.reason
    assert "category_non_inferiority_fail" in failing.reason
    assert "holdout_count_fail" in failing.reason


def test_gap_15_reconvergence_threshold_and_ratio_are_consistent() -> None:
    threshold = gamma_threshold(alpha_cat=2.0 / 6.0, delta_norm=0.25)
    assert threshold == pytest.approx(0.125)
    assert compute_reconvergence_ratio(1404, 546) > 1.0


class _StatusRecorder:
    def __init__(self) -> None:
        self.statuses: list[str] = []

    def set_conservation_status(self, status: str) -> None:
        self.statuses.append(status)


def test_gap_16_amber_red_auto_pause_signal_reaches_scorer() -> None:
    scorer = _StatusRecorder()
    monitor = ConservationMonitor(scorer=scorer)
    monitor.update_conservation_signal("AMBER")
    monitor.update_conservation_signal("RED")
    assert monitor.conservation_status == "RED"
    assert scorer.statuses == ["AMBER", "RED"]
