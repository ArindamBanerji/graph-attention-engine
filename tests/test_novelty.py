"""
Tests for gae.novelty.
"""

import numpy as np
import pytest

from gae.novelty import NearestNeighborNovelty


def test_first_decision_maximally_novel():
    tracker = NearestNeighborNovelty()

    novelty = tracker.compute_novelty(np.array([0.2, 0.4], dtype=np.float64), 0)

    assert novelty == pytest.approx(1.0)


def test_identical_decision_zero_novelty():
    tracker = NearestNeighborNovelty()
    f = np.array([0.2, 0.4, 0.6], dtype=np.float64)
    tracker.record(f, 0)

    novelty = tracker.compute_novelty(f, 0)

    assert novelty == pytest.approx(0.0, abs=1e-12)


def test_distant_decision_high_novelty():
    tracker = NearestNeighborNovelty()
    tracker.record(np.array([0.0, 0.0, 0.0], dtype=np.float64), 0)

    novelty = tracker.compute_novelty(np.array([1.0, 1.0, 1.0], dtype=np.float64), 0)

    assert abs(novelty - np.sqrt(3.0)) < 1e-6


def test_novelty_uses_nearest_not_mean():
    tracker = NearestNeighborNovelty()
    tracker.record(np.array([0.0, 0.0], dtype=np.float64), 0)
    tracker.record(np.array([10.0, 10.0], dtype=np.float64), 0)

    novelty = tracker.compute_novelty(np.array([0.0, 1.0], dtype=np.float64), 0)

    assert novelty == pytest.approx(1.0)


def test_record_stores_decision():
    tracker = NearestNeighborNovelty()

    tracker.record(np.array([0.1, 0.2], dtype=np.float64), 0)

    assert tracker.get_history_size(0) == 1


def test_record_trims_to_max_look():
    tracker = NearestNeighborNovelty(max_look=3)

    for i in range(10):
        tracker.record(np.array([float(i)], dtype=np.float64), 0)

    assert tracker.get_history_size(0) == 3
    assert len(tracker._novelty_scores[0]) == 3


def test_novelty_rate_empty():
    tracker = NearestNeighborNovelty()

    rate = tracker.get_novelty_rate(0)

    assert rate == pytest.approx(0.0)


def test_novelty_rate_all_novel():
    tracker = NearestNeighborNovelty(threshold=0.5)

    tracker.record(np.array([0.0], dtype=np.float64), 0)
    tracker.record(np.array([2.0], dtype=np.float64), 0)
    tracker.record(np.array([4.0], dtype=np.float64), 0)

    assert tracker.get_novelty_rate(0) == pytest.approx(1.0)


def test_novelty_rate_all_redundant():
    tracker = NearestNeighborNovelty(threshold=0.5)
    base = np.array([0.3, 0.4], dtype=np.float64)
    tracker.record(base, 0)
    tracker.record(base.copy(), 0)
    tracker.record(base.copy(), 0)

    assert tracker.get_novelty_rate(0, window=2) == pytest.approx(0.0)


def test_novelty_rate_respects_window():
    tracker = NearestNeighborNovelty(max_look=100, threshold=0.05)

    for i in range(49):
        tracker.record(np.array([float(i) * 1.0], dtype=np.float64), 0)
    tracker.record(np.array([1000.0], dtype=np.float64), 0)
    for i in range(50):
        tracker.record(np.array([1000.0 + 0.001 * float(i)], dtype=np.float64), 0)

    assert tracker.get_novelty_rate(0, window=100) == pytest.approx(0.5)
    assert tracker.get_novelty_rate(0, window=50) == pytest.approx(0.0)


def test_accumulator_grows():
    tracker = NearestNeighborNovelty()

    tracker.record(np.array([0.0], dtype=np.float64), 0)
    tracker.record(np.array([1.0], dtype=np.float64), 0)

    assert tracker.get_accumulator(0) == pytest.approx(2.0)


def test_accumulator_reset():
    tracker = NearestNeighborNovelty()
    tracker.record(np.array([0.0], dtype=np.float64), 0)
    tracker.record(np.array([1.0], dtype=np.float64), 0)

    tracker.reset_accumulator(0)

    assert tracker.get_accumulator(0) == pytest.approx(0.0)


def test_accumulator_resumes_after_reset():
    tracker = NearestNeighborNovelty()
    tracker.record(np.array([0.0], dtype=np.float64), 0)
    tracker.reset_accumulator(0)
    tracker.record(np.array([3.0], dtype=np.float64), 0)

    assert tracker.get_accumulator(0) == pytest.approx(3.0)


def test_categories_independent():
    tracker = NearestNeighborNovelty()
    tracker.record(np.array([0.0, 0.0], dtype=np.float64), 0)
    tracker.record(np.array([9.0, 9.0], dtype=np.float64), 1)

    novelty0 = tracker.compute_novelty(np.array([0.0, 1.0], dtype=np.float64), 0)
    novelty1 = tracker.compute_novelty(np.array([9.0, 10.0], dtype=np.float64), 1)

    assert novelty0 == pytest.approx(1.0)
    assert novelty1 == pytest.approx(1.0)


def test_rejects_invalid_params():
    with pytest.raises(ValueError, match="max_look must be >= 1"):
        NearestNeighborNovelty(max_look=0)
    with pytest.raises(ValueError, match="threshold must be >= 0"):
        NearestNeighborNovelty(threshold=-0.1)
    with pytest.raises(ValueError, match="n_categories must be >= 1"):
        NearestNeighborNovelty(n_categories=0)


def test_compute_does_not_modify_history():
    tracker = NearestNeighborNovelty()
    tracker.record(np.array([0.0, 0.0], dtype=np.float64), 0)
    size_before = tracker.get_history_size(0)

    tracker.compute_novelty(np.array([1.0, 1.0], dtype=np.float64), 0)

    assert tracker.get_history_size(0) == size_before
    assert len(tracker._novelty_scores[0]) == 1


def test_record_copies_input():
    tracker = NearestNeighborNovelty()
    f = np.array([0.2, 0.3], dtype=np.float64)
    tracker.record(f, 0)
    f[0] = 9.9

    assert np.allclose(tracker._history[0][0], np.array([0.2, 0.3], dtype=np.float64))


def test_high_dimensional_d20():
    tracker = NearestNeighborNovelty()
    base = np.linspace(0.0, 0.95, 20, dtype=np.float64)
    tracker.record(base, 0)

    novelty = tracker.compute_novelty(base + 0.1, 0)

    assert np.isfinite(novelty)
    assert novelty > 0.0
