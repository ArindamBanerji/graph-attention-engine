"""Tests for optional Prometheus metrics over ProfileScorer observability."""

from __future__ import annotations

import builtins
import importlib.util
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from prometheus_client import CollectorRegistry

from gae import ProfileScorer, build_profile_scorer
from gae.metrics import RGIMetrics


def make_scorer() -> ProfileScorer:
    """Build a deterministic three-category Eq. 4b scorer."""
    categories = ["alpha", "beta", "gamma"]
    actions = ["accept", "review"]
    centroids = {
        category: {
            "accept": [0.2, 0.2],
            "review": [0.8, 0.8],
        }
        for category in categories
    }
    return build_profile_scorer(categories, actions, centroids, n_factors=2)


def sample_value(registry: CollectorRegistry, name: str, **labels: str) -> float:
    """Read one sample from an isolated Prometheus registry."""
    value = registry.get_sample_value(name, labels)
    assert value is not None
    return value


def test_metrics_match_scorer() -> None:
    scorer = make_scorer()
    update = scorer.update(np.array([0.4, 0.4]), 0, 0, correct=True)
    scorer.set_conservation_status("AMBER")
    registry = CollectorRegistry(auto_describe=False)
    metrics = RGIMetrics(scorer, registry=registry)

    metrics.update()

    assert sample_value(registry, "rgi_verified_count", category="alpha") == scorer.verified_count(0)
    assert sample_value(registry, "rgi_accuracy", category="alpha") == scorer.accuracy(0)
    assert sample_value(registry, "rgi_centroid_drift", category="alpha") == update.centroid_delta_norm
    assert sample_value(registry, "rgi_conservation_status", category="alpha") == 1.0


def test_latency_histogram() -> None:
    scorer = make_scorer()
    registry = CollectorRegistry(auto_describe=False)
    metrics = RGIMetrics(scorer, registry=registry)

    with metrics.score_timer():
        scorer.score(np.array([0.3, 0.3]), category_index=0)

    assert sample_value(registry, "rgi_score_latency_seconds_count") == 1.0
    assert sample_value(registry, "rgi_score_latency_seconds_sum") >= 0.0


def test_score_timer_is_also_a_decorator() -> None:
    scorer = make_scorer()
    registry = CollectorRegistry(auto_describe=False)
    metrics = RGIMetrics(scorer, registry=registry)

    @metrics.score_timer()
    def score_once() -> None:
        scorer.score(np.array([0.3, 0.3]), category_index=0)

    score_once()

    assert sample_value(registry, "rgi_score_latency_seconds_count") == 1.0


def test_multiple_categories() -> None:
    scorer = make_scorer()
    for category_index in range(scorer.n_categories):
        scorer.update(np.array([0.4, 0.4]), category_index, 0, correct=True)
    registry = CollectorRegistry(auto_describe=False)
    metrics = RGIMetrics(scorer, registry=registry)

    metrics.update()

    for category in ("alpha", "beta", "gamma"):
        assert sample_value(registry, "rgi_verified_count", category=category) == 1.0
        assert sample_value(registry, "rgi_accuracy", category=category) == 1.0


def test_zero_verified_categories_export_zero_accuracy() -> None:
    registry = CollectorRegistry(auto_describe=False)
    metrics = RGIMetrics(make_scorer(), registry=registry)

    metrics.update()

    for category in ("alpha", "beta", "gamma"):
        assert sample_value(registry, "rgi_verified_count", category=category) == 0.0
        assert sample_value(registry, "rgi_accuracy", category=category) == 0.0


def test_update_after_verify() -> None:
    scorer = make_scorer()
    registry = CollectorRegistry(auto_describe=False)
    metrics = RGIMetrics(scorer, registry=registry)

    scorer.update(np.array([0.4, 0.4]), 1, 0, correct=True)
    scorer.update(np.array([0.6, 0.6]), 1, 0, correct=False, gt_action_index=1)
    metrics.update()

    assert scorer.verified_count(1) == 2
    assert scorer.accuracy(1) == 0.5
    assert sample_value(registry, "rgi_verified_count", category="beta") == 2.0
    assert sample_value(registry, "rgi_accuracy", category="beta") == 0.5


def test_agentevolver_phase_placeholder() -> None:
    registry = CollectorRegistry(auto_describe=False)
    metrics = RGIMetrics(make_scorer(), registry=registry)

    metrics.set_agentevolver_phase("candidate-a", "Promoted")

    assert sample_value(registry, "rgi_agentevolver_phase", variant="candidate-a") == 1.0


def test_observability_counters_survive_pickle_and_old_state() -> None:
    scorer = make_scorer()
    scorer.update(np.array([0.4, 0.4]), 0, 0, correct=True)
    scorer.update(np.array([0.6, 0.6]), 0, 0, correct=False, gt_action_index=1)

    restored = pickle.loads(pickle.dumps(scorer))

    assert restored.verified_count(0) == 2
    assert restored.accuracy(0) == 0.5
    assert restored.centroid_drift(0) == scorer.centroid_drift(0)

    old_state = scorer.__getstate__()
    for attribute in ("_verified_count", "_correct_count", "_centroid_drift"):
        old_state.pop(attribute)
    legacy = ProfileScorer.__new__(ProfileScorer)
    legacy.__setstate__(old_state)

    assert legacy.verified_count(0) == 0
    assert legacy.accuracy(0) == 0.0
    assert legacy.centroid_drift(0) == 0.0


def test_no_prometheus_graceful(monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = Path(__file__).parents[1] / "gae" / "metrics.py"
    spec = importlib.util.spec_from_file_location("_gae_metrics_missing_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "prometheus_client":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"graph-attention-engine\[prometheus\]"):
        spec.loader.exec_module(module)
