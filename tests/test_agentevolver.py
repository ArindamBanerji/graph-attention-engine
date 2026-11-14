"""Tests for GAE's scorer-level AgentEvolver comparison primitive."""

from __future__ import annotations

import dataclasses
import inspect
import numpy as np
import pytest

from gae import AgentEvolver, ComparisonMode, ProfileScorer


def scorer() -> ProfileScorer:
    return ProfileScorer(
        mu=np.full((2, 2, 1), 0.5),
        actions=["approve", "review"],
        categories=["alerts", "invoices"],
    )


def record(target: ProfileScorer, category: int, correct_count: int, total: int) -> None:
    for index in range(total):
        correct = index < correct_count
        target.update(
            np.array([0.6]), category, 0, correct, None if correct else 1
        )


def populated(base_counts: tuple[int, int], candidate_counts: tuple[int, int], total: int = 10) -> tuple[ProfileScorer, ProfileScorer]:
    base, candidate = scorer(), scorer()
    for category, (base_correct, candidate_correct) in enumerate(zip(base_counts, candidate_counts)):
        record(base, category, base_correct, total)
        record(candidate, category, candidate_correct, total)
    return base, candidate


def test_promote_all_categories() -> None:
    base, candidate = populated((7, 7), (8, 8))
    result = AgentEvolver(base, candidate, min_decisions=20).evaluate()
    assert result.verdict == "promoted"
    assert result.delta == {"alerts": pytest.approx(0.1), "invoices": pytest.approx(0.1)}


def test_rollback_regression() -> None:
    base, candidate = populated((7, 7), (8, 6))
    assert AgentEvolver(base, candidate, min_decisions=20).evaluate().verdict == "rolled_back"


def test_waiting_insufficient_decisions() -> None:
    base, candidate = populated((7, 7), (8, 8))
    result = AgentEvolver(base, candidate, min_decisions=21).evaluate()
    assert result.verdict == "waiting"
    assert "20/21" in result.reason


def test_waiting_conservation_not_green() -> None:
    base, candidate = populated((7, 7), (8, 8))
    candidate._conservation_status = "YELLOW"
    result = AgentEvolver(base, candidate, min_decisions=20).evaluate()
    assert result.verdict == "waiting"
    assert result.candidate_conservation == "YELLOW"


def test_timeout() -> None:
    base, candidate = populated((7, 7), (8, 8))
    evolver = AgentEvolver(base, candidate, min_decisions=20, timeout_seconds=0.001)
    # Advance the stored monotonic anchor directly: freezegun can make a
    # wall-clock sleep non-deterministic when the full suite runs.
    evolver._started_at -= 0.01
    assert evolver.evaluate().verdict == "timeout"


def test_weighted_average_mode() -> None:
    base, candidate = scorer(), scorer()
    record(base, 0, 7, 10)
    record(candidate, 0, 9, 10)
    record(base, 1, 1, 1)
    record(candidate, 1, 0, 1)
    result = AgentEvolver(
        base, candidate, min_decisions=11, comparison_mode=ComparisonMode.WEIGHTED_AVERAGE
    ).evaluate()
    assert result.verdict == "promoted"


def test_any_improvement_mode() -> None:
    base, candidate = populated((7, 7), (8, 7))
    result = AgentEvolver(
        base, candidate, min_decisions=20, comparison_mode=ComparisonMode.ANY_IMPROVEMENT
    ).evaluate()
    assert result.verdict == "promoted"


def test_any_improvement_rejects_large_regression() -> None:
    base, candidate = populated((7, 7), (8, 6))
    result = AgentEvolver(
        base, candidate, min_decisions=20, comparison_mode=ComparisonMode.ANY_IMPROVEMENT
    ).evaluate()
    assert result.verdict == "rolled_back"


def test_history_immutable() -> None:
    base, candidate = populated((7, 7), (8, 8))
    evolver = AgentEvolver(base, candidate, min_decisions=20)
    first, second = evolver.evaluate(), evolver.evaluate()
    assert len(evolver.history) == 2
    assert evolver.history == [first, second]
    assert dataclasses.is_dataclass(first)
    with pytest.raises(TypeError):
        first.delta["alerts"] = 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        first.verdict = "waiting"


def test_no_module_globals() -> None:
    import gae.agentevolver as module

    mutable = [
        name for name, value in vars(module).items()
        if not name.startswith("__") and isinstance(value, (dict, list, set))
    ]
    assert mutable == []


def test_docstring_mentions_sdk() -> None:
    assert "copilot_sdk" in inspect.getdoc(AgentEvolver).lower()
