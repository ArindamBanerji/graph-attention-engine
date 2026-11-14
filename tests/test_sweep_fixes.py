"""Regression reproductions for v0.8.0 sweep findings F1 through F7."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from gae import AgentEvolver, ComparisonMode, EvaluationResult, ProfileScorer, build_profile_scorer
from gae.integrations.airflow import RGIScoreOperator, RGIVerifyOperator
from gae.state import InMemoryStateBackend, VersionConflictError


def make(backend=None, **kwargs):
    return build_profile_scorer(
        categories=kwargs.pop("categories", ["a"]),
        actions=kwargs.pop("actions", ["x", "y"]),
        factors=kwargs.pop("factors", ["f"]),
        backend=backend, scorer_id="shared", **kwargs,
    )


def verify(scorer, category=0, action=0, correct=True):
    return scorer.update(np.array([.9]), category, action, correct,
                         gt_action_index=None if correct else 1 - action)


def test_f1_stale_save_rejected():
    backend = InMemoryStateBackend()
    s = make(backend)
    for _ in range(10):
        verify(s)
    u = make(backend)
    u.load()
    verify(u)
    expected, version = backend.load_category("shared", "a")
    s.save()
    actual, new_version = backend.load_category("shared", "a")
    assert actual == expected
    assert actual["verified_count"] == 11
    assert s.verified_count(0) == 11
    assert int(new_version) > int(version)
    assert s._backend_versions["a"] == new_version


class ConflictingBackend(InMemoryStateBackend):
    fail = False
    attempts = 0

    def save_category(self, scorer_id, category, state, version):
        if self.fail:
            self.attempts += 1
            raise VersionConflictError("concurrent writer")
        return super().save_category(scorer_id, category, state, version)


def test_f1_save_retry_is_bounded():
    backend = ConflictingBackend()
    s = make(backend)
    verify(s)
    backend.fail = True
    with pytest.raises(VersionConflictError):
        s.save()
    assert backend.attempts == 4
    assert s.verified_count(0) == 1


def test_f2_airflow_uses_persisted_state():
    backend = InMemoryStateBackend()
    config = dict(categories=["routing"], actions=["approve", "flag"],
                  factors=["freshness", "quality"], backend=backend, scorer_id="pipeline")
    s = build_profile_scorer(**config)
    for _ in range(50):
        s.update(np.array([.9, .8]), 0, 1, True)
    s2 = build_profile_scorer(**config)
    assert s2.score(np.array([.9, .8]), 0).action_index == 1
    assert s2.verified_count(0) == 50
    op = RGIScoreOperator(task_id="score", scorer_config=config, category="routing",
                          factors_callable=lambda ctx: [.9, .8])
    decision = op.execute({})
    assert decision["action_index"] == 1
    verifier = RGIVerifyOperator(
        task_id="verify", scorer_config=config,
        outcome_callable=lambda ctx: {"decision": decision, "correct": True},
    )
    assert verifier.execute({})["accuracy"] == 1.0
    assert backend.load_category("pipeline", "routing")[0]["verified_count"] == 51


def test_f2_direct_constructor_loads_and_opt_out_defers():
    backend = InMemoryStateBackend()
    s = make(backend)
    verify(s, action=1)
    fresh = ProfileScorer(np.full((1, 2, 1), .5), ["x", "y"], categories=["a"],
                          factors=["f"], backend=backend, scorer_id="shared")
    assert fresh.verified_count(0) == 1
    deferred = make(backend, load_backend=False)
    assert deferred.verified_count(0) == 0
    deferred.load()
    assert deferred.verified_count(0) == 1


def test_f3_reordered_actions_rejected():
    backend = InMemoryStateBackend()
    s = make(backend)
    verify(s, action=1)
    s.save()
    with pytest.raises(ValueError, match="Action axis mismatch"):
        make(backend, actions=["y", "x"])
    deferred = make(backend, actions=["y", "x"], load_backend=False)
    before = deferred.centroids.copy()
    with pytest.raises(ValueError, match="Action axis mismatch"):
        deferred.load()
    np.testing.assert_array_equal(deferred.centroids, before)


def test_f3_reordered_factors_rejected():
    backend = InMemoryStateBackend()
    s = make(backend, factors=["first", "second"])
    s.update(np.array([.9, .2]), 0, 1, True)
    with pytest.raises(ValueError, match="Factor axis mismatch"):
        make(backend, factors=["second", "first"])


def test_f3_legacy_schema_missing_loads():
    backend = InMemoryStateBackend()
    s = make(backend)
    verify(s)
    state, version = backend.load_category("shared", "a")
    del state["_schema"]
    backend.save_category("shared", "a", state, version)
    assert make(backend).verified_count(0) == 1


class FailingBackend(InMemoryStateBackend):
    fail = False

    def save_category(self, scorer_id, category, state, version):
        if self.fail:
            raise OSError("disk full")
        return super().save_category(scorer_id, category, state, version)


def test_f4_failed_write_restores_local_state():
    backend = FailingBackend()
    s = make(backend)
    verify(s)
    state = s._extract_category_state(0)
    runtime = s._runtime_state_for_retry()
    persisted = backend.load_all("shared")
    backend.fail = True
    with pytest.raises(OSError, match="disk full"):
        verify(s, action=1, correct=False)
    assert s._extract_category_state(0) == state
    assert s._runtime_state_for_retry() == runtime
    assert backend.load_all("shared") == persisted
    backend.fail = False
    verify(s)
    assert s.verified_count(0) == 2


def test_f4_failed_first_write_does_not_leak_learning():
    backend = FailingBackend()
    s = make(backend)
    backend.fail = True
    with pytest.raises(OSError):
        verify(s, action=1)
    assert s.verified_count(0) == s.decision_count == 0
    assert np.all(s.mu == .5)
    assert backend.load_all("shared") == {}


def test_f5_reordered_categories_aligned_by_name():
    base = make(categories=["a", "b"])
    cand = make(categories=["b", "a"])
    for _ in range(50):
        for s in (base, cand):
            verify(s, category=0)
            verify(s, category=1, correct=False)
    result = AgentEvolver(base, cand, min_decisions=10).evaluate()
    assert result.verdict == "rolled_back"
    assert result.delta == {"a": -1.0, "b": 1.0}


def test_f5_mismatched_category_names_rejected():
    with pytest.raises(ValueError, match="identical category"):
        AgentEvolver(make(categories=["a", "b"]), make(categories=["a", "c"]))


def test_f5_weighted_alignment_uses_candidate_volumes():
    base = make(categories=["a", "b"])
    cand = make(categories=["b", "a"])
    for _ in range(10):
        verify(base, 0, correct=False)
        verify(cand, 1)
    verify(base, 1)
    verify(cand, 0, correct=False)
    result = AgentEvolver(base, cand, min_decisions=1,
                          comparison_mode=ComparisonMode.WEIGHTED_AVERAGE).evaluate()
    assert result.verdict == "promoted"
    assert result.candidate_verified == 11


def test_f6_direct_result_maps_are_frozen_and_copied():
    source = {"a": .8}
    r = EvaluationResult("waiting", source, source, source, 50, "YELLOW", 3600, "test")
    source["a"] = 999
    for values in (r.base_accuracy, r.candidate_accuracy, r.delta):
        assert values["a"] == .8
        with pytest.raises(TypeError):
            values["a"] = 999
    with pytest.raises(FrozenInstanceError):
        r.verdict = "promoted"


def test_f7_zero_timeout_at_identical_clock_tick(monkeypatch):
    import gae.agentevolver as module

    monkeypatch.setattr(module.time, "monotonic", lambda: 100.0)
    assert AgentEvolver(make(), make(), timeout_seconds=0).evaluate().verdict == "timeout"
