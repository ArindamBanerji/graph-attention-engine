"""StateBackend integration tests for ProfileScorer's retry-by-replay path."""

from __future__ import annotations

import numpy as np
import pytest

from gae import ProfileScorer, build_profile_scorer
from gae.state import InMemoryStateBackend, VersionConflictError


def make_scorer(
    backend: InMemoryStateBackend | None = None, scorer_id: str = "backend-test"
) -> ProfileScorer:
    return build_profile_scorer(
        categories=["alerts", "invoices"],
        actions=["approve", "review"],
        factors=["severity", "confidence"],
        backend=backend,
        scorer_id=scorer_id,
    )


def update(scorer: ProfileScorer, category_index: int, correct: bool = True) -> None:
    scorer.update(
        np.array([0.8, 0.2]),
        category_index,
        0,
        correct=correct,
        gt_action_index=None if correct else 1,
    )


def test_backend_roundtrip() -> None:
    backend = InMemoryStateBackend()
    scorer = make_scorer(backend)

    update(scorer, 0)
    update(scorer, 1, correct=False)

    states = backend.load_all("backend-test")
    assert set(states) == {"alerts", "invoices"}
    alert_state, _ = states["alerts"]
    invoice_state, _ = states["invoices"]
    assert alert_state["verified_count"] == 1
    assert alert_state["correct_count"] == 1
    assert invoice_state["verified_count"] == 1
    assert invoice_state["correct_count"] == 0
    assert np.asarray(alert_state["centroids"]).shape == (2, 2)
    assert np.asarray(invoice_state["counts"]).shape == (2,)


class ConflictOnceBackend(InMemoryStateBackend):
    """Create one concurrent category write before rejecting the stale save."""

    def __init__(self) -> None:
        super().__init__()
        self.conflicts = 0

    def save_category(self, scorer_id: str, category: str, state: dict, version: str) -> str:
        if self.conflicts == 0 and version != "0":
            self.conflicts += 1
            current, current_version = self.load_category(scorer_id, category)
            current["verified_count"] += 1
            current["correct_count"] += 1
            current["counts"][0] += 1
            super().save_category(scorer_id, category, current, current_version)
        return super().save_category(scorer_id, category, state, version)


def test_backend_cas_retry_reloads_and_replays() -> None:
    backend = ConflictOnceBackend()
    scorer = make_scorer(backend)
    update(scorer, 0)

    update(scorer, 0)

    state, _ = backend.load_category("backend-test", "alerts")
    assert backend.conflicts == 1
    assert state["verified_count"] == 3
    assert state["correct_count"] == 3
    assert state["counts"][0] == 3
    assert scorer.verified_count(0) == 3


def test_backend_persistence_save_then_load() -> None:
    backend = InMemoryStateBackend()
    scorer = make_scorer(backend)
    update(scorer, 0)
    update(scorer, 1, correct=False)
    scorer.save()

    restored = make_scorer(backend)
    restored.load()

    np.testing.assert_allclose(restored.centroids, scorer.centroids)
    np.testing.assert_array_equal(restored.counts, scorer.counts)
    assert restored.verified_count(0) == 1
    assert restored.accuracy(0) == 1.0
    assert restored.verified_count(1) == 1
    assert restored.accuracy(1) == 0.0


def test_no_backend_local_pickle_roundtrip(tmp_path) -> None:
    scorer = make_scorer()
    update(scorer, 0)
    path = scorer.save(tmp_path / "scorer.pkl")

    restored = make_scorer()
    restored.load(path)

    np.testing.assert_allclose(restored.centroids, scorer.centroids)
    np.testing.assert_array_equal(restored.counts, scorer.counts)
    assert restored.verified_count(0) == 1


def test_backend_load_empty_keeps_clean_initial_state() -> None:
    scorer = make_scorer(InMemoryStateBackend())

    scorer.load()

    assert scorer.decision_count == 0
    assert scorer.verified_count(0) == 0
    assert scorer.verified_count(1) == 0
    assert np.all(scorer.centroids == 0.5)


def test_backend_with_observability() -> None:
    backend = InMemoryStateBackend()
    scorer = make_scorer(backend)
    update(scorer, 0)
    update(scorer, 0, correct=False)

    restored = make_scorer(backend)
    restored.load()

    assert restored.verified_count(0) == 2
    assert restored.accuracy(0) == 0.5
    assert restored.centroid_drift(0) > 0.0


class AlwaysConflictBackend(InMemoryStateBackend):
    def __init__(self) -> None:
        super().__init__()
        self.save_attempts = 0

    def save_category(self, scorer_id: str, category: str, state: dict, version: str) -> str:
        self.save_attempts += 1
        raise VersionConflictError("forced conflict")


def test_retry_exhaustion_after_three_retries() -> None:
    backend = AlwaysConflictBackend()
    scorer = make_scorer(backend)

    with pytest.raises(VersionConflictError, match="forced conflict"):
        update(scorer, 0)

    assert backend.save_attempts == 4
    assert scorer.verified_count(0) == 0
    assert scorer.decision_count == 0


def test_category_name_mapping() -> None:
    backend = InMemoryStateBackend()
    scorer = make_scorer(backend, scorer_id="named-scorer")

    update(scorer, 1)

    assert set(backend.load_all("named-scorer")) == {"invoices"}


def test_mixed_save_writes_backend_and_local_copy(tmp_path) -> None:
    backend = InMemoryStateBackend()
    scorer = make_scorer(backend)
    update(scorer, 0)

    path = scorer.save(tmp_path / "migration.pkl")

    assert path is not None and path.is_file()
    assert "alerts" in backend.load_all("backend-test")
    local_restored = make_scorer()
    local_restored.load(path)
    assert local_restored.verified_count(0) == 1
