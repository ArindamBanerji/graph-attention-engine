"""Dependency-free Judgment Memory loop for the LangChain integration.

The callback uses this same ProfileScorer flow. Install the ``langchain`` extra
only when connecting it to a LangChain agent.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from gae import build_profile_scorer, conservation_status, load_state, save_state
from gae.store import LearningState


CATEGORIES = ["retrieval", "generation", "routing"]
ACTIONS = ["use", "skip", "escalate", "retry"]
FACTORS = ["relevance", "confidence", "recency", "source_trust", "complexity"]


def build_scorer():
    """Build the neutral geometry used by the callback (blog Eq. 4-final)."""
    centroids = {
        category: {action: [0.5] * len(FACTORS) for action in ACTIONS}
        for category in CATEGORIES
    }
    scorer = build_profile_scorer(CATEGORIES, ACTIONS, centroids, len(FACTORS))
    scorer.categories = list(CATEGORIES)
    assert scorer.centroids.shape == (len(CATEGORIES), len(ACTIONS), len(FACTORS))
    return scorer


def main() -> None:
    """Run 20 decisions and 15 verified Eq. 4b-final updates."""
    scorer = build_scorer()
    baseline = scorer.centroids.copy()
    assert baseline.shape == scorer.centroids.shape
    rng = np.random.default_rng(42)
    decisions: list[tuple[np.ndarray, int, object]] = []

    probe = np.array([0.82, 0.74, 0.63, 0.88, 0.31], dtype=np.float64)
    assert probe.shape == (len(FACTORS),)
    day_1 = scorer.score(probe, 0).probabilities.copy()

    for index in range(20):
        factors = rng.uniform(0.3, 0.9, size=len(FACTORS))
        assert factors.shape == (len(FACTORS),)
        category_index = index % len(CATEGORIES)
        result = scorer.score(factors, category_index)
        decisions.append((factors, category_index, result))

    correct_count = 0
    for index, (factors, category_index, result) in enumerate(decisions[:15]):
        correct = index not in {3, 7, 11, 14}
        gt_action = None
        if not correct:
            gt_action = (result.action_index + 1 + index) % len(ACTIONS)
            if gt_action == result.action_index:
                gt_action = (gt_action + 1) % len(ACTIONS)
        scorer.update(
            factors,
            category_index,
            result.action_index,
            correct,
            gt_action_index=gt_action,
            confidence=result.confidence,
        )
        correct_count += int(correct)

    check = conservation_status(
        verified_count=15,
        correct_count=correct_count,
        total_decisions=20,
        penalty_ratio=5.0,
        categories_with_data=len(CATEGORIES),
        total_categories=len(CATEGORIES),
    )
    day_20 = scorer.score(probe, 0).probabilities.copy()
    drift = scorer.centroids - baseline
    assert drift.shape == baseline.shape
    cell_norms = np.linalg.norm(drift, axis=2)
    assert cell_norms.shape == (len(CATEGORIES), len(ACTIONS))
    iks = 100.0 * min(float(np.mean(cell_norms)) / 0.20, 1.0)

    print("Judgment Memory quickstart")
    print(f"Decisions: 20 | Verified: 15 | Accuracy: {correct_count / 15:.1%}")
    print(f"Conservation: {check.status} | IKS: {iks:.2f}")
    print("Compounding curve: 0 verified -> 5 -> 10 -> 15 verified")
    print("Day 1 probabilities :", np.round(day_1, 4).tolist())
    print("Day 20 probabilities:", np.round(day_20, 4).tolist())

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "jm_state.json"
        shape = scorer.centroids.shape
        save_state(
            LearningState(
                weights=scorer.centroids.reshape(-1).copy(),
                step=15,
                metadata={"shape": list(shape)},
            ),
            path,
        )
        restored_state = load_state(path)
        restored = restored_state.weights.reshape(tuple(restored_state.metadata["shape"]))
        assert restored.shape == shape
        np.testing.assert_allclose(restored, scorer.centroids)
        print(f"State reload preserved {restored_state.step} verified updates: yes")


if __name__ == "__main__":
    main()

