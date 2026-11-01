"""End-to-end smoke test for the public graph-attention-engine API.

Run after installing the base distribution:

    pip install graph-attention-engine
    python examples/pypi_smoke_test.py

Optional LangChain and MCP dependencies are detected without making the
corresponding checks fail when they are not installed.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from gae import (
    ProfileScorer,
    ScoringResult,
    build_profile_scorer,
    compute_judgment,
    conservation_status,
    load_state,
    save_state,
)


CATEGORIES = ["retrieval", "generation", "routing"]
ACTIONS = ["use", "skip", "escalate"]
FACTORS = ["relevance", "confidence", "recency", "source_trust", "complexity"]
CENTROIDS = {
    category: {
        "use": [0.85, 0.80, 0.75, 0.90, 0.25],
        "skip": [0.25, 0.30, 0.35, 0.20, 0.45],
        "escalate": [0.55, 0.45, 0.40, 0.50, 0.90],
    }
    for category in CATEGORIES
}
VECTORS = [
    np.asarray(values, dtype=np.float64)
    for values in (
        [0.82, 0.78, 0.72, 0.88, 0.28],
        [0.28, 0.32, 0.38, 0.22, 0.42],
        [0.58, 0.48, 0.44, 0.52, 0.86],
        [0.76, 0.72, 0.68, 0.80, 0.34],
        [0.32, 0.36, 0.40, 0.28, 0.48],
        [0.54, 0.46, 0.42, 0.48, 0.82],
        [0.88, 0.82, 0.78, 0.92, 0.22],
        [0.22, 0.26, 0.30, 0.18, 0.40],
        [0.60, 0.50, 0.46, 0.54, 0.88],
        [0.80, 0.76, 0.70, 0.84, 0.30],
    )
]


def main() -> int:
    """Run all nine checks and return nonzero if any check fails."""
    results: list[tuple[bool, str, str]] = []
    scorer: ProfileScorer | None = None
    scored: list[tuple[np.ndarray, int, ScoringResult]] = []

    def check(label: str, function: Callable[[], str]) -> None:
        try:
            detail = function()
            results.append((True, label, detail))
        except Exception as exc:  # each smoke check must remain isolated
            results.append((False, label, f"{type(exc).__name__}: {exc}"))

    print("=== GAE v0.7.26 PyPI Smoke Test ===")

    def package_check() -> str:
        installed = version("graph-attention-engine")
        if installed != "0.7.26":
            raise RuntimeError(f"expected 0.7.26, found {installed}")
        return installed

    check("Package version", package_check)

    def scorer_check() -> str:
        nonlocal scorer
        scorer = build_profile_scorer(CATEGORIES, ACTIONS, CENTROIDS, len(FACTORS))
        scorer.categories = list(CATEGORIES)
        expected = (len(CATEGORIES), len(ACTIONS), len(FACTORS))
        if scorer.centroids.shape != expected:
            raise RuntimeError(f"unexpected centroid shape {scorer.centroids.shape}")
        return "3x3x5"

    check("ProfileScorer", scorer_check)

    def score_check() -> str:
        if scorer is None:
            raise RuntimeError("ProfileScorer check did not complete")
        for index, vector in enumerate(VECTORS):
            category_index = index % len(CATEGORIES)
            scored.append((vector, category_index, scorer.score(vector, category_index)))
        last = scored[-1][2]
        return f"action={last.action_name}, confidence={last.confidence:.3f}"

    check("Score", score_check)

    def update_check() -> str:
        if scorer is None or len(scored) != 10:
            raise RuntimeError("ten scored decisions are required")
        for index, (vector, category_index, result) in enumerate(scored[:8]):
            correct = index < 6
            gt_action_index = None if correct else (result.action_index + 1) % len(ACTIONS)
            scorer.update(
                vector,
                category_index,
                result.action_index,
                correct,
                gt_action_index=gt_action_index,
                confidence=result.confidence,
            )
        if scorer.decision_count != 8:
            raise RuntimeError(f"expected 8 updates, found {scorer.decision_count}")
        return "8 verified (6 correct, 2 incorrect)"

    check("Update", update_check)

    def conservation_check() -> str:
        status = conservation_status(
            verified_count=8,
            correct_count=6,
            total_decisions=10,
            penalty_ratio=5.0,
            categories_with_data=3,
            total_categories=3,
        )
        if status.status not in {"RED", "AMBER", "GREEN"}:
            raise RuntimeError(f"unexpected status {status.status}")
        return status.status

    check("Conservation", conservation_check)

    def judgment_check() -> str:
        if scorer is None or not scored:
            raise RuntimeError("a scored decision is required")
        vector, category_index, result = scored[-1]
        judgment = compute_judgment(
            result,
            vector,
            scorer.centroids,
            category_index,
            FACTORS,
            ACTIONS,
        )
        return f"action={judgment.action}, confidence_tier={judgment.confidence_tier}"

    check("Judgment", judgment_check)

    def persistence_check() -> str:
        if scorer is None:
            raise RuntimeError("ProfileScorer check did not complete")
        shape = scorer.centroids.shape
        with TemporaryDirectory(prefix="gae-pypi-smoke-") as temp_dir:
            directory = Path(temp_dir)
            bootstrap_path = directory / "bootstrap.json"
            saved_path = directory / "state.json"
            bootstrap_path.write_text(
                json.dumps(
                    {
                        "weights": scorer.centroids.reshape(-1).tolist(),
                        "step": scorer.decision_count,
                        "converged": True,
                        "metadata": {"centroid_shape": list(shape)},
                    }
                ),
                encoding="utf-8",
            )
            state = load_state(bootstrap_path)
            save_state(state, saved_path)
            restored = load_state(saved_path)
            restored_centroids = restored.weights.reshape(shape)
            if not np.allclose(restored_centroids, scorer.centroids):
                raise RuntimeError("restored centroids differ from saved centroids")
        return "centroids match"

    check("State save/load", persistence_check)

    def langchain_check() -> str:
        try:
            from gae.integrations.langchain import JudgmentMemoryCallback
        except ImportError:
            return "optional dependency not installed (handled)"
        callback = JudgmentMemoryCallback(CATEGORIES, ACTIONS, FACTORS)
        if callback.centroids.shape != (3, 3, 5):
            raise RuntimeError("callback centroid shape is incorrect")
        return "available"

    check("LangChain callback", langchain_check)

    def mcp_check() -> str:
        try:
            from gae.integrations.mcp import create_server

            server: Any = create_server(CATEGORIES, ACTIONS, FACTORS)
        except ImportError:
            return "optional dependency not installed (handled)"
        if not hasattr(server, "judgment_memory"):
            raise RuntimeError("server does not expose judgment_memory")
        return "available"

    check("MCP server", mcp_check)

    for passed, label, detail in results:
        marker = "PASS" if passed else "FAIL"
        print(f"[{marker}] {label}: {detail}")
    passed_count = sum(passed for passed, _, _ in results)
    print(f"=== {passed_count}/{len(results)} PASSED ===")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
