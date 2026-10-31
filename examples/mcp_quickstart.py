"""Call all six MCP Judgment Memory tools directly without FastMCP."""

from __future__ import annotations

import tempfile
from pathlib import Path

from gae.integrations.mcp import JudgmentMemoryService


CATEGORIES = ["retrieval", "generation", "routing"]
ACTIONS = ["use", "skip", "escalate", "retry"]
FACTORS = ["relevance", "confidence", "recency", "source_trust", "complexity"]


def factors_for(index: int) -> dict[str, float]:
    """Build deterministic Eq. 4-final factors for one example decision."""
    return {
        name: 0.3 + ((index + offset * 3) % 10) * 0.06
        for offset, name in enumerate(FACTORS)
    }


def main() -> None:
    """Run all six direct tools through a complete Eq. 4b-final loop."""
    with tempfile.TemporaryDirectory() as directory:
        state_path = Path(directory) / "jm_state.json"
        service = JudgmentMemoryService(
            CATEGORIES,
            ACTIONS,
            FACTORS,
            state_path=state_path,
            penalty_ratio=5.0,
        )
        scored: list[dict[str, object]] = []
        for index in range(20):
            scored.append(
                service.rgi_score(CATEGORIES[index % len(CATEGORIES)], factors_for(index))
            )
        day_1 = scored[0]["probabilities"]

        verification: dict[str, object] = {}
        for index, decision in enumerate(scored[:15]):
            correct = index not in {3, 7, 11, 14}
            gt_action = None if correct else ACTIONS[(index + 1) % len(ACTIONS)]
            if gt_action == decision["action"]:
                gt_action = ACTIONS[(index + 2) % len(ACTIONS)]
            verification = service.rgi_verify(
                str(decision["decision_id"]), correct, gt_action
            )

        conservation = service.rgi_conservation()
        decisions = service.rgi_decisions(limit=3)
        accuracy = service.rgi_accuracy()
        explanation = service.rgi_explain()
        day_20 = service.rgi_score(CATEGORIES[0], factors_for(0))["probabilities"]

        print("RGI MCP Judgment Memory quickstart")
        print("rgi_score:", scored[0])
        print("rgi_verify:", verification, "(15 verified outcomes applied)")
        print("rgi_conservation:", conservation)
        print("rgi_decisions:", decisions)
        print("rgi_accuracy:", accuracy)
        print("rgi_explain:", explanation)
        print("Day 1 probabilities :", day_1)
        print("Day 20 probabilities:", day_20)

        restored = JudgmentMemoryService(
            CATEGORIES,
            ACTIONS,
            FACTORS,
            state_path=state_path,
            penalty_ratio=5.0,
        )
        assert restored.rgi_accuracy()["total_verified"] == 15
        print("State reload preserved 15 verified outcomes: yes")


if __name__ == "__main__":
    main()

