"""hello-gae — your first five minutes with the Graph Attention Engine.

Run ``python hello_gae.py``.  This is deliberately standalone: no server,
external graph service, or SDK checkout is required.
"""

from __future__ import annotations

import sqlite3

from gae.calibration import conservation_status


def main() -> None:
    categories = ("priority", "normal", "low")
    with sqlite3.connect(":memory:") as graph:
        graph.execute(
            "CREATE TABLE decisions (category TEXT, action TEXT, correct INTEGER)"
        )
        for index in range(10):
            category = categories[index % len(categories)]
            action = "approve" if index % 3 else "review"
            correct = int(index != 2 and index != 7)
            graph.execute(
                "INSERT INTO decisions VALUES (?, ?, ?)",
                (category, action, correct),
            )
        graph.commit()
        verified, correct = graph.execute(
            "SELECT COUNT(*), COALESCE(SUM(correct), 0) FROM decisions"
        ).fetchone()
        status = conservation_status(
            verified_count=verified,
            correct_count=correct,
            total_decisions=verified,
            penalty_ratio=1.0,
            categories_with_data=len(categories),
            total_categories=len(categories),
        )
        print("hello-gae — in-memory SQLite graph")
        print(f"  decisions: {verified}  categories: {', '.join(categories)}")
        print(f"  correct: {correct}/{verified}")
        print(
            f"  conservation: {status.status} "
            f"signal={status.signal:.2f} theta_min={status.theta_min:.2f}"
        )


if __name__ == "__main__":
    main()
