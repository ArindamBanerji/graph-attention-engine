"""Unit tests for the optional Airflow ProfileScorer operators."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

import gae.integrations.airflow as airflow_integration
from gae.integrations.airflow import RGIScoreOperator, RGIVerifyOperator


SCORER_CONFIG = {
    "categories": ["routing"],
    "actions": ["accept", "reject"],
    "factors": ["quality", "urgency"],
}


def factor_callable_from_path(_context: dict[str, Any]) -> dict[str, float]:
    """Return named factors for dynamic-import coverage."""
    return {"quality": 0.8, "urgency": 0.2}


def test_score_operator_init() -> None:
    """The scoring operator exposes its templated configuration fields."""
    operator = RGIScoreOperator(
        task_id="score",
        scorer_config=SCORER_CONFIG,
        category="routing",
        factors_callable=factor_callable_from_path,
    )

    assert operator.template_fields == ("scorer_config", "category")
    assert operator.scorer_config == SCORER_CONFIG
    assert operator.category == "routing"


def test_score_operator_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    """execute builds a fresh scorer, calls score, and returns an XCom record."""
    scorer = MagicMock()
    scorer.categories = ["routing"]
    scorer.conservation_status = "GREEN"
    scorer.score.return_value = SimpleNamespace(
        action_name="accept", action_index=0, confidence=0.9
    )
    factory = MagicMock(return_value=scorer)
    monkeypatch.setattr(airflow_integration, "build_profile_scorer", factory)
    operator = RGIScoreOperator(
        task_id="score",
        scorer_config=SCORER_CONFIG,
        category="routing",
        factors_callable=factor_callable_from_path,
    )

    result = operator.execute({"run_id": "run-1"})

    factory.assert_called_once_with(**SCORER_CONFIG)
    np.testing.assert_allclose(scorer.score.call_args.args[0], [0.8, 0.2])
    assert scorer.score.call_args.args[1] == 0
    assert result == {
        "action": "accept",
        "action_index": 0,
        "category": "routing",
        "category_index": 0,
        "confidence": 0.9,
        "conservation_status": "GREEN",
        "decision_id": "score:run-1:routing",
        "factors": [0.8, 0.2],
    }


def test_verify_operator_execute(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verification passes a complete prior decision into scorer.update."""
    scorer = MagicMock()
    scorer.conservation_status = "GREEN"
    scorer.accuracy.return_value = 1.0
    monkeypatch.setattr(airflow_integration, "build_profile_scorer", lambda **_kwargs: scorer)
    operator = RGIVerifyOperator(
        task_id="verify",
        scorer_config=SCORER_CONFIG,
        outcome_callable=lambda _context: {
            "correct": True,
            "category_index": 0,
            "action_index": 1,
            "factors": {"quality": 0.7, "urgency": 0.3},
        },
    )

    result = operator.execute({})

    np.testing.assert_allclose(scorer.update.call_args.args[0], [0.7, 0.3])
    assert scorer.update.call_args.args[1:4] == (0, 1, True)
    assert result == {
        "verified": True,
        "correct": True,
        "conservation_status": "GREEN",
        "accuracy": 1.0,
    }


def test_factors_callable_string_import() -> None:
    """A dotted callable reference is resolved only during execute."""
    operator = RGIScoreOperator(
        task_id="score",
        scorer_config=SCORER_CONFIG,
        category="routing",
        factors_callable=f"{__name__}.factor_callable_from_path",
    )

    result = operator.execute({"run_id": "run-2"})

    assert result["factors"] == [0.8, 0.2]


def test_per_execution_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two executes construct two scorers rather than retaining mutable state."""
    scorers = []

    def factory(**_kwargs: Any) -> MagicMock:
        scorer = MagicMock()
        scorer.categories = ["routing"]
        scorer.conservation_status = "GREEN"
        scorer.score.return_value = SimpleNamespace(
            action_name="accept", action_index=0, confidence=0.5
        )
        scorers.append(scorer)
        return scorer

    monkeypatch.setattr(airflow_integration, "build_profile_scorer", factory)
    operator = RGIScoreOperator(
        task_id="score",
        scorer_config=SCORER_CONFIG,
        category="routing",
        factors_callable=lambda context: {
            "quality": context["quality"], "urgency": 0.1
        },
    )

    operator.execute({"quality": 0.2})
    operator.execute({"quality": 0.9})

    assert len(scorers) == 2
    assert scorers[0] is not scorers[1]
    np.testing.assert_allclose(scorers[0].score.call_args.args[0], [0.2, 0.1])
    np.testing.assert_allclose(scorers[1].score.call_args.args[0], [0.9, 0.1])


def test_conservation_in_output() -> None:
    """The score result exposes conservation status for downstream gates."""
    operator = RGIScoreOperator(
        task_id="score",
        scorer_config=SCORER_CONFIG,
        category="routing",
        factors_callable=factor_callable_from_path,
    )

    assert "conservation_status" in operator.execute({})


def test_no_module_globals() -> None:
    """The integration module has no retained ProfileScorer instance or state."""
    retained_scorers = [
        value
        for _name, value in vars(airflow_integration).items()
        if value.__class__.__name__ == "ProfileScorer"
    ]

    assert retained_scorers == []
