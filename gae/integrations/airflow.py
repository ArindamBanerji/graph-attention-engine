"""Airflow operators for per-execution GAE pipeline decision scoring.

Install ``graph-attention-engine[airflow]`` for Airflow's full BaseOperator.
The small fallback keeps this optional integration importable for local tests
and DAG-serialization tooling that does not execute tasks.

Reference: blog Eq. 4-final and Eq. 4b-final.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from gae.profile_scorer import build_profile_scorer

try:
    from airflow.models.baseoperator import BaseOperator  # type: ignore[import]
except ImportError:  # pragma: no cover - exercised when Airflow is unavailable
    class BaseOperator:  # type: ignore[no-redef]
        """Minimal BaseOperator-compatible fallback for optional imports.

        Reference: blog Eq. 4-final integration boundary.
        """

        template_fields: Sequence[str] = ()

        def __init__(self, task_id: str, **kwargs: Any) -> None:
            """Store task metadata without requiring an Airflow installation.

            Reference: blog Eq. 4-final integration boundary.
            """
            self.task_id = task_id
            self.params = kwargs


ContextCallable = Callable[[dict[str, Any]], dict[str, Any]]
CallableReference = ContextCallable | str


def _resolve_callable(value: CallableReference) -> ContextCallable:
    """Resolve a callable or dotted callable path at task execution time.

    Reference: blog Eq. 4-final pipeline factor assembly.
    """
    if callable(value):
        return value
    module_name, separator, attribute_name = value.rpartition(".")
    if not separator or not module_name or not attribute_name:
        raise ValueError("callable import paths must use 'package.module.callable'")
    resolved = getattr(importlib.import_module(module_name), attribute_name)
    if not callable(resolved):
        raise TypeError(f"resolved object {value!r} is not callable")
    return resolved


def _factor_vector(
    values: Mapping[str, Any] | Sequence[float], scorer_config: Mapping[str, Any]
) -> np.ndarray:
    """Convert configured named factors into an Eq. 4-final numeric vector.

    Reference: blog Eq. 4-final pipeline factor assembly.
    """
    if not isinstance(values, Mapping):
        return np.asarray(values, dtype=np.float64)
    if "factors" in values and isinstance(values["factors"], Sequence):
        return np.asarray(values["factors"], dtype=np.float64)

    factor_names = scorer_config.get("factors")
    if factor_names is None:
        return np.asarray(list(values.values()), dtype=np.float64)
    if not isinstance(factor_names, Sequence) or isinstance(factor_names, str):
        raise ValueError("scorer_config['factors'] must be a sequence of factor names")
    try:
        return np.asarray([values[str(name)] for name in factor_names], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"missing configured factor {exc.args[0]!r}") from exc


def _decision_id(context: Mapping[str, Any], task_id: str, category: str) -> str:
    """Build a stable execution-local decision identifier for downstream XComs.

    Reference: blog Eq. 4b-final outcome attribution.
    """
    explicit = context.get("decision_id")
    if explicit is not None:
        return str(explicit)
    dag_run = context.get("dag_run")
    run_id = getattr(dag_run, "run_id", None) if dag_run is not None else None
    run_id = run_id or context.get("run_id", "manual")
    return f"{task_id}:{run_id}:{category}"


class RGIScoreOperator(BaseOperator):
    """Score one pipeline decision with a scorer instantiated per execution.

    No mutable scorer is retained on the operator or module. A configured
    StateBackend is therefore consulted by the newly built scorer for each
    task attempt.

    Reference: blog Eq. 4-final.
    """

    template_fields = ("scorer_config", "category")

    def __init__(
        self,
        *,
        scorer_config: dict[str, Any],
        category: str,
        factors_callable: CallableReference,
        task_id: str,
        **kwargs: Any,
    ) -> None:
        """Store serializable score-task configuration without building a scorer.

        Reference: blog Eq. 4-final.
        """
        super().__init__(task_id=task_id, **kwargs)
        self.scorer_config = dict(scorer_config)
        self.category = category
        self.factors_callable = factors_callable

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Score one decision and return an XCom-compatible decision record.

        Reference: blog Eq. 4-final.
        """
        scorer = build_profile_scorer(**self.scorer_config)
        if scorer.categories is None:
            raise ValueError("score operator requires named scorer categories")
        try:
            category_index = scorer.categories.index(self.category)
        except (AttributeError, ValueError) as exc:
            raise ValueError(f"unknown scorer category {self.category!r}") from exc

        factor_values = _resolve_callable(self.factors_callable)(context)
        factors = _factor_vector(factor_values, self.scorer_config)
        result = scorer.score(factors, category_index)
        return {
            "action": result.action_name,
            "action_index": result.action_index,
            "category": self.category,
            "category_index": category_index,
            "confidence": result.confidence,
            "conservation_status": scorer.conservation_status,
            "decision_id": _decision_id(context, self.task_id, self.category),
            "factors": factors.tolist(),
        }


class RGIVerifyOperator(BaseOperator):
    """Verify one scored outcome with a fresh scorer for each task execution.

    ``outcome_callable`` must return ``correct`` and ``category_index`` plus
    ``factors`` and ``action_index``. The latter two may instead be supplied
    by an embedded ``decision`` mapping, normally the XCom record emitted by
    :class:`RGIScoreOperator`.

    Reference: blog Eq. 4b-final.
    """

    template_fields = ("scorer_config",)

    def __init__(
        self,
        *,
        scorer_config: dict[str, Any],
        outcome_callable: CallableReference,
        task_id: str,
        **kwargs: Any,
    ) -> None:
        """Store verification configuration without retaining scorer state.

        Reference: blog Eq. 4b-final.
        """
        super().__init__(task_id=task_id, **kwargs)
        self.scorer_config = dict(scorer_config)
        self.outcome_callable = outcome_callable

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        """Apply a verified outcome and return XCom-compatible learning status.

        Reference: blog Eq. 4b-final.
        """
        scorer = build_profile_scorer(**self.scorer_config)
        outcome = _resolve_callable(self.outcome_callable)(context)
        decision = outcome.get("decision", context.get("decision", {}))
        if not isinstance(decision, Mapping):
            raise ValueError("outcome decision must be a mapping when provided")

        category_index = int(outcome.get("category_index", decision.get("category_index", -1)))
        action_index = int(outcome.get("action_index", decision.get("action_index", -1)))
        raw_factors = outcome.get("factors", decision.get("factors"))
        if raw_factors is None:
            raise ValueError("outcome requires factors or a decision containing factors")
        if category_index < 0 or action_index < 0:
            raise ValueError("outcome requires category_index and action_index")

        factors = _factor_vector(raw_factors, self.scorer_config)
        correct = bool(outcome["correct"])
        gt_action_index = outcome.get("gt_action_index")
        confidence = outcome.get("confidence", decision.get("confidence"))
        scorer.update(
            factors,
            category_index,
            action_index,
            correct,
            gt_action_index=None if gt_action_index is None else int(gt_action_index),
            confidence=None if confidence is None else float(confidence),
        )
        return {
            "verified": True,
            "correct": correct,
            "conservation_status": scorer.conservation_status,
            "accuracy": scorer.accuracy(category_index),
        }
