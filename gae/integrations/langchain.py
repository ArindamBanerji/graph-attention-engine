"""LangChain callback bridge for GAE Judgment Memory.

This module is intentionally absent from :mod:`gae`'s root imports. Install
``graph-attention-engine[langchain]`` before importing it.

Reference: blog Eq. 4-final, Eq. 4b-final, and Eq. IKS.
"""

from __future__ import annotations

import logging
import threading
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError as exc:  # pragma: no cover - exercised in an isolated import test
    raise ImportError(
        "GAE's LangChain integration requires langchain-core. Install it with "
        "`pip install graph-attention-engine[langchain]`."
    ) from exc

from gae import (
    CentroidUpdate,
    ProfileScorer,
    build_profile_scorer,
    compute_judgment,
    conservation_status,
    load_state,
    save_state,
)
from gae.store import LearningState

logger = logging.getLogger(__name__)

FactorFunction = Callable[[str, str], Sequence[float]]
ConservationMode = Literal["warn", "raise", "log"]


def _json_safe(value: Any) -> Any:
    """Return a JSON-safe representation of a callback value (blog Eq. 4b-final audit trail)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


class JudgmentMemoryCallback(BaseCallbackHandler):
    """Add verified-outcome Judgment Memory to LangChain tool calls.

    Tool starts are scored with blog Eq. 4-final. Only explicit ``verify``
    calls update centroids through Eq. 4b-final.
    """

    def __init__(
        self,
        categories: Sequence[str],
        actions: Sequence[str],
        factors: Sequence[str],
        state_path: str | Path | None = None,
        penalty_ratio: float = 5.0,
        q_window: int = 400,
        tau: float = 0.1,
        on_conservation_red: ConservationMode = "warn",
        *,
        factor_fn: FactorFunction | None = None,
        tool_category_map: Mapping[str, str] | None = None,
    ) -> None:
        """Build a neutral scorer and optionally restore state (blog Eq. 4-final)."""
        super().__init__()
        self.categories = self._validated_names("categories", categories)
        self.actions = self._validated_names("actions", actions)
        self.factors = self._validated_names("factors", factors)
        if penalty_ratio <= 0.0:
            raise ValueError("penalty_ratio must be positive")
        if q_window <= 0:
            raise ValueError("q_window must be positive")
        if tau <= 0.0:
            raise ValueError("tau must be positive")
        if on_conservation_red not in {"warn", "raise", "log"}:
            raise ValueError("on_conservation_red must be 'warn', 'raise', or 'log'")

        neutral = {
            category: {
                action: [0.5] * len(self.factors) for action in self.actions
            }
            for category in self.categories
        }
        self.scorer: ProfileScorer = build_profile_scorer(
            self.categories,
            self.actions,
            neutral,
            len(self.factors),
        )
        self.scorer.categories = list(self.categories)
        self.scorer.tau = float(tau)
        assert self.scorer.centroids.shape == (
            len(self.categories), len(self.actions), len(self.factors)
        ), "neutral centroid tensor has an unexpected shape"

        self.state_path = Path(state_path) if state_path is not None else None
        self.penalty_ratio = float(penalty_ratio)
        self.q_window = int(q_window)
        self.tau = float(tau)
        self.on_conservation_red = on_conservation_red
        self.factor_fn = factor_fn
        self._tool_category_map = dict(tool_category_map or {})
        unknown_categories = set(self._tool_category_map.values()) - set(self.categories)
        if unknown_categories:
            raise ValueError(
                "tool_category_map contains unknown categories: "
                + ", ".join(sorted(unknown_categories))
            )

        self._lock = threading.RLock()
        self._decisions: list[dict[str, Any]] = []
        self._run_decisions: dict[str, str] = {}
        self._next_id = 1
        self._initial_centroids = self.scorer.centroids.copy()
        assert self._initial_centroids.shape == self.scorer.centroids.shape

        if self.state_path is not None and self.state_path.is_file():
            self.load()

    @staticmethod
    def _validated_names(label: str, values: Sequence[str]) -> list[str]:
        """Validate one named scorer axis (blog Eq. 4-final tensor definition)."""
        names = [str(value) for value in values]
        if not names:
            raise ValueError(f"{label} must not be empty")
        if any(not name for name in names):
            raise ValueError(f"{label} must not contain empty names")
        if len(set(names)) != len(names):
            raise ValueError(f"{label} must contain unique names")
        return names

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Score and record a tool decision using blog Eq. 4-final."""
        with self._lock:
            tool_name = str(serialized.get("name") or kwargs.get("name") or "tool")
            category_index = self._map_tool_to_category(tool_name)
            factor_vector = self._build_factors(input_str, tool_name)
            result = self.scorer.score(factor_vector, category_index)
            decision_id = f"jm-{self._next_id}"
            self._next_id += 1
            decision = {
                "id": decision_id,
                "category_index": category_index,
                "category": self.categories[category_index],
                "action_index": result.action_index,
                "action": result.action_name,
                "factors": factor_vector.tolist(),
                "probabilities": result.probabilities.tolist(),
                "confidence": result.confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "tool_name": tool_name,
                "input": input_str,
                "verified": False,
            }
            self._decisions.append(decision)
            run_id = kwargs.get("run_id")
            if run_id is not None:
                self._run_decisions[str(run_id)] = decision_id
            self._handle_conservation_red(self._conservation_check())

    def on_tool_end(self, output: Any, **kwargs: Any) -> None:
        """Attach tool output to its pending Eq. 4-final decision record."""
        with self._lock:
            decision: dict[str, Any] | None = None
            run_id = kwargs.get("run_id")
            if run_id is not None:
                decision_id = self._run_decisions.pop(str(run_id), None)
                if decision_id is not None:
                    decision = self._find_decision(decision_id)
            if decision is None:
                decision = next(
                    (item for item in reversed(self._decisions) if "output" not in item),
                    None,
                )
            if decision is not None:
                decision["output"] = _json_safe(output)

    def verify(
        self,
        decision_id: str = "last",
        correct: bool = True,
        gt_action: str | int | None = None,
    ) -> CentroidUpdate:
        """Apply one verified outcome through blog Eq. 4b-final and persist it."""
        with self._lock:
            update = self._verify_locked(decision_id, correct, gt_action)
            if self.state_path is not None:
                self.save()
            return update

    def verify_batch(self, verifications: list[dict[str, Any]]) -> list[CentroidUpdate]:
        """Apply verified outcomes through Eq. 4b-final and persist once."""
        with self._lock:
            updates = [
                self._verify_locked(
                    str(item.get("decision_id", "last")),
                    bool(item.get("correct", True)),
                    cast(str | int | None, item.get("gt_action")),
                )
                for item in verifications
            ]
            if self.state_path is not None:
                self.save()
            return updates

    def _verify_locked(
        self,
        decision_id: str,
        correct: bool,
        gt_action: str | int | None,
    ) -> CentroidUpdate:
        """Mutate one verified centroid while holding the Eq. 4b-final lock."""
        decision = self._resolve_decision(decision_id)
        if decision["verified"]:
            raise ValueError(f"decision {decision['id']} has already been verified")
        gt_action_index = self._action_index(gt_action)
        if not correct and gt_action_index == decision["action_index"]:
            raise ValueError("gt_action must differ from the predicted action when incorrect")
        factor_vector = np.asarray(decision["factors"], dtype=np.float64)
        assert factor_vector.shape == (len(self.factors),)
        update = self.scorer.update(
            factor_vector,
            int(decision["category_index"]),
            int(decision["action_index"]),
            bool(correct),
            gt_action_index=gt_action_index,
            confidence=float(decision["confidence"]),
        )
        decision["verified"] = True
        decision["correct"] = bool(correct)
        decision["gt_action_index"] = gt_action_index
        decision["update"] = asdict(update)
        check = self._conservation_check()
        self.scorer.set_conservation_status(check.status)
        return update

    def status(self) -> dict[str, Any]:
        """Return accuracy, conservation, and Eq. IKS learning diagnostics."""
        with self._lock:
            check = self._conservation_check()
            verified = [item for item in self._decisions if item["verified"]]
            correct_count = sum(bool(item.get("correct")) for item in verified)
            category_stats: dict[str, dict[str, Any]] = {}
            for index, category in enumerate(self.categories):
                items = [
                    item for item in verified if item["category_index"] == index
                ]
                category_correct = sum(bool(item.get("correct")) for item in items)
                category_stats[category] = {
                    "verified": len(items),
                    "correct": category_correct,
                    "accuracy": category_correct / len(items) if items else 0.0,
                }
            category_counts = {
                category: sum(item["category"] == category for item in self._decisions)
                for category in self.categories
            }
            top_category = (
                max(category_counts, key=lambda name: category_counts[name])
                if self._decisions
                else None
            )
            return {
                "decisions": len(self._decisions),
                "verified": len(verified),
                "accuracy": correct_count / len(verified) if verified else 0.0,
                "conservation": check.status,
                "conservation_check": check._asdict(),
                "iks": self._iks(),
                "top_category": top_category,
                "accuracy_by_category": category_stats,
            }

    def judgment(self, decision_id: str = "last") -> dict[str, Any]:
        """Explain a stored decision using the Judgment layer over Eq. 4-final."""
        with self._lock:
            decision = self._resolve_decision(decision_id)
            factor_vector = np.asarray(decision["factors"], dtype=np.float64)
            probabilities = np.asarray(decision["probabilities"], dtype=np.float64)
            assert factor_vector.shape == (len(self.factors),)
            assert probabilities.shape == (len(self.actions),)
            scoring_result = self.scorer.score(
                factor_vector, int(decision["category_index"])
            )
            scoring_result.action_index = int(decision["action_index"])
            scoring_result.action_name = str(decision["action"])
            scoring_result.probabilities = probabilities
            scoring_result.confidence = float(decision["confidence"])
            result = compute_judgment(
                scoring_result,
                factor_vector,
                self.scorer.centroids,
                int(decision["category_index"]),
                self.factors,
                self.actions,
            )
            return asdict(result)

    def save(self, path: str | Path | None = None) -> Path:
        """Persist centroids and decision audit state with blog Eq. 4b-final storage."""
        with self._lock:
            target = self._resolve_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            centroids = self.scorer.centroids
            assert centroids.ndim == 3
            metadata = {
                "schema": "gae.langchain.judgment-memory.v1",
                "categories": self.categories,
                "actions": self.actions,
                "factors": self.factors,
                "centroid_shape": list(centroids.shape),
                "initial_centroids": self._initial_centroids.tolist(),
                "counts": self.scorer.counts.tolist(),
                "scorer_decision_count": self.scorer.decision_count,
                "decisions": _json_safe(self._decisions),
                "next_id": self._next_id,
                "tau": self.tau,
                "penalty_ratio": self.penalty_ratio,
                "q_window": self.q_window,
            }
            state = LearningState(
                weights=centroids.reshape(-1).copy(),
                step=len(self._decisions),
                converged=self._conservation_check().status == "GREEN",
                metadata=metadata,
            )
            assert state.weights.shape == (centroids.size,)
            save_state(state, target)
            return target

    def load(self, path: str | Path | None = None) -> None:
        """Restore centroids and audit state saved after Eq. 4b-final updates."""
        with self._lock:
            target = self._resolve_path(path)
            state = load_state(target)
            metadata = state.metadata
            expected_axes = {
                "categories": self.categories,
                "actions": self.actions,
                "factors": self.factors,
            }
            for key, expected in expected_axes.items():
                if metadata.get(key) != expected:
                    raise ValueError(f"state {key} do not match callback configuration")
            shape = tuple(int(value) for value in metadata["centroid_shape"])
            expected_shape = (
                len(self.categories), len(self.actions), len(self.factors)
            )
            if shape != expected_shape or state.weights.size != int(np.prod(shape)):
                raise ValueError("state centroid shape does not match callback configuration")
            centroids = state.weights.reshape(shape)
            assert centroids.shape == expected_shape
            self.scorer.centroids = centroids
            initial = np.asarray(metadata.get("initial_centroids"), dtype=np.float64)
            if initial.shape != expected_shape:
                raise ValueError("state initial centroid shape is invalid")
            self._initial_centroids = initial.copy()
            counts = np.asarray(metadata.get("counts"), dtype=np.int64)
            if counts.shape != (len(self.categories), len(self.actions)):
                raise ValueError("state count matrix shape is invalid")
            self.scorer.counts = counts.copy()
            self.scorer.decision_count = int(metadata.get("scorer_decision_count", 0))
            decisions = metadata.get("decisions", [])
            if not isinstance(decisions, list):
                raise ValueError("state decisions must be a list")
            self._decisions = [dict(item) for item in decisions]
            self._next_id = int(metadata.get("next_id", len(self._decisions) + 1))
            self._run_decisions.clear()
            self.scorer.set_conservation_status(self._conservation_check().status)

    def _resolve_path(self, path: str | Path | None) -> Path:
        """Resolve an explicit or configured persistence path (blog Eq. 4b-final)."""
        if path is not None:
            return Path(path)
        if self.state_path is None:
            raise ValueError("no state path was provided")
        return self.state_path

    def _map_tool_to_category(self, tool_name: str) -> int:
        """Map a tool to its category axis for blog Eq. 4-final."""
        mapped = self._tool_category_map.get(tool_name)
        if mapped is not None:
            return self.categories.index(mapped)
        return hash(tool_name) % len(self.categories)

    def _build_factors(self, input_str: str, tool_name: str) -> np.ndarray:
        """Build the factor vector consumed by blog Eq. 4-final."""
        raw = (
            self.factor_fn(input_str, tool_name)
            if self.factor_fn is not None
            else [0.5] * len(self.factors)
        )
        vector = np.asarray(raw, dtype=np.float64)
        assert vector.shape == (len(self.factors),), (
            f"factor_fn returned shape {vector.shape}; expected ({len(self.factors)},)"
        )
        if not np.all(np.isfinite(vector)):
            raise ValueError("factor_fn returned NaN or Inf")
        if np.any(vector < 0.0) or np.any(vector > 1.0):
            raise ValueError("factor values must be in [0, 1]")
        return vector

    def _action_index(self, action: str | int | None) -> int | None:
        """Resolve a verified ground-truth action for blog Eq. 4b-final."""
        if action is None:
            return None
        if isinstance(action, int):
            if not 0 <= action < len(self.actions):
                raise ValueError(f"gt_action index {action} is out of range")
            return action
        try:
            return self.actions.index(action)
        except ValueError as exc:
            raise ValueError(f"unknown gt_action: {action}") from exc

    def _find_decision(self, decision_id: str) -> dict[str, Any] | None:
        """Find one Eq. 4-final audit record without raising."""
        return next(
            (item for item in self._decisions if item["id"] == decision_id),
            None,
        )

    def _resolve_decision(self, decision_id: str) -> dict[str, Any]:
        """Resolve ``last`` or an explicit Eq. 4-final decision identifier."""
        if not self._decisions:
            raise ValueError("no decisions have been scored")
        if decision_id == "last":
            return self._decisions[-1]
        decision = self._find_decision(decision_id)
        if decision is None:
            raise KeyError(f"unknown decision_id: {decision_id}")
        return decision

    def _conservation_check(self):
        """Calculate the verified-signal conservation law for Eq. 4b-final."""
        verified = [item for item in self._decisions if item["verified"]]
        correct_count = sum(bool(item.get("correct")) for item in verified)
        categories_with_data = len({item["category_index"] for item in verified})
        return conservation_status(
            len(verified),
            correct_count,
            len(self._decisions),
            self.penalty_ratio,
            window=self.q_window,
            categories_with_data=categories_with_data,
            total_categories=len(self.categories),
        )

    def _handle_conservation_red(self, check: Any) -> None:
        """Apply the configured RED response to the Eq. 4b-final signal gate."""
        self.scorer.set_conservation_status(check.status)
        if check.status != "RED":
            return
        message = (
            "GAE conservation status is RED "
            f"(signal={check.signal}, theta_min={check.theta_min})"
        )
        if self.on_conservation_red == "raise":
            raise RuntimeError(message)
        logger.warning(message)
        if self.on_conservation_red == "warn":
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _iks(self) -> float:
        """Compute the normalized centroid-drift score from blog Eq. IKS."""
        delta = self.scorer.centroids - self._initial_centroids
        assert delta.shape == self._initial_centroids.shape
        cell_norms = np.linalg.norm(delta, axis=2)
        assert cell_norms.shape == (len(self.categories), len(self.actions))
        mean_drift = float(np.mean(cell_norms))
        return round(100.0 * min(mean_drift / 0.20, 1.0), 2)

    @property
    def decisions(self) -> int:
        """Return the number of Eq. 4-final tool decisions scored."""
        with self._lock:
            return len(self._decisions)

    @property
    def accuracy(self) -> float:
        """Return accuracy over outcomes verified for Eq. 4b-final."""
        return float(self.status()["accuracy"])

    @property
    def conservation(self) -> str:
        """Return the current Eq. 4b-final conservation state."""
        return str(self.status()["conservation"])

    @property
    def centroids(self) -> np.ndarray:
        """Return a thread-safe copy of the Eq. 4-final centroid geometry."""
        with self._lock:
            result = self.scorer.centroids.copy()
            assert result.shape == self.scorer.centroids.shape
            return result

