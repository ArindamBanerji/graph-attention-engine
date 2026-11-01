"""FastMCP bridge for GAE Judgment Memory.

The dependency-neutral :class:`JudgmentMemoryService` owns scoring and state.
``create_server`` registers its six methods when FastMCP is installed.

Reference: blog Eq. 4-final, Eq. 4b-final, and Eq. IKS.
"""

from __future__ import annotations

import threading
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover - tested with a mocked registry
    FastMCP = None  # type: ignore[assignment,misc]

from gae import (
    ProfileScorer,
    build_profile_scorer,
    compute_judgment,
    conservation_status,
    load_state,
    save_state,
)
from gae.store import LearningState

STATE_SCHEMA = "gae.langchain.judgment-memory.v1"
TOOL_NAMES = (
    "rgi_score",
    "rgi_verify",
    "rgi_conservation",
    "rgi_decisions",
    "rgi_accuracy",
    "rgi_explain",
)


class JudgmentMemoryService:
    """Dependency-neutral implementation of the six Judgment Memory tools."""

    def __init__(
        self,
        categories: Sequence[str],
        actions: Sequence[str],
        factors: Sequence[str],
        state_path: str | Path | None = None,
        penalty_ratio: float = 5.0,
        q_window: int = 400,
        tau: float = 0.1,
    ) -> None:
        """Build a neutral scorer and restore compatible Eq. 4b-final state."""
        self.categories = self._validated_names("categories", categories)
        self.actions = self._validated_names("actions", actions)
        self.factors = self._validated_names("factors", factors)
        if penalty_ratio <= 0.0:
            raise ValueError("penalty_ratio must be positive")
        if q_window <= 0:
            raise ValueError("q_window must be positive")
        if tau <= 0.0:
            raise ValueError("tau must be positive")

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
        expected_shape = (
            len(self.categories), len(self.actions), len(self.factors)
        )
        assert self.scorer.centroids.shape == expected_shape

        self.state_path = Path(state_path) if state_path is not None else None
        self.penalty_ratio = float(penalty_ratio)
        self.q_window = int(q_window)
        self.tau = float(tau)
        self._lock = threading.RLock()
        self._decisions: list[dict[str, Any]] = []
        self._next_id = 1
        self._initial_centroids = self.scorer.centroids.copy()
        assert self._initial_centroids.shape == expected_shape
        if self.state_path is not None and self.state_path.is_file():
            self.load()

    @staticmethod
    def _validated_names(label: str, values: Sequence[str]) -> list[str]:
        """Validate one name axis in the Eq. 4-final centroid tensor."""
        names = [str(value).strip() for value in values]
        if not names or any(not name for name in names):
            raise ValueError(f"{label} must contain non-empty names")
        if len(set(names)) != len(names):
            raise ValueError(f"{label} must contain unique names")
        return names

    def rgi_score(self, category: str, factors: dict[str, float]) -> dict[str, Any]:
        """Score one decision; use before choosing an action."""
        with self._lock:
            try:
                category_index = self.categories.index(category)
            except ValueError as exc:
                raise ValueError(f"unknown category: {category}") from exc
            supplied = set(factors)
            expected = set(self.factors)
            if supplied != expected:
                missing = sorted(expected - supplied)
                extra = sorted(supplied - expected)
                raise ValueError(
                    f"factors must match configured names; missing={missing}, extra={extra}"
                )
            vector = np.asarray([factors[name] for name in self.factors], dtype=np.float64)
            assert vector.shape == (len(self.factors),)
            if not np.all(np.isfinite(vector)):
                raise ValueError("factor values must be finite")
            if np.any(vector < 0.0) or np.any(vector > 1.0):
                raise ValueError("factor values must be in [0, 1]")
            result = self.scorer.score(vector, category_index)
            decision_id = f"jm-{self._next_id}"
            self._next_id += 1
            decision = {
                "id": decision_id,
                "category_index": category_index,
                "category": category,
                "action_index": result.action_index,
                "action": result.action_name,
                "factors": vector.tolist(),
                "probabilities": result.probabilities.tolist(),
                "confidence": result.confidence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "verified": False,
            }
            self._decisions.append(decision)
            return {
                "decision_id": decision_id,
                "action": result.action_name,
                "probabilities": {
                    name: float(result.probabilities[index])
                    for index, name in enumerate(self.actions)
                },
                "confidence": result.confidence,
                "entropy": result.entropy,
            }

    def rgi_verify(
        self,
        decision_id: str = "last",
        correct: bool = True,
        gt_action: str | None = None,
    ) -> dict[str, Any]:
        """Learn from a verified outcome; use only after checking the result."""
        with self._lock:
            decision = self._resolve_decision(decision_id)
            if decision["verified"]:
                raise ValueError(f"decision {decision['id']} has already been verified")
            gt_action_index: int | None = None
            if gt_action is not None:
                try:
                    gt_action_index = self.actions.index(gt_action)
                except ValueError as exc:
                    raise ValueError(f"unknown gt_action: {gt_action}") from exc
            if not correct and gt_action_index == decision["action_index"]:
                raise ValueError("gt_action must differ from the predicted action")
            vector = np.asarray(decision["factors"], dtype=np.float64)
            assert vector.shape == (len(self.factors),)
            update = self.scorer.update(
                vector,
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
            if self.state_path is not None:
                self.save()
            return {
                "decision_id": decision["id"],
                "centroid_delta": update.centroid_delta_norm,
                "gt_delta": update.gt_delta_norm,
                "decision_count": update.decision_count,
                "conservation": check.status,
                "outcome": update.outcome,
            }

    def rgi_conservation(self) -> dict[str, Any]:
        """Inspect learning-signal health; use before trusting adaptation."""
        with self._lock:
            check = self._conservation_check()
            verified = self._verified_decisions()
            correct_count = sum(bool(item.get("correct")) for item in verified)
            return {
                "status": check.status,
                "signal": check.signal,
                "theta_min": check.theta_min,
                "headroom": check.headroom,
                "passed": check.passed,
                "verified": len(verified),
                "correct": correct_count,
                "decisions": len(self._decisions),
            }

    def rgi_decisions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Return recent decisions; use to review the Judgment Memory audit trail."""
        if limit < 0:
            raise ValueError("limit must be non-negative")
        with self._lock:
            records = self._decisions[-limit:] if limit else []
            return [
                {
                    "id": item["id"],
                    "category": item["category"],
                    "action": item["action"],
                    "factors": {
                        name: item["factors"][index]
                        for index, name in enumerate(self.factors)
                    },
                    "verified": item["verified"],
                    **({"correct": item["correct"]} if item["verified"] else {}),
                    "timestamp": item["timestamp"],
                }
                for item in records
            ]

    def rgi_accuracy(self) -> dict[str, Any]:
        """Report verified accuracy and IKS; use to measure learning quality."""
        with self._lock:
            verified = self._verified_decisions()
            correct_count = sum(bool(item.get("correct")) for item in verified)
            per_category: dict[str, dict[str, Any]] = {}
            for index, category in enumerate(self.categories):
                records = [
                    item for item in verified if item["category_index"] == index
                ]
                category_correct = sum(bool(item.get("correct")) for item in records)
                per_category[category] = {
                    "verified": len(records),
                    "correct": category_correct,
                    "accuracy": category_correct / len(records) if records else 0.0,
                }
            return {
                "overall_accuracy": correct_count / len(verified) if verified else 0.0,
                "per_category": per_category,
                "total_verified": len(verified),
                "total_correct": correct_count,
                "iks": self._iks(),
            }

    def rgi_explain(self, decision_id: str = "last") -> dict[str, Any]:
        """Explain one decision; use to inspect its confidence and factor drivers."""
        with self._lock:
            decision = self._resolve_decision(decision_id)
            vector = np.asarray(decision["factors"], dtype=np.float64)
            probabilities = np.asarray(decision["probabilities"], dtype=np.float64)
            assert vector.shape == (len(self.factors),)
            assert probabilities.shape == (len(self.actions),)
            result = self.scorer.score(vector, int(decision["category_index"]))
            result.action_index = int(decision["action_index"])
            result.action_name = str(decision["action"])
            result.probabilities = probabilities
            result.confidence = float(decision["confidence"])
            judgment = compute_judgment(
                result,
                vector,
                self.scorer.centroids,
                int(decision["category_index"]),
                self.factors,
                self.actions,
            )
            return {
                "action": judgment.action,
                "confidence": judgment.confidence,
                "confidence_tier": judgment.confidence_tier,
                "dominant_factors": judgment.dominant_factors,
                "factor_contributions": judgment.factor_contributions,
                "rationale": judgment.rationale,
                "auto_approve": judgment.auto_approvable,
            }

    def save(self, path: str | Path | None = None) -> Path:
        """Persist the LangChain-compatible Eq. 4b-final state atomically."""
        with self._lock:
            target = self._resolve_path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            centroids = self.scorer.centroids
            assert centroids.ndim == 3
            metadata = {
                "schema": STATE_SCHEMA,
                "categories": self.categories,
                "actions": self.actions,
                "factors": self.factors,
                "centroid_shape": list(centroids.shape),
                "initial_centroids": self._initial_centroids.tolist(),
                "counts": self.scorer.counts.tolist(),
                "scorer_decision_count": self.scorer.decision_count,
                "decisions": self._decisions,
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
        """Load state written by MCP or LangChain after Eq. 4b-final updates."""
        with self._lock:
            state = load_state(self._resolve_path(path))
            metadata = state.metadata
            for key, expected in {
                "categories": self.categories,
                "actions": self.actions,
                "factors": self.factors,
            }.items():
                if metadata.get(key) != expected:
                    raise ValueError(f"state {key} do not match server configuration")
            expected_shape = (
                len(self.categories), len(self.actions), len(self.factors)
            )
            shape = tuple(int(value) for value in metadata["centroid_shape"])
            if shape != expected_shape or state.weights.size != int(np.prod(shape)):
                raise ValueError("state centroid shape does not match server configuration")
            centroids = state.weights.reshape(shape)
            assert centroids.shape == expected_shape
            self.scorer.centroids = centroids
            initial = np.asarray(metadata.get("initial_centroids"), dtype=np.float64)
            if initial.shape != expected_shape:
                raise ValueError("state initial centroid shape is invalid")
            self._initial_centroids = initial.copy()
            counts = np.asarray(metadata.get("counts"), dtype=np.int64)
            expected_counts = (len(self.categories), len(self.actions))
            if counts.shape != expected_counts:
                raise ValueError("state count matrix shape is invalid")
            self.scorer.counts = counts.copy()
            self.scorer.decision_count = int(metadata.get("scorer_decision_count", 0))
            decisions = metadata.get("decisions", [])
            if not isinstance(decisions, list):
                raise ValueError("state decisions must be a list")
            self._decisions = [dict(item) for item in decisions]
            self._next_id = int(metadata.get("next_id", len(self._decisions) + 1))
            self.scorer.set_conservation_status(self._conservation_check().status)

    @property
    def centroids(self) -> np.ndarray:
        """Return a copy of the current Eq. 4-final centroid tensor."""
        with self._lock:
            result = self.scorer.centroids.copy()
            assert result.shape == self.scorer.centroids.shape
            return result

    def _resolve_path(self, path: str | Path | None) -> Path:
        """Resolve the configured Eq. 4b-final persistence path."""
        if path is not None:
            return Path(path)
        if self.state_path is None:
            raise ValueError("no state path was provided")
        return self.state_path

    def _resolve_decision(self, decision_id: str) -> dict[str, Any]:
        """Resolve a stored Eq. 4-final decision by identifier."""
        if not self._decisions:
            raise ValueError("no decisions have been scored")
        if decision_id == "last":
            return self._decisions[-1]
        for decision in self._decisions:
            if decision["id"] == decision_id:
                return decision
        raise KeyError(f"unknown decision_id: {decision_id}")

    def _verified_decisions(self) -> list[dict[str, Any]]:
        """Return decisions admitted to the Eq. 4b-final learning signal."""
        return [item for item in self._decisions if item["verified"]]

    def _conservation_check(self):
        """Compute conservation for the verified Eq. 4b-final signal."""
        verified = self._verified_decisions()
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

    def _iks(self) -> float:
        """Compute normalized centroid drift from blog Eq. IKS."""
        delta = self.scorer.centroids - self._initial_centroids
        assert delta.shape == self._initial_centroids.shape
        cell_norms = np.linalg.norm(delta, axis=2)
        assert cell_norms.shape == (len(self.categories), len(self.actions))
        return round(100.0 * min(float(np.mean(cell_norms)) / 0.20, 1.0), 2)


def create_server(
    categories: Sequence[str],
    actions: Sequence[str],
    factors: Sequence[str],
    state_path: str | Path | None = None,
    penalty_ratio: float = 5.0,
    server_name: str = "rgi-judgment-memory",
    q_window: int = 400,
    tau: float = 0.1,
) -> Any:
    """Create a FastMCP server exposing the six Judgment Memory tools."""
    if FastMCP is None:
        raise ImportError(
            "GAE's MCP integration requires FastMCP. Install it with "
            "`pip install graph-attention-engine[mcp]`."
        )
    service = JudgmentMemoryService(
        categories=categories,
        actions=actions,
        factors=factors,
        state_path=state_path,
        penalty_ratio=penalty_ratio,
        q_window=q_window,
        tau=tau,
    )
    mcp = FastMCP(server_name)
    registrations = (
        (service.rgi_score, "Score a decision and return action probabilities."),
        (service.rgi_verify, "Learn from a verified decision outcome."),
        (service.rgi_conservation, "Report Judgment Memory conservation health."),
        (service.rgi_decisions, "Return recent Judgment Memory decisions."),
        (service.rgi_accuracy, "Report verified accuracy and institutional knowledge."),
        (service.rgi_explain, "Explain the confidence and factors behind a decision."),
    )
    for function, description in registrations:
        mcp.tool(name=function.__name__, description=description)(function)
        setattr(mcp, function.__name__, function)
    setattr(mcp, "judgment_memory", service)
    return mcp

