"""
Domain-agnostic per-category, per-dimension weight estimation.

This module estimates a `(C, D)` weight matrix from labeled decisions and a
fixed centroid tensor. The estimator is intentionally independent from higher-
level scoring classes so it can be reused across domains.
"""

from __future__ import annotations

from typing import Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np


Decision = Tuple[np.ndarray, int, int]


@runtime_checkable
class DKEstimator(Protocol):
    """Protocol for estimators that produce per-category `(C, D)` weights."""

    def estimate(
        self,
        decisions: Sequence[Decision],
        centroids: np.ndarray,
        n_categories: int,
        n_dims: int,
    ) -> np.ndarray:
        """Return estimated weights with shape `(n_categories, n_dims)`."""
        ...


class CoordinateDescentEstimator:
    """
    Coordinate-descent estimator for per-category dimension weights.

    Parameters
    ----------
    n_rounds
        Number of coordinate-descent passes. Must be at least 1.
    max_per_cat
        Maximum decisions used per category. Must be at least 10.
    candidates
        Positive candidate weight values tried for each dimension.
    tau
        Softmax temperature used in the internal accuracy objective.
    seed
        Optional seed for deterministic per-category subsampling.
    """

    def __init__(
        self,
        n_rounds: int = 5,
        max_per_cat: int = 400,
        candidates: Optional[Sequence[float]] = None,
        tau: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        if n_rounds < 1:
            raise ValueError(f"n_rounds must be >= 1, got {n_rounds}")
        if max_per_cat < 10:
            raise ValueError(f"max_per_cat must be >= 10, got {max_per_cat}")
        if candidates is None:
            candidates = (0.1, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0)
        candidate_array = np.asarray(tuple(candidates), dtype=np.float64)
        if candidate_array.ndim != 1 or candidate_array.size == 0:
            raise ValueError("candidates must be a non-empty 1-D sequence")
        if np.any(candidate_array <= 0.0):
            raise ValueError("all candidates must be > 0")
        if tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {tau}")

        self.n_rounds = n_rounds
        self.max_per_cat = max_per_cat
        self.candidates = candidate_array
        self.tau = float(tau)
        self.seed = seed

    def estimate(
        self,
        decisions: Sequence[Decision],
        centroids: np.ndarray,
        n_categories: int,
        n_dims: int,
    ) -> np.ndarray:
        """
        Estimate per-category dimension weights.

        Parameters
        ----------
        decisions
            Sequence of `(factor_vector, category_index, correct_action_index)`.
        centroids
            Either `(C, A, D)` or shared `(A, D)` centroid array.
        n_categories
            Number of categories `C`.
        n_dims
            Number of factor dimensions `D`.
        """
        if n_categories < 1:
            raise ValueError(f"n_categories must be >= 1, got {n_categories}")
        if n_dims < 1:
            raise ValueError(f"n_dims must be >= 1, got {n_dims}")

        centroid_array = np.asarray(centroids, dtype=np.float64)
        if centroid_array.ndim == 3:
            if centroid_array.shape[0] != n_categories:
                raise ValueError(
                    f"centroids.shape[0]={centroid_array.shape[0]} must equal "
                    f"n_categories={n_categories}"
                )
            if centroid_array.shape[2] != n_dims:
                raise ValueError(
                    f"centroids.shape[2]={centroid_array.shape[2]} must equal "
                    f"n_dims={n_dims}"
                )
        elif centroid_array.ndim == 2:
            if centroid_array.shape[1] != n_dims:
                raise ValueError(
                    f"centroids.shape[1]={centroid_array.shape[1]} must equal "
                    f"n_dims={n_dims}"
                )
        else:
            raise ValueError(
                f"centroids must have shape (C, A, D) or (A, D), got {centroid_array.shape}"
            )

        weights = np.ones((n_categories, n_dims), dtype=np.float64)

        for category_index in range(n_categories):
            category_decisions = [
                (np.asarray(factor_vector, dtype=np.float64), correct_action_index)
                for factor_vector, decision_category, correct_action_index in decisions
                if decision_category == category_index
            ]
            if not category_decisions:
                continue

            sampled = self._subsample_category(category_decisions, category_index)
            features = np.stack([factor_vector for factor_vector, _ in sampled], axis=0)
            correct_actions = np.asarray(
                [correct_action_index for _, correct_action_index in sampled],
                dtype=np.int64,
            )

            if features.shape[1] != n_dims:
                raise ValueError(
                    f"decision factor length {features.shape[1]} must equal n_dims={n_dims}"
                )

            cat_centroids = self._category_centroids(
                centroid_array,
                category_index=category_index,
            )
            if cat_centroids.shape[1] != n_dims:
                raise ValueError(
                    f"category centroid dims {cat_centroids.shape[1]} must equal n_dims={n_dims}"
                )
            if np.any(correct_actions < 0) or np.any(correct_actions >= cat_centroids.shape[0]):
                raise ValueError("correct_action_index out of bounds for provided centroids")

            category_weights = np.ones(n_dims, dtype=np.float64)
            best_accuracy = self._compute_accuracy(
                features=features,
                correct_actions=correct_actions,
                centroids=cat_centroids,
                weights=category_weights,
            )

            for _ in range(self.n_rounds):
                improved = False
                for dim_index in range(n_dims):
                    base_value = category_weights[dim_index]
                    best_dim_value = base_value
                    best_dim_accuracy = best_accuracy
                    for candidate in self.candidates:
                        trial_weights = category_weights.copy()
                        trial_weights[dim_index] = candidate
                        accuracy = self._compute_accuracy(
                            features=features,
                            correct_actions=correct_actions,
                            centroids=cat_centroids,
                            weights=trial_weights,
                        )
                        if accuracy > best_dim_accuracy:
                            best_dim_accuracy = accuracy
                            best_dim_value = float(candidate)
                    category_weights[dim_index] = best_dim_value
                    if best_dim_accuracy > best_accuracy:
                        best_accuracy = best_dim_accuracy
                        improved = True
                if not improved:
                    break

            weights[category_index] = category_weights

        return weights

    def _subsample_category(
        self,
        category_decisions: Sequence[Tuple[np.ndarray, int]],
        category_index: int,
    ) -> Sequence[Tuple[np.ndarray, int]]:
        """Apply deterministic per-category subsampling when needed."""
        if len(category_decisions) <= self.max_per_cat:
            return list(category_decisions)
        if self.seed is None:
            return list(category_decisions[: self.max_per_cat])

        rng = np.random.default_rng(self.seed + category_index)
        choice = rng.choice(len(category_decisions), size=self.max_per_cat, replace=False)
        return [category_decisions[int(idx)] for idx in np.sort(choice)]

    def _category_centroids(
        self,
        centroids: np.ndarray,
        category_index: int,
    ) -> np.ndarray:
        """Select `(A, D)` centroids for a category or return shared centroids."""
        if centroids.ndim == 3:
            return centroids[category_index]
        return centroids

    def _compute_accuracy(
        self,
        features: np.ndarray,
        correct_actions: np.ndarray,
        centroids: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """
        Compute mean classification accuracy for one category.

        Shapes
        ------
        features: `(N, D)`
        correct_actions: `(N,)`
        centroids: `(A, D)`
        weights: `(D,)`
        """
        features = np.asarray(features, dtype=np.float64)
        correct_actions = np.asarray(correct_actions, dtype=np.int64)
        centroids = np.asarray(centroids, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        if features.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {features.shape}")
        if centroids.ndim != 2:
            raise ValueError(f"centroids must be 2-D, got shape {centroids.shape}")
        if weights.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {weights.shape}")
        if correct_actions.ndim != 1:
            raise ValueError(
                f"correct_actions must be 1-D, got shape {correct_actions.shape}"
            )
        if features.shape[0] != correct_actions.shape[0]:
            raise ValueError("features and correct_actions must have the same length")
        if features.shape[1] != centroids.shape[1] or features.shape[1] != weights.shape[0]:
            raise ValueError("feature, centroid, and weight dimensions must match")
        if self.tau <= 0.0:
            raise ValueError(f"tau must be > 0, got {self.tau}")

        diff = features[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        weighted_diff = diff * weights[np.newaxis, np.newaxis, :]
        distances = np.sum(weighted_diff * diff, axis=2)
        logits = -distances / self.tau
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        predicted = np.argmax(probabilities, axis=1)
        return float(np.mean(predicted == correct_actions))
