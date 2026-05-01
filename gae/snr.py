"""
Signal-to-noise diagnostics for centroid geometry.

This module estimates a category-level classification ceiling from centroid
separation and per-factor noise:

    snr_c = separation_c / noise
    ceiling_c = Phi(snr_c / 2)

where Phi is the standard normal CDF. The /2 term is the equal-variance
two-class Gaussian Bayes approximation: when two action centroids are separated
by distance d and the effective noise scale is sigma_eff, the correct-routing
probability is approximated by Phi(d / (2*sigma_eff)).

For weighted kernels, the signal is measured in the same weighted geometry as
the noise:

    weighted_distance(x, y) = || w ⊙ (x - y) ||_2
    weighted_noise = sqrt(sum_j (sigma_j * w_j)^2)

No scipy dependency. NumPy only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from gae.calibration import compute_dominant_axis
from gae.profile_scorer import ProfileScorer

_AT_CEILING_THRESHOLD = 0.95
_NEAR_CEILING_THRESHOLD = 0.80

if not hasattr(np, "erf"):
    np.erf = np.vectorize(math.erf)


def _phi(x):
    """
    Standard normal CDF via erf.

    Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
    """
    x = np.asarray(x, dtype=np.float64)
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))


@dataclass
class CategorySNR:
    """
    Per-category signal-to-noise summary.

    Attributes
    ----------
    category_index : int
        Category axis index.
    category_name : str
        Human-readable category label.
    action_separation : float
        Mean pairwise action-centroid separation for this category.
    weighted_noise : float
        Effective deployment noise scale in the active kernel geometry.
    snr : float
        action_separation / weighted_noise.
    ceiling_estimate : float
        Phi(snr / 2) approximation of attainable routing accuracy.
    status : str
        One of healthy, near_ceiling, at_ceiling.
    weakest_action_pair : tuple[str, str]
        Lowest-separation action pair in this category.
    weakest_pair_distance : float
        Distance for weakest_action_pair in the active geometry.
    """

    category_index: int
    category_name: str
    action_separation: float
    weighted_noise: float
    snr: float
    ceiling_estimate: float
    status: str
    weakest_action_pair: Tuple[str, str]
    weakest_pair_distance: float


@dataclass
class SNRReport:
    """
    Full SNR report across all categories.

    Attributes
    ----------
    categories : list[CategorySNR]
        One entry per category.
    weighted_noise : float
        Effective deployment noise scale shared by all categories.
    factor_importance : dict[str, float]
        Per-factor dominant-axis score in [0, 1].
    mean_snr : float
        Mean category SNR.
    mean_ceiling_estimate : float
        Mean category ceiling estimate.
    status_counts : dict[str, int]
        Counts by status band.
    proposed_improvement : str
        Human-readable recommendation targeting the weakest category/pair.
    """

    categories: List[CategorySNR]
    weighted_noise: float
    factor_importance: Dict[str, float]
    mean_snr: float
    mean_ceiling_estimate: float
    status_counts: Dict[str, int]
    proposed_improvement: str


def _default_names(prefix: str, n: int) -> List[str]:
    return [f"{prefix}_{i}" for i in range(n)]


def _validate_vector(name: str, values: np.ndarray, expected_len: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    assert arr.shape == (expected_len,), (
        f"{name}.shape={arr.shape} != ({expected_len},)"
    )
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or Inf values")
    return arr


def _pairwise_distances(mu_c: np.ndarray, weights: np.ndarray) -> Tuple[List[float], List[Tuple[int, int, float]]]:
    """
    Compute all pairwise action distances inside one category.

    weighted_distance(a1, a2) = || weights ⊙ (mu[a1] - mu[a2]) ||_2
    """
    n_actions, n_factors = mu_c.shape
    assert weights.shape == (n_factors,), (
        f"weights.shape={weights.shape} != ({n_factors},)"
    )
    dists: List[float] = []
    triples: List[Tuple[int, int, float]] = []
    for a1 in range(n_actions):
        for a2 in range(a1 + 1, n_actions):
            diff = weights * (mu_c[a1] - mu_c[a2])
            dist = float(np.linalg.norm(diff))
            dists.append(dist)
            triples.append((a1, a2, dist))
    return dists, triples


def _status_for_ceiling(ceiling_estimate: float) -> str:
    if ceiling_estimate >= _AT_CEILING_THRESHOLD:
        return "at_ceiling"
    if ceiling_estimate >= _NEAR_CEILING_THRESHOLD:
        return "near_ceiling"
    return "healthy"


def compute_snr_report(
    centroids: np.ndarray,
    sigma: np.ndarray,
    kernel_weights=None,
    categories: Optional[List[str]] = None,
    actions: Optional[List[str]] = None,
    factor_names: Optional[List[str]] = None,
) -> SNRReport:
    """
    Compute category-level SNR and ceiling estimates from a centroid tensor.

    Parameters
    ----------
    centroids : np.ndarray, shape (C, A, d)
        Category-action-factor centroid tensor.
    sigma : np.ndarray, shape (d,)
        Per-factor noise scale.
    kernel_weights : array-like, shape (d,), optional
        Active kernel weights. None means uniform geometry (L2).
    categories : list[str], optional
        Category names. Defaults to cat_0..cat_{C-1}.
    actions : list[str], optional
        Action names. Defaults to action_0..action_{A-1}.
    factor_names : list[str], optional
        Factor names. Defaults to factor_0..factor_{d-1}.

    Returns
    -------
    SNRReport
        Category SNR summary plus global factor-importance and recommendation.
    """
    mu = np.asarray(centroids, dtype=np.float64)
    assert mu.ndim == 3, f"centroids must be 3-D (C, A, d), got shape {mu.shape}"
    if not np.all(np.isfinite(mu)):
        raise ValueError("centroids contain NaN or Inf values")

    n_categories, n_actions, n_factors = mu.shape
    sigma_arr = _validate_vector("sigma", sigma, n_factors)

    if kernel_weights is None:
        weights = np.ones(n_factors, dtype=np.float64)
    else:
        weights = _validate_vector("kernel_weights", kernel_weights, n_factors)

    if categories is None:
        categories = _default_names("cat", n_categories)
    if actions is None:
        actions = _default_names("action", n_actions)
    if factor_names is None:
        factor_names = _default_names("factor", n_factors)

    if len(categories) != n_categories:
        raise ValueError(
            f"categories length {len(categories)} != n_categories {n_categories}"
        )
    if len(actions) != n_actions:
        raise ValueError(
            f"actions length {len(actions)} != n_actions {n_actions}"
        )
    if len(factor_names) != n_factors:
        raise ValueError(
            f"factor_names length {len(factor_names)} != n_factors {n_factors}"
        )

    weighted_noise = float(np.sqrt(np.sum((sigma_arr * weights) ** 2)))
    if weighted_noise <= 0:
        raise ValueError("weighted_noise must be positive")

    diagnostics = None
    if kernel_weights is None:
        diagnostics = ProfileScorer(mu=mu.copy(), actions=list(actions), categories=list(categories)).diagnostics()

    category_reports: List[CategorySNR] = []
    weakest_category: Optional[CategorySNR] = None

    for c_idx in range(n_categories):
        mu_c = mu[c_idx]
        assert mu_c.shape == (n_actions, n_factors), (
            f"mu[{c_idx}].shape={mu_c.shape} != ({n_actions}, {n_factors})"
        )
        pair_dists, pair_triples = _pairwise_distances(mu_c, weights)
        weakest_pair = min(pair_triples, key=lambda item: item[2])

        if diagnostics is not None:
            action_separation = float(diagnostics["per_category"][c_idx]["separation"])
        else:
            action_separation = float(np.mean(pair_dists)) if pair_dists else 0.0

        snr = action_separation / weighted_noise
        ceiling_estimate = float(_phi(snr / 2.0))
        report = CategorySNR(
            category_index=c_idx,
            category_name=categories[c_idx],
            action_separation=action_separation,
            weighted_noise=weighted_noise,
            snr=float(snr),
            ceiling_estimate=ceiling_estimate,
            status=_status_for_ceiling(ceiling_estimate),
            weakest_action_pair=(
                actions[weakest_pair[0]],
                actions[weakest_pair[1]],
            ),
            weakest_pair_distance=float(weakest_pair[2]),
        )
        category_reports.append(report)
        if (
            weakest_category is None
            or report.weakest_pair_distance < weakest_category.weakest_pair_distance
        ):
            weakest_category = report

    dominant_axis = compute_dominant_axis(mu)
    assert dominant_axis.shape == (n_factors,), (
        f"dominant_axis.shape={dominant_axis.shape} != ({n_factors},)"
    )
    factor_importance = {
        factor_names[i]: float(dominant_axis[i])
        for i in range(n_factors)
    }

    weakest_factor_name = min(
        factor_importance.items(),
        key=lambda item: item[1],
    )[0]
    assert weakest_category is not None, "at least one category is required"
    proposed_improvement = (
        f"Target {weakest_category.category_name}: increase separation between "
        f"{weakest_category.weakest_action_pair[0]} and "
        f"{weakest_category.weakest_action_pair[1]} "
        f"(distance={weakest_category.weakest_pair_distance:.4f}, "
        f"ceiling={weakest_category.ceiling_estimate:.3f}). "
        f"Review low-leverage factor {weakest_factor_name} for clearer signal."
    )

    status_counts = {"healthy": 0, "near_ceiling": 0, "at_ceiling": 0}
    for category_report in category_reports:
        status_counts[category_report.status] += 1

    return SNRReport(
        categories=category_reports,
        weighted_noise=weighted_noise,
        factor_importance=factor_importance,
        mean_snr=float(np.mean([c.snr for c in category_reports])),
        mean_ceiling_estimate=float(
            np.mean([c.ceiling_estimate for c in category_reports])
        ),
        status_counts=status_counts,
        proposed_improvement=proposed_improvement,
    )
