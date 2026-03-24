"""
GAE Bootstrap — domain-agnostic synthetic calibration for ProfileScorer.

Takes a ProfileScorer whose centroids encode expert prior knowledge and runs
synthetic calibration rounds where the prior itself acts as oracle. Produces
a warm-started scorer that has "learned" from its own priors.

Deterministic given a seed. No domain-specific code.

Reference: docs/gae_design_v8_3.md §17; GAE-BOOT-1.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np

from gae.profile_scorer import ProfileScorer


def _assert_bootstrap_anchor_not_overwritten(filepath: str) -> None:
    """
    Raises RuntimeError if iks_bootstrap_soc.json already exists.
    Called before any write to the IKS anchor sidecar file.
    The IKS anchor is written ONCE at P28 Phase 2 bootstrap.
    Overwriting it causes IKS oscillation trap (T1 simulation failure:
    IKS locks at ~3.0 permanently). Hard architectural constraint.
    If intentional re-bootstrap needed: delete the file manually first.
    """
    if os.path.exists(filepath):
        raise RuntimeError(
            f"IKS anchor {filepath} already exists and cannot be "
            f"overwritten. It must be written exactly once at P28 "
            f"Phase 2 bootstrap. Delete the file manually if "
            f"intentional re-bootstrap is required."
        )


def write_iks_bootstrap_anchor(filepath: str, data: dict) -> None:
    """
    Write the IKS anchor sidecar file (iks_bootstrap_soc.json) exactly once.

    Calls _assert_bootstrap_anchor_not_overwritten() before writing to
    enforce the P28 Phase 2 single-write constraint. Raises RuntimeError
    if the file already exists.

    Args:
      filepath: Absolute or relative path to the anchor JSON file.
      data:     Dictionary to serialise as the anchor payload.

    Reference: T1 architecture constraint — μ₀ anchor freeze guard.
    """
    _assert_bootstrap_anchor_not_overwritten(filepath)
    with open(filepath, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


@dataclass
class BootstrapResult:
    """
    Result of a bootstrap_calibration() run.

    Reference: docs/gae_design_v8_3.md §17; GAE-BOOT-1.

    Attributes
    ----------
    n_decisions : int
        Total synthetic decisions made across all rounds.
    n_rounds : int
        Number of rounds completed.
    converged : bool
        True if final centroid drift < convergence_tol AND n_rounds > 0.
    final_drift : float
        Mean L2 drift of scorer.mu from the prior centroids after all rounds.
    decisions_per_category : dict[str, int]
        Number of synthetic decisions made per category name.
    metadata : dict
        seed, sigma, n_rounds_requested, timestamp.
    """

    n_decisions: int
    n_rounds: int
    converged: bool
    final_drift: float
    decisions_per_category: dict[str, int]
    metadata: dict


def bootstrap_calibration(
    scorer: ProfileScorer,
    categories: list[str],
    n_rounds: int = 10,
    samples_per_action: int = 5,
    sigma: float = 0.08,
    convergence_tol: float = 0.01,
    seed: int = 42,
) -> BootstrapResult:
    """
    Run synthetic calibration rounds using the scorer's own prior as oracle.

    Algorithm:
      1. Capture prior centroids: prior_mu = scorer.mu.copy()
         Shape: (n_categories, n_actions, n_factors)
      2. For each round in range(n_rounds):
         For each (category_idx, category_name):
           For each action_idx in range(n_actions):
             Sample `samples_per_action` factor vectors:
               f = clip(N(prior_mu[cat, act, :], sigma), 0.0, 1.0)
             For each f:
               oracle_action_idx = argmin ||f - prior_mu[cat, a, :]||^2
               scorer.update(f, category_idx, oracle_action_idx, correct=True)
      3. Compute final_drift = mean(||scorer.mu - prior_mu||)
         converged = (n_rounds > 0) and (final_drift < convergence_tol)
      4. Return BootstrapResult.

    scorer is mutated in place. Clipping to [0.0, 1.0] is enforced by
    scorer.update() — not re-applied here.

    Args:
      scorer:           ProfileScorer instance. Mutated in place.
      categories:       Category name list. len must equal scorer.n_categories.
      n_rounds:         Number of calibration rounds. Default 10.
      samples_per_action: Factor vectors sampled per (category, action). Default 5.
      sigma:            Gaussian noise std around each centroid. Default 0.08.
      convergence_tol:  Drift threshold for convergence flag. Default 0.01.
      seed:             RNG seed for reproducibility. Default 42.

    Returns:
      BootstrapResult with drift, convergence flag, and decision counts.

    Raises:
      ValueError: if len(categories) != scorer.n_categories.

    Reference: docs/gae_design_v8_3.md §17; GAE-BOOT-1.
    """
    if len(categories) != scorer.n_categories:
        raise ValueError(
            f"len(categories)={len(categories)} != "
            f"scorer.n_categories={scorer.n_categories}"
        )

    n_categories = scorer.n_categories
    n_actions    = scorer.n_actions

    print(
        f"[BOOTSTRAP] Starting calibration: {n_categories} categories, "
        f"{n_actions} actions, {n_rounds} rounds, seed={seed}"
    )

    # Step 1: capture prior centroids
    prior_mu = scorer.mu.copy()
    assert prior_mu.shape == scorer.mu.shape, (
        f"prior_mu.shape={prior_mu.shape} != scorer.mu.shape={scorer.mu.shape}"
    )

    rng = np.random.default_rng(seed)

    # Track decisions per category
    decisions_per_category: dict[str, int] = {cat: 0 for cat in categories}
    n_decisions = 0

    # Step 2: calibration rounds
    for round_idx in range(n_rounds):
        for cat_idx, cat_name in enumerate(categories):
            for act_idx in range(n_actions):
                # Sample factor vectors around this action's prior centroid
                centroid = prior_mu[cat_idx, act_idx, :]   # shape (n_factors,)
                assert centroid.shape == (scorer.n_factors,), (
                    f"centroid.shape={centroid.shape} != ({scorer.n_factors},)"
                )

                # f ~ clip(N(centroid, sigma), 0, 1), shape (samples, n_factors)
                noise = rng.normal(
                    loc=0.0,
                    scale=sigma,
                    size=(samples_per_action, scorer.n_factors),
                )
                samples = np.clip(centroid + noise, 0.0, 1.0)
                assert samples.shape == (samples_per_action, scorer.n_factors), (
                    f"samples.shape={samples.shape} != "
                    f"({samples_per_action}, {scorer.n_factors})"
                )

                for s_idx in range(samples_per_action):
                    f = samples[s_idx]
                    assert f.shape == (scorer.n_factors,), (
                        f"f.shape={f.shape} != ({scorer.n_factors},)"
                    )

                    # Oracle: nearest centroid in the prior (not the live mu)
                    diff = f - prior_mu[cat_idx]          # (n_actions, n_factors)
                    assert diff.shape == (n_actions, scorer.n_factors), (
                        f"diff.shape={diff.shape} != ({n_actions}, {scorer.n_factors})"
                    )
                    dists = np.sum(diff ** 2, axis=1)     # (n_actions,)
                    assert dists.shape == (n_actions,), (
                        f"dists.shape={dists.shape} != ({n_actions},)"
                    )
                    oracle_action_idx = int(np.argmin(dists))

                    # Update scorer with confirmed-correct decision
                    scorer.update(
                        f=f,
                        category_index=cat_idx,
                        action_index=oracle_action_idx,
                        correct=True,
                    )

                    n_decisions += 1
                    decisions_per_category[cat_name] += 1

        # Compute drift after this round for progress log
        drift = float(np.mean(np.abs(scorer.mu - prior_mu)))
        print(
            f"[BOOTSTRAP] Round {round_idx + 1}/{n_rounds} complete — "
            f"drift={drift:.4f}"
        )

    # Step 3: final drift and convergence
    final_drift_arr = np.abs(scorer.mu - prior_mu)
    assert final_drift_arr.shape == prior_mu.shape, (
        f"final_drift_arr.shape={final_drift_arr.shape} != {prior_mu.shape}"
    )
    final_drift = float(np.mean(final_drift_arr))
    converged   = (n_rounds > 0) and (final_drift < convergence_tol)

    print(
        f"[BOOTSTRAP] Complete: {n_decisions} decisions, "
        f"converged={converged}, final_drift={final_drift:.4f}"
    )

    return BootstrapResult(
        n_decisions=n_decisions,
        n_rounds=n_rounds,
        converged=converged,
        final_drift=final_drift,
        decisions_per_category=decisions_per_category,
        metadata={
            "seed": seed,
            "sigma": sigma,
            "n_rounds_requested": n_rounds,
            "timestamp": time.time(),
        },
    )
