"""
GAE Enrichment Advisor — rank factors by expected Day-1 accuracy lift.

Reference: V-CLAIM60-THRESHOLD (W-share calibration sweep, 2026-03-24).
"""
from __future__ import annotations

_SIGMA_BENCHMARKS_DEFAULT = {
    "asset_criticality": 0.060,
    "threat_intel":      0.080,
    "pattern_history":   0.090,
    "time_anomaly":      0.065,
    "device_trust":      0.090,
    "travel_match":      0.110,
}


def rank_enrichment_opportunities(
    sigma_profile: dict,
    mu_star_spread: dict,
    sigma_benchmarks: dict | None = None,
) -> list[dict]:
    """
    Rank factors by expected Day-1 accuracy lift from enrichment.

    enrichment_score_j = spread_j × W_gain_j
    where W_gain_j = 1/sigma_target_j² − 1/sigma_current_j²

    Priority classification:
      enrichment_score > 50  → 'high (>=+5pp expected)'
      enrichment_score > 20  → 'moderate (+2 to +5pp expected)'
      else                   → 'low (<+2pp expected)'

    Parameters
    ----------
    sigma_profile : dict
        factor_name → current σ (measured, pre-enrichment).
    mu_star_spread : dict
        factor_name → max-min spread of μ* values across all (cat, act) pairs.
    sigma_benchmarks : dict or None
        factor_name → target σ (best-observed deployment). Uses built-in
        defaults if None.

    Returns
    -------
    list[dict]
        Sorted descending by enrichment_score. Each entry has keys:
        factor, current_sigma, target_sigma, w_gain, spread, score, priority.
    """
    if sigma_benchmarks is None:
        sigma_benchmarks = _SIGMA_BENCHMARKS_DEFAULT

    rows = []
    for factor, sigma_current in sigma_profile.items():
        sigma_target = sigma_benchmarks.get(factor)
        if sigma_target is None:
            continue
        spread = mu_star_spread.get(factor, 0.0)
        w_gain = max(0.0, 1.0 / sigma_target ** 2 - 1.0 / sigma_current ** 2)
        score  = spread * w_gain

        if score > 50:
            priority = "high (>=+5pp expected)"
        elif score > 20:
            priority = "moderate (+2 to +5pp expected)"
        else:
            priority = "low (<+2pp expected)"

        rows.append(dict(
            factor=factor,
            current_sigma=round(sigma_current, 4),
            target_sigma=round(sigma_target, 4),
            w_gain=round(w_gain, 2),
            spread=round(spread, 4),
            score=round(score, 2),
            priority=priority,
        ))

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows
