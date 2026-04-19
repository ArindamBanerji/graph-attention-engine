"""
GAE primitives — Tier 1 building blocks.

Implements the scaled dot-product attention mechanism:

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) ) V        [Eq. 1, blog]

where:
    Q : query matrix  (n, d_k)
    K : key matrix    (m, d_k)
    V : value matrix  (m, d_v)

    softmax is applied row-wise over the (n, m) logit matrix.

Reference: docs/gae_design_v10_6.md §2.1; blog Eq. 1.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

def compute_entropy(p: np.ndarray, eps: float = 1e-10) -> float:
    """Shannon entropy of probability distribution.

    H(p) = -sum(p * log(p + eps))

    eps guard prevents log(0) = -inf when softmax underflows to exact zero
    on extreme logits.

    Args:
        p: np.ndarray, probability distribution (sums to 1)
        eps: float, small constant for numerical stability

    Returns:
        float, entropy in nats (natural log)
    """
    p_safe = np.maximum(p, eps)
    return float(-np.sum(p_safe * np.log(p_safe)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically-stable row-wise softmax.

    Implements:  softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))

    Reference: docs/gae_design_v10_6.md §2.1 (stability note); blog Eq. 1 denominator.

    Parameters
    ----------
    x : np.ndarray
        Input array of any shape.
    axis : int, default -1
        Axis along which softmax is computed.

    Returns
    -------
    np.ndarray
        Same shape as *x*; values sum to 1.0 along *axis*.
    """
    assert isinstance(x, np.ndarray), "softmax: x must be np.ndarray"
    shift = x - x.max(axis=axis, keepdims=True)
    exp_x = np.exp(shift)
    result = exp_x / exp_x.sum(axis=axis, keepdims=True)
    assert result.shape == x.shape, (
        f"softmax: output shape {result.shape} != input shape {x.shape}"
    )
    return result


# ---------------------------------------------------------------------------
# scaled_dot_product_attention
# ---------------------------------------------------------------------------

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention (numpy implementation).

    Implements Eq. 1 from the blog:

        scores  = Q K^T / sqrt(d_k)                          [Eq. 1a]
        weights = softmax(scores + mask)                     [Eq. 1b]
        output  = weights V                                  [Eq. 1c]

    Reference: docs/gae_design_v10_6.md §2.1; blog Eq. 1.

    Parameters
    ----------
    Q : np.ndarray, shape (n, d_k)
        Query matrix.
    K : np.ndarray, shape (m, d_k)
        Key matrix.
    V : np.ndarray, shape (m, d_v)
        Value matrix.
    mask : np.ndarray or None, shape (n, m)
        Optional additive mask (e.g. -inf for padding positions).
        Added to raw logits *before* softmax.

    Returns
    -------
    output : np.ndarray, shape (n, d_v)
        Attention-weighted value vectors.
    weights : np.ndarray, shape (n, m)
        Attention weight matrix (rows sum to 1).

    Raises
    ------
    AssertionError
        On shape mismatches between Q, K, V, or mask.
    """
    assert Q.ndim == 2, f"Q must be 2-D, got shape {Q.shape}"
    assert K.ndim == 2, f"K must be 2-D, got shape {K.shape}"
    assert V.ndim == 2, f"V must be 2-D, got shape {V.shape}"

    n, d_k = Q.shape
    m, d_k_k = K.shape
    m_v, d_v = V.shape

    assert d_k == d_k_k, (
        f"Q and K must share d_k dimension: Q.shape={Q.shape}, K.shape={K.shape}"
    )
    assert m == m_v, (
        f"K and V must share sequence dimension: K.shape={K.shape}, V.shape={V.shape}"
    )

    # Eq. 1a — raw logits, shape (n, m)
    scale = np.sqrt(d_k).astype(Q.dtype) if hasattr(Q.dtype, 'type') else np.sqrt(d_k)
    logits = Q @ K.T / np.sqrt(float(d_k))
    assert logits.shape == (n, m), (
        f"logits shape {logits.shape} != expected ({n}, {m})"
    )

    # Optional additive mask — Eq. 1b
    if mask is not None:
        assert mask.shape == (n, m), (
            f"mask shape {mask.shape} != logits shape ({n}, {m})"
        )
        logits = logits + mask

    # Eq. 1b — attention weights, shape (n, m)
    weights = softmax(logits, axis=-1)
    assert weights.shape == (n, m), (
        f"weights shape {weights.shape} != expected ({n}, {m})"
    )

    # Eq. 1c — output, shape (n, d_v)
    output = weights @ V
    assert output.shape == (n, d_v), (
        f"output shape {output.shape} != expected ({n}, {d_v})"
    )

    return output, weights
