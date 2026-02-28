"""Tests for gae.primitives — softmax and scaled_dot_product_attention."""

import numpy as np
import pytest

from gae.primitives import softmax, scaled_dot_product_attention


# ---------------------------------------------------------------------------
# softmax
# ---------------------------------------------------------------------------

class TestSoftmax:
    def test_output_shape_preserved(self):
        x = np.random.randn(4, 6)
        out = softmax(x)
        assert out.shape == (4, 6)

    def test_rows_sum_to_one(self):
        x = np.random.randn(5, 8)
        out = softmax(x, axis=-1)
        np.testing.assert_allclose(out.sum(axis=-1), np.ones(5), atol=1e-6)

    def test_cols_sum_to_one_when_axis_0(self):
        x = np.random.randn(5, 8)
        out = softmax(x, axis=0)
        np.testing.assert_allclose(out.sum(axis=0), np.ones(8), atol=1e-6)

    def test_all_outputs_positive(self):
        x = np.random.randn(3, 10)
        out = softmax(x)
        assert np.all(out > 0)

    def test_numerically_stable_large_values(self):
        # Without the max-shift trick, exp(1000) overflows to inf
        x = np.array([[1000.0, 1001.0, 1002.0]])
        out = softmax(x)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out.sum(axis=-1), [1.0], atol=1e-6)

    def test_uniform_input_gives_uniform_output(self):
        x = np.zeros((4,))
        out = softmax(x)
        np.testing.assert_allclose(out, np.full(4, 0.25), atol=1e-9)

    def test_rejects_non_array(self):
        with pytest.raises(AssertionError):
            softmax([1.0, 2.0, 3.0])  # list, not ndarray


# ---------------------------------------------------------------------------
# scaled_dot_product_attention
# ---------------------------------------------------------------------------

class TestScaledDotProductAttention:
    def setup_method(self):
        rng = np.random.default_rng(42)
        self.n, self.m, self.d_k, self.d_v = 4, 6, 8, 16
        self.Q = rng.standard_normal((self.n, self.d_k))
        self.K = rng.standard_normal((self.m, self.d_k))
        self.V = rng.standard_normal((self.m, self.d_v))

    def test_output_shapes(self):
        out, w = scaled_dot_product_attention(self.Q, self.K, self.V)
        assert out.shape == (self.n, self.d_v)
        assert w.shape == (self.n, self.m)

    def test_weights_sum_to_one(self):
        _, w = scaled_dot_product_attention(self.Q, self.K, self.V)
        np.testing.assert_allclose(w.sum(axis=-1), np.ones(self.n), atol=1e-6)

    def test_weights_non_negative(self):
        _, w = scaled_dot_product_attention(self.Q, self.K, self.V)
        assert np.all(w >= 0)

    def test_mask_zeros_out_position(self):
        # Applying -inf mask to last column should zero out that weight
        mask = np.zeros((self.n, self.m))
        mask[:, -1] = -np.inf
        _, w = scaled_dot_product_attention(self.Q, self.K, self.V, mask=mask)
        np.testing.assert_allclose(w[:, -1], np.zeros(self.n), atol=1e-6)

    def test_self_attention_identity(self):
        # When Q == K == V and d_k == d_v, output must have same shape as input
        n, d = 5, 8
        X = np.random.randn(n, d)
        out, _ = scaled_dot_product_attention(X, X, X)
        assert out.shape == (n, d)

    def test_shape_mismatch_q_k_raises(self):
        Q = np.random.randn(4, 8)
        K = np.random.randn(6, 10)   # wrong d_k
        V = np.random.randn(6, 16)
        with pytest.raises(AssertionError):
            scaled_dot_product_attention(Q, K, V)

    def test_shape_mismatch_k_v_raises(self):
        Q = np.random.randn(4, 8)
        K = np.random.randn(6, 8)
        V = np.random.randn(5, 16)   # wrong m
        with pytest.raises(AssertionError):
            scaled_dot_product_attention(Q, K, V)

    def test_mask_shape_mismatch_raises(self):
        mask = np.zeros((self.n, self.m + 1))   # wrong second dim
        with pytest.raises(AssertionError):
            scaled_dot_product_attention(self.Q, self.K, self.V, mask=mask)

    def test_1d_inputs_raise(self):
        with pytest.raises(AssertionError):
            scaled_dot_product_attention(
                np.random.randn(8),
                np.random.randn(6, 8),
                np.random.randn(6, 16),
            )
