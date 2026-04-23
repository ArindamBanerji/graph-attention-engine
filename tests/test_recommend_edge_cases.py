import pytest
import numpy as np
from gae.kernel_selector import KernelSelector, KernelRecommendation

D = 6
SIGMA = np.ones(D)


@pytest.fixture
def ks():
    return KernelSelector(d=D, sigma_per_factor=SIGMA)


def test_recommend_returns_kernel_recommendation(ks):
    rec = ks.recommend()
    assert isinstance(rec, KernelRecommendation)


def test_recommend_kernel_name_is_valid(ks):
    rec = ks.recommend()
    assert rec.recommended_kernel in ("l2", "diagonal", "shrinkage")


def test_recommend_method_is_rule(ks):
    rec = ks.recommend()
    assert rec.method == "rule"


def test_recommend_stable_across_calls(ks):
    r1 = ks.recommend()
    r2 = ks.recommend()
    assert r1.recommended_kernel == r2.recommended_kernel


def test_recommend_after_recording(ks):
    mu = np.full((1, 4, D), 0.5)
    actions = ["a", "b", "c", "d"]
    factors = np.random.rand(D)
    ks.record_comparison(factors, 0, mu, 0, actions)
    rec = ks.recommend()
    assert isinstance(rec.recommended_kernel, str)
