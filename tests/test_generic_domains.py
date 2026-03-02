"""
tests/test_generic_domains.py

Comprehensive test suite proving GAE is domain-agnostic.

All factor names, action names, and domain labels are drawn from NON-SOC
scenarios (supply-chain, finance, healthcare, logistics).  The GAE library
must not know or care which domain is driving it.

Groups
------
1  Multi-Domain Scoring       — score_entity works for arbitrary domains
2  Learning Across Domains    — independent states diverge; 20:1 asymmetry holds
3  Convergence Scenarios      — correct/incorrect/mixed/noisy outcome streams
4  Edge Cases                 — extreme shapes, zero vectors, temperature limits
5  Persistence Round-Trip     — save/reload preserves W, names, count, defaults
6  FactorComputer Protocol    — arbitrary-domain mock computers satisfy Protocol
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from gae.calibration import CalibrationProfile
from gae.contracts import PropertySpec, SchemaContract
from gae.convergence import get_convergence_metrics
from gae.factors import FactorComputer, assemble_factor_vector
from gae.learning import ALPHA, EPSILON, LAMBDA_NEG, W_CLAMP, LearningState
from gae.scoring import ScoringResult, score_entity


# ---------------------------------------------------------------------------
# Shared RNG — deterministic seeds keep tests hermetic
# ---------------------------------------------------------------------------

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _make_state(
    n_actions: int,
    n_factors: int,
    names: list[str] | None = None,
    seed: int = 42,
) -> LearningState:
    rng = _rng(seed)
    W = rng.standard_normal((n_actions, n_factors)) * 0.3
    if names is None:
        names = [f"factor_{i}" for i in range(n_factors)]
    return LearningState(W=W, n_actions=n_actions, n_factors=n_factors,
                         factor_names=names, profile=CalibrationProfile())


def _make_f(n_factors: int, seed: int = 7) -> np.ndarray:
    rng = _rng(seed)
    return np.clip(rng.random((1, n_factors)).astype(np.float64), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Persistence helpers (learning.LearningState has no built-in save/load)
# ---------------------------------------------------------------------------

def _save_learning_state(state: LearningState, path: Path) -> None:
    data = {
        "W": state.W.tolist(),
        "n_actions": state.n_actions,
        "n_factors": state.n_factors,
        "factor_names": state.factor_names,
        "decision_count": state.decision_count,
        "discount_strength": state.discount_strength,
        "epsilon_vector": state.epsilon_vector.tolist(),
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_learning_state(path: Path) -> LearningState:
    data = json.loads(path.read_text(encoding="utf-8"))
    return LearningState(
        W=np.array(data["W"], dtype=np.float64),
        n_actions=data["n_actions"],
        n_factors=data["n_factors"],
        factor_names=list(data["factor_names"]),
        profile=CalibrationProfile(),
        decision_count=data["decision_count"],
        discount_strength=data["discount_strength"],
        epsilon_vector=np.array(data["epsilon_vector"], dtype=np.float64),
    )


# ===========================================================================
# GROUP 1 — Multi-Domain Scoring
# ===========================================================================

class TestMultiDomainScoring:

    def test_supply_chain_po_scoring(self):
        """Supply-chain purchase-order: 6 factors, 3 actions, no SOC vocabulary."""
        rng = _rng(1)
        W_s2p = rng.standard_normal((3, 6)) * 0.3
        actions_s2p = ["approve", "flag", "dual_source"]
        f_s2p = np.array([[
            0.9,   # supplier_reliability
            0.2,   # price_variance
            0.7,   # geo_risk
            0.8,   # delivery_history
            0.6,   # contract_compliance
            0.4,   # volume_discount
        ]])

        result = score_entity(f_s2p, W_s2p, actions_s2p, tau=0.3)

        assert isinstance(result, ScoringResult)
        assert result.selected_action in actions_s2p
        assert abs(result.action_probabilities.flatten().sum() - 1.0) < 1e-6
        assert result.action_probabilities.shape == (1, 3)
        assert 0.0 < result.confidence <= 1.0

    def test_financial_compliance_scoring(self):
        """Financial compliance: 4 factors, 5 actions."""
        rng = _rng(2)
        W_fin = rng.standard_normal((5, 4)) * 0.3
        actions_fin = ["approve", "review", "hold", "escalate", "reject"]
        f_fin = np.array([[0.1, 0.8, 0.5, 0.3]])  # regulatory, size, counterparty, anomaly

        result = score_entity(f_fin, W_fin, actions_fin, tau=0.2)

        assert result.selected_action in actions_fin
        np.testing.assert_allclose(result.action_probabilities.sum(), 1.0, atol=1e-6)
        assert result.action_probabilities.shape == (1, 5)

    def test_healthcare_triage_scoring(self):
        """Healthcare triage: 8 factors, 3 actions."""
        rng = _rng(3)
        W_health = rng.standard_normal((3, 8)) * 0.3
        actions_health = ["routine", "urgent", "emergency"]
        f_health = np.clip(rng.random((1, 8)), 0.0, 1.0)

        result = score_entity(f_health, W_health, actions_health, tau=0.25)

        assert result.selected_action in actions_health
        np.testing.assert_allclose(result.action_probabilities.sum(), 1.0, atol=1e-6)
        assert result.factor_vector.shape == (1, 8)

    def test_logistics_routing_scoring(self):
        """Logistics route scoring: 5 factors, 4 actions."""
        rng = _rng(4)
        W_log = rng.standard_normal((4, 5)) * 0.3
        actions_log = ["standard", "express", "hold_for_batch", "reroute"]
        f_log = np.array([[0.6, 0.4, 0.9, 0.2, 0.7]])  # capacity, urgency, weather, cost, reliability

        result = score_entity(f_log, W_log, actions_log, tau=0.3)

        assert result.selected_action in actions_log
        assert result.raw_scores.shape == (1, 4)
        assert result.temperature == 0.3

    def test_factor_vector_r4_preserved_after_scoring(self):
        """Requirement R4: factor_vector in result must equal the input f."""
        rng = _rng(5)
        W = rng.standard_normal((3, 6)) * 0.3
        f = np.clip(rng.random((1, 6)), 0.0, 1.0)
        f_original = f.copy()

        result = score_entity(f, W, ["a", "b", "c"], tau=0.25)
        np.testing.assert_array_equal(result.factor_vector, f_original)


# ===========================================================================
# GROUP 2 — Learning Across Domains
# ===========================================================================

class TestLearningAcrossDomains:

    def test_independent_states_diverge_after_training(self):
        """SOC and S2P states trained on domain-specific outcomes diverge."""
        soc_names = ["travel", "asset", "threat", "time", "device", "pattern"]
        s2p_names = ["reliability", "price", "geo", "delivery", "compliance", "volume"]

        state_soc = _make_state(4, 6, soc_names, seed=10)
        state_s2p = _make_state(3, 6, s2p_names, seed=10)  # same W seed

        f_soc = np.array([[0.9, 0.3, 0.8, 0.2, 0.7, 0.6]])
        f_s2p = np.array([[0.1, 0.9, 0.2, 0.8, 0.3, 0.4]])

        # Train each domain independently for 20 cycles
        rng = _rng(99)
        for _ in range(20):
            outcome_soc = +1 if rng.random() > 0.3 else -1
            state_soc.update(0, "action_0", outcome_soc, f_soc)
            outcome_s2p = +1 if rng.random() < 0.3 else -1
            state_s2p.update(0, "action_0", outcome_s2p, f_s2p)

        # Same initial W but different training → different final matrices
        # (Compare only the overlapping 3×6 portion)
        diff = np.linalg.norm(state_soc.W[:3, :] - state_s2p.W)
        assert diff > 0.1, f"States should diverge, diff={diff:.4f}"

    def test_asymmetry_ratio_holds_across_domain_sizes(self):
        """20:1 asymmetry holds for all (n_actions, n_factors) combinations."""
        for n_actions in [2, 3, 4, 5, 8]:
            for n_factors in [3, 6, 10, 20]:
                rng = _rng(n_actions * 100 + n_factors)
                W = rng.standard_normal((n_actions, n_factors)) * 0.3
                f = np.clip(rng.random((1, n_factors)), 0.05, 0.95)

                # Two separate states, identical initial W
                state_pos = LearningState(W=W.copy(), n_actions=n_actions,
                                          n_factors=n_factors,
                                          factor_names=[f"f{i}" for i in range(n_factors)],
                                          profile=CalibrationProfile())
                state_neg = LearningState(W=W.copy(), n_actions=n_actions,
                                          n_factors=n_factors,
                                          factor_names=[f"f{i}" for i in range(n_factors)],
                                          profile=CalibrationProfile())

                u_correct = state_pos.update(0, "action_0", +1, f)
                u_incorrect = state_neg.update(0, "action_0", -1, f)

                ratio = (np.linalg.norm(u_incorrect.delta_applied) /
                         np.linalg.norm(u_correct.delta_applied))
                assert 15.0 < ratio < 25.0, (
                    f"Asymmetry broken at {n_actions}×{n_factors}: ratio={ratio:.2f}"
                )

    def test_domains_use_independent_weight_matrices(self):
        """Mutating one domain's W does not affect another."""
        state_a = _make_state(4, 6, seed=20)
        state_b = _make_state(4, 6, seed=20)  # same initial W

        W_b_before = state_b.W.copy()
        f = _make_f(6)

        # Train only state_a
        for _ in range(10):
            state_a.update(0, "action_0", +1, f)

        # state_b must be unchanged
        np.testing.assert_array_equal(state_b.W, W_b_before)

    def test_factor_names_are_domain_labels_only(self):
        """factor_names are arbitrary strings — engine treats them as opaque keys."""
        exotic_names = ["μ_latency", "carbon_intensity", "élan", "信頼度", "Δprice"]
        state = _make_state(2, 5, exotic_names, seed=30)
        assert state.factor_names == exotic_names

        f = _make_f(5)
        u = state.update(0, "act_0", +1, f)
        assert u is not None
        assert u.factor_vector.shape == (1, 5)


# ===========================================================================
# GROUP 3 — Convergence Scenarios
# ===========================================================================

class TestConvergenceScenarios:

    def test_100_all_correct_converges(self):
        """Steady stream of correct outcomes → W stabilises → converged=True.

        Uses epsilon_vector=0.05 (A2 fast-decay mode) so W reaches its
        analytical steady state (W_ss = ALPHA*f/epsilon = 0.4*f << W_CLAMP)
        within ~60 steps and the norm is flat by step 100 (stability << 0.05).
        """
        state = LearningState(
            W=np.zeros((3, 4)),
            n_actions=3,
            n_factors=4,
            factor_names=[f"factor_{i}" for i in range(4)],
            profile=CalibrationProfile(),
            epsilon_vector=np.full(4, 0.05),
        )
        f = np.array([[0.6, 0.8, 0.4, 0.7]])

        for _ in range(100):
            state.update(0, "action_0", +1, f)

        m = get_convergence_metrics(state)
        assert m["accuracy"] == pytest.approx(1.0), "all-correct → accuracy must be 1.0"
        assert m["stability"] < 0.05, (
            f"W should stabilise after 100 correct updates, stability={m['stability']:.4f}"
        )
        assert m["converged"] is True

    def test_100_all_incorrect_w_stays_bounded(self):
        """All-incorrect stream drives W toward −W_CLAMP; clamping prevents explosion."""
        state = _make_state(3, 4, seed=51)
        f = np.array([[0.6, 0.8, 0.4, 0.7]])

        for _ in range(100):
            state.update(0, "action_0", -1, f)

        # Clamping must hold
        assert np.all(state.W >= -W_CLAMP - 1e-9), "W must not go below -W_CLAMP"
        assert np.all(state.W <= W_CLAMP + 1e-9), "W must not exceed +W_CLAMP"
        # All outcomes were incorrect → accuracy = 0
        m = get_convergence_metrics(state)
        assert m["accuracy"] == pytest.approx(0.0)
        assert m["converged"] is False

    def test_70_30_mixed_w_bounded_not_converged(self):
        """70% correct, 30% incorrect — 20:1 asymmetry means net-negative signal.

        The W norm is bounded (clamping works) and converged=False because
        accuracy 0.7 < ACCURACY_THRESHOLD 0.8.
        """
        state = _make_state(3, 4, seed=52)
        f = np.array([[0.6, 0.8, 0.4, 0.7]])
        rng = _rng(52)

        for _ in range(100):
            outcome = +1 if rng.random() < 0.70 else -1
            state.update(0, "action_0", outcome, f)

        assert np.all(np.abs(state.W) <= W_CLAMP + 1e-9), "W must stay bounded"
        m = get_convergence_metrics(state)
        # accuracy near 0.70 (last 20 updates) — below convergence threshold
        assert m["accuracy"] <= 0.8
        assert m["converged"] is False

    def test_alternating_outcomes_does_not_converge(self):
        """Alternating +1/−1 outcomes create oscillation — stability stays high."""
        state = _make_state(3, 4, seed=53)
        f = np.array([[0.6, 0.8, 0.4, 0.7]])

        for i in range(100):
            outcome = +1 if i % 2 == 0 else -1
            state.update(0, "action_0", outcome, f)

        m = get_convergence_metrics(state)
        assert m["converged"] is False, (
            "Alternating outcomes should not satisfy convergence criterion"
        )
        # accuracy over last 20 alternating = 0.5 < 0.8
        assert m["accuracy"] == pytest.approx(0.5)

    def test_convergence_metrics_complete_after_long_training(self):
        """All metric keys present and numeric after 200 mixed-domain updates."""
        state = _make_state(4, 6, seed=54)
        f = _make_f(6)
        rng = _rng(54)

        for _ in range(200):
            outcome = +1 if rng.random() > 0.2 else -1
            idx = int(rng.integers(0, 4))
            state.update(idx, f"action_{idx}", outcome, f)

        m = get_convergence_metrics(state)
        for key in ("decisions", "weight_norm", "stability", "accuracy",
                    "converged", "provisional_dimensions", "pending_autonomous"):
            assert key in m, f"Missing key: {key}"
        assert m["decisions"] == 200


# ===========================================================================
# GROUP 4 — Edge Cases
# ===========================================================================

class TestEdgeCases:

    def test_single_factor_scoring(self):
        """n_factors=1 — scoring still produces valid distribution."""
        W = np.array([[0.5], [-0.3]])   # 2 actions, 1 factor
        f = np.array([[0.8]])
        result = score_entity(f, W, ["yes", "no"], tau=0.25)
        assert result.selected_action in ["yes", "no"]
        np.testing.assert_allclose(result.action_probabilities.sum(), 1.0, atol=1e-6)

    def test_single_action_always_selected_probability_one(self):
        """n_actions=1 — selected action has probability 1.0."""
        W = np.array([[0.3, -0.1, 0.5]])   # 1 action, 3 factors
        f = np.array([[0.7, 0.2, 0.9]])
        result = score_entity(f, W, ["only_option"], tau=0.25)
        assert result.selected_action == "only_option"
        assert result.confidence == pytest.approx(1.0)
        np.testing.assert_allclose(result.action_probabilities, [[1.0]], atol=1e-9)

    def test_large_scale_20_factors_10_actions(self):
        """20 factors × 10 actions — scoring works at larger scale."""
        rng = _rng(60)
        W = rng.standard_normal((10, 20)) * 0.3
        f = np.clip(rng.random((1, 20)), 0.0, 1.0)
        actions = [f"action_{i}" for i in range(10)]

        result = score_entity(f, W, actions, tau=0.25)

        assert result.action_probabilities.shape == (1, 10)
        assert result.selected_action in actions
        np.testing.assert_allclose(result.action_probabilities.sum(), 1.0, atol=1e-6)

    def test_zero_factor_vector_no_nan(self):
        """f=zeros — softmax of uniform logits → uniform distribution, no NaN."""
        rng = _rng(61)
        W = rng.standard_normal((3, 4)) * 0.3
        f = np.zeros((1, 4))

        result = score_entity(f, W, ["a", "b", "c"], tau=0.25)

        assert not np.any(np.isnan(result.action_probabilities)), "NaN in zero-f output"
        np.testing.assert_allclose(result.action_probabilities.sum(), 1.0, atol=1e-6)

    def test_all_ones_factor_vector_valid_probabilities(self):
        """f=ones — all probabilities valid, sum to 1."""
        rng = _rng(62)
        W = rng.standard_normal((3, 4)) * 0.3
        f = np.ones((1, 4))

        result = score_entity(f, W, ["a", "b", "c"], tau=0.25)

        assert np.all(result.action_probabilities > 0)
        np.testing.assert_allclose(result.action_probabilities.sum(), 1.0, atol=1e-6)

    def test_zero_weight_matrix_uniform_distribution(self):
        """W=0 → all raw scores equal → uniform distribution."""
        W = np.zeros((4, 5))
        f = _make_f(5)
        actions = ["a", "b", "c", "d"]

        result = score_entity(f, W, actions, tau=0.25)

        expected = np.full((1, 4), 0.25)
        np.testing.assert_allclose(result.action_probabilities, expected, atol=1e-9)

    def test_low_temperature_sharpens_distribution(self):
        """tau=0.01 → near-argmax, one probability dominates."""
        rng = _rng(63)
        W = rng.standard_normal((3, 4)) * 1.0  # larger weights for clearer winner
        f = _make_f(4)

        result_sharp = score_entity(f, W, ["a", "b", "c"], tau=0.01)
        result_flat = score_entity(f, W, ["a", "b", "c"], tau=10.0)

        # sharp: max prob >> flat: max prob
        assert result_sharp.confidence > result_flat.confidence
        # flat: all probs should be close to uniform (1/3)
        np.testing.assert_allclose(result_flat.action_probabilities,
                                   np.full((1, 3), 1/3), atol=0.05)

    def test_expand_weight_matrix_5_times_shape_correct(self):
        """expand_weight_matrix called 5× → shape correct, original columns intact."""
        state = _make_state(3, 4, seed=70)
        W_orig = state.W.copy()

        for i in range(5):
            state.expand_weight_matrix(f"discovered_dim_{i}")

        assert state.W.shape == (3, 9)
        assert state.n_factors == 9
        assert len(state.factor_names) == 9
        assert state.epsilon_vector.shape == (9,)
        # Original first 4 columns must be unchanged
        np.testing.assert_array_equal(state.W[:, :4], W_orig)

    def test_expand_then_score_works(self):
        """After expansion, score_entity accepts the new (wider) f."""
        state = _make_state(3, 4, seed=71)
        state.expand_weight_matrix("extra_dim")

        f_new = np.clip(_rng(71).random((1, 5)), 0.0, 1.0)
        result = score_entity(f_new, state.W, ["a", "b", "c"], tau=0.25)

        assert result.selected_action in ["a", "b", "c"]

    def test_update_after_expand_preserves_shapes(self):
        """update() after expand_weight_matrix keeps W shape consistent."""
        state = _make_state(2, 3, seed=72)
        state.expand_weight_matrix("new_factor")

        f = np.clip(_rng(72).random((1, 4)), 0.0, 1.0)
        u = state.update(0, "action_0", +1, f)

        assert state.W.shape == (2, 4)
        assert u.W_before.shape == (2, 4)
        assert u.W_after.shape == (2, 4)


# ===========================================================================
# GROUP 5 — Persistence Round-Trip
# ===========================================================================

class TestPersistenceRoundTrip:

    def test_save_load_w_matches_exactly(self, tmp_path):
        """W values survive JSON round-trip without floating-point loss."""
        state = _make_state(4, 6, seed=80)
        f = _make_f(6)
        for _ in range(10):
            state.update(0, "action_0", +1, f)

        path = tmp_path / "state_g.json"
        _save_learning_state(state, path)
        loaded = _load_learning_state(path)

        np.testing.assert_array_almost_equal(loaded.W, state.W, decimal=12)

    def test_save_load_factor_names_preserved(self, tmp_path):
        """factor_names survive round-trip exactly."""
        names = ["carbon_intensity", "latency_score", "vendor_tier",
                 "regulatory_fit", "cost_index", "delivery_reliability"]
        state = _make_state(3, 6, names, seed=81)
        path = tmp_path / "state_names.json"
        _save_learning_state(state, path)
        loaded = _load_learning_state(path)

        assert loaded.factor_names == names

    def test_save_load_decision_count_preserved(self, tmp_path):
        """decision_count is restored correctly."""
        state = _make_state(3, 4, seed=82)
        f = _make_f(4)
        for _ in range(7):
            state.update(0, "action_0", +1, f)

        path = tmp_path / "state_count.json"
        _save_learning_state(state, path)
        loaded = _load_learning_state(path)

        assert loaded.decision_count == 7

    def test_save_load_hardening_defaults_present(self, tmp_path):
        """discount_strength and epsilon_vector survive round-trip."""
        state = _make_state(3, 4, seed=83)
        path = tmp_path / "state_hard.json"
        _save_learning_state(state, path)
        loaded = _load_learning_state(path)

        assert loaded.discount_strength == pytest.approx(0.0)
        np.testing.assert_allclose(loaded.epsilon_vector,
                                   np.full(4, EPSILON), atol=1e-12)

    def test_continue_training_after_load(self, tmp_path):
        """Training after a round-trip produces valid WeightUpdate records."""
        state = _make_state(3, 4, seed=84)
        f = _make_f(4)
        for _ in range(10):
            state.update(0, "action_0", +1, f)

        path = tmp_path / "state_resume.json"
        _save_learning_state(state, path)
        loaded = _load_learning_state(path)

        # Should not raise; should return a valid WeightUpdate
        u = loaded.update(0, "action_0", +1, f)
        assert u is not None
        assert u.W_after.shape == (3, 4)
        assert loaded.decision_count == 11

    def test_round_trip_n_actions_n_factors_preserved(self, tmp_path):
        """n_actions and n_factors survive round-trip."""
        state = _make_state(5, 8, seed=85)
        path = tmp_path / "state_shape.json"
        _save_learning_state(state, path)
        loaded = _load_learning_state(path)

        assert loaded.n_actions == 5
        assert loaded.n_factors == 8
        assert loaded.W.shape == (5, 8)


# ===========================================================================
# GROUP 6 — FactorComputer Protocol (arbitrary domains)
# ===========================================================================

class TestFactorComputerProtocol:
    """All mock computers use non-SOC domain vocabulary."""

    def test_supplier_reliability_factor_satisfies_protocol(self):
        class SupplierReliabilityFactor:
            async def compute(self, entity_id: str, context: Any = None) -> float:
                return 0.85

        assert isinstance(SupplierReliabilityFactor(), FactorComputer)

    def test_transaction_size_factor_satisfies_protocol(self):
        class TransactionSizeFactor:
            async def compute(self, entity_id: str, context: Any = None) -> float:
                return 0.60

        assert isinstance(TransactionSizeFactor(), FactorComputer)

    def test_patient_acuity_factor_satisfies_protocol(self):
        class PatientAcuityFactor:
            async def compute(self, entity_id: str, context: Any = None) -> float:
                return 0.95

        assert isinstance(PatientAcuityFactor(), FactorComputer)

    def test_all_three_mock_computers_are_factor_computers(self):
        """Bulk isinstance check across three unrelated domain computers."""
        class C1:
            async def compute(self, entity_id: str, context: Any = None) -> float:
                return 0.1

        class C2:
            async def compute(self, entity_id: str, context: Any = None) -> float:
                return 0.2

        class C3:
            async def compute(self, entity_id: str, context: Any = None) -> float:
                return 0.3

        for cls in (C1, C2, C3):
            assert isinstance(cls(), FactorComputer), f"{cls.__name__} failed Protocol check"

    def test_non_computer_fails_protocol_check(self):
        """Objects without compute() do not satisfy FactorComputer."""
        class NotAComputer:
            def score(self, x: float) -> float:
                return x

        assert not isinstance(NotAComputer(), FactorComputer)

    def test_assemble_factor_vector_with_supply_chain_schema(self):
        """assemble_factor_vector works with a supply-chain SchemaContract."""
        schema = SchemaContract(
            node_type="purchase_order",
            properties=(
                PropertySpec("supplier_reliability"),
                PropertySpec("transaction_volume"),
                PropertySpec("geo_risk"),
            ),
        )
        raw = {"supplier_reliability": 0.8, "transaction_volume": 0.3, "geo_risk": 0.9}

        v = assemble_factor_vector(raw, schema)

        assert v.shape == (3,)
        assert v[0] == pytest.approx(0.8)
        assert v[1] == pytest.approx(0.3)
        assert v[2] == pytest.approx(0.9)
        assert v.dtype == np.float64

    def test_assemble_factor_vector_with_healthcare_schema(self):
        """assemble_factor_vector works with healthcare property names."""
        schema = SchemaContract(
            node_type="patient_encounter",
            properties=(
                PropertySpec("acuity_score"),
                PropertySpec("wait_time_normalized"),
                PropertySpec("resource_availability"),
                PropertySpec("comorbidity_index"),
            ),
        )
        raw = {
            "acuity_score": 0.9,
            "wait_time_normalized": 0.6,
            "resource_availability": 0.4,
            "comorbidity_index": 0.7,
        }

        v = assemble_factor_vector(raw, schema)

        assert v.shape == (4,)
        np.testing.assert_allclose(v, [0.9, 0.6, 0.4, 0.7], atol=1e-9)

    def test_assemble_factor_vector_optional_default_domain_agnostic(self):
        """Optional factors with domain-specific names fall back to defaults."""
        schema = SchemaContract(
            node_type="logistics_shipment",
            properties=(
                PropertySpec("carrier_score"),
                PropertySpec("weather_risk", required=False, default_value=0.5),
            ),
        )
        raw = {"carrier_score": 0.7}   # weather_risk absent

        v = assemble_factor_vector(raw, schema)

        assert v[0] == pytest.approx(0.7)
        assert v[1] == pytest.approx(0.5)   # default applied

    def test_factor_vector_fed_directly_into_score_entity(self):
        """Full pipeline: Schema → assemble → reshape → score_entity."""
        schema = SchemaContract(
            node_type="financial_transaction",
            properties=(
                PropertySpec("regulatory_match"),
                PropertySpec("counterparty_risk"),
                PropertySpec("pattern_anomaly"),
            ),
        )
        raw = {"regulatory_match": 0.1, "counterparty_risk": 0.8, "pattern_anomaly": 0.4}

        v = assemble_factor_vector(raw, schema)           # shape (3,)
        f = v.reshape(1, -1)                              # → (1, 3) for score_entity

        rng = _rng(90)
        W = rng.standard_normal((3, 3)) * 0.3
        actions = ["approve", "hold", "reject"]

        result = score_entity(f, W, actions, tau=0.25)

        assert result.selected_action in actions
        np.testing.assert_allclose(result.action_probabilities.sum(), 1.0, atol=1e-6)
