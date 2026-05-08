"""
GAE evolution ledger tests — domain-agnostic variant lifecycle tracking.

Covers:
- Ledger CRUD: create variant, update shadow results, promote, reject
- Pattern origin: source_copilot / source_rule / warm_start_prior round-trip
- Lifecycle paths: created→shadow→promoted and created→shadow→rejected
- Empty ledger returns empty results
"""

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from gae.evolution import (
    ARTIFACT_CONTEXT_POLICY,
    ARTIFACT_ROUTING_RULE,
    ARTIFACT_SCORING_THRESHOLD,
    PROMOTION_APPROVED,
    PROMOTION_REJECTED,
    SHADOW_RESULT,
    SHADOW_STARTED,
    VALID_ARTIFACT_TYPES,
    VALID_EVENT_TYPES,
    VARIANT_CREATED,
    get_evolution_summary,
    get_recent_events,
    get_shadow_summary,
    get_variant_history,
    rebuild_shadow_index,
    record_evolution_event,
    reset_evolution_ledger,
)


def _run(coro):
    return asyncio.run(coro)


def _client(rows=None):
    mock = AsyncMock()
    mock.run_query = AsyncMock(return_value=rows or [])
    return mock


def _last_query(mock):
    return mock.run_query.call_args.args[0]


def _empty_summary():
    return {
        "variants_generated":    0,
        "variants_promoted":     0,
        "variants_rejected":     0,
        "variants_rolled_back":  0,
        "shadow_batches":        0,
        "shadow_started":        0,
        "by_artifact_type":      {},
        "avg_shadow_win_rate":   0.0,
        "total_shadow_decisions": 0,
    }


def setup_function():
    reset_evolution_ledger()


# ── Ledger CRUD ───────────────────────────────────────────────────────────────

def test_create_variant_writes_age_create_query():
    mock = _client()
    event = _run(record_evolution_event(
        mock,
        VARIANT_CREATED,
        "variant_a",
        ARTIFACT_ROUTING_RULE,
        "routing rule candidate for DataOps",
    ))
    query = _last_query(mock)
    assert "CREATE (e:EvolutionEvent" in query
    assert "variant_id" in query
    assert "artifact_type" in query
    assert "MERGE" not in query
    assert "$" not in query
    assert event["event_type"] == VARIANT_CREATED
    assert event["variant_id"] == "variant_a"


def test_shadow_result_updates_shadow_index():
    _run(record_evolution_event(
        _client(), SHADOW_STARTED, "v1", ARTIFACT_ROUTING_RULE, "started"
    ))
    _run(record_evolution_event(
        _client(), SHADOW_RESULT, "v1", ARTIFACT_ROUTING_RULE, "result",
        metadata={"wins": 7, "total": 10},
    ))
    summary = get_shadow_summary("v1")
    assert summary["shadow_tested"] is True
    assert summary["shadow_active"] is False
    assert summary["wins"]     == 7
    assert summary["total"]    == 10
    assert summary["win_rate"] == 0.7


def test_promote_variant_event_recorded():
    mock = _client()
    event = _run(record_evolution_event(
        mock, PROMOTION_APPROVED, "v_promote", ARTIFACT_ROUTING_RULE,
        "promoted after shadow win",
    ))
    assert event["event_type"] == PROMOTION_APPROVED
    assert "promotion_approved" in _last_query(mock)


def test_reject_variant_event_recorded():
    mock = _client()
    event = _run(record_evolution_event(
        mock, PROMOTION_REJECTED, "v_reject", ARTIFACT_SCORING_THRESHOLD,
        "rejected — win_rate below gate",
    ))
    assert event["event_type"] == PROMOTION_REJECTED
    assert "promotion_rejected" in _last_query(mock)


def test_all_valid_event_types_accepted():
    for et in VALID_EVENT_TYPES:
        mock = _client()
        event = _run(record_evolution_event(
            mock, et, f"variant_{et}", ARTIFACT_CONTEXT_POLICY, "accepted",
            metadata={"win": True} if et == SHADOW_RESULT else None,
        ))
        assert event["event_type"] == et


def test_all_valid_artifact_types_accepted():
    for at in VALID_ARTIFACT_TYPES:
        event = _run(record_evolution_event(
            _client(), VARIANT_CREATED, f"variant_{at}", at, "artifact ok",
        ))
        assert event["artifact_type"] == at


def test_invalid_event_type_raises():
    with pytest.raises(ValueError):
        _run(record_evolution_event(
            _client(), "pattern_learned", "v", ARTIFACT_ROUTING_RULE, "bad",
        ))


def test_invalid_artifact_type_raises():
    with pytest.raises(ValueError):
        _run(record_evolution_event(
            _client(), VARIANT_CREATED, "v", "centroid", "bad",
        ))


def test_empty_variant_id_raises():
    with pytest.raises(ValueError):
        _run(record_evolution_event(
            _client(), VARIANT_CREATED, "", ARTIFACT_ROUTING_RULE, "bad",
        ))


# ── Pattern origin: cross-copilot transfer ────────────────────────────────────

def test_source_copilot_stored_in_query_and_returned():
    mock = _client()
    event = _run(record_evolution_event(
        mock, VARIANT_CREATED, "v_xcopilot", ARTIFACT_ROUTING_RULE,
        "warm-start from SOC",
        source_copilot="soc",
        source_rule="routing_rule_v7",
        warm_start_prior={"win_rate": 0.83, "sample_size": 120},
    ))
    query = _last_query(mock)
    assert "soc" in query
    assert "routing_rule_v7" in query
    assert event["source_copilot"] == "soc"
    assert event["source_rule"]    == "routing_rule_v7"
    assert event["warm_start_prior"]["win_rate"] == 0.83


def test_warm_start_prior_stored_as_json():
    mock = _client()
    prior = {"win_rate": 0.75, "categories": ["cred", "malware"]}
    _run(record_evolution_event(
        mock, VARIANT_CREATED, "v_ws", ARTIFACT_CONTEXT_POLICY, "warm start",
        warm_start_prior=prior,
    ))
    query = _last_query(mock)
    assert json.dumps(prior, sort_keys=True, separators=(",", ":")) in query


def test_pattern_origin_round_trips_through_row_to_event():
    rows = [{
        "id":               "e1",
        "event_type":       VARIANT_CREATED,
        "variant_id":       "v_rt",
        "artifact_type":    ARTIFACT_ROUTING_RULE,
        "description":      "created",
        "source_copilot":   "purchasing",
        "source_rule":      "rule_p3",
        "warm_start_prior": '{"threshold":0.9}',
        "before_state": "{}", "after_state": "{}", "metadata": "{}",
        "graph_context": "{}", "timestamp_epoch": 1,
    }]
    events = _run(get_variant_history(_client(rows), "v_rt"))
    assert events[0]["source_copilot"]          == "purchasing"
    assert events[0]["source_rule"]             == "rule_p3"
    assert events[0]["warm_start_prior"]["threshold"] == 0.9


def test_cross_copilot_chain_soc_to_dataops():
    mock = _client()
    # SOC promotes a variant
    soc_event = _run(record_evolution_event(
        mock, PROMOTION_APPROVED, "v_soc_promoted", ARTIFACT_ROUTING_RULE,
        "SOC routing rule v7 promoted",
        source_copilot="soc",
        warm_start_prior={"win_rate": 0.88, "sample_size": 200},
    ))
    # DataOps creates a warm-started variant from SOC's result
    dataops_event = _run(record_evolution_event(
        mock, VARIANT_CREATED, "v_dataops_warmstart", ARTIFACT_ROUTING_RULE,
        "DataOps warm-start from SOC routing_rule",
        source_copilot="soc",
        source_rule="v_soc_promoted",
        warm_start_prior=soc_event["warm_start_prior"],
    ))
    assert dataops_event["source_rule"]             == "v_soc_promoted"
    assert dataops_event["warm_start_prior"]["win_rate"] == 0.88


def test_pattern_origin_fields_absent_when_omitted():
    event = _run(record_evolution_event(
        _client(), VARIANT_CREATED, "v_no_origin", ARTIFACT_ROUTING_RULE, "no origin",
    ))
    assert event["source_copilot"]   is None
    assert event["source_rule"]      is None
    assert event["warm_start_prior"] == {}


# ── Lifecycle: created → shadow → promoted ────────────────────────────────────

def test_lifecycle_created_shadow_promoted():
    variant_id = "v_lifecycle_pass"

    _run(record_evolution_event(
        _client(), VARIANT_CREATED, variant_id, ARTIFACT_ROUTING_RULE, "created",
    ))
    _run(record_evolution_event(
        _client(), SHADOW_STARTED, variant_id, ARTIFACT_ROUTING_RULE, "shadow started",
    ))
    assert get_shadow_summary(variant_id)["shadow_active"] is True

    _run(record_evolution_event(
        _client(), SHADOW_RESULT, variant_id, ARTIFACT_ROUTING_RULE, "shadow result",
        metadata={"wins": 8, "total": 10},
    ))
    assert get_shadow_summary(variant_id)["shadow_active"]  is False
    assert get_shadow_summary(variant_id)["shadow_tested"]  is True
    assert get_shadow_summary(variant_id)["win_rate"]       == 0.8

    promoted = _run(record_evolution_event(
        _client(), PROMOTION_APPROVED, variant_id, ARTIFACT_ROUTING_RULE, "promoted",
    ))
    assert promoted["event_type"] == PROMOTION_APPROVED


# ── Lifecycle: created → shadow → rejected ────────────────────────────────────

def test_lifecycle_created_shadow_rejected():
    variant_id = "v_lifecycle_fail"

    _run(record_evolution_event(
        _client(), VARIANT_CREATED, variant_id, ARTIFACT_SCORING_THRESHOLD, "created",
    ))
    _run(record_evolution_event(
        _client(), SHADOW_STARTED, variant_id, ARTIFACT_SCORING_THRESHOLD, "shadow",
    ))
    _run(record_evolution_event(
        _client(), SHADOW_RESULT, variant_id, ARTIFACT_SCORING_THRESHOLD, "result",
        metadata={"wins": 2, "total": 10},
    ))
    assert get_shadow_summary(variant_id)["win_rate"] == 0.2

    rejected = _run(record_evolution_event(
        _client(), PROMOTION_REJECTED, variant_id, ARTIFACT_SCORING_THRESHOLD, "rejected",
    ))
    assert rejected["event_type"] == PROMOTION_REJECTED


# ── Empty ledger ──────────────────────────────────────────────────────────────

def test_empty_ledger_get_shadow_summary_returns_none():
    assert get_shadow_summary("nonexistent") is None


def test_empty_ledger_get_evolution_summary_returns_zeros():
    summary = _run(get_evolution_summary(_client([])))
    assert summary == _empty_summary()


def test_empty_ledger_get_recent_events_returns_empty_list():
    events = _run(get_recent_events(_client([]), limit=10))
    assert events == []


def test_empty_ledger_get_variant_history_returns_empty_list():
    events = _run(get_variant_history(_client([]), "any_variant"))
    assert events == []


def test_empty_ledger_rebuild_shadow_index_returns_empty_dict():
    index = _run(rebuild_shadow_index(_client([])))
    assert index == {}


# ── No SOC-specific imports ───────────────────────────────────────────────────

def test_no_soc_imports_in_gae_evolution():
    import inspect
    import gae.evolution as evo_module
    source = inspect.getsource(evo_module)
    forbidden = ["from app.", "import app.", "triage", "SOCDomain", "alert_type"]
    assert not any(term in source for term in forbidden), (
        f"gae.evolution must not import SOC-specific code. Found: "
        + ", ".join(t for t in forbidden if t in source)
    )
