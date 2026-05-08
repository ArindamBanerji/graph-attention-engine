"""
Domain-agnostic evolution ledger for AgentEvolver variant lifecycle tracking.

Tracks variants through: created → shadow → promoted/rejected.
Used by SOC, DataOps, Purchasing, and any future copilot built on GAE.

Storage backend: AGE (PostgreSQL Cypher) via the caller-supplied neo4j_client.
The in-memory _SHADOW_INDEX is a read-optimised projection for hot-path queries.

Pattern origin fields (source_copilot, source_rule, warm_start_prior) enable
cross-copilot learning transfer — a variant promoted in SOC can warm-start a
DataOps variant with its shadow-tested context.
"""

from __future__ import annotations

import ast
import json
import math
import time
import uuid
from copy import deepcopy
from typing import Any, Optional


# ── AGE Cypher serialiser (inline — no app.graph_schema dependency) ──────────

def _S(val: Any) -> str:
    """Serialise a Python value for inline use in AGE Cypher.

    Handles: None, bool, int, float, str, list, tuple, numpy arrays.
    Lists are stored as JSON strings (AGE has no array property type).
    Strings are single-quoted with proper escaping.
    Raises ValueError on NaN/Inf (AGE cannot store these).
    """
    if val is None:
        return "null"
    if isinstance(val, bool):
        return "true" if val else "false"
    if isinstance(val, (int, float)):
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            raise ValueError(f"AGE cannot store NaN/Inf: {val!r}")
        return str(val)
    if isinstance(val, (list, tuple)):
        return "'" + json.dumps(val).replace("'", "\\'") + "'"
    if hasattr(val, "tolist"):  # numpy array
        return "'" + json.dumps(val.tolist()).replace("'", "\\'") + "'"
    s = str(val).replace("\\", "\\\\").replace("'", "\\'")
    return "'" + s + "'"


# ── Event type constants ──────────────────────────────────────────────────────

VARIANT_CREATED    = "variant_created"
SHADOW_STARTED     = "shadow_started"
SHADOW_RESULT      = "shadow_result"
PROMOTION_APPROVED = "promotion_approved"
PROMOTION_REJECTED = "promotion_rejected"
ROLLBACK           = "rollback"

VALID_EVENT_TYPES = {
    VARIANT_CREATED,
    SHADOW_STARTED,
    SHADOW_RESULT,
    PROMOTION_APPROVED,
    PROMOTION_REJECTED,
    ROLLBACK,
}

# ── Artifact type constants ───────────────────────────────────────────────────

ARTIFACT_ROUTING_RULE      = "routing_rule"
ARTIFACT_CONTEXT_POLICY    = "context_policy"
ARTIFACT_EVIDENCE_ORDER    = "evidence_order"
ARTIFACT_SCORING_THRESHOLD = "scoring_threshold"
ARTIFACT_PROMPT_MODULE     = "prompt_module"

VALID_ARTIFACT_TYPES = {
    ARTIFACT_ROUTING_RULE,
    ARTIFACT_CONTEXT_POLICY,
    ARTIFACT_EVIDENCE_ORDER,
    ARTIFACT_SCORING_THRESHOLD,
    ARTIFACT_PROMPT_MODULE,
}

# ── In-memory shadow index (read-optimised projection) ───────────────────────

_SHADOW_INDEX: dict[str, dict[str, Any]] = {}


# ── Private helpers ───────────────────────────────────────────────────────────

def _json_state(value: Any) -> str:
    if value is None:
        return "{}"
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _parse_json_state(value: Any) -> Any:
    if value is None or value == "":
        return {}
    if isinstance(value, (dict, list, int, float, bool)):
        return deepcopy(value)
    if not isinstance(value, str):
        return {}
    try:
        return json.loads(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, (dict, list, int, float, bool, str)):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return {}


def _event_id() -> str:
    return "evo_" + uuid.uuid4().hex


def _coerce_win_total(
    after_state: dict | None,
    metadata: dict | None,
    graph_context: dict | None,
) -> tuple[int, int]:
    sources = [metadata or {}, after_state or {}, graph_context or {}]
    for source in sources:
        if "wins" in source and "total" in source:
            try:
                wins  = max(int(source.get("wins")  or 0), 0)
                total = max(int(source.get("total") or 0), 0)
                return (min(wins, total), total)
            except (TypeError, ValueError):
                continue

    for source in sources:
        if "win" in source:
            return (1, 1) if bool(source.get("win")) else (0, 1)
        if "won" in source:
            return (1, 1) if bool(source.get("won")) else (0, 1)
        if "winner" in source:
            winner = str(source.get("winner") or "").lower()
            return (1, 1) if winner in {"variant", "candidate", "new"} else (0, 1)

    for source in sources:
        if "sample_size" in source:
            try:
                total     = max(int(source.get("sample_size") or 0), 0)
                variant   = float(source.get("variant_accuracy",  0.0) or 0.0)
                baseline  = float(source.get("baseline_accuracy", 0.0) or 0.0)
                improvement = float(source.get("improvement_pp",  0.0) or 0.0)
                return (total if variant > baseline or improvement > 0 else 0, total)
            except (TypeError, ValueError):
                continue

    return 0, 0


def _default_shadow_summary() -> dict[str, Any]:
    return {
        "shadow_active": False,
        "shadow_tested": False,
        "wins":     0,
        "total":    0,
        "win_rate": 0.0,
    }


def _apply_shadow_started(variant_id: str) -> None:
    entry = _SHADOW_INDEX.setdefault(variant_id, _default_shadow_summary())
    entry["shadow_active"] = True


def _apply_shadow_result(variant_id: str, wins: int, total: int) -> None:
    entry = _SHADOW_INDEX.setdefault(variant_id, _default_shadow_summary())
    entry["shadow_active"]  = False
    entry["shadow_tested"]  = True
    entry["wins"]  = int(entry.get("wins",  0)) + max(int(wins),  0)
    entry["total"] = int(entry.get("total", 0)) + max(int(total), 0)
    entry["win_rate"] = (
        round(entry["wins"] / entry["total"], 4) if entry["total"] else 0.0
    )


def _string_literal(value: str | None) -> str:
    return "null" if value is None else _S(value)


def _safe_float(value: Any, field_name: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be numeric") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field_name} must be finite")
    return numeric


def _event_type_filter(alias: str = "e") -> str:
    return "(" + " OR ".join(
        f"{alias}.event_type = {_S(event_type)}"
        for event_type in sorted(VALID_EVENT_TYPES)
    ) + ")"


def _shadow_event_filter(alias: str = "e") -> str:
    return (
        f"({alias}.event_type = {_S(SHADOW_STARTED)} "
        f"OR {alias}.event_type = {_S(SHADOW_RESULT)})"
    )


def _row_value(row: dict[str, Any], key: str, default: Any = None) -> Any:
    if key in row:
        return row.get(key)
    event = row.get("e")
    if isinstance(event, dict) and key in event:
        return event.get(key)
    return default


def _row_to_event(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id":             _row_value(row, "id"),
        "event_type":     _row_value(row, "event_type"),
        "variant_id":     _row_value(row, "variant_id"),
        "artifact_type":  _row_value(row, "artifact_type"),
        "triggered_by":   _row_value(row, "triggered_by"),
        "description":    _row_value(row, "description") or "",
        "before_state":   _parse_json_state(_row_value(row, "before_state")),
        "after_state":    _parse_json_state(_row_value(row, "after_state")),
        "graph_context":  _parse_json_state(_row_value(row, "graph_context")),
        "metadata":       _parse_json_state(_row_value(row, "metadata")),
        "impact":         _row_value(row, "impact") or "operational",
        "magnitude":      float(_row_value(row, "magnitude", 0.0) or 0.0),
        "timestamp":      _row_value(row, "timestamp"),
        "timestamp_epoch": int(_row_value(row, "timestamp_epoch", 0) or 0),
        # Pattern origin (cross-copilot transfer fields)
        "source_copilot":    _row_value(row, "source_copilot"),
        "source_rule":       _row_value(row, "source_rule"),
        "warm_start_prior":  _parse_json_state(_row_value(row, "warm_start_prior")),
    }


def _empty_evolution_summary() -> dict[str, Any]:
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


def _ensure_artifact_summary(
    summary: dict[str, Any], artifact_type: Any
) -> dict[str, Any] | None:
    if not artifact_type:
        return None
    artifact_key = str(artifact_type)
    by_artifact = summary["by_artifact_type"]
    if artifact_key not in by_artifact:
        by_artifact[artifact_key] = {
            "generated":     0,
            "promoted":      0,
            "rejected":      0,
            "promotion_rate": 0.0,
        }
    return by_artifact[artifact_key]


def _coerce_summary_float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _event_return_payload(
    *,
    event_id:        str,
    event_type:      str,
    variant_id:      str,
    artifact_type:   str,
    description:     str,
    before_state:    dict | None,
    after_state:     dict | None,
    triggered_by:    str | None,
    graph_context:   dict | None,
    metadata:        dict | None,
    impact:          str,
    magnitude:       float,
    timestamp:       str,
    timestamp_epoch: int,
    source_copilot:    str | None,
    source_rule:       str | None,
    warm_start_prior:  dict | None,
) -> dict[str, Any]:
    return {
        "id":             event_id,
        "event_type":     event_type,
        "variant_id":     variant_id,
        "artifact_type":  artifact_type,
        "description":    description,
        "before_state":   deepcopy(before_state  or {}),
        "after_state":    deepcopy(after_state   or {}),
        "triggered_by":   triggered_by,
        "graph_context":  deepcopy(graph_context or {}),
        "metadata":       deepcopy(metadata      or {}),
        "impact":         impact,
        "magnitude":      float(magnitude),
        "timestamp":      timestamp,
        "timestamp_epoch": timestamp_epoch,
        # Pattern origin fields
        "source_copilot":   source_copilot,
        "source_rule":      source_rule,
        "warm_start_prior": deepcopy(warm_start_prior or {}),
    }


# ── Public API ────────────────────────────────────────────────────────────────

async def record_evolution_event(
    neo4j_client,
    event_type:   str,
    variant_id:   str,
    artifact_type: str,
    description:  str,
    before_state:  dict | None = None,
    after_state:   dict | None = None,
    triggered_by:  str | None = None,
    graph_context: dict | None = None,
    metadata:      dict | None = None,
    impact:        str   = "operational",
    magnitude:     float = 0.0,
    timestamp_override: Optional[float] = None,
    # Cross-copilot pattern origin (additive — SOC callers omit these)
    source_copilot:   str | None  = None,
    source_rule:      str | None  = None,
    warm_start_prior: dict | None = None,
) -> dict[str, Any]:
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(f"Invalid evolution event_type: {event_type}")
    if artifact_type not in VALID_ARTIFACT_TYPES:
        raise ValueError(f"Invalid evolution artifact_type: {artifact_type}")
    if not variant_id or not str(variant_id).strip():
        raise ValueError("variant_id is required")

    event_id       = _event_id()
    ts_epoch       = timestamp_override if timestamp_override is not None else time.time() * 1000
    timestamp_epoch = int(ts_epoch)
    timestamp      = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(timestamp_epoch / 1000))

    before_json      = _json_state(before_state)
    after_json       = _json_state(after_state)
    graph_ctx_json   = _json_state(graph_context)
    metadata_json    = _json_state(metadata)
    warm_start_json  = _json_state(warm_start_prior)
    magnitude_value  = _safe_float(magnitude, "magnitude")

    query = (
        "CREATE (e:EvolutionEvent {"
        f"id: {_S(event_id)}, "
        f"event_type: {_S(event_type)}, "
        f"variant_id: {_S(variant_id)}, "
        f"artifact_type: {_S(artifact_type)}, "
        f"triggered_by: {_string_literal(triggered_by)}, "
        f"description: {_S(description)}, "
        f"before_state: {_S(before_json)}, "
        f"after_state: {_S(after_json)}, "
        f"graph_context: {_S(graph_ctx_json)}, "
        f"metadata: {_S(metadata_json)}, "
        f"impact: {_S(impact)}, "
        f"magnitude: {magnitude_value}, "
        f"timestamp: {_S(timestamp)}, "
        f"timestamp_epoch: {timestamp_epoch}, "
        f"source_copilot: {_string_literal(source_copilot)}, "
        f"source_rule: {_string_literal(source_rule)}, "
        f"warm_start_prior: {_S(warm_start_json)}"
        "}) RETURN e.id AS id"
    )
    await neo4j_client.run_query(query)

    if event_type == SHADOW_STARTED:
        _apply_shadow_started(variant_id)
    elif event_type == SHADOW_RESULT:
        wins, total = _coerce_win_total(after_state, metadata, graph_context)
        _apply_shadow_result(variant_id, wins, total)

    return _event_return_payload(
        event_id=event_id,
        event_type=event_type,
        variant_id=variant_id,
        artifact_type=artifact_type,
        description=description,
        before_state=before_state,
        after_state=after_state,
        triggered_by=triggered_by,
        graph_context=graph_context,
        metadata=metadata,
        impact=impact,
        magnitude=magnitude_value,
        timestamp=timestamp,
        timestamp_epoch=timestamp_epoch,
        source_copilot=source_copilot,
        source_rule=source_rule,
        warm_start_prior=warm_start_prior,
    )


async def rebuild_shadow_index(neo4j_client) -> dict[str, dict[str, Any]]:
    _SHADOW_INDEX.clear()
    rows = await neo4j_client.run_query(
        "MATCH (e:EvolutionEvent) "
        f"WHERE {_shadow_event_filter('e')} "
        "RETURN e.id AS id, e.event_type AS event_type, "
        "e.variant_id AS variant_id, e.after_state AS after_state, "
        "e.metadata AS metadata, e.graph_context AS graph_context, "
        "e.timestamp_epoch AS timestamp_epoch "
        "ORDER BY e.timestamp_epoch ASC"
    )
    for row in rows or []:
        event_type = _row_value(row, "event_type")
        variant_id = _row_value(row, "variant_id")
        if event_type not in {SHADOW_STARTED, SHADOW_RESULT}:
            continue
        if not variant_id:
            continue
        if event_type == SHADOW_STARTED:
            _apply_shadow_started(variant_id)
        else:
            after_state   = _parse_json_state(_row_value(row, "after_state"))
            metadata      = _parse_json_state(_row_value(row, "metadata"))
            graph_context = _parse_json_state(_row_value(row, "graph_context"))
            wins, total   = _coerce_win_total(after_state, metadata, graph_context)
            _apply_shadow_result(variant_id, wins, total)
    return deepcopy(_SHADOW_INDEX)


def get_shadow_summary(variant_id: str) -> dict[str, Any] | None:
    summary = _SHADOW_INDEX.get(variant_id)
    return deepcopy(summary) if summary is not None else None


async def get_variant_history(
    neo4j_client, variant_id: str
) -> list[dict[str, Any]]:
    if not variant_id or not str(variant_id).strip():
        raise ValueError("variant_id is required")
    rows = await neo4j_client.run_query(
        "MATCH (e:EvolutionEvent) "
        f"WHERE e.variant_id = {_S(variant_id)} AND {_event_type_filter('e')} "
        "RETURN e.id AS id, e.event_type AS event_type, "
        "e.variant_id AS variant_id, e.artifact_type AS artifact_type, "
        "e.triggered_by AS triggered_by, e.description AS description, "
        "e.before_state AS before_state, e.after_state AS after_state, "
        "e.graph_context AS graph_context, e.metadata AS metadata, "
        "e.impact AS impact, e.magnitude AS magnitude, "
        "e.timestamp AS timestamp, e.timestamp_epoch AS timestamp_epoch, "
        "e.source_copilot AS source_copilot, e.source_rule AS source_rule, "
        "e.warm_start_prior AS warm_start_prior "
        "ORDER BY e.timestamp_epoch ASC"
    )
    return [_row_to_event(row) for row in rows or []]


async def get_recent_events(
    neo4j_client, limit: int = 20
) -> list[dict[str, Any]]:
    safe_limit = min(max(int(limit or 20), 1), 100)
    rows = await neo4j_client.run_query(
        "MATCH (e:EvolutionEvent) "
        f"WHERE {_event_type_filter('e')} "
        "RETURN e.id AS id, e.event_type AS event_type, "
        "e.variant_id AS variant_id, e.artifact_type AS artifact_type, "
        "e.triggered_by AS triggered_by, e.description AS description, "
        "e.before_state AS before_state, e.after_state AS after_state, "
        "e.graph_context AS graph_context, e.metadata AS metadata, "
        "e.impact AS impact, e.magnitude AS magnitude, "
        "e.timestamp AS timestamp, e.timestamp_epoch AS timestamp_epoch, "
        "e.source_copilot AS source_copilot, e.source_rule AS source_rule, "
        "e.warm_start_prior AS warm_start_prior "
        "ORDER BY e.timestamp_epoch DESC "
        f"LIMIT {safe_limit}"
    )
    return [_row_to_event(row) for row in rows or []]


async def get_evolution_summary(neo4j_client) -> dict[str, Any]:
    """Return aggregate lifecycle statistics (approximate above 10 000 nodes)."""
    summary = _empty_evolution_summary()
    try:
        rows = await neo4j_client.run_query(
            "MATCH (e:EvolutionEvent) "
            f"WHERE {_event_type_filter('e')} "
            "RETURN e.event_type AS event_type, e.variant_id AS variant_id, "
            "e.artifact_type AS artifact_type, e.graph_context AS graph_context, "
            "e.metadata AS metadata "
            "LIMIT 10000"
        )
    except Exception:
        return summary

    shadow_win_rates: list[float] = []
    for row in rows or []:
        event_type    = _row_value(row, "event_type")
        artifact_type = _row_value(row, "artifact_type")

        if event_type == VARIANT_CREATED:
            summary["variants_generated"] += 1
            art = _ensure_artifact_summary(summary, artifact_type)
            if art is not None:
                art["generated"] += 1
        elif event_type == PROMOTION_APPROVED:
            summary["variants_promoted"] += 1
            art = _ensure_artifact_summary(summary, artifact_type)
            if art is not None:
                art["promoted"] += 1
        elif event_type == PROMOTION_REJECTED:
            summary["variants_rejected"] += 1
            art = _ensure_artifact_summary(summary, artifact_type)
            if art is not None:
                art["rejected"] += 1
        elif event_type == ROLLBACK:
            summary["variants_rolled_back"] += 1
        elif event_type == SHADOW_RESULT:
            summary["shadow_batches"] += 1
            graph_context = _parse_json_state(_row_value(row, "graph_context"))
            if isinstance(graph_context, dict):
                wr = _coerce_summary_float(graph_context.get("win_rate"))
                if wr is not None:
                    shadow_win_rates.append(wr)
                ss = _coerce_summary_float(graph_context.get("sample_size"))
                if ss is not None:
                    summary["total_shadow_decisions"] += max(int(ss), 0)
        elif event_type == SHADOW_STARTED:
            summary["shadow_started"] += 1

    for art in summary["by_artifact_type"].values():
        denom = art["promoted"] + art["rejected"]
        art["promotion_rate"] = round(art["promoted"] / denom, 4) if denom else 0.0

    if shadow_win_rates:
        summary["avg_shadow_win_rate"] = round(
            sum(shadow_win_rates) / len(shadow_win_rates), 4
        )
    return summary


def reset_evolution_ledger() -> None:
    _SHADOW_INDEX.clear()
