"""Tests for the optional FastMCP Judgment Memory server."""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import pytest

import gae.integrations.mcp as mcp_module
from gae.integrations.mcp import JudgmentMemoryService, TOOL_NAMES, create_server
from gae.integrations.mcp_cli import parse_args


FASTMCP_INSTALLED = importlib.util.find_spec("fastmcp") is not None


class FakeFastMCP:
    """Minimal decorator registry for dependency-independent tests."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.tools: dict[str, Callable[..., Any]] = {}

    def tool(
        self, name: str | None = None, description: str | None = None
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        def register(function: Callable[..., Any]) -> Callable[..., Any]:
            self.tools[name or function.__name__] = function
            return function

        return register


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> FakeFastMCP:
    monkeypatch.setattr(mcp_module, "FastMCP", FakeFastMCP)
    return create_server(
        categories=["retrieval", "generation", "routing"],
        actions=["use", "skip", "escalate"],
        factors=["relevance", "confidence", "recency"],
    )


def factor_values(seed: float = 0.6) -> dict[str, float]:
    """Return one complete factor mapping."""
    return {"relevance": seed, "confidence": 0.7, "recency": 0.4}


def test_create_server(server: FakeFastMCP) -> None:
    assert set(server.tools) == set(TOOL_NAMES)
    assert server.judgment_memory.scorer is not None  # type: ignore[attr-defined]


def test_score_tool(server: FakeFastMCP) -> None:
    result = server.rgi_score("retrieval", factor_values())  # type: ignore[attr-defined]
    assert result["action"] in {"use", "skip", "escalate"}
    assert set(result["probabilities"]) == {"use", "skip", "escalate"}
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["decision_id"] == "jm-1"


def test_verify_tool(server: FakeFastMCP) -> None:
    score = server.rgi_score("retrieval", factor_values(0.9))  # type: ignore[attr-defined]
    result = server.rgi_verify(score["decision_id"], True)  # type: ignore[attr-defined]
    assert result["centroid_delta"] > 0.0
    assert result["decision_count"] == 1
    assert result["conservation"] in {"RED", "AMBER", "GREEN"}


def test_conservation_tool(server: FakeFastMCP) -> None:
    result = server.rgi_conservation()  # type: ignore[attr-defined]
    assert {"status", "theta_min", "headroom", "passed", "verified", "correct"} <= result.keys()


def test_decisions_tool(server: FakeFastMCP) -> None:
    for index in range(5):
        server.rgi_score("retrieval", factor_values(0.3 + index * 0.1))  # type: ignore[attr-defined]
    decisions = server.rgi_decisions(limit=3)  # type: ignore[attr-defined]
    assert len(decisions) == 3
    assert [item["id"] for item in decisions] == ["jm-3", "jm-4", "jm-5"]


def test_accuracy_tool(server: FakeFastMCP) -> None:
    identifiers = []
    for index in range(10):
        result = server.rgi_score(  # type: ignore[attr-defined]
            ["retrieval", "generation", "routing"][index % 3],
            factor_values(0.35 + index * 0.04),
        )
        identifiers.append(result["decision_id"])
    for index, decision_id in enumerate(identifiers[:7]):
        correct = index < 5
        server.rgi_verify(  # type: ignore[attr-defined]
            decision_id,
            correct,
            None if correct else "skip",
        )
    accuracy = server.rgi_accuracy()  # type: ignore[attr-defined]
    assert accuracy["total_verified"] == 7
    assert accuracy["overall_accuracy"] == pytest.approx(5 / 7)
    assert set(accuracy["per_category"]) == {"retrieval", "generation", "routing"}


def test_explain_tool(server: FakeFastMCP) -> None:
    server.rgi_score("generation", factor_values())  # type: ignore[attr-defined]
    result = server.rgi_explain()  # type: ignore[attr-defined]
    assert {"action", "confidence", "confidence_tier", "dominant_factors", "factor_contributions", "rationale", "auto_approve"} <= result.keys()


def test_save_load_state(tmp_path: Path) -> None:
    path = tmp_path / "mcp_state.json"
    service = JudgmentMemoryService(
        ["retrieval", "generation"], ["use", "skip"], ["relevance", "confidence"], path
    )
    result = service.rgi_score("retrieval", {"relevance": 0.9, "confidence": 0.3})
    service.rgi_verify(result["decision_id"], True)
    expected = service.centroids
    restored = JudgmentMemoryService(
        ["retrieval", "generation"], ["use", "skip"], ["relevance", "confidence"], path
    )
    np.testing.assert_allclose(restored.centroids, expected)
    assert restored.rgi_accuracy()["total_verified"] == 1


def test_state_compat_with_langchain(tmp_path: Path) -> None:
    from gae.integrations.langchain import JudgmentMemoryCallback

    path = tmp_path / "shared_state.json"
    callback = JudgmentMemoryCallback(
        categories=["retrieval", "generation"],
        actions=["use", "skip"],
        factors=["relevance", "confidence"],
        state_path=path,
        factor_fn=lambda _input, _tool: [0.8, 0.3],
        tool_category_map={"search": "retrieval"},
        on_conservation_red="log",
    )
    callback.on_tool_start({"name": "search"}, "query")
    callback.verify(correct=True)
    service = JudgmentMemoryService(
        ["retrieval", "generation"], ["use", "skip"], ["relevance", "confidence"], path
    )
    np.testing.assert_allclose(service.centroids, callback.centroids)
    assert service.rgi_decisions()[0]["id"] == "jm-1"

    reverse_path = tmp_path / "mcp_to_langchain.json"
    mcp_service = JudgmentMemoryService(
        ["retrieval", "generation"],
        ["use", "skip"],
        ["relevance", "confidence"],
        reverse_path,
    )
    scored = mcp_service.rgi_score(
        "generation", {"relevance": 0.2, "confidence": 0.9}
    )
    mcp_service.rgi_verify(scored["decision_id"], True)
    restored_callback = JudgmentMemoryCallback(
        categories=["retrieval", "generation"],
        actions=["use", "skip"],
        factors=["relevance", "confidence"],
        state_path=reverse_path,
        on_conservation_red="log",
    )
    np.testing.assert_allclose(restored_callback.centroids, mcp_service.centroids)
    assert restored_callback.decisions == 1


def test_unknown_category(server: FakeFastMCP) -> None:
    with pytest.raises(ValueError, match="unknown category"):
        server.rgi_score("unknown", factor_values())  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    "factors",
    [
        {"relevance": 0.5, "confidence": 0.5},
        {"relevance": 0.5, "confidence": 0.5, "recency": 0.5, "extra": 0.5},
    ],
)
def test_unknown_factor(server: FakeFastMCP, factors: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="factors must match"):
        server.rgi_score("retrieval", factors)  # type: ignore[attr-defined]


def test_cli_args() -> None:
    config = parse_args(
        [
            "--categories", "retrieval,generation,routing",
            "--actions", "use,skip,escalate",
            "--factors", "relevance,confidence,recency",
            "--state", "state.json",
            "--penalty-ratio", "7.0",
            "--http",
            "--port", "8080",
        ]
    )
    assert config.categories == ["retrieval", "generation", "routing"]
    assert config.penalty_ratio == 7.0
    assert config.transport == "http"
    assert config.port == 8080


def test_config_file(tmp_path: Path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "rgi.yaml"
    path.write_text(
        "categories: [retrieval, generation]\n"
        "actions: [use, skip]\n"
        "factors: [relevance, confidence]\n"
        "state_path: ./jm.json\n"
        "penalty_ratio: 6.0\n"
        "q_window: 300\n"
        "tau: 0.2\n",
        encoding="utf-8",
    )
    config = parse_args(["--config", str(path)])
    assert config.actions == ["use", "skip"]
    assert config.state_path == "./jm.json"
    assert config.q_window == 300
    assert config.tau == 0.2


@pytest.mark.skipif(not FASTMCP_INSTALLED, reason="FastMCP is optional")
def test_real_mcp_server() -> None:
    from fastmcp import FastMCP as RealFastMCP

    server = create_server(["a"], ["x"], ["f"])
    assert isinstance(server, RealFastMCP)


@pytest.mark.skipif(not FASTMCP_INSTALLED, reason="FastMCP is optional")
def test_tool_schemas() -> None:
    server = create_server(["a"], ["x", "y"], ["f1", "f2"])
    tools = asyncio.run(server.list_tools())
    assert {tool.name for tool in tools} == set(TOOL_NAMES)
    for tool in tools:
        assert tool.parameters["type"] == "object"
        assert len(tool.description or "") < 100

