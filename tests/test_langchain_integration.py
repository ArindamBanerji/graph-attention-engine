"""Tests for the optional LangChain Judgment Memory callback."""

from __future__ import annotations

import builtins
import importlib.util
import logging
import sys
import threading
import types
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest


LANGCHAIN_CORE_INSTALLED = importlib.util.find_spec("langchain_core") is not None
LANGCHAIN_INSTALLED = importlib.util.find_spec("langchain") is not None

if not LANGCHAIN_CORE_INSTALLED:
    callbacks_module = types.ModuleType("langchain_core.callbacks")

    class MockBaseCallbackHandler:
        """Minimal callback base used when the optional dependency is absent."""

    setattr(callbacks_module, "BaseCallbackHandler", MockBaseCallbackHandler)
    core_module = types.ModuleType("langchain_core")
    setattr(core_module, "callbacks", callbacks_module)
    sys.modules["langchain_core"] = core_module
    sys.modules["langchain_core.callbacks"] = callbacks_module

from gae.integrations.langchain import JudgmentMemoryCallback


def make_callback(**kwargs: Any) -> JudgmentMemoryCallback:
    """Build a small callback fixture."""
    options: dict[str, Any] = {
        "categories": ["retrieval", "generation", "routing"],
        "actions": ["use", "skip", "escalate"],
        "factors": ["relevance", "confidence", "recency"],
        "on_conservation_red": "log",
    }
    options.update(kwargs)
    return JudgmentMemoryCallback(**options)


def score(callback: JudgmentMemoryCallback, name: str = "search") -> str:
    """Record one tool decision and return its identifier."""
    callback.on_tool_start({"name": name}, "find late suppliers")
    return f"jm-{callback.decisions}"


def test_callback_init() -> None:
    callback = make_callback()
    assert callback.scorer is not None
    assert callback.centroids.shape == (3, 3, 3)


def test_score_decision() -> None:
    callback = make_callback()
    score(callback)
    assert callback.decisions == 1
    assert callback.status()["top_category"] in callback.categories


def test_verify() -> None:
    callback = make_callback(factor_fn=lambda _input, _tool: [0.8, 0.4, 0.6])
    score(callback)
    before = callback.centroids
    update = callback.verify(correct=True)
    assert update.outcome == "applied"
    assert update.centroid_delta_norm > 0.0
    assert not np.array_equal(before, callback.centroids)


def test_verify_batch_saves_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    callback = make_callback(state_path=tmp_path / "state.json")
    first = score(callback, "search")
    second = score(callback, "write")
    calls = 0
    original_save = callback.save

    def counted_save(path: str | Path | None = None) -> Path:
        nonlocal calls
        calls += 1
        return original_save(path)

    monkeypatch.setattr(callback, "save", counted_save)
    updates = callback.verify_batch(
        [
            {"decision_id": first, "correct": True},
            {"decision_id": second, "correct": True},
        ]
    )
    assert len(updates) == 2
    assert calls == 1


def test_conservation_red_warn(caplog: pytest.LogCaptureFixture) -> None:
    callback = make_callback(on_conservation_red="warn")
    with warnings.catch_warnings(record=True) as caught, caplog.at_level(logging.WARNING):
        score(callback)
    assert any("conservation status is RED" in str(item.message) for item in caught)
    assert "conservation status is RED" in caplog.text


def test_conservation_red_raise() -> None:
    callback = make_callback(on_conservation_red="raise")
    with pytest.raises(RuntimeError, match="conservation status is RED"):
        score(callback)


def test_status() -> None:
    callback = make_callback()
    score(callback)
    callback.verify(correct=True)
    status = callback.status()
    assert {
        "decisions",
        "verified",
        "accuracy",
        "conservation",
        "iks",
        "top_category",
        "accuracy_by_category",
    } <= status.keys()
    assert callback.accuracy == 1.0


def test_judgment() -> None:
    callback = make_callback()
    score(callback)
    judgment = callback.judgment()
    assert judgment["action"] in callback.actions
    assert 0.0 <= judgment["confidence"] <= 1.0
    assert judgment["factor_contributions"]


def test_save_load(tmp_path: Path) -> None:
    state_path = tmp_path / "jm_state.json"
    callback = make_callback(
        state_path=state_path,
        factor_fn=lambda _input, _tool: [0.7, 0.4, 0.8],
    )
    score(callback)
    callback.verify(correct=True)
    expected = callback.centroids

    restored = make_callback(state_path=state_path)
    np.testing.assert_allclose(restored.centroids, expected)
    assert restored.decisions == 1
    assert restored.accuracy == 1.0


def test_thread_safety() -> None:
    callback = make_callback()
    threads = [
        threading.Thread(target=callback.on_tool_start, args=({"name": f"tool-{i}"}, "x"))
        for i in range(40)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert callback.decisions == 40


def test_factor_fn_custom() -> None:
    callback = make_callback(factor_fn=lambda input_str, _tool: [len(input_str) / 10, 0.2, 0.9])
    callback.on_tool_start({"name": "search"}, "hello")
    decision = callback._resolve_decision("last")
    assert decision["factors"] == [0.5, 0.2, 0.9]


def test_tool_category_map() -> None:
    callback = make_callback(tool_category_map={"search": "retrieval"})
    callback.on_tool_start({"name": "search"}, "hello")
    assert callback._resolve_decision("last")["category"] == "retrieval"


def test_optional_import_has_clear_error(monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = Path(__file__).parents[1] / "gae" / "integrations" / "langchain.py"
    spec = importlib.util.spec_from_file_location("_gae_langchain_missing_test", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    real_import = builtins.__import__

    def blocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "langchain_core.callbacks":
            raise ImportError("blocked for test")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    with pytest.raises(ImportError, match=r"graph-attention-engine\[langchain\]"):
        spec.loader.exec_module(module)


@pytest.mark.skipif(not LANGCHAIN_CORE_INSTALLED, reason="langchain-core is optional")
def test_real_callback_handler() -> None:
    from langchain_core.callbacks import BaseCallbackHandler

    assert isinstance(make_callback(), BaseCallbackHandler)


@pytest.mark.skipif(
    not (LANGCHAIN_CORE_INSTALLED and LANGCHAIN_INSTALLED),
    reason="full LangChain agent package is optional",
)
def test_with_fake_agent() -> None:
    from langchain.agents import AgentExecutor, create_react_agent
    from langchain_core.language_models.fake import FakeListLLM
    from langchain_core.prompts import PromptTemplate
    from langchain_core.tools import Tool

    callback = make_callback(tool_category_map={"lookup": "retrieval"})
    llm = FakeListLLM(
        responses=[
            "Thought: I should search.\nAction: lookup\nAction Input: suppliers",
            "Thought: I have the result.\nFinal Answer: supplier-a",
        ]
    )
    tools = [Tool(name="lookup", func=lambda query: f"found {query}", description="lookup")]
    prompt = PromptTemplate.from_template(
        "Answer with tools.\n{tools}\nTool names: {tool_names}\n"
        "Question: {input}\n{agent_scratchpad}"
    )
    executor = AgentExecutor(
        agent=create_react_agent(llm, tools, prompt),
        tools=tools,
        verbose=False,
    )
    result = executor.invoke({"input": "Find supplier"}, config={"callbacks": [callback]})
    assert result["output"] == "supplier-a"
    assert callback.decisions == 1

