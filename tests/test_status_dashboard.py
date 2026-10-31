"""Tests for the fault-tolerant RGI status dashboard."""

from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

_DASHBOARD_PATH = Path(__file__).parents[1] / "examples" / "status_dashboard.py"
_SPEC = importlib.util.spec_from_file_location("rgi_status_dashboard", _DASHBOARD_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load status dashboard from {_DASHBOARD_PATH}")
dashboard = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = dashboard
_SPEC.loader.exec_module(dashboard)


def test_core_engine_check() -> None:
    items = dashboard.check_core_engine()
    assert len(items) == 2
    assert all(item.status == "ok" for item in items)
    assert "ProfileScorer" in items[0].detail
    assert "conservation=" in items[0].detail


def test_integration_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_import(name: str) -> Any:
        if name.endswith("langchain"):
            return SimpleNamespace()
        if name.endswith("mcp"):
            return SimpleNamespace(FastMCP=object())
        raise ImportError(name)

    monkeypatch.setattr(dashboard.importlib, "import_module", fake_import)
    items = dashboard.check_integrations()
    assert [item.status for item in items] == ["ok", "ok"]
    assert [item.name for item in items] == ["LangChain bridge", "MCP server"]


def test_external_api_checks(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(method: str, url: str, **kwargs: Any) -> Any:
        status = 503 if "stlouisfed" in url else 200
        return SimpleNamespace(status_code=status)

    monkeypatch.setattr(dashboard.httpx, "request", fake_request)
    items = dashboard.check_external_apis()
    by_name = {item.name: item for item in items}
    assert by_name["SAP S/4HANA"].status == "partial"
    assert by_name["Celonis"].status == "partial"
    assert by_name["Weather (Open-Meteo)"].status == "ok"
    assert by_name["FRED"].status == "unavailable"


def test_json_output(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(
        dashboard,
        "collect_status",
        lambda copilots_only=False: [
            dashboard.StatusItem("CORE ENGINE", "GAE", "ok", "v0.7.25")
        ],
    )
    assert dashboard.main(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"][0]["name"] == "GAE"
    assert payload["summary"] == {"ok": 1, "partial": 0, "unavailable": 0}


def test_failed_checks_dont_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail() -> list[Any]:
        raise RuntimeError("offline")

    for name in (
        "check_core_engine",
        "check_integrations",
        "check_copilots",
        "check_database",
        "check_external_apis",
        "check_connectors",
        "check_fixtures",
        "check_mcp_ecosystem",
    ):
        monkeypatch.setattr(dashboard, name, fail)
    items = dashboard.collect_status()
    assert len(items) == 8
    assert all(item.status == "unavailable" for item in items)
    rendered = dashboard.render_text(items, color=False)
    assert "[--]" in rendered
    assert "offline" in rendered
