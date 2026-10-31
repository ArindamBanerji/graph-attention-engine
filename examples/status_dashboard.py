"""Print a fault-tolerant status report for the RGI demonstration stack."""

from __future__ import annotations

import argparse
import importlib
import json
import socket
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import httpx
import numpy as np

TIMEOUT_SECONDS = 3.0
REPO_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPO_ROOT.parent

GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"


@dataclass(frozen=True)
class StatusItem:
    """One status line in the human and JSON reports."""

    group: str
    name: str
    status: str
    detail: str


def _item(group: str, name: str, status: str, detail: str) -> StatusItem:
    """Build one normalized status item."""
    return StatusItem(group=group, name=name, status=status, detail=detail)


def check_core_engine() -> list[StatusItem]:
    """Exercise GAE import, ProfileScorer, conservation, and persistence."""
    items: list[StatusItem] = []
    try:
        import gae
        from gae import build_profile_scorer, conservation_status

        names = ["use", "skip"]
        centroids = {"test": {name: [0.5, 0.5] for name in names}}
        scorer = build_profile_scorer(["test"], names, centroids, 2)
        factors = np.array([0.6, 0.4], dtype=np.float64)
        assert factors.shape == (2,)
        result = scorer.score(factors, 0)
        conservation = conservation_status(
            50,
            40,
            50,
            penalty_ratio=5.0,
            categories_with_data=1,
            total_categories=1,
        )
        items.append(
            _item(
                "CORE ENGINE",
                "GAE",
                "ok",
                f"v{gae.__version__}; ProfileScorer={result.action_name}; conservation={conservation.status}",
            )
        )
    except Exception as exc:
        items.append(_item("CORE ENGINE", "GAE", "unavailable", str(exc)))

    try:
        from gae import load_state, save_state
        from gae.store import LearningState

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = LearningState(weights=np.array([0.2, 0.8]), step=2)
            save_state(state, path)
            restored = load_state(path)
            assert restored.step == 2
            np.testing.assert_allclose(restored.weights, state.weights)
        items.append(_item("CORE ENGINE", "State persistence", "ok", "save/load working"))
    except Exception as exc:
        items.append(_item("CORE ENGINE", "State persistence", "unavailable", str(exc)))
    return items


def _integration_status(module_name: str, display_name: str) -> StatusItem:
    """Detect an optional integration without allowing import failures to escape."""
    try:
        module = importlib.import_module(module_name)
        if display_name == "MCP server" and getattr(module, "FastMCP", None) is None:
            return _item("INTEGRATIONS", display_name, "unavailable", "FastMCP not installed")
        return _item("INTEGRATIONS", display_name, "ok", "available")
    except Exception as exc:
        return _item("INTEGRATIONS", display_name, "unavailable", str(exc))


def check_integrations() -> list[StatusItem]:
    """Check the LangChain and MCP optional bridges."""
    return [
        _integration_status("gae.integrations.langchain", "LangChain bridge"),
        _integration_status("gae.integrations.mcp", "MCP server"),
    ]


def _http_status(
    group: str,
    name: str,
    method: str,
    url: str,
    reachable_status: str = "ok",
) -> StatusItem:
    """Make one bounded HTTP availability request."""
    try:
        response = httpx.request(
            method,
            url,
            timeout=TIMEOUT_SECONDS,
            follow_redirects=True,
        )
        if response.status_code < 500:
            return _item(
                group,
                name,
                reachable_status,
                f"reachable (HTTP {response.status_code})",
            )
        return _item(group, name, "unavailable", f"HTTP {response.status_code}")
    except Exception as exc:
        return _item(group, name, "unavailable", f"not reachable: {exc}")


def check_copilots() -> list[StatusItem]:
    """Probe all four local copilot health endpoints concurrently."""
    endpoints = {
        "Trading (8010)": "http://127.0.0.1:8010/api/health",
        "Purchasing (8020)": "http://127.0.0.1:8020/api/health",
        "DataOps (8030)": "http://127.0.0.1:8030/api/health",
        "S2P (8002)": "http://127.0.0.1:8002/api/health",
    }
    return _parallel_http("COPILOT BACKENDS", endpoints, method="GET")


def check_database() -> list[StatusItem]:
    """Check whether PostgreSQL+AGE accepts a TCP connection."""
    try:
        with socket.create_connection(("127.0.0.1", 5433), timeout=TIMEOUT_SECONDS):
            pass
        return [_item("DATABASE", "PostgreSQL+AGE (5433)", "ok", "reachable")]
    except Exception as exc:
        return [_item("DATABASE", "PostgreSQL+AGE (5433)", "unavailable", f"not reachable: {exc}")]


def _parallel_http(
    group: str,
    endpoints: dict[str, str],
    partial_names: set[str] | None = None,
    method: str | None = None,
) -> list[StatusItem]:
    """Run independent three-second HTTP probes concurrently."""
    partial_names = partial_names or set()
    ordered_names = list(endpoints)
    results: dict[str, StatusItem] = {}
    with ThreadPoolExecutor(max_workers=len(endpoints)) as executor:
        futures = {
            executor.submit(
                _http_status,
                group,
                name,
                method or ("GET" if name == "Weather (Open-Meteo)" else "HEAD"),
                url,
                "partial" if name in partial_names else "ok",
            ): name
            for name, url in endpoints.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as exc:
                results[name] = _item(group, name, "unavailable", str(exc))
    return [results[name] for name in ordered_names]


def check_external_apis() -> list[StatusItem]:
    """Probe public data portals without credentials or mutating calls."""
    endpoints = {
        "SAP S/4HANA": "https://api.sap.com",
        "Celonis": "https://developer.celonis.com",
        "Weather (Open-Meteo)": "https://api.open-meteo.com/v1/forecast?latitude=0&longitude=0",
        "FRED": "https://api.stlouisfed.org",
    }
    return _parallel_http(
        "EXTERNAL DATA SOURCES",
        endpoints,
        partial_names={"SAP S/4HANA", "Celonis"},
    )


def _connector_status(name: str, module_candidates: Sequence[str], file_candidates: Sequence[Path]) -> StatusItem:
    """Detect an importable connector or an accessible sibling-repository implementation."""
    for module_name in module_candidates:
        try:
            importlib.import_module(module_name)
            return _item("CONNECTORS", name, "ok", f"importable: {module_name}")
        except Exception:
            continue
    existing = next((path for path in file_candidates if path.is_file()), None)
    if existing is not None:
        return _item("CONNECTORS", name, "partial", f"code present: {existing}")
    return _item("CONNECTORS", name, "unavailable", "not built")


def check_connectors() -> list[StatusItem]:
    """Detect QuickBooks, Toast POS, and Alpaca connector implementations."""
    purchasing = WORKSPACE_ROOT / "copilot-sdk" / "apps" / "purchasing" / "backend" / "app" / "connectors"
    trading = WORKSPACE_ROOT / "copilot-sdk" / "apps" / "trading" / "backend" / "app" / "connectors"
    return [
        _connector_status(
            "QuickBooks",
            ["app.connectors.qbo_connector", "apps.purchasing.backend.app.connectors.qbo_connector"],
            [purchasing / "qbo_connector.py"],
        ),
        _connector_status(
            "Toast POS",
            ["app.connectors.toast", "apps.purchasing.backend.app.connectors.toast"],
            [purchasing / "toast.py"],
        ),
        _connector_status(
            "Alpaca",
            ["app.connectors.alpaca", "apps.trading.backend.app.connectors.alpaca"],
            [trading / "alpaca.py", trading / "alpaca_connector.py"],
        ),
    ]


def check_fixtures() -> list[StatusItem]:
    """Count captured JSON fixtures in the local fixture directory."""
    fixture_dir = REPO_ROOT / "fixtures"
    files = sorted(fixture_dir.glob("*.json")) if fixture_dir.is_dir() else []
    if files:
        return [_item("FIXTURES", "Fixture files", "ok", f"{len(files)} JSON files")]
    return [_item("FIXTURES", "Fixture files", "unavailable", "no fixture files found")]


def check_mcp_ecosystem() -> list[StatusItem]:
    """Probe expected public listing pages for the MCP server."""
    endpoints = {
        "Official registry": "https://registry.modelcontextprotocol.io/servers/io.github.arindambanerji/rgi-judgment-memory",
        "Smithery": "https://smithery.ai/server/@arindambanerji/rgi-judgment-memory",
        "Glama": "https://glama.ai/mcp/servers/@arindambanerji/rgi-judgment-memory",
    }
    return _parallel_http("MCP ECOSYSTEM", endpoints)


def collect_status(copilots_only: bool = False) -> list[StatusItem]:
    """Collect every requested status check without propagating failures."""
    if copilots_only:
        return check_copilots()
    checks: list[Callable[[], list[StatusItem]]] = [
        check_core_engine,
        check_integrations,
        check_copilots,
        check_database,
        check_external_apis,
        check_connectors,
        check_fixtures,
        check_mcp_ecosystem,
    ]
    items: list[StatusItem] = []
    for check in checks:
        try:
            items.extend(check())
        except Exception as exc:
            items.append(_item("DIAGNOSTICS", check.__name__, "unavailable", str(exc)))
    return items


def _marker(item: StatusItem, color: bool = True) -> str:
    """Render one colored status marker."""
    marker, ansi = {
        "ok": ("[OK]", GREEN),
        "partial": ("[~~]", YELLOW),
        "unavailable": ("[--]", RED),
    }[item.status]
    return f"{ansi}{marker}{RESET}" if color else marker


def render_text(items: Sequence[StatusItem], color: bool = True) -> str:
    """Render the boxed terminal dashboard with ANSI status colors."""
    width = 62
    border = "+" + "=" * width + "+"
    lines = [border, f"|  {'RGI System Status':<{width - 2}}|", border]
    current_group = ""
    for item in items:
        if item.group != current_group:
            lines.extend([f"|{'':{width}}|", f"|  {item.group:<{width - 2}}|"])
            current_group = item.group
        visible = f"{_marker(item, color=False)} {item.name}: {item.detail}"
        clipped = visible[: width - 4]
        marker = _marker(item, color=color)
        colored = marker + clipped[4:]
        lines.append(f"|  {colored}{' ' * max(0, width - 2 - len(clipped))}|")
    lines.extend([f"|{'':{width}}|", border])
    return "\n".join(lines)


def render_json(items: Sequence[StatusItem]) -> str:
    """Render a stable machine-readable status payload."""
    return json.dumps(
        {
            "status": [asdict(item) for item in items],
            "summary": {
                key: sum(item.status == key for item in items)
                for key in ("ok", "partial", "unavailable")
            },
        },
        indent=2,
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the dashboard command-line parser."""
    parser = argparse.ArgumentParser(description="Check RGI system connections.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument(
        "--copilots-only",
        action="store_true",
        help="Check only local copilot health endpoints",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run checks and print either the terminal dashboard or JSON."""
    args = build_parser().parse_args(argv)
    items = collect_status(copilots_only=args.copilots_only)
    print(render_json(items) if args.json else render_text(items))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
