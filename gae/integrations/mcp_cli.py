"""Command-line entry point for the GAE Judgment Memory MCP server."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from gae.integrations.mcp import create_server


@dataclass(frozen=True)
class ServerConfig:
    """Validated CLI configuration for Eq. 4-final MCP serving."""

    categories: list[str]
    actions: list[str]
    factors: list[str]
    state_path: str | None
    penalty_ratio: float
    q_window: int
    tau: float
    server_name: str
    transport: str
    port: int


def _csv_names(value: str | Sequence[str] | None) -> list[str] | None:
    """Normalize CLI or YAML names for the Eq. 4-final tensor axes."""
    if value is None:
        return None
    raw = value.split(",") if isinstance(value, str) else list(value)
    names = [str(item).strip() for item in raw if str(item).strip()]
    return names or None


def build_parser() -> argparse.ArgumentParser:
    """Build the standalone Eq. 4-final MCP argument parser."""
    parser = argparse.ArgumentParser(
        prog="rgi-mcp-server",
        description="Serve GAE Judgment Memory over the Model Context Protocol.",
    )
    parser.add_argument("--config", type=Path, help="YAML configuration file")
    parser.add_argument("--categories", help="Comma-separated category names")
    parser.add_argument("--actions", help="Comma-separated action names")
    parser.add_argument("--factors", help="Comma-separated factor names")
    parser.add_argument("--state", dest="state_path", help="Judgment Memory state file")
    parser.add_argument("--penalty-ratio", type=float, help="Conservation penalty ratio")
    parser.add_argument("--q-window", type=int, help="Conservation quality window")
    parser.add_argument("--tau", type=float, help="Scoring temperature")
    parser.add_argument("--server-name", help="MCP server name")
    transports = parser.add_mutually_exclusive_group()
    transports.add_argument("--stdio", action="store_true", help="Use stdio transport (default)")
    transports.add_argument("--http", action="store_true", help="Use HTTP/SSE transport")
    parser.add_argument("--port", type=int, help="HTTP port (default: 8080)")
    return parser


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load optional YAML configuration for the Eq. 4-final server."""
    try:
        import yaml  # type: ignore[import]
    except ImportError as exc:
        raise ImportError("YAML config requires PyYAML: pip install pyyaml") from exc
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError("config file must contain a YAML mapping")
    return dict(data)


def parse_args(argv: Sequence[str] | None = None) -> ServerConfig:
    """Merge CLI and YAML values into a validated Eq. 4-final server config."""
    parser = build_parser()
    args = parser.parse_args(argv)
    config = _load_yaml(args.config) if args.config is not None else {}

    categories = _csv_names(args.categories) or _csv_names(config.get("categories"))
    actions = _csv_names(args.actions) or _csv_names(config.get("actions"))
    factors = _csv_names(args.factors) or _csv_names(config.get("factors"))
    missing = [
        name
        for name, value in (
            ("categories", categories),
            ("actions", actions),
            ("factors", factors),
        )
        if not value
    ]
    if missing:
        parser.error("missing required configuration: " + ", ".join(missing))

    state_path = args.state_path if args.state_path is not None else config.get("state_path")
    penalty_ratio = (
        args.penalty_ratio
        if args.penalty_ratio is not None
        else float(config.get("penalty_ratio", 5.0))
    )
    q_window = args.q_window if args.q_window is not None else int(config.get("q_window", 400))
    tau = args.tau if args.tau is not None else float(config.get("tau", 0.1))
    server_name = args.server_name or str(config.get("server_name", "rgi-judgment-memory"))
    port = args.port if args.port is not None else int(config.get("port", 8080))
    if not 1 <= port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return ServerConfig(
        categories=list(categories or []),
        actions=list(actions or []),
        factors=list(factors or []),
        state_path=str(state_path) if state_path is not None else None,
        penalty_ratio=penalty_ratio,
        q_window=q_window,
        tau=tau,
        server_name=server_name,
        transport="http" if args.http else "stdio",
        port=port,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the configured Eq. 4-final FastMCP server."""
    config = parse_args(argv)
    server = create_server(
        categories=config.categories,
        actions=config.actions,
        factors=config.factors,
        state_path=config.state_path,
        penalty_ratio=config.penalty_ratio,
        server_name=config.server_name,
        q_window=config.q_window,
        tau=config.tau,
    )
    if config.transport == "stdio":
        server.run(transport="stdio")
        return
    if hasattr(server, "settings"):
        server.settings.port = config.port
    try:
        server.run(transport="http")
    except ValueError:
        server.run(transport="sse")


if __name__ == "__main__":
    main()

