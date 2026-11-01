# Graph Attention Engine (GAE)

Standalone, pip-installable Python library implementing Cross-Graph Attention.
Zero domain knowledge. Zero external dependencies (numpy only). Apache 2.0.

## Install

```bash
pip install graph-attention-engine
# or editable dev install:
pip install -e ".[dev]"
```

## Quick start

```python
import numpy as np
from gae import scaled_dot_product_attention

Q = np.random.randn(4, 8)   # (seq, d_k)
K = np.random.randn(6, 8)
V = np.random.randn(6, 16)
out, weights = scaled_dot_product_attention(Q, K, V)
# out: (4, 16)  weights: (4, 6)
```

## Math reference

Full specification: `docs/gae_design_v5.md`
Blog: https://www.dakshineshwari.net/post/cross-graph-attention-mathematical-foundation-with-experimental-validation

## MCP Server

Graph Attention Engine includes a built-in [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that exposes decision scoring as MCP tools. Any MCP-compatible client (Claude Desktop, Cursor, etc.) can connect and use GAE's cross-graph attention scoring directly.

### Install

```bash
pip install "graph-attention-engine[mcp]"
```

### Usage

```bash
# stdio transport (for Claude Desktop, Cursor)
rgi-mcp-server --categories alert_triage,incident_response --actions investigate,escalate --factors severity,confidence,blast_radius

# HTTP transport (for remote/hosted deployments)
rgi-mcp-server --http --port 8080 --categories alert_triage,incident_response --actions investigate,escalate --factors severity,confidence,blast_radius
```

### Docker

```bash
docker build -f Dockerfile.mcp -t gae-mcp .
docker run -p 8080:8080 gae-mcp
```

### MCP Tools

| Tool | Description |
|---|---|
| `score_decision` | Score a decision using cross-graph attention and compounding intelligence |
| `get_trajectory` | Retrieve the learning trajectory showing how decisions compound over time |
| `get_conservation_status` | Check conservation law compliance and scoring integrity |

### Claude Desktop Configuration

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "graph-attention-engine": {
      "command": "rgi-mcp-server",
      "args": ["--categories", "alert_triage,incident_response", "--actions", "investigate,escalate", "--factors", "severity,confidence,blast_radius"]
    }
  }
}
```

## Testing

```bash
pytest tests/ -v
```
