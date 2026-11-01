# RGI Judgment Memory — MCP Server

Add compounding Judgment Memory to any AI agent via Model Context Protocol.

## Install

```bash
pip install graph-attention-engine[mcp]
```

## Configure (Claude Desktop / Cursor)

Add to your MCP client config:

```json
{
  "mcpServers": {
    "rgi-judgment-memory": {
      "command": "rgi-mcp-server",
      "args": [
        "--categories", "retrieval,generation,routing",
        "--actions", "use,skip,escalate",
        "--factors", "relevance,confidence,recency,source_trust,complexity",
        "--state", "./jm_state.json"
      ]
    }
  }
}
```

## What it does

Six tools score decisions, verify outcomes, check conservation status, view
decision history, report accuracy, and explain any decision.

The system learns from your verified outcomes. Day 90 is measurably better
than Day 1.

## Links

- PyPI: https://pypi.org/project/graph-attention-engine/
- GitHub: https://github.com/ArindamBanerji/graph-attention-engine
- Website: https://rgi.dakshineshwari.net
