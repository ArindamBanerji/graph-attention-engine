#!/bin/bash
set -e
echo "=== Step 1: Login to MCP registry ==="
mcp-publisher login --github
echo "=== Step 2: Validate server.json ==="
mcp-publisher validate mcp-registry/server.json
echo "=== Step 3: Publish ==="
mcp-publisher publish mcp-registry/server.json
echo "=== Step 4: Verify listing ==="
echo "Check: https://registry.modelcontextprotocol.io/servers/io.github.arindambanerji/rgi-judgment-memory"
echo ""
echo "=== Next: Smithery ==="
echo "Run: smithery mcp publish <your-server-url> -n arindambanerji/rgi-judgment-memory"
echo ""
echo "=== Next: Glama ==="
echo "Visit glama.ai, find auto-crawled listing, claim and verify"
echo ""
echo "Done. Downstream directories (mcp.so, PulseMCP) auto-ingest from the official registry."
