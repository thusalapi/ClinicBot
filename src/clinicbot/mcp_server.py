"""Expose ClinicBot's tools over MCP (Model Context Protocol).

Our agent calls tools.py directly. That works, but only *our* agent can use
those tools. MCP is a standard way to publish tools so that *any* MCP client
can discover and call them: Claude Desktop, Claude Code, IDEs, other agent
frameworks, and (Phase 4) agent platforms like WSO2 Agent Manager.

    ┌──────────────┐   list_tools / call_tool   ┌────────────────┐
    │  MCP client  │ ─────── JSON-RPC ────────▶ │ this server    │──▶ tools.py ──▶ clinic DB
    │ (any agent)  │ ◀────── results ────────── │ (stdio / HTTP) │
    └──────────────┘                            └────────────────┘

Two transports:
  stdio            the client starts this program and talks over stdin/stdout
                   (local tools: Claude Desktop, Claude Code)
  streamable-http  a normal web server other machines can reach (deployments)

Run:  uv run clinicbot-mcp            (stdio)
      uv run clinicbot-mcp --http     (http://127.0.0.1:8765/mcp)
"""

import argparse

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from clinicbot.tools import SIDE_EFFECT_TOOLS, TOOL_SCHEMAS, TOOLS

server = MCPServer(
    name="clinicbot",
    instructions="Tools for Sunrise Family Clinic: clinic information, doctors, "
    "availability, booking and cancelling appointments.",
)

# Register every tool from tools.py, so there's one source of truth for both
# our own agent and MCP clients.
for schema in TOOL_SCHEMAS:
    name = schema["function"]["name"]
    server.add_tool(
        TOOLS[name],
        name=name,
        description=schema["function"]["description"],
        # Annotations let clients decide which tools need a human to approve.
        # (Hints only: a client may ignore them, so real enforcement still
        # belongs on the server side or in a gateway; see Phase 4.)
        annotations=ToolAnnotations(
            read_only_hint=name not in SIDE_EFFECT_TOOLS,
            destructive_hint=name == "cancel_appointment",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="clinicbot-mcp")
    parser.add_argument("--http", action="store_true", help="serve over streamable HTTP")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    if args.http:
        server.run(transport="streamable-http", host="127.0.0.1", port=args.port)
    else:
        server.run()  # stdio
