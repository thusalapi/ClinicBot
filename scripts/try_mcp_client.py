"""A minimal MCP *client*: the other side of the protocol.

It starts the ClinicBot MCP server as a subprocess (stdio transport), asks which
tools exist, and calls two of them. This is exactly what Claude Desktop or an
agent platform does under the hood.

    uv run python scripts/try_mcp_client.py
"""

import asyncio
import json
from datetime import date, timedelta

from mcp import ClientSession, StdioServerParameters, stdio_client


async def main() -> None:
    params = StdioServerParameters(command="uv", args=["run", "clinicbot-mcp"])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            print("Tools the server offers:")
            for tool in (await session.list_tools()).tools:
                hints = tool.annotations
                kind = "read-only" if hints and hints.read_only_hint else "CHANGES DATA"
                print(f"  - {tool.name:<22} [{kind}]")
                print(f"      params: {list(tool.input_schema.get('properties', {}))}")

            friday = next(
                date.today() + timedelta(days=i)
                for i in range(1, 8)
                if (date.today() + timedelta(days=i)).weekday() == 4
            )
            print(f"\ncall_tool check_availability(kevin, {friday}):")
            result = await session.call_tool(
                "check_availability", {"doctor_id": "kevin", "date": friday.isoformat()}
            )
            print("  " + result.content[0].text)

            print("\ncall_tool search_clinic_info('parking'):")
            result = await session.call_tool("search_clinic_info", {"query": "parking"})
            first = json.loads(result.content[0].text)["passages"][0]
            print(f"  top passage: {first['source']} :: {first['section']}")


if __name__ == "__main__":
    asyncio.run(main())
