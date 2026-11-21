"""Expose the assistant's guarded tools to local MCP clients over stdio.

Claude Code, Codex, Claude Desktop (Cowork), LM Studio and other agents can use the owner's local files, Mail,
Messages, Notes and Calendar entries, memory, browser, schedules and connectors through the same policy and
approval path the built-in assistant uses. `--tools search` offers only search, reading and memory, for agents
that should find local things and nothing more. The bridge holds the admin token, so it must only be configured
for clients the owner runs locally; remote clients use the daemon's Streamable HTTP endpoint instead
(gateway.mcp_app).
"""

from __future__ import annotations

import asyncio
import json

from ..core import paths
from ..core.config import Config
from .gateway import admin_request

SEARCH_TOOLS = ("files_search", "file_read", "file_view", "file_listen", "memory_search", "memory_save", "memory_forget")


def tool_names(selection: str, described: list[dict]) -> list[dict]:
    """The tools offered: all of them, or the search, reading and memory ones."""
    if selection == "search":
        return [t for t in described if t["name"] in SEARCH_TOOLS]
    return described


def main(tools: str = "all") -> None:
    from mcp import types
    from mcp.server.lowlevel import Server
    from mcp.server.stdio import stdio_server

    config = Config.load()
    base_url = f"http://127.0.0.1:{config.get('gateway.admin_port')}"
    token = paths.admin_token()
    server = Server("coderunner" if tools == "all" else "coderunner-search")

    def call(method: str, path: str, body=None):
        return admin_request(method, path, base_url=base_url, token=token, json_body=body)

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        described = tool_names(tools, await asyncio.to_thread(call, "GET", "/api/tools"))
        return [types.Tool(name=t["name"], description=t["description"], inputSchema=t["parameters"])
                for t in described]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list:
        result = await asyncio.to_thread(call, "POST", f"/api/tools/{name}",
                                         {"arguments": arguments or {}, "client": "mcp"})
        text = result.get("result", "")
        content: list = [types.TextContent(type="text", text=text if isinstance(text, str) else json.dumps(text))]
        content += [types.ImageContent(type="image", data=image["data_b64"], mimeType=image["mime"])
                    for image in result.get("images", [])]
        return content

    async def run() -> None:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())

    asyncio.run(run())
