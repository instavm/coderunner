import asyncio
import os
import shutil
import subprocess
import time
import socket
from typing import Any

from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerSse
from agents.model_settings import ModelSettings


# async def run(mcp_server: MCPServer):
#     agent = Agent(
#         name="Assistant",
#         instructions="Use the tools to answer the questions.",
#         mcp_servers=[mcp_server],
#         model_settings=ModelSettings(tool_choice="required"),
#     )

#     # Use the `add` tool to add two numbers
#     message = "list files in current directory using python"
#     print(f"Running: {message}")
#     result = await Runner.run(starting_agent=agent, input=message)
#     print(result.final_output)

#     # Run the `get_weather` tool
#     message = "Fetch ETH price on 15th June 2025. First pip install libraries needed like yfinance, then write the code and fetch data."
#     print(f"\n\nRunning: {message}")
#     result = await Runner.run(starting_agent=agent, input=message)
#     print(result.final_output)

#     # Run the `get_secret_word` tool
#     message = "What's the secret word?"
#     print(f"\n\nRunning: {message}")
#     result = await Runner.run(starting_agent=agent, input=message)
#     print(result.final_output)


async def run(mcp_server: MCPServer):
    agent = Agent(
        name="Assistant",
        instructions="Use the tools to answer the questions.",
        mcp_servers=[mcp_server],
        model_settings=ModelSettings(tool_choice="required"),
    )

    while True:
        message = input("Enter your prompt (or 'exit' to quit): ")
        if message.lower() == 'exit':
            print("Exiting...")
            break

        print(f"Running: {message}")
        result = await Runner.run(starting_agent=agent, input=message)
        print(result.final_output)

def resolve_with_system_dns(hostname):
    try:
        return socket.gethostbyname(hostname)
    except socket.gaierror as e:
        print(f"Error resolving {hostname}: {e}")
        return None


async def main():
    hostname = "coderunner.local"
    address = resolve_with_system_dns(hostname)
    async with MCPServerSse(
        name="SSE Python Server",
        params={
            "url": f"http://{address}:8222/sse",
            "sse_read_timeout": 60,
            "timeout": 60,
        },
    ) as server:
        trace_id = gen_trace_id()
        with trace(workflow_name="SSE Example", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}\n")
            await run(server)


if __name__ == "__main__":
    asyncio.run(main())
