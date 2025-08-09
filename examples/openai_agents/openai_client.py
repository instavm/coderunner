import asyncio
import socket


from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStreamableHttp
from agents.model_settings import ModelSettings



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
    async with MCPServerStreamableHttp(
        name="SSE Python Server",
        params={
            "url": f"http://{address}:8222/mcp",
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
