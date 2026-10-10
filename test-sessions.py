#!/usr/bin/env python3

import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from threading import Lock


BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "http://coderunner.local:8222"


def request(path, method="GET", payload=None, headers=None):
    data = json.dumps(payload).encode() if payload is not None else None
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers=request_headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.headers, response.read().decode()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.headers, exc.read().decode()


def parse_sse(body):
    for line in body.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    raise AssertionError(f"No SSE data in response: {body[:200]}")


class MCPClient:
    def __init__(self):
        status, headers, body = request(
            "/mcp",
            "POST",
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "session-test", "version": "1"},
                },
            },
            {"Accept": "application/json, text/event-stream"},
        )
        assert status == 200
        self.session_id = headers["mcp-session-id"]
        self.headers = {
            "Accept": "application/json, text/event-stream",
            "mcp-session-id": self.session_id,
        }
        initialized = parse_sse(body)
        assert initialized["result"]["protocolVersion"] == "2025-06-18"
        request(
            "/mcp",
            "POST",
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            self.headers,
        )
        self.next_id = 2
        self.lock = Lock()

    def rpc(self, method, params=None):
        with self.lock:
            request_id = self.next_id
            self.next_id += 1
        status, _, body = request(
            "/mcp",
            "POST",
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
                "params": params or {},
            },
            self.headers,
        )
        assert status == 200
        message = parse_sse(body)
        assert message["id"] == request_id
        return message

    def tool(self, name, arguments=None):
        return self.rpc(
            "tools/call",
            {"name": name, "arguments": arguments or {}},
        )["result"]


def tool_text(result):
    return result["content"][0]["text"].strip()


def main():
    client = MCPClient()

    tools = client.rpc("tools/list")["result"]["tools"]
    tools_by_name = {tool["name"]: tool for tool in tools}
    expected = {
        "execute_python_code",
        "start_python_session",
        "list_python_sessions",
        "stop_python_session",
    }
    assert expected <= tools_by_name.keys()
    assert all("outputSchema" in tool for tool in tools)
    execute_schema = tools_by_name["execute_python_code"]["inputSchema"]["properties"]
    session_id_types = {schema["type"] for schema in execute_schema["session_id"]["anyOf"]}
    assert session_id_types == {"string", "null"}

    active = ["alpha", "beta"]
    for session_id in active:
        result = client.tool("start_python_session", {"session_id": session_id})
        assert not result.get("isError")

    client.tool(
        "execute_python_code",
        {"session_id": "alpha", "command": "value = 10"},
    )
    client.tool(
        "execute_python_code",
        {"session_id": "beta", "command": "value = 20"},
    )
    assert tool_text(client.tool(
        "execute_python_code",
        {"session_id": "alpha", "command": "print(value)"},
    )) == "10"
    assert tool_text(client.tool(
        "execute_python_code",
        {"session_id": "beta", "command": "print(value)"},
    )) == "20"

    error = tool_text(client.tool(
        "execute_python_code",
        {"session_id": "alpha", "command": "raise ValueError('expected')"},
    ))
    assert error.startswith("Error: Execution Error:")
    assert tool_text(client.tool(
        "execute_python_code",
        {"session_id": "alpha", "command": "print(value)"},
    )) == "10"

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as executor:
        calls = [
            executor.submit(
                client.tool,
                "execute_python_code",
                {"session_id": session_id, "command": "import time; time.sleep(2); print('done')"},
            )
            for session_id in active
        ]
        assert all(tool_text(call.result()) == "done" for call in calls)
    assert time.monotonic() - started < 4.5

    for session_id in ("gamma", "delta", "epsilon"):
        client.tool("start_python_session", {"session_id": session_id})
        active.append(session_id)
    overflow = client.tool("start_python_session", {"session_id": "overflow"})
    assert overflow.get("isError") is True

    listed = client.tool("list_python_sessions")
    structured = listed["structuredContent"]["result"]
    assert {item["session_id"] for item in structured} == set(active)

    for session_id in active:
        client.tool("stop_python_session", {"session_id": session_id})

    request(
        "/execute",
        "POST",
        {"code": "from pathlib import Path\nPath('/app/uploads/retry-check').unlink(missing_ok=True)"},
    )
    status, _, body = request(
        "/execute",
        "POST",
        {
            "code": (
                "from pathlib import Path\n"
                "path = Path('/app/uploads/retry-check')\n"
                "path.write_text(path.read_text() + 'x' if path.exists() else 'x')\n"
                "raise ValueError('expected')"
            )
        },
    )
    assert status == 200 and json.loads(body)["stderr"].startswith("Error: Execution Error:")
    status, _, body = request(
        "/execute",
        "POST",
        {"code": "print(len(open('/app/uploads/retry-check').read()))"},
    )
    assert json.loads(body)["stdout"].strip() == "1"

    status, _, body = request(
        "/v1/sessions/session",
        "POST",
        {"session_id": "rest_session"},
    )
    assert status == 201 and json.loads(body)["status"] == "active"
    status, _, body = request(
        "/execute",
        "POST",
        {"session_id": "rest_session", "code": "rest_value = 7"},
    )
    assert status == 200
    status, _, body = request(
        "/execute",
        "POST",
        {"session_id": "rest_session", "code": "print(rest_value)"},
    )
    assert json.loads(body)["stdout"].strip() == "7"
    status, _, body = request("/v1/sessions/session?session_id=rest_session")
    assert status == 200 and json.loads(body)["status"] == "active"
    status, _, body = request(
        "/v1/sessions/session?session_id=rest_session",
        "DELETE",
    )
    assert status == 200 and json.loads(body)["status"] == "stopped"

    print("session and MCP contract tests passed")


if __name__ == "__main__":
    main()
