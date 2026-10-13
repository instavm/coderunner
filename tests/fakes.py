"""Shared test helpers: a scripted OpenAI-compatible server and a services stub."""

from __future__ import annotations

import json
import pathlib
import socket
import tempfile
import threading
import time

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from coderunner.core.audit import AuditLog
from coderunner.core.config import Config
from coderunner.safety.egress import EgressGuard
from coderunner.safety.guard import Guard
from coderunner.safety.policy import Policy
from coderunner.safety.secrets_store import SecretStore
from coderunner.core.store import Store


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class FakeOpenAI:
    """Replies come from `script`: each item is a message dict or a callable(request_json) -> message dict.

    Requests from the step judge (coderunner.safety.decision) are answered by `judge(state) -> bool` instead,
    so they do not consume the script; by default every step is judged ordinary.
    """

    def __init__(self):
        self.script: list = []
        self.requests: list[dict] = []
        self.judged: list[str] = []
        self.judge = lambda state: False
        self.port = free_port()

        async def chat(request: Request):
            body = await request.json()
            messages = body.get("messages") or [{}]
            if str(messages[0].get("content", "")).startswith("Evaluate the supplied decision task"):
                state = json.loads(messages[-1]["content"])["state"]
                self.judged.append(state)
                answer = "B" if self.judge(state) else "A"
                return JSONResponse({"choices": [{"message": {"role": "assistant", "content": answer}}]})
            if body.get("max_tokens") == 1:      # the daemon warming the model's cache: not part of the script
                return JSONResponse({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
            self.requests.append(body)
            item = self.script.pop(0) if self.script else {"role": "assistant", "content": "done"}
            message = item(body) if callable(item) else item
            if "_status" in message:            # a runtime failure, e.g. {"_status": 500, "error": "Compute error."}
                return JSONResponse({"error": {"message": message.get("error", "")}}, status_code=message["_status"])
            if body.get("stream"):
                async def events():
                    chunk = {"choices": [{"delta": {"content": message.get("content", "")}}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(events(), media_type="text/event-stream")
            return JSONResponse({"choices": [{"message": message, "finish_reason": "stop"}],
                                 "usage": {"completion_tokens": 3}})

        async def embeddings(request: Request):
            body = await request.json()
            self.requests.append(body)
            return JSONResponse({"data": [{"embedding": [1.0, 0.0]} for _ in body["input"]]})

        app = Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"]),
                                Route("/v1/embeddings", embeddings, methods=["POST"])])
        self.server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=self.port, log_level="error"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    def __enter__(self):
        self.thread.start()
        for _ in range(100):
            if self.server.started:
                return self
            time.sleep(0.05)
        raise RuntimeError("fake server did not start")

    def __exit__(self, *exc):
        self.server.should_exit = True
        self.thread.join(5)


def tool_call(name: str, arguments: dict, call_id: str = "call_1") -> dict:
    return {"role": "assistant", "content": "", "tool_calls": [
        {"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}}]}


class Svc:
    """The attributes the gateway, tools and agent read."""

    def __init__(self, data: dict | None = None):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.config = Config(data or {}, path=self.tmp / "config.json")
        self.store = Store(":memory:")
        self.audit = AuditLog(self.tmp / "audit.jsonl")
        self.policy = Policy(self.config, self.store, self.audit)
        self.secrets = SecretStore(self.tmp / "secrets.json")
        self.egress = EgressGuard(self.config, self.policy, self.audit)
        self.guard = Guard()
        self.api_token = lambda: "sandbox-token-0123456789abcdefghijklmnop"
        self.admin_token = lambda: "admin-token-0123456789abcdefghijklmnopq"
        self.llm = None
        self.llm_api_key = ""
        self.embedding_model = ""
        self.vision = False
        self.sandbox = None
        self.browser = None
        self.memory = None
        self.scheduler = None
        self.files = None
        self.connectors = []
        self.channels = {}
        self.registry = None
        self.agent = None
