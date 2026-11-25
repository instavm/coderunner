"""Shared test helpers: a scripted OpenAI-compatible server and a services stub."""

from __future__ import annotations

import asyncio
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
from coderunner.agent.compaction import SUMMARIZE
from coderunner.server.titles import SYSTEM as TITLE_SYSTEM

SUMMARY = ("## Goal\nThe owner's request.\n## Constraints & Preferences\n(none)\n## Progress\n### Done\n- Read the pages so far.\n"
           "### In progress\n- The rest.\n### Blocked\n(none)\n## Key Decisions\n(none)\n## Key facts found\n(none yet)\n"
           "## Next Steps\n- Carry on.")


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
        self.titled: list[dict] = []            # requests for a thread's name (server/titles.py), apart from the script
        self.title = lambda prompt: "Fake title"
        # Requests for a summary of the conversation (agent.compaction), apart from the script: summary(body)
        # gives the reply, a message dict or its text.
        self.summarized: list[dict] = []
        self.summary = lambda body: SUMMARY
        self.served = ["test"]                  # the models /v1/models lists
        self.authorizations: list[str] = []
        self.port = free_port()

        async def chat(request: Request):
            body = await request.json()
            messages = body.get("messages") or [{}]
            if str(messages[0].get("content", "")).startswith("Evaluate the supplied decision task"):
                state = json.loads(messages[-1]["content"])["state"]
                self.judged.append(state)
                answer = "B" if self.judge(state) else "A"
                return JSONResponse({"choices": [{"message": {"role": "assistant", "content": answer}}]})
            if str(messages[0].get("content", "")).startswith(TITLE_SYSTEM):     # a thread's name, not a step
                self.titled.append(body)
                return JSONResponse({"choices": [{"message": {"role": "assistant",
                                                              "content": self.title(messages[-1]["content"])}}]})
            if body.get("max_tokens") == 1:      # the daemon warming the model's cache: not part of the script
                return JSONResponse({"choices": [{"message": {"role": "assistant", "content": "ok"}}]})
            if str(messages[-1].get("content", "")) == SUMMARIZE:
                self.summarized.append(body)
                item = self.summary(body)
                message = {"role": "assistant", "content": item} if isinstance(item, str) else dict(item)
                if "_status" in message:
                    return JSONResponse({"error": {"message": message.get("error", "")}}, status_code=message["_status"])
                if "_delay" in message:
                    await asyncio.sleep(message.pop("_delay"))
                return JSONResponse({"choices": [{"message": message, "finish_reason": "stop"}],
                                     "usage": {"completion_tokens": len(message.get("content") or "") // 4}})
            self.requests.append(body)
            item = self.script.pop(0) if self.script else {"role": "assistant", "content": "done"}
            message = item(body) if callable(item) else item
            if "_delay" in message:              # a slow reply, without holding up other requests
                await asyncio.sleep(message["_delay"])
                message = {k: v for k, v in message.items() if k != "_delay"}
            if "_status" in message:            # a runtime failure, e.g. {"_status": 500, "error": "Compute error."}
                return JSONResponse({"error": {"message": message.get("error", "")}}, status_code=message["_status"])
            if body.get("stream"):
                async def events():
                    chunk = {"choices": [{"delta": {"content": message.get("content", "")}}]}
                    yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                return StreamingResponse(events(), media_type="text/event-stream")
            return JSONResponse({"choices": [{"message": message, "finish_reason": message.pop("finish_reason", "stop")}],
                                 "usage": {"completion_tokens": 3}})

        async def embeddings(request: Request):
            body = await request.json()
            self.requests.append(body)
            return JSONResponse({"data": [{"embedding": [1.0, 0.0]} for _ in body["input"]]})

        self.warmed: list[dict] = []
        self.slot_actions: list[tuple[int, str, str]] = []

        async def chat_or_warm(request: Request):
            body = await request.json()
            if body.get("max_tokens") == 1:
                self.warmed.append(body)
            return await chat(request)

        async def slots(request: Request):       # llama.cpp's slot save and restore
            body = await request.json()
            self.slot_actions.append((int(request.path_params["id"]), request.query_params.get("action", ""),
                                      body.get("filename", "")))
            return JSONResponse({"id_slot": int(request.path_params["id"])})

        async def models(request: Request):       # what an OpenAI-compatible server lists
            self.authorizations.append(request.headers.get("authorization", ""))
            return JSONResponse({"object": "list", "data": [{"id": name} for name in self.served]})

        async def props(request: Request):        # llama.cpp's own description of itself
            return JSONResponse({"default_generation_settings": {"n_ctx": 65536}, "total_slots": 2})

        app = Starlette(routes=[Route("/v1/chat/completions", chat_or_warm, methods=["POST"]),
                                Route("/v1/embeddings", embeddings, methods=["POST"]),
                                Route("/v1/models", models, methods=["GET"]),
                                Route("/props", props, methods=["GET"]),
                                Route("/health", lambda request: JSONResponse({"status": "ok"}), methods=["GET"]),
                                Route("/slots/{id:int}", slots, methods=["POST"])])
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
