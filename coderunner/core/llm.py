"""OpenAI-compatible chat client tuned for local runtimes.

Ollama, llama.cpp's llama-server, mlx-lm/mlx-vlm and vLLM all expose
/v1/chat/completions. Local models sometimes emit tool calls as text instead
of structured `tool_calls`, and some include reasoning in <think> blocks; both
are normalised here so the agent loop sees one shape.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any, Awaitable, Callable

import httpx

THINK = re.compile(r"<think>.*?</think>\s*", re.S)
TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def parse_text_tool_calls(content: str, tool_names: set[str]) -> tuple[str, list[dict]]:
    """Extract tool calls written as text by models whose template was not parsed."""
    calls = []
    for raw in TOOL_CALL_TAG.findall(content):
        data = _loads(raw)
        if isinstance(data, dict) and data.get("name") in tool_names:
            calls.append(data)
    remaining = TOOL_CALL_TAG.sub("", content).strip()
    if not calls:
        stripped = remaining.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = _loads(stripped)
        if isinstance(data, dict) and data.get("name") in tool_names and "arguments" in data:
            calls, remaining = [data], ""
    normalised = []
    for call in calls:
        arguments = call.get("arguments", call.get("parameters", {}))
        normalised.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {"name": call["name"],
                         "arguments": arguments if isinstance(arguments, str) else json.dumps(arguments)},
        })
    return remaining, normalised


class LLMError(Exception):
    pass


class LLM:
    def __init__(self, base_url: str, model: str, api_key: str = "", timeout: float = 600):
        self.base_url = base_url.rstrip("/")
        self.model = model
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout, headers=headers, trust_env=False)
        self._always_thinks = False     # the runtime refused a request to answer without reasoning
        # Restarts a local runtime that crashed or got stuck (a GPU out of memory leaves llama.cpp unable to
        # compute until it is restarted); returns whether it is serving again. Set by the daemon.
        self.recover: Callable[[], Awaitable[bool]] | None = None

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   tool_choice: str | dict | None = None, think: bool | None = None, **options: Any) -> dict:
        """think=False asks a reasoning model to answer directly (None: the model's default)."""
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, **options}
        quiet = think is False and not self._always_thinks
        if quiet:
            # Ollama reads reasoning_effort, llama.cpp reads chat_template_kwargs.
            payload.update(reasoning_effort="none", chat_template_kwargs={"enable_thinking": False})
        if tools:
            payload["tools"] = tools
            if tool_choice:
                payload["tool_choice"] = tool_choice
        response = await self._post(payload)
        if response.status_code == 400 and quiet:
            self._always_thinks = True
            return await self.chat(messages, tools, tool_choice, **options)
        if response.status_code >= 400:
            raise LLMError(f"Model runtime returned {response.status_code}: {response.text[:500]}")
        data = response.json()
        choice = (data.get("choices") or [{}])[0]
        message = dict(choice.get("message") or {})
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        content = THINK.sub("", content)
        tool_calls = message.get("tool_calls") or []
        if not tool_calls and tools:
            names = {t["function"]["name"] for t in tools}
            content, tool_calls = parse_text_tool_calls(content, names)
        return {"content": content.strip(), "tool_calls": tool_calls, "usage": data.get("usage", {}),
                "timings": data.get("timings", {}), "finish_reason": choice.get("finish_reason")}

    async def _post(self, payload: dict) -> httpx.Response:
        """One request; when the runtime is down or failing, it is restarted once and the request retried, so
        a long task keeps its progress."""
        for attempt in (1, 2):
            try:
                response = await self._client.post("/chat/completions", json=payload)
            except httpx.HTTPError as exc:
                if attempt == 1 and self.recover is not None and await self.recover():
                    continue
                raise LLMError(f"Model runtime at {self.base_url} is unreachable: {exc}") from exc
            if response.status_code >= 500 and attempt == 1 and self.recover is not None and await self.recover():
                continue
            return response
        return response

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        response = await self._client.post("/embeddings", json={"model": model or self.model, "input": texts})
        if response.status_code >= 400:
            raise LLMError(f"Embedding request failed with {response.status_code}: {response.text[:200]}")
        return [item["embedding"] for item in response.json()["data"]]

    async def list_models(self) -> list[str]:
        response = await self._client.get("/models")
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", [])]

    async def aclose(self) -> None:
        await self._client.aclose()
