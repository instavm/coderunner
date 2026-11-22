"""The assistant's model on Anthropic's API instead of this computer (remote mode, models/remote.py).

The agent loop, the step judge and the injection guard speak the chat-completions shape of the local runtimes
(core/llm.py). This client takes that shape and speaks the Messages API through Anthropic's own SDK: system
text apart from the messages, tool calls as tool_use blocks, their results as tool_result blocks in one user
turn, pictures as image blocks. Nothing here is private: every request carries the conversation, page text
and file contents to Anthropic.

Thinking is switched off and the conversation is sent as plain text and tool blocks: the agent loop edits
earlier turns (old screenshots and pages are dropped), and a thinking block is only valid while everything
before it is unchanged. How much the model deliberates is set with the effort level instead.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from typing import Any

from .llm import LLMError

log = logging.getLogger("coderunner")
MODEL = "claude-haiku-5-5"
EFFORT = "medium"
MAX_TOKENS = 16000          # an answer or a step; the SDK's non-streaming timeout allows this much
CACHE = {"type": "ephemeral"}


def _image(url: str) -> dict | None:
    """A data: URL as an image block (pictures reach the model only as data, never as addresses to fetch)."""
    if not url.startswith("data:") or ";base64," not in url:
        return None
    media, data = url[5:].split(";base64,", 1)
    return {"type": "image", "source": {"type": "base64", "media_type": media or "image/jpeg", "data": data}}


def _blocks(content: Any) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content.strip() else []
    blocks = []
    for part in content or []:
        if part.get("type") == "text" and (part.get("text") or "").strip():
            blocks.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            image = _image((part.get("image_url") or {}).get("url", ""))
            if image:
                blocks.append(image)
    return blocks


def convert(messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Chat-completions messages as (system blocks, Messages API messages). Turns of the same side that follow
    one another become one turn, so every result of an assistant turn's tool calls is in the user turn after it,
    ahead of anything else said there."""
    system: list[dict] = []
    out: list[dict] = []

    def add(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": list(blocks)})

    for message in messages:
        role, content = message.get("role"), message.get("content")
        if role == "system":
            system += _blocks(content)
        elif role == "tool":
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
            add("user", [{"type": "tool_result", "tool_use_id": message.get("tool_call_id", ""),
                          "content": text or "(no output)"}])
        elif role == "assistant":
            blocks = _blocks(content)
            for call in message.get("tool_calls") or []:
                function = call.get("function") or {}
                arguments = function.get("arguments") or {}
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments or "{}")
                    except json.JSONDecodeError:
                        arguments = {}
                blocks.append({"type": "tool_use", "id": call.get("id", ""), "name": function.get("name", ""),
                               "input": arguments if isinstance(arguments, dict) else {}})
            add("assistant", blocks)
        else:
            add("user", _blocks(content))
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": [{"type": "text", "text": "(continuing)"}]})
    return system, out


def tools_of(tools: list[dict] | None) -> list[dict]:
    """Function tools as the Messages API declares them."""
    found = []
    for tool in tools or []:
        function = tool.get("function") or {}
        found.append({"name": function.get("name", ""), "description": function.get("description", ""),
                      "input_schema": function.get("parameters") or {"type": "object", "properties": {}}})
    return found


class Claude:
    """The chat interface of core.llm.LLM, served by Anthropic's API."""

    remote = True

    def __init__(self, api_key: str, model: str = MODEL, effort: str = EFFORT, timeout: float = 180,
                 http_client=None):
        import anthropic
        self._sdk = anthropic
        self.model, self.effort = model, effort
        self.base_url = "https://api.anthropic.com"
        self.recover = None                 # nothing of ours to restart
        self.usage = {"requests": 0, "input_tokens": 0, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0, "output_tokens": 0}
        self.last: dict = {}                # the model and request id of the latest answer, as the API gave them
        self._seconds: list[float] = []     # how long the latest steps took
        self._client = anthropic.AsyncAnthropic(api_key=api_key, timeout=timeout, max_retries=2,
                                                **({"http_client": http_client} if http_client else {}))

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   tool_choice: str | dict | None = None, think: bool | None = None, **options: Any) -> dict:
        system, converted = convert(messages)
        request: dict[str, Any] = {
            "model": self.model, "max_tokens": int(options.get("max_tokens") or MAX_TOKENS), "messages": converted,
            "thinking": {"type": "disabled"}, "output_config": {"effort": options.get("effort") or self.effort},
            # The conversation so far is read back from the cache on the next step.
            "cache_control": CACHE,
        }
        if system:
            # Tools are rendered before the system text: this one mark keeps both, whatever the messages do.
            system[-1] = {**system[-1], "cache_control": CACHE}
            request["system"] = system
        if tools:
            request["tools"] = tools_of(tools)
            if tool_choice == "none":
                request["tool_choice"] = {"type": "none"}
        started = time.monotonic()
        reply = await self._create(request)
        self._seconds = [*self._seconds[-49:], time.monotonic() - started]
        if reply.stop_reason == "refusal":
            details = getattr(reply, "stop_details", None)
            why = getattr(details, "category", None) or "its usage policy"
            raise LLMError(f"Claude declined this request ({why}).")
        text = "".join(block.text for block in reply.content if block.type == "text")
        calls = [{"id": block.id, "type": "function",
                  "function": {"name": block.name, "arguments": json.dumps(block.input, ensure_ascii=False)}}
                 for block in reply.content if block.type == "tool_use"]
        if reply.stop_reason == "max_tokens" and calls:
            calls = calls[:-1]              # the last call was cut off mid-way: its arguments are not whole
        usage = reply.usage
        read, made = int(usage.cache_read_input_tokens or 0), int(usage.cache_creation_input_tokens or 0)
        self.usage["requests"] += 1
        for key, value in (("input_tokens", usage.input_tokens), ("cache_read_input_tokens", read),
                           ("cache_creation_input_tokens", made), ("output_tokens", usage.output_tokens)):
            self.usage[key] += int(value or 0)
        return {"content": text.strip(), "tool_calls": calls, "timings": {}, "finish_reason": reply.stop_reason,
                "usage": {"prompt_tokens": int(usage.input_tokens or 0) + read + made,
                          "completion_tokens": int(usage.output_tokens or 0),
                          "cache_read_input_tokens": read, "cache_creation_input_tokens": made}}

    def step_seconds(self) -> float | None:
        """The median time of the latest steps, as measured here (None before the first)."""
        return round(statistics.median(self._seconds), 1) if self._seconds else None

    async def _create(self, request: dict):
        sdk = self._sdk
        try:
            reply = await self._client.messages.create(**request)
            usage = reply.usage
            self.last = {"model": reply.model, "request_id": getattr(reply, "_request_id", None)}
            # What answered and what it cost, never what was said.
            log.info("anthropic %s %s stop=%s in=%s cache_read=%s cache_write=%s out=%s", reply.model,
                     self.last["request_id"], reply.stop_reason, usage.input_tokens,
                     usage.cache_read_input_tokens, usage.cache_creation_input_tokens, usage.output_tokens)
            return reply
        except sdk.AuthenticationError as exc:
            raise LLMError("Anthropic did not accept the key. Set it again in the model settings.") from exc
        except sdk.PermissionDeniedError as exc:
            raise LLMError("This Anthropic key may not use that model.") from exc
        except sdk.NotFoundError as exc:
            raise LLMError(f"Anthropic has no model named {self.model}.") from exc
        except sdk.RateLimitError as exc:
            raise LLMError("Anthropic is limiting requests on this key right now. Try again in a minute.") from exc
        except sdk.BadRequestError as exc:
            raise LLMError(f"Anthropic refused the request: {str(getattr(exc, 'message', exc))[:300]}") from exc
        except sdk.APIStatusError as exc:
            raise LLMError(f"Anthropic's API returned {exc.status_code}. Try again shortly.") from exc
        except sdk.APIConnectionError as exc:
            raise LLMError("Anthropic's API could not be reached. Check the internet connection.") from exc

    async def ask(self, system: str, turns: list[dict], max_tokens: int = 16) -> str:
        """A short answer to a fixed question (the step judge's letter, the guard's): low effort, no tools."""
        reply = await self._create({"model": self.model, "max_tokens": max_tokens, "messages": convert(turns)[1],
                                    "system": [{"type": "text", "text": system, "cache_control": CACHE}],
                                    "thinking": {"type": "disabled"}, "output_config": {"effort": "low"}})
        if reply.stop_reason == "refusal":
            raise LLMError("Claude declined to judge this step.")
        return "".join(block.text for block in reply.content if block.type == "text").strip()

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        raise LLMError("Anthropic's API has no embeddings; file search and memory use the local embedding model.")

    async def list_models(self) -> list[str]:
        return [self.model]

    async def aclose(self) -> None:
        await self._client.close()
