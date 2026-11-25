"""OpenAI-compatible chat client tuned for local runtimes.

Ollama, llama.cpp's llama-server, mlx-lm/mlx-vlm and vLLM all expose
/v1/chat/completions. Local models sometimes emit tool calls as text instead
of structured `tool_calls`, and some include reasoning in <think> blocks; both
are normalised here so the agent loop sees one shape.
"""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from typing import Any, Awaitable, Callable

import httpx

THINK = re.compile(r"<think>.*?</think>\s*", re.S)
MARK = "QZX7"                               # where the probe's request sits in a rendered prompt
EFFORTS = ("medium", "high", "low")         # reasoning efforts tried for one that renders like reasoning off
# A server's answer to a request for a model it does not serve (llama.cpp's router, llama-swap, vLLM, Ollama).
MODEL_MISSING = re.compile(r"model\b.{0,60}\b(not found|does not exist|not exist|unknown|no such|not available|"
                           r"not loaded|not served|invalid)|\b(unknown|invalid|no such) model", re.I | re.S)
TOOL_CALL_TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def _loads(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


# Qwen3.5's own call form (as Qwen3-Coder's): <function=name><parameter=key>value</parameter>...</function>
QWEN_CALL = re.compile(r"<function=([\w.-]+)>(.*?)</function>", re.S)
QWEN_PARAMETER = re.compile(r"<parameter=([\w.-]+)>\s*(.*?)\s*</parameter>", re.S)


def parse_text_tool_calls(content: str, tool_names: set[str]) -> tuple[str, list[dict]]:
    """Extract tool calls written as text by models whose template was not parsed: JSON inside <tool_call>, or
    Qwen3.5's own XML form, which the runtime returns as text when it could not read it."""
    calls = []
    for raw in TOOL_CALL_TAG.findall(content):
        data = _loads(raw)
        if isinstance(data, dict) and data.get("name") in tool_names:
            calls.append(data)
    if not calls:
        for name, body in QWEN_CALL.findall(content):
            if name in tool_names:
                arguments = {}
                for key, value in QWEN_PARAMETER.findall(body):
                    loaded = _loads(value)
                    arguments[key] = value if loaded is None else loaded
                calls.append({"name": name, "arguments": arguments})
        if calls:
            content = QWEN_CALL.sub("", content).replace("<tool_call>", "").replace("</tool_call>", "")
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


KEEP_MISSES = 3         # restores in a row that the server did not resume from, before slot keeping stops


class SlotKeeper:
    """Keeps a task's slot on a llama.cpp server through a side request (the step judge's) in another slot. On the owner's server such a request made the next step of the task read its whole
    conversation again: 268 of 278 steps after a judge call in the bench runs of 24-25 November started from
    nothing (~33 s each at ~510 tokens a second), 46% of all the time spent reading prompts.

    Before a side request, each idle task slot holding a conversation is kept in a file (POST /slots/N
    ?action=save, needs the server's --slot-save-path); the task's next request puts it back first (?action=
    restore), only when that request extends the one before (otherwise the server reads from the change anyway).
    A server without --slot-save-path refuses the first save and keeping stops; so does one whose restored
    slots the next request does not resume from (KEEP_MISSES in a row). A save and a restore are a file written
    and read on the server, where reading the conversation again is ~40 s at 20,000 tokens."""

    def __init__(self, llm: "LLM", root: str):
        self._llm = llm
        self._root = root
        self.on = True
        self.misses = 0
        self._turns: dict[int, int] = {}        # slot -> requests answered in it
        self._saved: dict[int, int] = {}        # slot -> its request count when it was last kept
        self._dirty: set[int] = set()           # kept slots a side request may have emptied since
        self._busy: set[int] = set()            # slots with a request in flight (not idle: not emptied)
        self._sent: dict[int, list[dict]] = {}  # slot -> the messages of its last request
        self._lock = asyncio.Lock()

    def _file(self, slot: int) -> str:
        return f"task-{slot}-{re.sub(r'[^A-Za-z0-9_.-]', '_', self._llm.model)[:80]}.bin"

    async def _action(self, slot: int, action: str) -> dict | None:
        try:
            response = await self._llm._client.post(f"{self._root}/slots/{slot}", params={"action": action},
                                               json={"filename": self._file(slot)}, timeout=60)
        except httpx.HTTPError:
            return None
        if response.status_code >= 400:
            if action == "save" and response.status_code in (400, 404, 405, 501):
                self.on = False         # no --slot-save-path, the slots endpoint off, or not llama.cpp
            return None
        try:
            data = response.json()
        except ValueError:
            return {}
        return data if isinstance(data, dict) else {}

    async def before_side(self, side_slot: int | None) -> None:
        """Keep every idle task slot that changed since it was last kept, before a request in side_slot."""
        if not self.on:
            return
        async with self._lock:
            for slot, turns in list(self._turns.items()):
                if slot == side_slot or slot in self._busy:
                    continue
                if self._saved.get(slot) != turns:
                    if await self._action(slot, "save") is None:
                        self._saved.pop(slot, None)
                        if not self.on:
                            return
                        continue
                    self._saved[slot] = turns
                self._dirty.add(slot)

    async def before_turn(self, slot: int, messages: list[dict]) -> int:
        """Put the slot back as it was kept when a side request came since; returns the tokens restored."""
        self._busy.add(slot)
        if not self.on or slot not in self._dirty:
            return 0
        async with self._lock:
            self._dirty.discard(slot)
            sent = self._sent.get(slot)
            if self._saved.get(slot) != self._turns.get(slot) or not sent or messages[:len(sent)] != sent:
                return 0                # an edited conversation is read from the change whatever is restored
            done = await self._action(slot, "restore")
            return int((done or {}).get("n_restored") or 0) if done is not None else 0

    def after_turn(self, slot: int, messages: list[dict], restored: int, cached: int | None, ok: bool) -> None:
        self._busy.discard(slot)
        if not ok:
            return
        self._turns[slot] = self._turns.get(slot, 0) + 1
        self._sent[slot] = list(messages)
        if restored and cached is not None:
            # A restored slot the server did not resume from (a hybrid model's state that cannot be rolled back
            # to where the new prompt differs) costs the save and the restore for nothing.
            self.misses = 0 if cached >= restored // 2 else self.misses + 1
            if self.misses >= KEEP_MISSES:
                self.on = False


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
        # Another name for the model when the server stops serving this one (an upgrade on the owner's server
        # renames it); None when there is none. renamed is told the new name. Set for a model server of the owner's.
        self.rename: Callable[[], Awaitable[str | None]] | None = None
        self.slot_base = 0            # this install's first slot on a shared server (models.remote.slots)
        self.renamed: Callable[[str], None] | None = None
        self.think_kwargs: dict = {}        # template settings for a step with reasoning (see settle_thinking)
        self.steady_thinking = False        # no setting keeps the prompt the same: a task keeps to one
        self._settled = False
        # Keeps the tasks' slots through side requests in other slots (the owner's server; see SlotKeeper).
        self.keeper: SlotKeeper | None = None

    def keep_slots(self) -> SlotKeeper:
        """Turn slot keeping on for this client (a llama.cpp server whose slots tasks keep to)."""
        if self.keeper is None:
            root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
            self.keeper = SlotKeeper(self, root)
        return self.keeper

    async def side_request(self, slot: int | None) -> None:
        """Called before a request in another slot than the tasks' (the step judge's)."""
        if self.keeper is not None:
            await self.keeper.before_side(slot)

    async def settle_thinking(self) -> None:
        """Some chat templates write the reasoning setting at the top of the system prompt (Bonsai's: "Reasoning
        effort is set to xhigh…" with reasoning on, nothing with it off), so a step with reasoning after one
        without read the whole conversation again: about 40 s at 20,000 tokens on the owner's server, twice for
        every failed step. The template is rendered both ways (llama.cpp's /apply-template); where the prompts
        differ, a reasoning effort that renders like reasoning off is used, or else steady_thinking is set.
        Runtimes without /apply-template are left as they are. Once per client."""
        if self._settled:
            return
        self._settled = True
        root = self.base_url[:-3] if self.base_url.endswith("/v1") else self.base_url
        probe = [{"role": "system", "content": "S"}, {"role": "user", "content": MARK}]
        tool = [{"type": "function", "function": {"name": "f", "description": "d",
                                                  "parameters": {"type": "object", "properties": {}}}}]

        async def head(kwargs: dict) -> str | None:
            body = {"messages": probe, "tools": tool, **({"chat_template_kwargs": kwargs} if kwargs else {})}
            response = await self._client.post(f"{root}/apply-template", json=body, timeout=10)
            prompt = str(response.json().get("prompt") or "") if response.status_code == 200 else ""
            return prompt[:prompt.find(MARK)] if MARK in prompt else None

        try:
            off = await head({"enable_thinking": False})
            if off is None:
                return
            for kwargs in ({}, *({"reasoning_effort": effort} for effort in EFFORTS)):
                if await head(kwargs) == off:
                    self.think_kwargs = kwargs
                    return
            self.steady_thinking = True
        except (httpx.HTTPError, ValueError):
            return

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   tool_choice: str | dict | None = None, think: bool | None = None, **options: Any) -> dict:
        """think=False asks a reasoning model to answer directly (None: the model's default)."""
        slot = options.get("id_slot")
        if isinstance(slot, int) and self.slot_base:
            slot = options["id_slot"] = slot + self.slot_base      # this install's slots on a shared server
        if self.keeper is None or not isinstance(slot, int):
            return await self._chat(messages, tools, tool_choice, think, **options)
        restored = await self.keeper.before_turn(slot, messages)
        reply: dict | None = None
        try:
            reply = await self._chat(messages, tools, tool_choice, think, **options)
        finally:
            cached = (reply or {}).get("timings", {}).get("cache_n")
            self.keeper.after_turn(slot, messages, restored, cached if isinstance(cached, int) else None,
                                   reply is not None)
        if restored:
            reply["restored"] = restored
        return reply

    async def _chat(self, messages: list[dict], tools: list[dict] | None = None,
                    tool_choice: str | dict | None = None, think: bool | None = None, **options: Any) -> dict:
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, **options}
        quiet = think is False and not self._always_thinks
        if quiet:
            # Ollama reads reasoning_effort, llama.cpp reads chat_template_kwargs.
            payload.update(reasoning_effort="none", chat_template_kwargs={"enable_thinking": False})
        elif self.think_kwargs:
            payload["chat_template_kwargs"] = dict(self.think_kwargs)
        if tools:
            payload["tools"] = tools
            payload["parallel_tool_calls"] = True        # several steps in one reply: llama.cpp parses only one without
            if tool_choice:
                payload["tool_choice"] = tool_choice
        response = await self._post(payload)
        if response.status_code in (400, 404) and self.rename is not None and MODEL_MISSING.search(response.text):
            name = await self.rename()
            if name and name != self.model:
                self.model = payload["model"] = name
                if self.renamed is not None:
                    self.renamed(name)
                response = await self._post(payload)
        if response.status_code == 400 and quiet:
            self._always_thinks = True
            return await self._chat(messages, tools, tool_choice, **options)
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

    async def ask(self, system: str, turns: list[dict], max_tokens: int = 16, slot: int | None = None,
                  timeout: float = 20) -> str:
        """A short answer to a small side question (a thread's name), reasoning off, as Claude.ask. Sent once:
        a runtime that is down is not restarted for it, nor a renamed model looked for. slot: llama.cpp's slot
        to answer in (one no task is using)."""
        payload: dict[str, Any] = {"model": self.model, "stream": False, "max_tokens": max_tokens, "temperature": 0,
                                   "messages": [{"role": "system", "content": system}, *turns],
                                   "reasoning_effort": "none", "chat_template_kwargs": {"enable_thinking": False},
                                   **({"id_slot": slot} if slot is not None else {})}
        try:
            response = await self._client.post("/chat/completions", json=payload, timeout=timeout)
        except httpx.HTTPError as exc:
            raise LLMError(f"Model runtime at {self.base_url} is unreachable: {type(exc).__name__}") from exc
        if response.status_code >= 400:
            raise LLMError(f"Model runtime returned {response.status_code}: {response.text[:200]}")
        message = ((response.json().get("choices") or [{}])[0].get("message") or {})
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        return THINK.sub("", content).strip()

    async def _post(self, payload: dict) -> httpx.Response:
        """One request; when the runtime is down or failing, it is restarted once and the request retried, so
        a long task keeps its progress."""
        for attempt in (1, 2):
            try:
                response = await self._send(payload)
            except httpx.HTTPError as exc:
                if attempt == 1 and self.recover is not None and await self.recover():
                    continue
                raise LLMError(f"Model runtime at {self.base_url} is unreachable: {type(exc).__name__} {exc}".strip()) from exc
            if response.status_code >= 500 and attempt == 1 and self.recover is not None and await self.recover():
                continue
            return response
        return response

    async def _send(self, payload: dict) -> httpx.Response:
        try:
            return await self._client.post("/chat/completions", json=payload)
        except httpx.RemoteProtocolError:
            # llama.cpp's server closes a connection left idle for 5 s, and a request sent on it as it closes
            # comes back "disconnected without sending a response" (2 of 10 reuses after 4.9-5.0 s idle): a
            # browser driver's step lands there often. The server is fine; a new connection carries it.
            return await self._client.post("/chat/completions", json=payload)

    async def embed(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        body = {"model": model or self.model, "input": texts}
        try:
            response = await self._client.post("/embeddings", json=body)
        except httpx.TransportError:
            if self.recover is None or not await self.recover():
                raise
            response = await self._client.post("/embeddings", json=body)
        if response.status_code >= 400:
            raise LLMError(f"Embedding request failed with {response.status_code}: {response.text[:200]}")
        return [item["embedding"] for item in response.json()["data"]]

    async def list_models(self) -> list[str]:
        response = await self._client.get("/models")
        response.raise_for_status()
        return [m["id"] for m in response.json().get("data", [])]

    async def copy_slot(self, source: int, targets) -> int:
        """llama.cpp only: keep what one slot has read (the instructions, after the warm-up) in a file under
        --slot-save-path, and put it in the other slots. Returns how many were filled; elsewhere, or on any
        failure, none, and those slots read the instructions themselves when first used."""
        root = self.base_url.removesuffix("/v1")
        try:
            saved = await self._client.post(f"{root}/slots/{source}", params={"action": "save"},
                                             json={"filename": f"instructions-{self.model}.bin"})
        except httpx.HTTPError:
            return 0
        if saved.status_code >= 400:
            return 0
        return sum([await self.reset_slot(target) for target in targets])

    async def reset_slot(self, slot: int) -> bool:
        """Put a slot back to holding only the instructions (the file copy_slot kept). All slots share one
        context, so a finished task's conversation left in its slot crowds the running ones: two stale
        conversations of 15-18K tokens each filled 32K and stopped a task. Restoring is about 0.1 s; a slot
        the server cannot fill (no file, another runtime) is left as it is."""
        root = self.base_url.removesuffix("/v1")
        try:
            done = await self._client.post(f"{root}/slots/{slot}", params={"action": "restore"},
                                           json={"filename": f"instructions-{self.model}.bin"})
        except httpx.HTTPError:
            return False
        return done.status_code < 400

    async def aclose(self) -> None:
        await self._client.aclose()
