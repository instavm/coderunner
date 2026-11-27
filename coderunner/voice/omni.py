"""Client helpers for the LFM3-Omni llama.cpp sidecar: phrasing, TTS, and slot locking."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import secrets
import time
from typing import AsyncIterator

import httpx

from ..models.runtimes import VoiceRuntime

log = logging.getLogger("coderunner.voice.omni")

# Markdown the TTS model would otherwise read aloud as punctuation.
_BOLD = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)|_(.+?)_")
_HEADING = re.compile(r"^#{1,6}\s+", re.M)
_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_CODE = re.compile(r"`([^`]+)`")
_TABLE_ROW = re.compile(r"^\|.+\|$", re.M)
_LIST_MARK = re.compile(r"^[\s]*[-*+]\s+", re.M)
_URL = re.compile(r"https?://\S+")

PHRASE_SYSTEM = (
    "You write one short sentence a person would say out loud while working. "
    "Use the same language as the request. No quotes, no markdown, no lists. "
    "If nothing new is worth saying, reply with NOTHING."
)


def clean_for_speech(text: str) -> str:
    """Strip markdown markers so TTS does not read asterisks and raw URLs."""
    text = _LINK.sub(r"\1", text or "")
    text = _BOLD.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _ITALIC.sub(lambda m: m.group(1) or m.group(2) or "", text)
    text = _CODE.sub(r"\1", text)
    text = _HEADING.sub("", text)
    text = _URL.sub("", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    return text


def looks_like_table(text: str) -> bool:
    rows = _TABLE_ROW.findall(text or "")
    return len(rows) >= 3 or (len(rows) >= 2 and bool(_LIST_MARK.search(text or "")))


BUSY = "Voice is in use. Try again in a moment."


class VoiceBusy(RuntimeError):
    """The voice model cannot be used right now (another use holds it, or the assistant needs the memory)."""


class OmniSlot:
    """One spoken line at a time on the voice server's single llama.cpp slot."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.busy_reason = ""

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    async def acquire(self, reason: str, *, wait: bool = False):
        if self._lock.locked() and not wait:
            raise VoiceBusy(BUSY)
        await self._lock.acquire()
        self.busy_reason = reason
        return self

    def release(self) -> None:
        self.busy_reason = ""
        if self._lock.locked():
            self._lock.release()

    def hold(self, reason: str, *, wait: bool = False):
        """Async context manager: `async with slot.hold("talk"):`."""
        slot = self

        class _Hold:
            async def __aenter__(self):
                await slot.acquire(reason, wait=wait)
                return slot

            async def __aexit__(self, *exc):
                slot.release()

        return _Hold()


class SpeechStore:
    """Ephemeral PCM clips for the web page (never written to SQLite)."""

    TTL_S = 120

    def __init__(self):
        self._items: dict[str, dict] = {}

    def put(self, *, text: str, pcm: bytes, rate: int = 24000, conversation: str = "",
            kind: str = "aside") -> str:
        self._expire()
        sid = secrets.token_urlsafe(12)
        self._items[sid] = {"text": text, "pcm": pcm, "rate": rate, "conversation": conversation,
                            "kind": kind, "ts": time.time()}
        return sid

    def get(self, sid: str) -> dict | None:
        self._expire()
        return self._items.get(sid)

    def pop(self, sid: str) -> dict | None:
        self._expire()
        return self._items.pop(sid, None)

    def _expire(self) -> None:
        cutoff = time.time() - self.TTL_S
        for key in [k for k, v in self._items.items() if v["ts"] < cutoff]:
            self._items.pop(key, None)


async def phrase(runtime: VoiceRuntime, *, prompt: str, context: str = "",
                 max_tokens: int = 48) -> str | None:
    """One natural spoken sentence, or None when the model says NOTHING."""
    await asyncio.to_thread(runtime.ensure)
    messages = [
        {"role": "system", "content": PHRASE_SYSTEM},
        {"role": "user", "content": (f"Request: {prompt}\n\nWhat just happened: {context}"
                                     if context else f"Request: {prompt}\n\nSay what you will do.")},
    ]
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        response = await client.post(
            f"{runtime.base_url}/chat/completions",
            json={"model": runtime.alias(),
                  "messages": messages, "max_tokens": max_tokens, "temperature": 0.2,
                  "chat_template_kwargs": {"enable_thinking": False}},
        )
        response.raise_for_status()
        text = ((response.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    text = clean_for_speech(text).strip().strip('"').strip("'")
    if not text or text.upper() == "NOTHING" or text.upper().startswith("NOTHING"):
        return None
    return text.split("\n")[0][:280]


async def synthesize(runtime: VoiceRuntime, text: str, *, voice: str = "female",
                     lang: str = "en", stream: bool = False) -> tuple[bytes, int]:
    """Render text to PCM16 mono via POST /tts. Returns (pcm_bytes, sample_rate)."""
    await asyncio.to_thread(runtime.ensure)
    payload = {"input": text, "voice": voice if voice in ("female", "male") else "female",
               "lang": lang or "en", "response_format": "pcm16", "stream": stream}
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        response = await client.post(runtime.tts_url(), json=payload)
        response.raise_for_status()
        rate = _rate_from_content_type(response.headers.get("content-type") or "")
        return response.content, rate


async def synthesize_stream(runtime: VoiceRuntime, text: str, *, voice: str = "female",
                            lang: str = "en") -> AsyncIterator[bytes]:
    await asyncio.to_thread(runtime.ensure)
    payload = {"input": text, "voice": voice if voice in ("female", "male") else "female",
               "lang": lang or "en", "response_format": "pcm16", "stream": True}
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        async with client.stream("POST", runtime.tts_url(), json=payload) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk


def _rate_from_content_type(value: str) -> int:
    match = re.search(r"rate=(\d+)", value or "")
    return int(match.group(1)) if match else 24000


def speech_lang(config) -> str:
    raw = str(config.get("voice.language") or "").strip().lower()
    if not raw or raw == "auto":
        return "en"
    return raw.split("-")[0][:8]


class VoiceTurns:
    """Where the voice model and the assistant's model do not both fit in memory (voice.omni_exclusive with the
    assistant's model on this computer), they take turns, as the browser driver does (agent.driver.Swap): the
    assistant's model is unloaded while the voice model is in use and loaded again after. Voice is used only
    when the assistant is idle, and gives way at once when a message comes in (make_way). Elsewhere the voice
    model sits beside the assistant's and frees its memory when idle."""

    def __init__(self, daemon, runtime: VoiceRuntime):
        self.daemon = daemon
        self.runtime = runtime
        self.users = 0
        self.planner = None             # the assistant's runtime while it is unloaded for voice
        self.planner_was_running = False
        self.yielding = asyncio.Event()  # set: the assistant needs its model back; voice users finish now
        self._restored = asyncio.Event()
        self._restored.set()
        self._lock = asyncio.Lock()

    @property
    def config(self):
        return self.daemon.config

    def exclusive(self) -> bool:
        if not self.config.get("voice.omni_exclusive"):
            return False
        if not self.config.get("models.chat"):
            return False                # search mode: no model of the assistant's on this computer
        remote = self.config.get("models.remote") or {}
        return not remote.get("provider")

    def assistant_busy(self) -> bool:
        tasks = getattr(self.daemon, "tasks", None)
        if tasks is not None and (getattr(tasks, "running", None) or getattr(tasks, "waiting", None)):
            return True
        turns = getattr(getattr(self.daemon, "svc", None), "turns", None)
        return bool(getattr(turns, "_busy", False))

    @contextlib.asynccontextmanager
    async def use(self):
        """Hold the voice model for one use (one spoken line, or a phrased line and its speech)."""
        if not self.exclusive():
            await asyncio.to_thread(self.runtime.ensure)
            yield self
            return
        async with self._lock:
            if self.yielding.is_set() or (self.users == 0 and self.assistant_busy()):
                raise VoiceBusy("The assistant is working. Voice is back when it is done.")
            self.users += 1
            try:
                if self.planner is None:
                    self._restored.clear()
                    await asyncio.to_thread(self._unload_planner)
                await asyncio.to_thread(self.runtime.ensure)
            except BaseException:
                self.users -= 1
                if self.users == 0:
                    await self._restore()
                raise
        try:
            yield self
        finally:
            async with self._lock:
                self.users -= 1
                if self.users == 0:
                    await self._restore()

    async def make_way(self, timeout: float = 120) -> None:
        """Before the assistant's model is asked: voice users end, and its model is loaded again."""
        if self._restored.is_set():
            return
        self.yielding.set()
        with contextlib.suppress(asyncio.TimeoutError):
            await asyncio.wait_for(self._restored.wait(), timeout)

    def _unload_planner(self) -> None:
        from ..models.runtimes import choose_runtime
        planner = choose_runtime(self.config)
        if planner is None:
            self.planner = False
            return
        self.planner_was_running = bool(planner.running())
        if getattr(planner, "name", "") == "ollama":
            # Ollama also serves the embedding model, so only the chat model is unloaded.
            unload = getattr(planner, "unload", None)
            llm = getattr(getattr(self.daemon, "svc", None), "llm", None)
            if unload and getattr(llm, "model", None):
                unload(llm.model)
        elif self.planner_was_running:
            planner.stop()
        self.planner = planner

    async def _restore(self) -> None:
        planner, self.planner = self.planner, None
        try:
            await asyncio.to_thread(self.runtime.stop)
            if planner and self.planner_was_running and getattr(planner, "name", "") != "ollama":
                await asyncio.to_thread(planner.start)
                # Started again, the server has nothing read: the tasks' slots get their instructions back.
                svc = getattr(self.daemon, "svc", None)
                reset = getattr(getattr(svc, "llm", None), "reset_slot", None)
                if reset is not None and getattr(svc, "turns", None) is not None:
                    from ..server.tasks import limit
                    for slot in range(limit(svc)):
                        with contextlib.suppress(Exception):
                            await reset(slot)
        except Exception as exc:
            log.warning("could not load the assistant's model again after voice: %s", exc)
        finally:
            self.planner_was_running = False
            self.yielding.clear()
            self._restored.set()
