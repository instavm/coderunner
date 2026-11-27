"""Client helpers for the LFM3-Omni llama.cpp sidecar: phrasing, TTS, and slot locking."""

from __future__ import annotations

import asyncio
import re
import secrets
import time
from typing import AsyncIterator

import httpx

from ..models.runtimes import VoiceRuntime

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


class OmniSlot:
    """One at a time on the voice server's single llama.cpp slot (Talk vs TTS)."""

    def __init__(self):
        self._lock = asyncio.Lock()
        self.busy_reason = ""

    @property
    def busy(self) -> bool:
        return self._lock.locked()

    async def acquire(self, reason: str, *, wait: bool = False):
        if self._lock.locked() and not wait:
            raise RuntimeError(self.busy_reason or "The voice model is busy.")
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
    runtime.ensure()
    messages = [
        {"role": "system", "content": PHRASE_SYSTEM},
        {"role": "user", "content": (f"Request: {prompt}\n\nWhat just happened: {context}"
                                     if context else f"Request: {prompt}\n\nSay what you will do.")},
    ]
    async with httpx.AsyncClient(timeout=60.0, trust_env=False) as client:
        response = await client.post(
            f"{runtime.base_url}/chat/completions",
            json={"model": (runtime._state().get("active") or {}).get("alias") or "lfm3-omni-2b",
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
    runtime.ensure()
    payload = {"input": text, "voice": voice if voice in ("female", "male") else "female",
               "lang": lang or "en", "response_format": "pcm16", "stream": stream}
    async with httpx.AsyncClient(timeout=120.0, trust_env=False) as client:
        response = await client.post(runtime.tts_url(), json=payload)
        response.raise_for_status()
        rate = _rate_from_content_type(response.headers.get("content-type") or "")
        return response.content, rate


async def synthesize_stream(runtime: VoiceRuntime, text: str, *, voice: str = "female",
                            lang: str = "en") -> AsyncIterator[bytes]:
    runtime.ensure()
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
