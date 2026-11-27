"""Client helpers for the LFM3-Omni llama.cpp sidecar: phrasing, TTS, and slot locking."""

from __future__ import annotations

import asyncio
import base64
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


NARRATE_SYSTEM = (
    "You are the owner's assistant. You are doing a task for them on the computer and talking with them while "
    "you work, the way a helpful friend on a call would. Write the one thing you would say out loud now: first "
    "person, warm and natural, contractions, at most 15 words. Vary your wording and never open the way your "
    "earlier lines did. No quotes, markdown, lists, links or emoji, and no stock phrases like Certainly or "
    "Great question. Use the language the request is in. If there is nothing worth saying, reply NOTHING."
)

NEWS = {
    "opening": "You are starting now. Say briefly what you will look at first.",
    "progress": "What you are doing now: {text}",
    "finding": "You just found this: {text}. Tell them the key fact.",
    "needs_you": "You need the owner to do something: {text}. Tell them plainly and kindly what to do.",
    "heard": "The owner just told you: {text}. Acknowledge it in a few words.",
    "waiting": "Nothing new to report yet; you are still on it. Reassure them in a few words, or reply NOTHING.",
}


async def narrate(runtime: VoiceRuntime, *, request: str, kind: str, text: str = "", said=(),
                  max_tokens: int = 40) -> str | None:
    """One natural spoken line about the task, knowing what was asked and what was already said; None when
    there is nothing worth saying."""
    await asyncio.to_thread(runtime.ensure)
    earlier = " / ".join(said) if said else "nothing yet"
    news = NEWS.get(kind, NEWS["progress"]).format(text=text[:400])
    messages = [{"role": "system", "content": NARRATE_SYSTEM},
                {"role": "user", "content": f"Request: {request[:400]}\nYou said so far: {earlier}\nNow: {news}"}]
    async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
        response = await client.post(
            f"{runtime.base_url}/chat/completions",
            json={"model": runtime.alias(), "messages": messages, "max_tokens": max_tokens, "temperature": 0.7,
                  "chat_template_kwargs": {"enable_thinking": False}},
        )
        response.raise_for_status()
        line = ((response.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""
    line = clean_for_speech(line).strip().strip('"').strip("'").split("\n")[0].strip()
    if not line or line.upper().startswith("NOTHING"):
        return None
    return line[:200]


HEAR_SYSTEM = "Transcribe the user's speech exactly, word for word. Reply with the transcript only."
TALK_SYSTEM = "You are the voice of the owner's assistant on a live call. Reply exactly like the examples: a SAY line and a DO line."
# Worked examples rather than rules: a 2B model follows these closely (measured: right DO on every case tried,
# corrections said the right way round, about 0.3 s a reply once this prefix is cached).
TALK_SHOTS = [
 ("Status: nothing in progress\nThey said: Find me a cheap flight from Delhi to Goa next Friday.", "SAY: Sure, looking for cheap flights from Delhi to Goa next Friday.\nDO: yes"),
 ("Status: working on cheap flights from Delhi to Goa. Latest: comparing fares on MakeMyTrip.\nThey said: How's it going?", "SAY: Getting there, I'm comparing fares on MakeMyTrip right now.\nDO: no"),
 ("Status: working on hotels in Jaipur. Latest: reading reviews.\nThey said: Actually, make it Udaipur.", "SAY: Got it, Udaipur instead of Jaipur.\nDO: yes"),
 ("Status: working on a table at Olive in Bandra. Latest: opening the booking page.\nThey said: No, I said Colaba, not Bandra.", "SAY: Sorry about that, Colaba it is.\nDO: yes"),
 ("Status: done. Found two hotels under 5,000 rupees.\nThey said: Thank you, that's all.", "SAY: You're welcome! Just say if you need anything else.\nDO: no"),
 ("Status: nothing in progress\nThey said: Kal Mumbai ka mausam batao.", "SAY: Haan, Mumbai ka kal ka mausam dekh rahi hoon.\nDO: yes"),
]


async def _complete(runtime: VoiceRuntime, messages: list, max_tokens: int, temperature: float) -> str:
    async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
        response = await client.post(
            f"{runtime.base_url}/chat/completions",
            json={"model": runtime.alias(), "messages": messages, "max_tokens": max_tokens,
                  "temperature": temperature, "chat_template_kwargs": {"enable_thinking": False}},
        )
        response.raise_for_status()
        return ((response.json().get("choices") or [{}])[0].get("message") or {}).get("content") or ""


async def hear(runtime: VoiceRuntime, wav: bytes) -> str:
    """What the owner said, heard by LFM3-Omni itself (16 kHz mono WAV): better on names of places and services
    than the dictation engines (IRCTC, Indiranagar, Patna, measured on the same recordings)."""
    await asyncio.to_thread(runtime.ensure)
    audio = {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode(), "format": "wav"}}
    text = await _complete(runtime, [{"role": "system", "content": HEAR_SYSTEM},
                                     {"role": "user", "content": [audio]}], 120, 0.0)
    return " ".join(text.split()).strip().strip('"')


# LFM3-Omni is a chat model: asked for a transcript it sometimes answers instead ("Sure, I can check the quick
# commerce websites for you. Which ones are you looking at?"). Words only the assistant would say to the owner:
_REPLY = re.compile(
    r"\bfor you\b|\bhow (?:can|may) i help\b|\bi can help\b|\bi(?:'d| would) be (?:happy|glad)\b|\bhappy to help\b"
    r"|\bas an ai\b|\bi(?:'m| am) an ai\b|\bi (?:can't|cannot|couldn't) (?:hear|understand)\b|\bwould you like\b"
    r"|\bwhich ones? are you\b|\bare you looking (?:at|for)\b|\bwhat would you like\b|\bis there anything else\b"
    r"|(?:^|[.!?]\s+)(?:sure|certainly|of course|absolutely|no problem|okay|ok|alright|got it)[,!.]?\s+"
    r"i(?:'ll| will| can|'m| am)\b",
    re.I)
_WORD = re.compile(r"[\w']+")


def reply_shaped(text: str) -> bool:
    """Whether a "transcript" reads as the assistant answering the owner rather than the owner's words."""
    return bool(_REPLY.search(text or ""))


def pick_heard(omni: str, dictated: str) -> str:
    """The owner's words for a Talk turn, from what LFM3-Omni heard and what the dictation engine wrote of the
    same recording. Omni's, which gets names of places and services right, when it is a transcript: it does not
    read as a reply, and it says nearly what dictation heard (a few words apart, misheard names, not a sentence
    more). Otherwise dictation's; with no dictation, Omni's unless it is a reply (then nothing: ask again)."""
    omni, dictated = " ".join((omni or "").split()), " ".join((dictated or "").split())
    if not omni:
        return dictated
    if not dictated:
        return "" if reply_shaped(omni) else omni
    if reply_shaped(omni):
        return dictated
    said, heard = _WORD.findall(omni.lower()), _WORD.findall(dictated.lower())
    known = set(heard)
    extra = sum(1 for word in said if word not in known)
    if extra > max(2, len(said) // 3) or len(said) > len(heard) + max(3, len(heard) // 2):
        return dictated
    return omni


async def talk_reply(runtime: VoiceRuntime, heard: str, status: str) -> tuple[str, bool]:
    """What to say back at once, and whether there is work for the assistant (a new task, or a change to the one
    at work); progress questions, thanks and small talk are answered here and go no further."""
    messages = [{"role": "system", "content": TALK_SYSTEM}]
    for asked, answered in TALK_SHOTS:
        messages += [{"role": "user", "content": asked}, {"role": "assistant", "content": answered}]
    messages.append({"role": "user", "content": f"Status: {status}\nThey said: {heard}"})
    text = await _complete(runtime, messages, 60, 0.4)
    say = re.search(r"SAY:\s*(.+)", text)
    do = re.search(r"DO:\s*(\w+)", text)
    line = clean_for_speech(say.group(1)).strip() if say else ""
    return line[:200], (do.group(1).lower() == "yes") if do else True


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
