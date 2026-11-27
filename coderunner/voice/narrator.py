"""Speak natural asides while a task runs, without blocking the planner.

Subscribes to the daemon EventBus. Phrases short sentences on LFM3-Omni and stores
ephemeral PCM for the web page. Never writes audio into SQLite task_events.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from ..agent.loop import FINDING
from ..models.pool import ModelPool
from ..models.runtimes import VoiceRuntime
from .omni import (
    OmniSlot,
    SpeechStore,
    VoiceBusy,
    VoiceTurns,
    clean_for_speech,
    looks_like_table,
    phrase,
    speech_lang,
    synthesize,
)

log = logging.getLogger("coderunner.voice.narrator")

COALESCE_S = 8.0
FINAL = frozenset({"done", "failed", "cancelled"})
SPOKEN = frozenset({"started", "note", "interim"}) | FINAL


class TaskSpeaker:
    def __init__(self, daemon):
        self.daemon = daemon
        self.config = daemon.config
        self.runtime = VoiceRuntime(self.config)
        self.slot = OmniSlot()
        self.turns = VoiceTurns(daemon, self.runtime)
        self.speech = SpeechStore()
        self.pool = ModelPool(self.config)
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._spoken: dict[str, str] = {}       # conversation -> last spoken text
        self._last_finding_ts: dict[str, float] = {}
        self._pending_finding: dict[str, str] = {}
        self._subs: set[asyncio.Queue] = set()

    def start(self) -> None:
        if self._task is not None:
            return
        self._queue = asyncio.Queue()
        self._task = asyncio.create_task(self._run())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None
        self._queue = None

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subs.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subs.discard(queue)

    def on_event(self, event: dict) -> None:
        """Called from EventBus.publish path (sync); schedules work without blocking the agent."""
        if event.get("type") not in SPOKEN or self._queue is None:
            return
        if not (self.config.get("voice.speak_tasks") and self.config.get("voice.omni_model")):
            return
        if not self.runtime.ready():
            return
        self._queue.put_nowait(dict(event))

    async def speak_text(self, text: str, *, conversation: str = "", kind: str = "reply") -> dict | None:
        """On-demand Listen / CLI: phrase cleanup + TTS. Returns speech metadata for the client, which plays it
        itself (no live event, so it is not heard twice)."""
        cleaned = clean_for_speech(text or "").strip()
        if not cleaned:
            return None
        async with self.slot.hold(f"speaking:{kind}"), self.turns.use():
            if looks_like_table(cleaned):
                said = await self._phrase(prompt=cleaned[:500],
                                          context="Summarize this table in one or two spoken sentences.")
                if not said:
                    return None
                cleaned = said
            return await self._tts(cleaned, conversation=conversation, kind=kind, announce=False)

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            try:
                await self._handle(event)
            except asyncio.CancelledError:
                raise
            except VoiceBusy as exc:
                log.info("voice aside skipped: %s", exc)
            except Exception as exc:
                log.warning("voice aside failed: %s", exc)

    async def _handle(self, event: dict) -> None:
        kind = event.get("type")
        conversation = str(event.get("conversation") or "")
        if kind in FINAL:
            try:
                await self._final(conversation, event)
            finally:
                self._spoken.pop(conversation, None)
                self._last_finding_ts.pop(conversation, None)
                self._pending_finding.pop(conversation, None)
            return
        if not self._may_speak_mid_task():
            return
        if kind == "started":
            prompt = str(event.get("prompt") or "")
            if prompt:
                await self._say(conversation, "opening", prompt=prompt)
        elif kind == "note":
            text = str(event.get("text") or "").strip()
            if text and (event.get("tell_owner") or FINDING.search(text)):
                await self._finding(conversation, text)
        elif kind == "interim":
            text = str(event.get("text") or "").strip()
            if text:
                await self._answer(conversation, text, kind="interim")

    async def _final(self, conversation: str, event: dict) -> None:
        text = str(event.get("reply") or "").strip()
        if not text:
            return
        already = (self._spoken.get(conversation) or "").strip()
        if already and text == already:
            return
        if already and text.startswith(already):
            text = text[len(already):].strip()
        if text:
            await self._answer(conversation, text, kind=str(event.get("type") or "done"))

    def _may_speak_mid_task(self) -> bool:
        """Asides while a task runs need the voice model beside the assistant's (not taking turns with it)."""
        return self.pool.can_speak_during_task()

    async def _finding(self, conversation: str, text: str) -> None:
        now = time.time()
        last = self._last_finding_ts.get(conversation) or 0
        self._pending_finding[conversation] = text
        if now - last < COALESCE_S:
            return
        self._last_finding_ts[conversation] = now
        pending = self._pending_finding.pop(conversation, text)
        await self._say(conversation, "finding", prompt=pending, context=f"Finding: {pending}")

    async def _answer(self, conversation: str, text: str, *, kind: str) -> None:
        cleaned = clean_for_speech(text)
        if not cleaned:
            return
        self._spoken[conversation] = text
        if looks_like_table(cleaned):
            await self._say(conversation, kind, prompt=cleaned[:800],
                            context="Summarize what this table shows in one or two spoken sentences.")
        else:
            await self._say(conversation, kind, text=cleaned)

    async def _say(self, conversation: str, kind: str, *, text: str = "", prompt: str = "",
                   context: str = "") -> dict | None:
        """One spoken line: phrased by the model from `prompt` when no `text` is given. A line that cannot be
        said within a few seconds (Talk holds the model, or the assistant needs the memory) is dropped."""
        for _ in range(40):
            if not self.slot.busy:
                break
            await asyncio.sleep(0.25)
        if self.slot.busy:
            return None
        async with self.slot.hold(f"speaking:{kind}"), self.turns.use():
            if not text:
                text = await self._phrase(prompt=prompt, context=context) or ""
                if not text:
                    return None
            return await self._tts(text, conversation=conversation, kind=kind)

    async def _phrase(self, *, prompt: str, context: str = "") -> str | None:
        try:
            return await phrase(self.runtime, prompt=prompt, context=context)
        except Exception as exc:
            log.info("phrase skipped: %s", exc)
            return None

    async def _tts(self, text: str, *, conversation: str, kind: str, announce: bool = True) -> dict | None:
        voice = self.config.get("voice.voice") or "female"
        pcm, rate = await synthesize(self.runtime, text, voice=voice, lang=speech_lang(self.config))
        sid = self.speech.put(text=text, pcm=pcm, rate=rate, conversation=conversation, kind=kind)
        meta = {"id": sid, "text": text, "rate": rate, "conversation": conversation, "kind": kind,
                "bytes": len(pcm)}
        if announce:
            self._announce(meta)
        return meta

    def _announce(self, meta: dict) -> None:
        # Ephemeral notify for the open app; not persisted to SQLite.
        for queue in list(self._subs):
            if queue.qsize() < 50:
                queue.put_nowait(meta)
        bus = getattr(self.daemon, "bus", None)
        if bus is not None and hasattr(bus, "notify_live"):
            bus.notify_live({"type": "voice_utterance", "conversation": meta.get("conversation"),
                             "utterance": {k: meta[k] for k in ("id", "text", "rate", "kind") if k in meta}})


def attach_bus(bus, speaker: TaskSpeaker) -> None:
    """Wrap EventBus.publish so the narrator sees events without changing call sites."""
    original = bus.publish

    def publish(event: dict) -> dict:
        out = original(event)
        with_context = out if isinstance(out, dict) else event
        try:
            speaker.on_event(with_context)
        except Exception:
            pass
        return out

    bus.publish = publish  # type: ignore[method-assign]
