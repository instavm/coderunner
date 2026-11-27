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
    clean_for_speech,
    looks_like_table,
    phrase,
    speech_lang,
    synthesize,
)

log = logging.getLogger("coderunner.voice.narrator")

COALESCE_S = 8.0
FINAL = frozenset({"done", "failed", "cancelled"})


class TaskSpeaker:
    def __init__(self, daemon):
        self.daemon = daemon
        self.config = daemon.config
        self.runtime = VoiceRuntime(self.config)
        self.slot = OmniSlot()
        self.speech = SpeechStore()
        self.pool = ModelPool(self.config)
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None
        self._spoken: dict[str, str] = {}       # conversation -> last spoken text
        self._last_finding_ts: dict[str, float] = {}
        self._pending_finding: dict[str, str] = {}
        self._utterances: asyncio.Queue = asyncio.Queue()  # for SSE/subscribers
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
        if not self.config.get("voice.speak_tasks"):
            return
        if not (self.config.get("voice.omni_model") and self.runtime.ready()):
            return
        kind = event.get("type")
        if kind not in ("started", "note", "interim", "done", "failed", "cancelled"):
            return
        if self._queue is not None:
            self._queue.put_nowait(dict(event))

    async def speak_text(self, text: str, *, conversation: str = "", kind: str = "reply") -> dict | None:
        """On-demand Listen / CLI: phrase cleanup + TTS. Returns speech metadata for the client."""
        cleaned = clean_for_speech(text or "").strip()
        if not cleaned:
            return None
        if looks_like_table(cleaned):
            said = await self._phrase(prompt=cleaned[:500], context="Summarize this table in one or two spoken sentences.")
            if not said:
                return None
            cleaned = said
        return await self._tts(cleaned, conversation=conversation, kind=kind)

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            try:
                await self._handle(event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("voice aside failed: %s", exc)

    async def _handle(self, event: dict) -> None:
        kind = event.get("type")
        conversation = str(event.get("conversation") or "")
        if kind == "started":
            prompt = str(event.get("prompt") or "")
            if not prompt or not self._may_speak_mid_task():
                return
            said = await self._phrase(prompt=prompt)
            if said:
                await self._tts(said, conversation=conversation, kind="opening")
            return
        if kind == "note":
            text = str(event.get("text") or "").strip()
            if not text or not self._may_speak_mid_task():
                return
            if event.get("tell_owner") or FINDING.search(text):
                await self._finding(conversation, text)
            return
        if kind == "interim":
            text = str(event.get("text") or "").strip()
            if not text or not self._may_speak_mid_task():
                return
            await self._answer(conversation, text, kind="interim")
            return
        if kind in FINAL:
            text = str(event.get("reply") or "").strip()
            if not text:
                return
            # Exclusive Macs: speak only after the task ends (chat slot free).
            if self.pool.role_plan().omni_exclusive and not self.pool.can_speak_during_task():
                await self._answer(conversation, text, kind="done")
                return
            already = self._spoken.get(conversation) or ""
            if already and text.strip() == already.strip():
                return
            if already and text.startswith(already):
                text = text[len(already):].strip()
            if text:
                await self._answer(conversation, text, kind=kind)
            self._spoken.pop(conversation, None)
            self._last_finding_ts.pop(conversation, None)
            self._pending_finding.pop(conversation, None)

    def _may_speak_mid_task(self) -> bool:
        return self.pool.can_speak_during_task() or not self.pool.role_plan().omni_exclusive

    async def _finding(self, conversation: str, text: str) -> None:
        now = time.time()
        last = self._last_finding_ts.get(conversation) or 0
        self._pending_finding[conversation] = text
        if now - last < COALESCE_S:
            return
        self._last_finding_ts[conversation] = now
        pending = self._pending_finding.pop(conversation, text)
        said = await self._phrase(prompt=pending, context=f"Finding: {pending}")
        if said:
            await self._tts(said, conversation=conversation, kind="finding")

    async def _answer(self, conversation: str, text: str, *, kind: str) -> None:
        cleaned = clean_for_speech(text)
        if looks_like_table(cleaned):
            said = await self._phrase(prompt=cleaned[:800],
                                      context="Summarize what this table shows in one or two spoken sentences.")
            if not said:
                return
            cleaned = said
        if not cleaned:
            return
        self._spoken[conversation] = text
        await self._tts(cleaned, conversation=conversation, kind=kind)

    async def _phrase(self, *, prompt: str, context: str = "") -> str | None:
        if self.slot.busy:
            return None
        try:
            await self.slot.acquire("phrasing")
            return await phrase(self.runtime, prompt=prompt, context=context)
        except Exception as exc:
            log.info("phrase skipped: %s", exc)
            return None
        finally:
            self.slot.release()

    async def _tts(self, text: str, *, conversation: str, kind: str) -> dict | None:
        if self.slot.busy:
            # Wait briefly for Talk to finish rather than dropping a final answer.
            for _ in range(40):
                if not self.slot.busy:
                    break
                await asyncio.sleep(0.25)
            if self.slot.busy:
                return None
        try:
            await self.slot.acquire(f"speaking:{kind}")
            voice = self.config.get("voice.voice") or "female"
            pcm, rate = await synthesize(self.runtime, text, voice=voice, lang=speech_lang(self.config))
        except Exception as exc:
            log.info("tts skipped: %s", exc)
            return None
        finally:
            self.slot.release()
        sid = self.speech.put(text=text, pcm=pcm, rate=rate, conversation=conversation, kind=kind)
        meta = {"id": sid, "text": text, "rate": rate, "conversation": conversation, "kind": kind,
                "bytes": len(pcm)}
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
