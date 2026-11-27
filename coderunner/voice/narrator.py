"""Speak while a task works, so the owner hears what it is doing without watching.

Subscribes to the daemon's EventBus. For a conversation the owner talks in (Talk follows it) or every task when
spoken replies are on, it says: a short opening line, what it is doing now (the task's own status lines and plan
steps, at most one every GAP_S, never the same twice), findings as they come, the moment it needs the owner (an
approval, a page they must finish), "Got it" when it reads what they said meanwhile, and the answer. Lines are
spoken by LFM3-Omni and kept as ephemeral PCM for the page; nothing is written to SQLite.
"""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import re
import time

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

GAP_S = 10.0                    # at least this long between two progress lines
FOLLOW_S = 30 * 60              # a Talk conversation is followed this long after its last word
FINAL = frozenset({"done", "failed", "cancelled"})
NEEDS_YOU = frozenset({"approval.requested", "browser.handoff"})
ACTIVITY = NEEDS_YOU | {"agent.pivot"}
SPOKEN = frozenset({"started", "note", "interim", "plan", "steer_taken", "activity"}) | FINAL
STATUS = "_status"              # the narrator's own queued progress line


def first_sentence(text: str, limit: int = 160) -> str:
    text = clean_for_speech(text or "").replace("\n", " ").strip()
    text = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def doing_step(steps) -> str:
    """The plan step marked doing, from a plan event (a list of dicts, or its text as the event carries it)."""
    if isinstance(steps, str):
        try:
            steps = ast.literal_eval(steps)
        except (ValueError, SyntaxError, MemoryError, RecursionError):
            try:
                steps = json.loads(steps)
            except ValueError:
                return ""
    for item in steps if isinstance(steps, list) else []:
        if isinstance(item, dict) and item.get("status") == "doing":
            return str(item.get("step") or "")
    return ""


class TaskSpeaker:
    def __init__(self, daemon):
        self.daemon = daemon
        self.config = daemon.config
        self.runtime = VoiceRuntime(self.config)
        self.slot = OmniSlot()
        self.turns = VoiceTurns(daemon, self.runtime)
        self.speech = SpeechStore()
        self.pool = ModelPool(self.config)
        self.following: dict[str, float] = {}   # conversation -> until when Talk follows it
        self._queue: asyncio.Queue | None = None
        self._tasks: list[asyncio.Task] = []
        self._running: set[str] = set()         # conversations with a task at work
        self._status: dict[str, str] = {}       # conversation -> latest progress line not yet said
        self._said: dict[str, tuple[float, str]] = {}   # conversation -> (when, what) it last said
        self._step: dict[str, str] = {}         # conversation -> the plan step last put in a line
        self._spoken: dict[str, str] = {}       # conversation -> answer text already said (interim)
        self._subs: set[asyncio.Queue] = set()

    def start(self) -> None:
        if self._tasks:
            return
        self._queue = asyncio.Queue()
        self._tasks = [asyncio.create_task(self._run()), asyncio.create_task(self._pulse())]

    def stop(self) -> None:
        for task in self._tasks:
            task.cancel()
        self._tasks = []
        self._queue = None

    def subscribe(self) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue()
        self._subs.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        self._subs.discard(queue)

    # ---- who is listening ----------------------------------------------------------------------------------
    def follow(self, conversation: str) -> None:
        if conversation:
            self.following[conversation] = time.time() + FOLLOW_S

    def unfollow(self, conversation: str) -> None:
        self.following.pop(conversation, None)
        self._status.pop(conversation, None)

    def wants(self, conversation: str) -> bool:
        if self.config.get("voice.speak_tasks"):
            return True
        until = self.following.get(conversation)
        if until is None:
            return False
        if until < time.time():
            self.following.pop(conversation, None)
            return False
        return True

    def on_event(self, event: dict) -> None:
        """Called from EventBus.publish (sync): cheap checks, then the work is queued, never done here."""
        kind = event.get("type")
        if kind not in SPOKEN or self._queue is None:
            return
        if kind == "activity" and event.get("event") not in ACTIVITY:
            return
        if not self.config.get("voice.omni_model") or not self.wants(str(event.get("conversation") or "")):
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

    # ---- the work ------------------------------------------------------------------------------------------
    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            event = await self._queue.get()
            try:
                await self._handle(event)
            except asyncio.CancelledError:
                raise
            except VoiceBusy as exc:
                log.info("voice line skipped: %s", exc)
            except Exception as exc:
                log.warning("voice line failed: %s", exc)

    async def _pulse(self) -> None:
        """Puts the latest progress line of each working conversation in the queue once GAP_S has passed."""
        while True:
            await asyncio.sleep(1.0)
            if self._queue is None:
                continue
            now = time.time()
            for conversation, text in list(self._status.items()):
                if conversation not in self._running:
                    self._status.pop(conversation, None)
                elif now - self._said.get(conversation, (0.0, ""))[0] >= GAP_S and not self.slot.busy:
                    self._status.pop(conversation, None)
                    self._queue.put_nowait({"type": STATUS, "conversation": conversation, "text": text})

    async def _handle(self, event: dict) -> None:
        kind = event.get("type")
        conversation = str(event.get("conversation") or "")
        if kind in FINAL:
            self._running.discard(conversation)
            self._status.pop(conversation, None)
            self._step.pop(conversation, None)
            try:
                await self._final(conversation, event)
            finally:
                self._spoken.pop(conversation, None)
            return
        if kind == "started":
            self._running.add(conversation)
            self._spoken.pop(conversation, None)
        if conversation in self.following:
            self.follow(conversation)       # still talking with it: followed on
        if not self._may_speak_mid_task():
            return
        if kind == "started":
            prompt = str(event.get("prompt") or "")
            if prompt:
                await self._say(conversation, "opening", prompt=prompt)
        elif kind == STATUS:
            if conversation in self._running and self.wants(conversation):
                await self._say(conversation, "progress", text=str(event.get("text") or ""))
        elif kind == "note":
            text = str(event.get("text") or "").strip()
            if not text:
                return
            if event.get("tell_owner") or FINDING.search(text):
                if len(text) > 200:
                    await self._say(conversation, "finding", prompt=text[:800], context=f"Finding: {text[:800]}")
                else:
                    await self._say(conversation, "finding", text=text)
            else:
                self._status[conversation] = first_sentence(text)
        elif kind == "plan":
            step = doing_step(event.get("steps"))
            if step and step != self._step.get(conversation) and conversation not in self._status:
                self._step[conversation] = step
                self._status[conversation] = first_sentence(step)
        elif kind == "steer_taken":
            await self._say(conversation, "heard", text="Got it.")
        elif kind == "interim":
            text = str(event.get("text") or "").strip()
            if text:
                await self._answer(conversation, text, kind="interim")
        elif kind == "activity":
            name = event.get("event")
            if name == "approval.requested":
                summary = first_sentence(str(event.get("summary") or "something"), 200)
                await self._say(conversation, "needs_you", text=f"I need your OK. {summary}")
            elif name == "browser.handoff":
                reason = first_sentence(str(event.get("reason") or "this page needs you"), 200)
                await self._say(conversation, "needs_you", text=f"Over to you on the screen. {reason}")
            elif name == "agent.pivot" and event.get("host"):
                self._status[conversation] = f"{event['host']} is blocking me, so I am trying another way."

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
        """Lines while a task works need the voice model beside the assistant's (not taking turns with it)."""
        return self.pool.can_speak_during_task()

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
        said within a few seconds (Listen holds the model, or the assistant needs the memory) is dropped."""
        if text and text == self._said.get(conversation, (0.0, ""))[1]:
            return None                     # never the same line twice in a row
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
            meta = await self._tts(text, conversation=conversation, kind=kind)
        self._said[conversation] = (time.time(), text)
        return meta

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
