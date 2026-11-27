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

import httpx

from ..agent.loop import FINDING
from ..models.pool import ModelPool
from collections import deque

from ..models.runtimes import VoiceRuntime
from .omni import (
    OmniSlot,
    SpeechStore,
    VoiceBusy,
    VoiceTurns,
    clean_for_speech,
    hear,
    looks_like_table,
    narrate,
    phrase,
    pick_heard,
    speech_lang,
    synthesize,
    talk_reply,
)

log = logging.getLogger("coderunner.voice.narrator")

GAP_S = 10.0                    # at least this long between two progress lines
QUIET_S = 15.0                  # this long without a word while it works: a short check-in
QUIET_MAX = 2                   # at most this many check-ins in a row without real news
FOLLOW_S = 30 * 60              # a Talk conversation is followed this long after its last word
FINAL = frozenset({"done", "failed", "cancelled"})
NEEDS_YOU = frozenset({"approval.requested", "browser.handoff"})
ACTIVITY = NEEDS_YOU | {"agent.pivot"}
SPOKEN = frozenset({"started", "note", "interim", "plan", "steer", "steer_taken", "activity"}) | FINAL
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
        self._request: dict[str, str] = {}      # conversation -> what the owner asked
        self._recent: dict[str, deque] = {}     # conversation -> the last lines said, so the next is not alike
        self._heard: dict[str, str] = {}        # conversation -> what the owner said while it works
        self._quiet: dict[str, int] = {}        # conversation -> check-ins since the last real news
        self._warming: asyncio.Task | None = None
        self._answered: dict[str, str] = {}     # conversation -> the start of its last answer
        self._acked: set[tuple[str, str]] = set()   # (conversation, "started" | "steer"): Talk already replied
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

    def acknowledge(self, conversation: str, steer: bool) -> None:
        """Talk has already answered this request out loud: no opening line (or "got it") for it as well."""
        self._acked.add((conversation, "steer" if steer else "started"))

    def status_for(self, conversation: str) -> str:
        """Where the work in this conversation stands, in a line, for Talk to answer "how is it going"."""
        request = first_sentence(self._request.get(conversation, ""), 160)
        if conversation in self._running:
            latest = self._status.get(conversation) or self._said.get(conversation, (0.0, ""))[1]
            return f"working on {request}" + (f" Latest: {first_sentence(latest, 120)}" if latest else "")
        if conversation in self._answered:
            return f"done. {self._answered[conversation]}"
        return "nothing in progress"

    async def talk_turn(self, wav: bytes, conversation: str, dictation=None) -> tuple[str, str, bool]:
        """One thing the owner said in Talk: what was heard, what to say back at once, whether there is work.
        dictation: the dictation engine's transcript of the same recording, being written meanwhile (an awaitable
        of a string): the owner's words are Omni's only where they are a transcript, not an answer (pick_heard)."""
        async with self.slot.hold("talk"), self.turns.use():
            try:
                omni = await hear(self.runtime, wav)
            except (httpx.HTTPError, ValueError) as exc:
                if dictation is None:
                    raise
                log.warning("talk: the voice model did not hear it (%s); dictation's words are used", exc)
                omni = ""
            heard = pick_heard(omni, await dictation if dictation is not None else "")
            if not heard:
                return "", "", False
            say, do = await talk_reply(self.runtime, heard, self.status_for(conversation))
        return heard, say, do

    async def say_now(self, text: str, conversation: str, kind: str = "talk") -> dict | None:
        """Speak a line for the page that asked (it plays it itself: no live event)."""
        async with self.slot.hold(f"speaking:{kind}"), self.turns.use():
            meta = await self._tts(text, conversation=conversation, kind=kind, announce=False)
        self._said[conversation] = (time.time(), text)
        self._recent.setdefault(conversation, deque(maxlen=3)).append(text)
        return meta

    def warm(self) -> None:
        """Load the voice model ahead of the first line (Talk opened), where it can sit beside the assistant's:
        otherwise that line waits for the load."""
        if self._warming is not None and not self._warming.done():
            return
        if self.turns.exclusive() or not self.runtime.ready():
            return
        try:
            self._warming = asyncio.get_running_loop().create_task(asyncio.to_thread(self.runtime.ensure))
        except RuntimeError:
            pass

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
            for conversation in list(self._running):
                last = self._said.get(conversation, (0.0, ""))[0]
                if (last and conversation not in self._status and now - last >= QUIET_S and not self.slot.busy
                        and self._quiet.get(conversation, 0) < QUIET_MAX and self.wants(conversation)):
                    self._quiet[conversation] = self._quiet.get(conversation, 0) + 1
                    self._said[conversation] = (now, self._said[conversation][1])   # one check-in per quiet spell
                    self._queue.put_nowait({"type": STATUS, "conversation": conversation, "waiting": True})

    async def _handle(self, event: dict) -> None:
        kind = event.get("type")
        conversation = str(event.get("conversation") or "")
        if kind in FINAL:
            self._running.discard(conversation)
            for kept in (self._status, self._step, self._heard, self._quiet, self._recent):
                kept.pop(conversation, None)
            try:
                await self._final(conversation, event)
            finally:
                self._spoken.pop(conversation, None)
            return
        if kind == "started":
            self._running.add(conversation)
            self._spoken.pop(conversation, None)
            self._request[conversation] = str(event.get("prompt") or "")
            self._quiet.pop(conversation, None)
        if kind == "steer":
            self._heard[conversation] = str(event.get("text") or "")[:300]
            return
        if conversation in self.following:
            self.follow(conversation)       # still talking with it: followed on
        if not self._may_speak_mid_task():
            return
        if kind == "started":
            if (conversation, "started") in self._acked:
                self._acked.discard((conversation, "started"))
            elif self._request.get(conversation):
                await self._say(conversation, "opening", news="")
        elif kind == STATUS:
            if conversation in self._running and self.wants(conversation):
                if event.get("waiting"):
                    await self._say(conversation, "waiting", news="")
                else:
                    self._quiet.pop(conversation, None)
                    await self._say(conversation, "progress", news=str(event.get("text") or ""))
        elif kind == "note":
            text = str(event.get("text") or "").strip()
            if not text:
                return
            if event.get("tell_owner") or FINDING.search(text):
                self._quiet.pop(conversation, None)
                await self._say(conversation, "finding", news=text[:800], fallback=first_sentence(text, 200))
            else:
                self._status[conversation] = first_sentence(text)
        elif kind == "plan":
            step = doing_step(event.get("steps"))
            if step and step != self._step.get(conversation) and conversation not in self._status:
                self._step[conversation] = step
                self._status[conversation] = first_sentence(step)
        elif kind == "steer_taken":
            heard = self._heard.pop(conversation, "")
            if (conversation, "steer") in self._acked:
                self._acked.discard((conversation, "steer"))
                return
            await self._say(conversation, "heard", news=heard or "a change to the request", fallback="Got it.")
        elif kind == "interim":
            text = str(event.get("text") or "").strip()
            if text:
                await self._answer(conversation, text, kind="interim")
        elif kind == "activity":
            name = event.get("event")
            self._quiet.pop(conversation, None)
            if name == "approval.requested":
                summary = first_sentence(str(event.get("summary") or "something"), 200)
                await self._say(conversation, "needs_you", news=f"approve this on the screen: {summary}",
                                fallback=f"I need your OK on the screen. {summary}")
            elif name == "browser.handoff":
                reason = first_sentence(str(event.get("reason") or "this page needs you"), 200)
                await self._say(conversation, "needs_you", news=f"finish this on the screen: {reason}",
                                fallback=f"Over to you on the screen. {reason}")
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
        self._answered[conversation] = first_sentence(text, 200)
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

    async def _say(self, conversation: str, kind: str, *, text: str = "", prompt: str = "", context: str = "",
                   news: str | None = None, fallback: str = "") -> dict | None:
        """One spoken line. `text` is said as it is (answers); `news` is told to the voice model, which says it
        the way a person would, knowing the request and the lines before it (a line it finds not worth saying is
        skipped, or `fallback` said instead); `prompt` is phrased into one sentence (tables). A line that cannot
        be said within a few seconds (Listen holds the model, or the assistant needs the memory) is dropped."""
        if text and text == self._said.get(conversation, (0.0, ""))[1]:
            return None                     # never the same line twice in a row
        for _ in range(40):
            if not self.slot.busy:
                break
            await asyncio.sleep(0.25)
        if self.slot.busy:
            return None
        async with self.slot.hold(f"speaking:{kind}"), self.turns.use():
            if news is not None:
                recent = self._recent.setdefault(conversation, deque(maxlen=3))
                try:
                    text = await narrate(self.runtime, request=self._request.get(conversation, ""), kind=kind,
                                         text=news, said=tuple(recent)) or fallback
                except Exception as exc:
                    log.info("narration skipped: %s", exc)
                    text = fallback
            elif not text:
                text = await self._phrase(prompt=prompt, context=context) or ""
            if not text:
                return None
            meta = await self._tts(text, conversation=conversation, kind=kind)
        self._said[conversation] = (time.time(), text)
        self._recent.setdefault(conversation, deque(maxlen=3)).append(text)
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
