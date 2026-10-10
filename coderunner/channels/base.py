"""Messaging channel interface.

A channel delivers the owner's messages to the assistant and sends replies
and approval prompts back. Only paired owner identities may instruct the
assistant; anything else is ignored (or, when a channel is configured to
accept it, handled as untrusted input). Long polling or local daemons are
used everywhere so no public webhook is ever required.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import pathlib
import random
import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from ..config import Config
from ..policy import Approval
from ..voice import STT_HINT


@dataclass
class IncomingMessage:
    channel: str
    chat_id: str
    sender: str
    text: str
    sender_is_owner: bool
    images: list[tuple[str, str]] = field(default_factory=list)  # (mime, base64)
    audio_path: str | None = None


MessageHandler = Callable[[IncomingMessage], Awaitable[str | None]]


class Channel:
    name = "base"
    text_limit = 4000  # characters per outgoing message

    def __init__(self, settings: dict, svc: Any, on_message: MessageHandler):
        self.settings = settings or {}
        self.svc = svc
        self.on_message = on_message

    # Lifecycle ------------------------------------------------------------
    def configured(self) -> bool:
        return False

    async def start(self) -> None:
        raise NotImplementedError

    async def stop(self) -> None:
        pass

    # Outbound -------------------------------------------------------------
    async def send(self, to: str, text: str) -> None:
        raise NotImplementedError

    async def notify_approval(self, approval: Approval) -> None:
        """Ask every owner to decide. Text channels accept `approve <id> [scope]` / `deny <id>`.

        The whole summary is always sent (numbered when it needs several messages):
        the owner must see everything they are approving.
        """
        scopes = "once" if approval.risk == "high" else "once|session|day|always"
        text = (f"Approval needed ({approval.kind}):\n{approval.summary}\n\n"
                f"Reply `approve {approval.id}` or `deny {approval.id}`. Scope: {scopes}.")
        for owner in self.owners():
            try:
                for chunk in numbered_chunks(text, self.text_limit):
                    await self.send(owner, chunk)
            except Exception as exc:  # one unreachable owner must not stop the others
                self.svc.audit.write("channel.error", channel=self.name, stage="notify", error=str(exc)[:500])

    # Owners and pairing -----------------------------------------------------
    def owners(self) -> list[str]:
        return [str(o) for o in self.settings.get("owners", [])]

    def is_owner(self, sender: str) -> bool:
        return str(sender) in self.owners()

    def start_pairing(self) -> str:
        code = "".join(secrets.choice("23456789") for _ in range(6))
        self.svc.store.set_kv(f"pair:{self.name}", {"code": code, "expires": time.time() + 600})
        return code

    def try_pair(self, sender: str, text: str) -> bool:
        """`/pair 123456` or `pair 123456` from a new identity adds it as an owner."""
        parts = text.strip().split()
        if len(parts) != 2 or parts[0].lower().lstrip("/") != "pair":
            return False
        key = f"pair:{self.name}"
        pending = self.svc.store.get_kv(key)
        if not pending or pending["expires"] < time.time():
            return False
        if not secrets.compare_digest(parts[1], pending["code"]):
            pending["failures"] = pending.get("failures", 0) + 1
            # A few wrong guesses burn the code so it cannot be brute forced.
            self.svc.store.set_kv(key, None if pending["failures"] >= 5 else pending)
            return False
        self.svc.store.set_kv(key, None)
        # Edit the file as it is now: the daemon's in-memory copy may predate CLI changes.
        dotted = f"channels.{self.name}.owners"
        current = Config.load(self.svc.config.path)
        owners = [str(o) for o in current.get(dotted, []) or []]
        if str(sender) not in owners:
            owners.append(str(sender))
        current.set(dotted, owners)
        current.save()
        self.svc.config.set(dotted, owners)
        self.settings["owners"] = owners
        self.svc.audit.write("channel.paired", channel=self.name, owner=str(sender))
        return True

    def handle_decision_text(self, sender: str, text: str) -> str | None:
        """Owner replies like `approve 1a2b3c4d day` resolve approvals. Returns a confirmation or None."""
        if not self.is_owner(sender):
            return None
        parts = text.strip().split()
        if len(parts) not in (2, 3) or parts[0].lower() not in ("approve", "deny", "allow"):
            return None
        allow = parts[0].lower() != "deny"
        scope = parts[2].lower() if len(parts) == 3 else "once"
        try:
            approval = self.svc.policy.decide(parts[1], allow, scope, decided_by=f"{self.name}:{sender}")
        except (KeyError, ValueError) as exc:
            return f"Could not apply that: {exc}"
        return f"{'Approved' if allow else 'Denied'} ({approval.scope}): {approval.summary[:200]}"


# ---- shared by the channel implementations ----------------------------------

log = logging.getLogger("coderunner.channels")

PAIRED = "Paired. I'll take instructions from this chat."
TURN_FAILED = "Sorry, something went wrong while handling that message."
NO_VOICE = "Voice notes aren't set up yet, so please send text for now. " + STT_HINT
NO_WORDS = "I couldn't make out any words in that voice note."
MAX_MEDIA = 20 * 1024 * 1024
Build = Callable[[], Awaitable["IncomingMessage | str | None"]]


def backoff_delay(attempt: int, cap: float = 60.0) -> float:
    """Exponential backoff with jitter, capped at `cap` seconds."""
    return min(cap, 2.0 ** min(attempt, 6)) * random.uniform(0.5, 1.0)


def utf16_len(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2


def numbered_chunks(text: str, limit: int, size: Callable[[str], int] = len) -> list[str]:
    """split_text, prefixing "(i/n) " when the text needs more than one message."""
    chunks = split_text(text, limit, size)
    if len(chunks) > 1:
        chunks = split_text(text, limit - 12, size)  # room for the prefix
        chunks = [f"({i}/{len(chunks)}) {chunk}" for i, chunk in enumerate(chunks, 1)]
    return chunks


def split_text(text: str, limit: int, size: Callable[[str], int] = len) -> list[str]:
    """Split into chunks no larger than `limit` (measured by `size`), preferring line then word breaks."""
    chunks = []
    while size(text) > limit:
        lo, hi = 1, min(len(text), limit)
        while lo < hi:  # longest prefix that fits
            mid = (lo + hi + 1) // 2
            lo, hi = (mid, hi) if size(text[:mid]) <= limit else (lo, mid - 1)
        cut = text.rfind("\n", 0, lo + 1)
        if cut <= lo // 2:
            cut = text.rfind(" ", 0, lo + 1)
        if cut <= lo // 2:
            chunks.append(text[:lo])
            text = text[lo:]
        else:
            chunks.append(text[:cut])
            text = text[cut + 1:]
    chunks.append(text)
    return [chunk for chunk in chunks if chunk.strip()]


class PollingChannel(Channel):
    """A channel with background receive loops, per-chat agent turns and backoff.

    Agent turns run as tasks so the receive loop keeps reading approval replies
    while the agent is blocked waiting for one.
    """

    def __init__(self, settings: dict, svc: Any, on_message):
        super().__init__(settings, svc, on_message)
        self.retry_delay: Callable[[int], float] = backoff_delay
        self._loops: set[asyncio.Task] = set()
        self._turns: set[asyncio.Task] = set()
        self._locks: dict[str, asyncio.Lock] = {}
        self._scratch: pathlib.Path | None = None

    def launch(self, coro) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._loops.add(task)
        task.add_done_callback(self._loops.discard)
        return task

    async def stop(self) -> None:
        tasks = [*self._loops, *self._turns]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self._scratch is not None:
            shutil.rmtree(self._scratch, ignore_errors=True)
            self._scratch = None

    async def drain(self) -> None:
        """Wait until every submitted agent turn has finished."""
        while self._turns:
            await asyncio.gather(*list(self._turns), return_exceptions=True)

    def redact(self, text: str) -> str:
        return text

    def error(self, stage: str, exc: BaseException | str, **fields: Any) -> None:
        message = self.redact(str(exc) or type(exc).__name__)[:500]
        log.warning("%s %s: %s", self.name, stage, message)
        self.svc.audit.write("channel.error", channel=self.name, stage=stage, error=message, **fields)

    def scratch(self, suffix: str) -> pathlib.Path:
        """A fresh path in this channel's private temp dir (removed on stop)."""
        if self._scratch is None:
            self._scratch = pathlib.Path(tempfile.mkdtemp(prefix=f"coderunner-{self.name}-"))
        return self._scratch / f"{secrets.token_hex(8)}{suffix}"

    async def gate(self, sender: str, text: str, reply_to: str, *, allow_pair: bool = True) -> bool:
        """Pairing, approval replies and the owner check. True when the message is for the agent."""
        text = (text or "").strip()
        if allow_pair and text and self.try_pair(sender, text):
            await self.send(reply_to, PAIRED)
            return False
        confirmation = self.handle_decision_text(sender, text) if text else None
        if confirmation:
            await self.send(reply_to, confirmation)
            return False
        if not self.is_owner(sender):
            self.svc.audit.write("channel.ignored", channel=self.name, sender=str(sender))
            return False
        return True

    def submit(self, chat_id: str, reply_to: str, build: Build) -> asyncio.Task:
        """Build the message (downloads, transcription) and run the agent turn, one at a time per chat."""
        lock = self._locks.setdefault(chat_id, asyncio.Lock())

        async def turn() -> None:
            async with lock:
                try:
                    incoming = await build()
                    reply = await self.on_message(incoming) if isinstance(incoming, IncomingMessage) else incoming
                    if reply:
                        await self.reply(reply_to, reply, incoming)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.error("turn", exc)
                    with contextlib.suppress(Exception):
                        await self.send(reply_to, TURN_FAILED)

        task = asyncio.create_task(turn())
        self._turns.add(task)
        task.add_done_callback(self._turns.discard)
        return task

    async def reply(self, to: str, text: str, incoming: IncomingMessage | str | None) -> None:
        await self.send(to, text)

    def can_transcribe(self) -> bool:
        voice = getattr(self.svc, "voice", None)
        return voice is not None and voice.available()

    async def voice_text(self, path, *, discard: bool = False) -> str | None:
        """Transcript of a voice note, or None when no local speech engine is set up.

        Channels pass the transcript as the message text and leave `audio_path` unset,
        so the audio is transcribed exactly once. `discard` deletes a downloaded copy.
        """
        try:
            return await self.svc.voice.transcribe(path) if self.can_transcribe() else None
        finally:
            if discard:
                pathlib.Path(path).unlink(missing_ok=True)
