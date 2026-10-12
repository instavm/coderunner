"""iMessage channel through `imsg` (github.com/openclaw/imsg, macOS 14+).

Setup: `brew install steipete/tap/imsg`. Sign this Mac's Messages into a
dedicated Apple ID (messages you send from your own Apple ID show up here as
`is_from_me` and are ignored, which prevents reply loops). Grant the process
that runs the assistant Full Disk Access (to read chat.db) and
Automation > Messages (to send). Enable `channels.imessage`, then text
`pair <code>` to that Apple ID from your phone.

`imsg watch --json` streams one JSON row per line; the last rowid is kept in
the store so a restart resumes where it stopped. Replies go through
`imsg send` with the text as a single argv element, never through a shell.

Owners are matched by phone number or email, so the transport matters. An
iMessage sender is authenticated by Apple; an SMS (or RCS) sender number is
easy to spoof, and Text Message Forwarding puts those messages in the same
chat.db. Only messages that provably arrived over iMessage are accepted: a
`service` field of "iMessage" when imsg provides one, otherwise a chat GUID
starting with "iMessage;". SMS, RCS and ambiguous chats (newer macOS may use
"any;-;" GUIDs) are ignored and audited unless `allow_sms: true`, which
accepts that spoofing risk. Replies are sent with `--service=imessage` for
the same reason unless `service` is set explicitly.

Group chats stay off by default. With `allow_groups: true` the owner's
messages in a group are obeyed and the assistant's replies, which may contain
private data from memory, connectors or files, are sent to every member.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import json
import os
from typing import Any

from ..process import find_binary, run, terminate
from .base import IncomingMessage, NO_VOICE, NO_WORDS, PollingChannel, split_text

CURSOR = "imessage:rowid"
MAX_IMAGE = 10 * 1024 * 1024
HINTS = {
    "unable to open database file": "Grant Full Disk Access to the app or process that runs the assistant "
                                    "(System Settings > Privacy & Security > Full Disk Access), then restart it.",
    "authorization denied": "Grant Full Disk Access to the process that runs the assistant, then restart it.",
    "not authorized to send apple events": "Allow the assistant's process to control Messages "
                                           "(System Settings > Privacy & Security > Automation).",
    "-1743": "Allow the assistant's process to control Messages (Privacy & Security > Automation).",
}


def permission_hint(detail: str) -> str:
    lowered = detail.lower()
    return next((f" Hint: {hint}" for needle, hint in HINTS.items() if needle in lowered), "")


def via_imessage(row: dict) -> bool:
    """True only when the row provably arrived over iMessage (not SMS, RCS or an ambiguous chat)."""
    service = row.get("service")
    if service:
        return str(service).lower() == "imessage"
    return str(row.get("chat_guid") or "").startswith("iMessage;")


class IMessageSendError(RuntimeError):
    pass


class IMessageChannel(PollingChannel):
    """Settings: enabled, owners (phone numbers in E.164 or Apple ID emails), binary (default "imsg"),
    allow_sms (default false), service (auto|imessage|sms; default imessage), allow_groups (default false),
    attachments (default true)."""

    name = "imessage"
    text_limit = 3000
    retry_wait = 2.0  # seconds before retrying a send that provably never started

    def __init__(self, settings: dict, svc: Any, on_message):
        super().__init__(settings, svc, on_message)
        self._proc: asyncio.subprocess.Process | None = None

    @property
    def binary(self) -> str | None:
        return find_binary("imsg", self.settings.get("binary"))

    def configured(self) -> bool:
        return bool(self.settings.get("enabled", True) and self.binary)

    async def start(self) -> None:
        self.launch(self._run())

    async def stop(self) -> None:
        await super().stop()
        await terminate(self._proc)

    # Receiving ------------------------------------------------------------
    async def _run(self) -> None:
        attempt = 0
        while True:
            try:
                rows = await self.watch_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error("watch", exc)
                rows = 0
            attempt = 1 if rows else attempt + 1
            await asyncio.sleep(self.retry_delay(attempt))

    async def watch_once(self) -> int:
        """Run `imsg watch` until it exits. Returns the number of rows read."""
        binary = self.binary
        if not binary:
            raise FileNotFoundError("imsg is not installed (brew install steipete/tap/imsg)")
        argv = [binary, "watch", "--json"]
        cursor = self.svc.store.get_kv(CURSOR)
        if cursor:
            argv += ["--since-rowid", str(int(cursor))]
        if self.settings.get("attachments", True):
            argv.append("--attachments")
        proc = self._proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=1 << 22)
        tail: collections.deque[str] = collections.deque(maxlen=10)
        stderr = asyncio.create_task(_collect(proc.stderr, tail))
        rows = 0
        try:
            async for raw in proc.stdout:
                if not raw.strip():
                    continue
                rows += 1
                try:
                    await self.handle_row(json.loads(raw))
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.error("row", exc)
            code = await proc.wait()
            await stderr
        finally:
            stderr.cancel()
            await terminate(proc)
            self._proc = None
        if code != 0:
            detail = " | ".join(tail)[-400:]
            self.error("watch", f"imsg watch exited with {code}: {detail}{permission_hint(detail)}")
        return rows

    async def handle_row(self, row: dict) -> None:
        rowid = row.get("id")
        if isinstance(rowid, int) and rowid > (self.svc.store.get_kv(CURSOR) or 0):
            self.svc.store.set_kv(CURSOR, rowid)  # saved first so a crash never replays an instruction
        if row.get("is_from_me") or row.get("is_reaction"):
            return
        if not self.settings.get("allow_sms") and not via_imessage(row):
            # SMS sender numbers can be spoofed, so they never count as the owner.
            self.svc.audit.write("channel.ignored", channel=self.name, sender=str(row.get("sender") or ""),
                                 reason="not_imessage")
            return
        group = bool(row.get("is_group"))
        if group and not self.settings.get("allow_groups"):
            return
        sender = str(row.get("sender") or "").strip()
        if not sender:
            return
        sender = sender.lower() if "@" in sender else sender
        reply_to = f"chat:{row.get('chat_id')}" if group else sender
        text = (row.get("text") or "").replace("\ufffc", "").strip()  # U+FFFC marks inline attachments
        if await self.gate(sender, text, reply_to, allow_pair=not group):
            self.submit(reply_to, reply_to, lambda: self._incoming(row, reply_to, sender, text))

    async def _incoming(self, row: dict, chat_id: str, sender: str, text: str) -> IncomingMessage | str | None:
        images: list[tuple[str, str]] = []
        audio = None
        for item in row.get("attachments") or []:
            path = os.path.expanduser(item.get("original_path") or "")
            mime = (item.get("mime_type") or "").lower()
            if item.get("missing") or not path or not os.path.isfile(path):
                continue
            if mime.startswith("image/") and len(images) < 4 and os.path.getsize(path) <= MAX_IMAGE:
                images.append(await self._image(path, mime))
            elif mime.startswith("audio/") and audio is None:
                audio = path
        if audio:
            transcript = await self.voice_text(audio)  # Messages owns this file: never delete it
            if transcript is None:
                return NO_VOICE
            if not transcript:
                return NO_WORDS
            text = f"{text}\n{transcript}".strip()
        if not text and not images:
            return None
        return IncomingMessage(self.name, chat_id, sender, text, True, images)

    async def _image(self, path: str, mime: str) -> tuple[str, str]:
        sips = find_binary("sips")
        if mime in ("image/heic", "image/heif") and sips:  # most models can't read HEIC
            target = self.scratch(".jpg")
            code, _, _ = await run([sips, "-s", "format", "jpeg", path, "--out", str(target)], 60)
            if code == 0 and target.exists():
                data = target.read_bytes()
                target.unlink()
                return "image/jpeg", base64.b64encode(data).decode()
        with open(path, "rb") as handle:
            return mime, base64.b64encode(handle.read()).decode()

    # Sending --------------------------------------------------------------
    async def send(self, to: str, text: str) -> None:
        for chunk in split_text(text.replace("\x00", ""), self.text_limit):
            await self._send_one(to, chunk)

    async def _send_one(self, to: str, text: str) -> None:
        binary = self.binary
        if not binary:
            raise IMessageSendError("imsg is not installed")
        # `--opt=value` keeps text that starts with "-" from being parsed as a flag.
        if to.startswith("chat:"):  # group sends keep the chat's own service
            argv = [binary, "send", f"--chat-id={to[5:]}", f"--text={text}", "--json"]
        else:
            service = self.settings.get("service") or ("auto" if self.settings.get("allow_sms") else "imessage")
            argv = [binary, "send", f"--to={to}", f"--text={text}", "--json", f"--service={service}"]
        for attempt in range(3):
            code, out, err = await run(argv, 90)  # a timeout leaves the outcome unknown, so it is not retried
            if code == 0:
                return
            detail = (err.strip() or out.strip())[-400:]
            # Only `not_started` is safe to retry; other failures may already have been delivered.
            if "not_started" in detail and attempt < 2:
                await asyncio.sleep(self.retry_wait * (attempt + 1))
                continue
            raise IMessageSendError(f"imsg send failed ({code}): {detail}{permission_hint(detail)}")


async def _collect(stream: asyncio.StreamReader, tail: collections.deque) -> None:
    async for raw in stream:
        line = raw.decode(errors="replace").strip()
        if line:
            tail.append(line)

