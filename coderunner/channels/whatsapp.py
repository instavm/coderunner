"""WhatsApp channel through `wacli` (github.com/openclaw/wacli). EXPERIMENTAL and opt-in.

WARNING: wacli links to WhatsApp as an unofficial WhatsApp Web client
(whatsmeow). That breaks WhatsApp's terms of service, and WhatsApp can ban a
number that uses unofficial clients or automation, without notice or appeal.
Use a secondary number you can afford to lose, never your main one. The
official Cloud API is not an option: it needs a public webhook and a business
number, and since 2026-01-15 WhatsApp's policy bans general-purpose assistants
on it. The channel stays off unless its settings have both `"enabled": true`
and `"accept_tos_risk": true`.

Setup: `brew install openclaw/tap/wacli`, then `wacli auth` (scan the QR code
from Linked devices on the assistant's phone), set `channels.whatsapp`, and
send `pair <code>` to that number from your own WhatsApp.

How it works: we run `wacli sync --follow` and point its webhook at a listener
on 127.0.0.1 in this process. Each POST is HMAC-signed with a secret made
fresh for every run, and unsigned requests are rejected. Message ids are
de-duplicated in the store. Replies go through `wacli send text` (delegated
to the running sync process), with link previews off.

Group chats stay off by default. With `allow_groups: true` the owner's
messages in a group are obeyed and the assistant's replies, which may contain
private data from memory, connectors or files, are sent to every member.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import contextlib
import hashlib
import hmac
import json
import pathlib
import secrets
import time
from typing import Any

from ..voice import find_binary, run, terminate
from .base import IncomingMessage, NO_VOICE, NO_WORDS, PollingChannel, split_text

SEEN_KEY = "whatsapp:seen"
MAX_BODY = 1 << 20
REASONS = {204: "No Content", 400: "Bad Request", 401: "Unauthorized", 404: "Not Found", 413: "Too Large"}


class WhatsAppError(RuntimeError):
    pass


def handle_of(jid: str) -> str:
    """`15551234567:3@s.whatsapp.net` -> `+15551234567`; other JIDs (LIDs, groups) stay as they are."""
    user, _, server = jid.partition("@")
    user = user.split(":")[0].split(".")[0]
    if server in ("s.whatsapp.net", "c.us") and user.isdigit():
        return f"+{user}"
    return f"{user}@{server}" if server else user


class WhatsAppChannel(PollingChannel):
    """Settings: enabled, accept_tos_risk (must be true), owners (E.164 numbers), binary (default "wacli"),
    account (wacli named account, optional), allow_groups (default false)."""

    name = "whatsapp"
    text_limit = 4000
    send_spacing = 1.5  # seconds between outgoing messages, across all sends

    def __init__(self, settings: dict, svc: Any, on_message):
        super().__init__(settings, svc, on_message)
        self._proc: asyncio.subprocess.Process | None = None
        self._server: asyncio.AbstractServer | None = None
        self._secret = secrets.token_hex(32)
        self._rejected = 0.0
        self._send_lock = asyncio.Lock()
        self._last_send = 0.0
        self.webhook_url = ""

    @property
    def binary(self) -> str | None:
        return find_binary("wacli", self.settings.get("binary"))

    def configured(self) -> bool:
        return bool(self.settings.get("enabled") and self.settings.get("accept_tos_risk") is True and self.binary)

    def _argv(self, *args: str) -> list[str]:
        argv = [self.binary or "wacli"]
        if self.settings.get("account"):
            argv.append(f"--account={self.settings['account']}")
        return [*argv, *args]

    async def start(self) -> None:
        self._secret = secrets.token_hex(32)
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        self.webhook_url = f"http://127.0.0.1:{self._server.sockets[0].getsockname()[1]}/wacli"
        self.launch(self._run())

    async def stop(self) -> None:
        await super().stop()
        await terminate(self._proc)
        if self._server is not None:
            self._server.close()
            self._server = None

    # Sync process ---------------------------------------------------------
    async def _run(self) -> None:
        attempt = 0
        while True:
            started = time.monotonic()
            try:
                logged_out = await self.sync_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error("sync", exc)
                logged_out = False
            if logged_out:
                self.error("sync", "WhatsApp unlinked this device. Run `wacli auth` again, then restart the assistant.")
                return
            attempt = 1 if time.monotonic() - started > 300 else attempt + 1
            await asyncio.sleep(self.retry_delay(attempt))

    async def sync_once(self) -> bool:
        """Run `wacli sync --follow` until it exits. Returns True when WhatsApp logged the device out."""
        argv = self._argv("sync", "--follow", "--events", "--max-reconnect=0", f"--webhook={self.webhook_url}",
                          f"--webhook-secret={self._secret}", "--webhook-allow-private")
        proc = self._proc = await asyncio.create_subprocess_exec(
            *argv, stdin=asyncio.subprocess.DEVNULL, stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.PIPE, limit=1 << 20)
        tail: collections.deque[str] = collections.deque(maxlen=10)
        logged_out = False
        try:
            async for raw in proc.stderr:  # --events: one {"event", "data", "ts"} object per line
                line = raw.decode(errors="replace").strip()
                try:
                    event = json.loads(line).get("event")
                except (ValueError, AttributeError):
                    event = None
                logged_out = logged_out or event == "logged_out"
                if line:
                    tail.append(line[:300])
            code = await proc.wait()
        finally:
            await terminate(proc)
            self._proc = None
        if code != 0:
            self.error("sync", f"wacli sync exited with {code}: {' | '.join(tail)[-500:]}")
        return logged_out

    # Webhook --------------------------------------------------------------
    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        status, payload = 400, None
        try:
            head = (await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)).decode("latin-1").split("\r\n")
            method, target = (head[0].split(" ") + ["", ""])[:2]
            headers = {k.strip().lower(): v.strip() for k, _, v in (line.partition(":") for line in head[1:] if line)}
            length = int(headers.get("content-length", "0"))
            if method != "POST" or target != "/wacli":
                status = 404
            elif not 0 < length <= MAX_BODY:
                status = 413
            else:
                body = await asyncio.wait_for(reader.readexactly(length), 10)
                expected = "sha256=" + hmac.new(self._secret.encode(), body, hashlib.sha256).hexdigest()
                if hmac.compare_digest(headers.get("x-wacli-signature", "").encode(), expected.encode()):
                    payload, status = json.loads(body), 204
                else:
                    status = 401
        except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, asyncio.TimeoutError, ValueError):
            status = 400
        with contextlib.suppress(Exception):
            writer.write(f"HTTP/1.1 {status} {REASONS[status]}\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                         .encode())
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        if status == 401 and time.monotonic() - self._rejected > 60:  # anything local can knock: audit sparingly
            self._rejected = time.monotonic()
            self.error("webhook", "rejected a request with a bad signature")
        if isinstance(payload, dict):
            try:
                await self.handle_payload(payload)
            except Exception as exc:
                self.error("webhook", exc)

    def _seen(self, msg_id: str) -> bool:
        seen = self.svc.store.get_kv(SEEN_KEY) or []
        if msg_id in seen:
            return True
        self.svc.store.set_kv(SEEN_KEY, (seen + [msg_id])[-500:])
        return False

    async def handle_payload(self, payload: dict) -> None:
        if payload.get("EventType", "message") != "message":
            return
        msg_id, chat = str(payload.get("ID") or ""), str(payload.get("Chat") or "")
        if not msg_id or self._seen(f"{chat}/{msg_id}"):
            return
        if payload.get("FromMe") or payload.get("ReactionToID") or payload.get("Revoked") or payload.get("Edited"):
            return
        group = chat.endswith("@g.us")
        if not chat.endswith(("@s.whatsapp.net", "@lid", "@g.us")) or (group and not self.settings.get("allow_groups")):
            return  # status broadcasts, channels, and groups unless allowed
        sender = handle_of(str(payload.get("SenderJID") or chat))
        reply_to = chat if group else sender
        media = payload.get("Media") or {}
        text = (media.get("Caption") or "") if media else (payload.get("Text") or "")
        if await self.gate(sender, text, reply_to, allow_pair=not group):
            self.submit(reply_to, reply_to, lambda: self._incoming(payload, media, chat, reply_to, sender, text))

    async def _incoming(self, payload: dict, media: dict, chat: str, chat_id: str, sender: str,
                        text: str) -> IncomingMessage | str | None:
        images: list[tuple[str, str]] = []
        kind, mime = (media.get("Type") or "").lower(), (media.get("MimeType") or "").lower()
        if kind == "audio" or mime.startswith("audio/"):
            if not self.can_transcribe():
                return NO_VOICE
            transcript = await self.voice_text(await self._download(chat, payload["ID"], ".ogg"), discard=True)
            if not transcript:
                return NO_WORDS
            text = f"{text}\n{transcript}".strip()
        elif kind == "image" or mime.startswith("image/"):
            path = await self._download(chat, payload["ID"], ".img")
            images.append((mime or "image/jpeg", base64.b64encode(path.read_bytes()).decode()))
            path.unlink(missing_ok=True)
        if not text and not images:
            return None
        return IncomingMessage(self.name, chat_id, sender, text, True, images)

    async def _download(self, chat: str, msg_id: str, suffix: str) -> pathlib.Path:
        # --read-only takes no store lock, so it works while `sync --follow` is running.
        target = self.scratch(suffix)
        code, _, err = await run(self._argv("--read-only", "media", "download", f"--chat={chat}", f"--id={msg_id}",
                                            f"--output={target}"), 120)
        if code != 0 or not target.exists():
            raise WhatsAppError(f"wacli media download failed ({code}): {err.strip()[-300:]}")
        return target

    # Sending --------------------------------------------------------------
    async def send(self, to: str, text: str) -> None:
        for chunk in split_text(text.replace("\x00", ""), self.text_limit):
            async with self._send_lock:  # wacli warns that rapid-fire sends raise the ban risk
                await asyncio.sleep(max(0.0, self._last_send + self.send_spacing - time.monotonic()))
                # `--opt=value` keeps text that starts with "-" from being parsed as a flag.
                code, out, err = await run(self._argv("--json", "send", "text", f"--to={to}", f"--message={chunk}",
                                                      "--no-preview"), 90)
                self._last_send = time.monotonic()
            try:
                ok = code == 0 and json.loads(out or "{}").get("success", True) is not False
            except ValueError:
                ok = code == 0
            if not ok:
                raise WhatsAppError(f"wacli send failed ({code}): {(err.strip() or out.strip())[-300:]}")
