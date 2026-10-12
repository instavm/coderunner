"""Telegram bot channel over Bot API long polling (no webhook is ever set).

Setup: create a bot with @BotFather, store its token as the secret
`telegram_bot_token`, set `channels.telegram.enabled`, then send
`/pair <code>` to the bot from a private chat. Only private chats from paired
user ids are obeyed by default; everything else is ignored without a reply.

The bot token is part of every request URL, so it is scrubbed from errors and
from httpx's request log.

Groups stay off by default. With `allow_groups: true` the owner's messages in a
group are obeyed and the assistant's replies, which may contain private data
from memory, connectors or files, are posted to the whole group for every
member to read.

"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import logging
import pathlib
import re
from typing import Any

import httpx

from ..safety.policy import Approval
from .base import (MAX_MEDIA, NO_VOICE, NO_WORDS, IncomingMessage, PollingChannel, numbered_chunks, split_text,
                   utf16_len)

# --------------------------------------------------------------------------- Telegram

API = "https://api.telegram.org"
_BOT_URL = re.compile(r"/bot\d+:[A-Za-z0-9_-]+")


class _RedactBotTokens(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if "/bot" in message:
            record.msg, record.args = _BOT_URL.sub("/bot<redacted>", message), ()
        return True


# httpx logs every request URL at INFO level.
if not any(isinstance(f, _RedactBotTokens) for f in logging.getLogger("httpx").filters):
    logging.getLogger("httpx").addFilter(_RedactBotTokens())


class TelegramError(RuntimeError):
    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class TelegramChannel(PollingChannel):
    """Settings: enabled, owners (user ids), token_secret (default "telegram_bot_token"),
    allow_groups (default false), voice_replies (default false), poll_timeout (default 50)."""

    name = "telegram"
    text_limit = 4096
    transport: httpx.AsyncBaseTransport | None = None  # tests inject httpx.MockTransport

    def __init__(self, settings: dict, svc: Any, on_message):
        super().__init__(settings, svc, on_message)
        self.poll_timeout = int(self.settings.get("poll_timeout", 50))
        self._token: str | None = None
        self._http: httpx.AsyncClient | None = None
        self._spoken: set[str] = set()  # chats whose current turn came from a voice note

    @property
    def token(self) -> str:
        if self._token is None:
            self._token = self.svc.secrets.get(self.settings.get("token_secret", "telegram_bot_token")) or ""
        return self._token

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(base_url=self.settings.get("api_base", API), transport=self.transport,
                                           timeout=httpx.Timeout(self.poll_timeout + 20, connect=10))
        return self._http

    @property
    def offset_key(self) -> str:
        return f"telegram:offset:{self.token.split(':')[0]}"  # update ids are per bot

    def configured(self) -> bool:
        return bool(self.settings.get("enabled", True) and self.token)

    def redact(self, text: str) -> str:
        return _BOT_URL.sub("/bot<redacted>", text.replace(self.token, "<redacted>") if self.token else text)

    async def start(self) -> None:
        self.launch(self._run())

    async def stop(self) -> None:
        await super().stop()
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def call(self, method: str, payload: dict | None = None, files: dict | None = None) -> Any:
        if files:
            response = await self.http.post(f"/bot{self.token}/{method}", data=payload, files=files)
        else:
            response = await self.http.post(f"/bot{self.token}/{method}", json=payload or {})
        try:
            body = response.json()
        except ValueError:
            raise TelegramError(f"{method}: HTTP {response.status_code}") from None
        if not body.get("ok"):
            retry = (body.get("parameters") or {}).get("retry_after")
            raise TelegramError(f"{method}: {body.get('description') or response.status_code}", retry)
        return body.get("result")

    # Receiving ------------------------------------------------------------
    async def _run(self) -> None:
        attempt = 0
        while True:
            try:
                await self.poll_once()
                attempt = 0
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempt += 1
                if attempt == 1 or attempt % 10 == 0:
                    self.error("poll", exc, attempt=attempt)
                await asyncio.sleep(min(60.0, getattr(exc, "retry_after", None) or self.retry_delay(attempt)))

    async def poll_once(self) -> None:
        payload: dict[str, Any] = {"timeout": self.poll_timeout, "allowed_updates": ["message", "callback_query"]}
        offset = self.svc.store.get_kv(self.offset_key)
        if offset is not None:
            payload["offset"] = offset + 1
        for update in await self.call("getUpdates", payload) or []:
            # Saved before handling: a crash mid-update must not replay an instruction.
            self.svc.store.set_kv(self.offset_key, update["update_id"])
            try:
                await self.handle_update(update)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.error("update", exc)

    async def handle_update(self, update: dict) -> None:
        if "callback_query" in update:
            await self.handle_callback(update["callback_query"])
            return
        message = update.get("message") or {}
        chat = message.get("chat") or {}
        sender, chat_id = str((message.get("from") or {}).get("id", "")), str(chat.get("id", ""))
        if not sender or not chat_id:
            return
        private = chat.get("type") == "private"
        if not private and not self.settings.get("allow_groups"):
            self.svc.audit.write("channel.ignored", channel=self.name, sender=sender, reason="group")
            return
        text = message.get("text") or message.get("caption") or ""
        if await self.gate(sender, text, chat_id, allow_pair=private):
            self.submit(chat_id, chat_id, lambda: self._incoming(message, chat_id, sender, text))

    async def _incoming(self, message: dict, chat_id: str, sender: str, text: str) -> IncomingMessage | str | None:
        images: list[tuple[str, str]] = []
        self._spoken.discard(chat_id)  # turns in one chat run one at a time, so this is per turn
        voice = message.get("voice") or message.get("audio")
        if voice:
            if not self.can_transcribe():
                return NO_VOICE
            path = await self.download(voice["file_id"], ".ogg", voice.get("file_size"))
            transcript = await self.voice_text(path, discard=True)
            if not transcript:
                return NO_WORDS
            text = f"{text}\n{transcript}".strip()
            self._spoken.add(chat_id)
        if message.get("photo"):
            largest = max(message["photo"], key=lambda p: (p.get("width", 0) * p.get("height", 0),
                                                            p.get("file_size", 0)))
            path = await self.download(largest["file_id"], ".jpg", largest.get("file_size"))
            images.append(("image/jpeg", base64.b64encode(path.read_bytes()).decode()))
            path.unlink(missing_ok=True)
        if not text and not images:
            return None
        return IncomingMessage(self.name, chat_id, sender, text, True, images)

    async def download(self, file_id: str, suffix: str, size: int | None = None) -> pathlib.Path:
        if (size or 0) > MAX_MEDIA:
            raise ValueError("file is larger than the 20 MB bot download limit")
        info = await self.call("getFile", {"file_id": file_id})
        if (info.get("file_size") or 0) > MAX_MEDIA or not info.get("file_path"):
            raise ValueError("file is unavailable or larger than 20 MB")
        target, received = self.scratch(suffix), 0
        async with self.http.stream("GET", f"/file/bot{self.token}/{info['file_path']}") as response:
            if response.status_code != 200:
                raise TelegramError(f"file download: HTTP {response.status_code}")
            with open(target, "wb") as handle:
                async for chunk in response.aiter_bytes():
                    received += len(chunk)
                    if received > MAX_MEDIA:
                        raise ValueError("file is larger than 20 MB")
                    handle.write(chunk)
        return target

    # Approvals ------------------------------------------------------------
    async def notify_approval(self, approval: Approval) -> None:
        buttons = [{"text": "Allow once", "callback_data": f"a:{approval.id}:once"}]
        if approval.risk != "high":
            buttons.append({"text": "Allow for a day", "callback_data": f"a:{approval.id}:day"})
        buttons.append({"text": "Deny", "callback_data": f"d:{approval.id}"})
        # The owner must see everything they approve: send all of it, buttons on the last part only.
        chunks = numbered_chunks(f"Approval needed ({approval.kind}):\n{approval.summary}", self.text_limit, utf16_len)
        for owner in self.owners():
            try:
                for index, chunk in enumerate(chunks, 1):
                    markup = {"reply_markup": {"inline_keyboard": [buttons]}} if index == len(chunks) else {}
                    await self._send_message({"chat_id": owner, "text": chunk, **markup})
            except Exception as exc:  # buttons are never sent without the full text before them
                self.error("notify", exc)

    async def handle_callback(self, query: dict) -> None:
        sender = str((query.get("from") or {}).get("id", ""))
        owner = self.is_owner(sender)
        with contextlib.suppress(TelegramError):
            await self.call("answerCallbackQuery", {"callback_query_id": query.get("id"),
                                                    **({} if owner else {"text": "Not allowed."})})
        if not owner:
            self.svc.audit.write("channel.ignored", channel=self.name, sender=sender, reason="callback")
            return
        action, approval_id, *rest = (query.get("data") or "").split(":") + ["", ""]
        if action not in ("a", "d") or not approval_id:
            return
        allow = action == "a"
        try:
            approval = self.svc.policy.decide(approval_id, allow, rest[0] if allow and rest[0] else "once",
                                              decided_by=f"{self.name}:{sender}")
            outcome = f"{'Approved' if allow else 'Denied'} ({approval.scope})."
        except (KeyError, ValueError) as exc:
            outcome = f"Not applied: {exc.args[0] if exc.args else exc}"
        message = query.get("message") or {}
        ids = {"chat_id": (message.get("chat") or {}).get("id"), "message_id": message.get("message_id")}
        if not all(ids.values()):
            return
        with contextlib.suppress(TelegramError):
            await self.call("editMessageReplyMarkup", {**ids, "reply_markup": {"inline_keyboard": []}})
        text = f"{message.get('text') or ''}\n\n{outcome}".strip()
        with contextlib.suppress(TelegramError):
            if utf16_len(text) <= self.text_limit:
                await self.call("editMessageText", {**ids, "text": text})
            else:  # never cut the prompt to make room for the outcome
                await self._send_message({"chat_id": ids["chat_id"], "text": outcome})

    # Sending --------------------------------------------------------------
    async def send(self, to: str, text: str) -> None:
        for chunk in split_text(text, self.text_limit, utf16_len):
            await self._send_message({"chat_id": to, "text": chunk})

    async def _send_message(self, payload: dict) -> Any:
        # No link previews: Telegram's servers would fetch any URL the model writes.
        payload = {**payload, "link_preview_options": {"is_disabled": True}}
        for attempt in range(3):
            try:
                return await self.call("sendMessage", payload)
            except TelegramError as exc:
                if not exc.retry_after or attempt == 2:
                    raise
                await asyncio.sleep(min(float(exc.retry_after), 60.0))

    async def reply(self, to: str, text: str, incoming: IncomingMessage | str | None) -> None:
        await self.send(to, text)
        voice = getattr(self.svc, "voice", None)
        spoken = isinstance(incoming, IncomingMessage) and incoming.chat_id in self._spoken
        if not (self.settings.get("voice_replies") and spoken and voice is not None and len(text) <= 1500):
            return
        try:
            if not voice.can_speak():
                return
            path = await voice.speak(text)
            try:
                with open(path, "rb") as handle:
                    await self.call("sendVoice", {"chat_id": to}, files={"voice": ("reply.ogg", handle, "audio/ogg")})
            finally:
                voice.discard(path)
        except Exception as exc:
            self.error("voice_reply", exc)
