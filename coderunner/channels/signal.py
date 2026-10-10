"""Signal channel through a local signal-cli daemon (v0.14+, HTTP mode).

We don't launch signal-cli. Register or link a number once (ideally a
dedicated one), then keep a single-account daemon running on loopback:

    signal-cli -a +15550001111 link -n coderunner      # or `register` + `verify`
    signal-cli -a +15550001111 daemon --http=127.0.0.1:8080

(for example as a launchd agent). The HTTP API has no authentication, so never
bind it to anything but 127.0.0.1. Then set `channels.signal` with the url and
account and send `pair <code>` from your own Signal account.

Inbound messages come from the Server-Sent Events stream at /api/v1/events;
the last event id is stored and sent back as Last-Event-ID on reconnect.
signal-cli replays its whole buffer when it doesn't know that id (after a
daemon restart), so envelopes are also de-duplicated by sender and timestamp.
Replies use JSON-RPC `send` on /api/v1/rpc.

Group chats stay off by default. With `allow_groups: true` the owner's
messages in a group are obeyed and the assistant's replies, which may contain
private data from memory, connectors or files, are sent to every member.
"""

from __future__ import annotations

import asyncio
import base64
import json
from typing import Any

import httpx

from .base import IncomingMessage, MAX_MEDIA, NO_VOICE, NO_WORDS, PollingChannel, split_text

EVENT_KEY = "signal:event_id"
SEEN_KEY = "signal:seen"
AUDIO_SUFFIX = {"audio/aac": ".aac", "audio/mp4": ".m4a", "audio/mpeg": ".mp3", "audio/ogg": ".ogg"}


class SignalError(RuntimeError):
    pass


class SignalChannel(PollingChannel):
    """Settings: enabled, url (e.g. "http://127.0.0.1:8080"), account (the daemon's number),
    owners (numbers or ACI uuids), allow_groups (default false)."""

    name = "signal"
    text_limit = 2000
    transport: httpx.AsyncBaseTransport | None = None  # tests inject httpx.MockTransport

    def __init__(self, settings: dict, svc: Any, on_message):
        super().__init__(settings, svc, on_message)
        self._http: httpx.AsyncClient | None = None
        self._rpc_id = 0

    @property
    def http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(base_url=self.settings["url"].rstrip("/"), transport=self.transport,
                                           timeout=httpx.Timeout(60, connect=5))
        return self._http

    def configured(self) -> bool:
        return bool(self.settings.get("enabled", True) and self.settings.get("url") and self.settings.get("account"))

    async def start(self) -> None:
        self.launch(self._run())

    async def stop(self) -> None:
        await super().stop()
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    async def rpc(self, method: str, params: dict) -> Any:
        self._rpc_id += 1
        response = await self.http.post("/api/v1/rpc", json={"jsonrpc": "2.0", "id": self._rpc_id,
                                                             "method": method, "params": params})
        if response.status_code == 201:  # notification accepted, no result
            return None
        if response.status_code != 200:
            raise SignalError(f"{method}: HTTP {response.status_code}")
        body = response.json()
        if body.get("error"):
            raise SignalError(f"{method}: {body['error'].get('message', body['error'])}")
        return body.get("result")

    # Receiving ------------------------------------------------------------
    async def _run(self) -> None:
        attempt = 0
        while True:
            try:
                events = await self.listen_once()
                attempt = 0 if events else attempt + 1
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempt += 1
                if attempt == 1 or attempt % 10 == 0:
                    self.error("events", exc, attempt=attempt)
            await asyncio.sleep(self.retry_delay(max(attempt, 1)))

    async def listen_once(self) -> int:
        """Read the event stream until the daemon closes it. Returns the number of events."""
        headers = {"Accept": "text/event-stream"}
        last = self.svc.store.get_kv(EVENT_KEY)
        if last:
            headers["Last-Event-ID"] = last
        count, event_id, data = 0, None, []
        # signal-cli sends a keep-alive every 15 s, so a long read timeout means a dead daemon.
        async with self.http.stream("GET", "/api/v1/events", params={"account": self.settings["account"]},
                                    headers=headers, timeout=httpx.Timeout(10, read=90)) as response:
            if response.status_code != 200:
                raise SignalError(f"events: HTTP {response.status_code}")
            async for line in response.aiter_lines():
                if line:
                    field, _, value = line.partition(":")
                    value = value[1:] if value.startswith(" ") else value
                    if field == "id":
                        event_id = value
                    elif field == "data":
                        data.append(value)
                    continue
                if data:
                    count += 1
                    if event_id:
                        self.svc.store.set_kv(EVENT_KEY, event_id)
                    try:
                        await self.handle_event(json.loads("\n".join(data)))
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:
                        self.error("event", exc)
                event_id, data = None, []
        return count

    def _seen(self, key: str) -> bool:
        seen = self.svc.store.get_kv(SEEN_KEY) or []
        if key in seen:
            return True
        self.svc.store.set_kv(SEEN_KEY, (seen + [key])[-500:])
        return False

    async def handle_event(self, payload: dict) -> None:
        # The SSE data is {"account", "envelope"}; also accept the JSON-RPC notification shapes.
        params = payload.get("params", payload)
        params = params.get("result", params)
        envelope = params.get("envelope") or {}
        data = envelope.get("dataMessage")
        if not data or data.get("reaction") or data.get("remoteDelete"):
            return  # receipts, typing, edits, and syncMessages from our own linked devices
        number, uuid = envelope.get("sourceNumber"), envelope.get("sourceUuid")
        if not (number or uuid) or number == self.settings.get("account"):
            return
        if self._seen(f"{uuid or number}:{envelope.get('timestamp')}"):
            return
        group = (data.get("groupInfo") or {}).get("groupId")
        if group and not self.settings.get("allow_groups"):
            return
        sender = next((i for i in (number, uuid) if i and self.is_owner(i)), number or uuid)
        reply_to = f"group:{group}" if group else sender
        text = data.get("message") or ""
        if await self.gate(sender, text, reply_to, allow_pair=not group):
            self.submit(reply_to, reply_to, lambda: self._incoming(data, reply_to, sender, text))

    async def _incoming(self, data: dict, chat_id: str, sender: str, text: str) -> IncomingMessage | str | None:
        images: list[tuple[str, str]] = []
        audio = None
        source = {"groupId": chat_id[6:]} if chat_id.startswith("group:") else {"recipient": sender}
        for item in data.get("attachments") or []:
            mime = (item.get("contentType") or "").lower()
            if not item.get("id") or (item.get("size") or 0) > MAX_MEDIA:
                continue
            if mime.startswith("image/") and len(images) < 4:
                images.append((mime, await self._attachment(item["id"], source)))
            elif (mime.startswith("audio/") or item.get("isVoiceNote")) and audio is None:
                audio = item
        if audio:
            if not self.can_transcribe():
                return NO_VOICE
            path = self.scratch(AUDIO_SUFFIX.get((audio.get("contentType") or "").lower(), ".m4a"))
            path.write_bytes(base64.b64decode(await self._attachment(audio["id"], source)))
            transcript = await self.voice_text(path, discard=True)
            if not transcript:
                return NO_WORDS
            text = f"{text}\n{transcript}".strip()
        if not text and not images:
            return None
        return IncomingMessage(self.name, chat_id, sender, text, True, images)

    async def _attachment(self, attachment_id: str, source: dict) -> str:
        result = await self.rpc("getAttachment", {"id": attachment_id, **source})
        data = result.get("data") if isinstance(result, dict) else result
        if not isinstance(data, str):
            raise SignalError("getAttachment returned no data")
        return data

    # Sending --------------------------------------------------------------
    async def send(self, to: str, text: str) -> None:
        target = {"groupId": to[6:]} if to.startswith("group:") else {"recipient": [to]}
        for chunk in split_text(text, self.text_limit):
            result = await self.rpc("send", {**target, "message": chunk})
            outcomes = [r.get("type") for r in (result or {}).get("results") or []]
            if outcomes and "SUCCESS" not in outcomes:
                raise SignalError(f"send failed: {', '.join(map(str, outcomes))}")
