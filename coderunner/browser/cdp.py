"""Chrome DevTools Protocol over --remote-debugging-pipe.

The browser reads commands on fd 3 and writes replies and events on fd 4,
each message a JSON object terminated by NUL. There is no TCP port, so
nothing else on the machine (or in the sandbox VM) can attach to the browser.

The transport is pluggable: anything with async start(on_message, on_close),
send(message: dict) and close() works, which lets tests use a fake browser.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Callable, Protocol

log = logging.getLogger(__name__)


class CDPError(Exception):
    def __init__(self, message: str, code: int | None = None):
        super().__init__(message)
        self.code = code


class Transport(Protocol):
    async def start(self, on_message: Callable[[dict], None], on_close: Callable[[], None]) -> None: ...
    def send(self, message: dict) -> None: ...
    def close(self) -> None: ...


class _FrameReader(asyncio.Protocol):
    def __init__(self, on_message: Callable[[dict], None], on_close: Callable[[], None]):
        self._on_message = on_message
        self._on_close = on_close
        self._buf = bytearray()
        self._scan = 0

    def data_received(self, data: bytes) -> None:
        self._buf += data
        while True:
            end = self._buf.find(b"\0", self._scan)
            if end < 0:
                self._scan = len(self._buf)
                return
            frame = bytes(self._buf[:end])
            del self._buf[:end + 1]
            self._scan = 0
            try:
                message = json.loads(frame)
            except ValueError:
                log.warning("dropping malformed CDP frame")
                continue
            self._on_message(message)

    def connection_lost(self, exc: Exception | None) -> None:
        self._on_close()


class _WriteProtocol(asyncio.Protocol):
    def __init__(self, on_close: Callable[[], None]):
        self._on_close = on_close

    def connection_lost(self, exc: Exception | None) -> None:
        self._on_close()


class PipeTransport:
    """NUL-delimited JSON over a pair of pipe fds owned by this process."""

    def __init__(self, read_fd: int, write_fd: int):
        self._read_fd = read_fd
        self._write_fd = write_fd
        self._reader: asyncio.ReadTransport | None = None
        self._writer: asyncio.WriteTransport | None = None

    async def start(self, on_message: Callable[[dict], None], on_close: Callable[[], None]) -> None:
        loop = asyncio.get_running_loop()
        self._reader, _ = await loop.connect_read_pipe(
            lambda: _FrameReader(on_message, on_close), os.fdopen(self._read_fd, "rb", buffering=0))
        self._writer, _ = await loop.connect_write_pipe(
            lambda: _WriteProtocol(on_close), os.fdopen(self._write_fd, "wb", buffering=0))

    def send(self, message: dict) -> None:
        if self._writer is None or self._writer.is_closing():
            raise CDPError("Browser connection is closed")
        self._writer.write(json.dumps(message, separators=(",", ":")).encode() + b"\0")

    def close(self) -> None:
        for transport in (self._writer, self._reader):
            if transport is not None and not transport.is_closing():
                transport.close()


class CDPPipe:
    def __init__(self, transport: Transport):
        self._transport = transport
        self._next_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._handlers: dict[str, list[Callable[[dict, str | None], Any]]] = {}
        self._waiters: list[tuple[str, str | None, Callable[[dict], bool] | None, asyncio.Future]] = []
        self._tasks: set[asyncio.Task] = set()
        self._close_callbacks: list[Callable[[], None]] = []
        self.closed = False

    async def start(self) -> "CDPPipe":
        await self._transport.start(self._dispatch, self._closed)
        return self

    async def send(self, method: str, params: dict | None = None, session_id: str | None = None,
                   timeout: float = 30.0) -> dict:
        if self.closed:
            raise CDPError("Browser connection is closed")
        self._next_id += 1
        message: dict[str, Any] = {"id": self._next_id, "method": method, "params": params or {}}
        if session_id:
            message["sessionId"] = session_id
        future = asyncio.get_running_loop().create_future()
        self._pending[self._next_id] = future
        try:
            self._transport.send(message)
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            raise CDPError(f"{method} timed out after {timeout:g}s") from None
        finally:
            self._pending.pop(message["id"], None)

    def on(self, method: str, callback: Callable[[dict, str | None], Any]) -> Callable[[], None]:
        """Call callback(params, session_id) for every event; coroutines run as tasks."""
        self._handlers.setdefault(method, []).append(callback)

        def unsubscribe() -> None:
            handlers = self._handlers.get(method, [])
            if callback in handlers:
                handlers.remove(callback)
        return unsubscribe

    def on_close(self, callback: Callable[[], None]) -> None:
        self._close_callbacks.append(callback)

    def wait_for_event(self, method: str, predicate: Callable[[dict], bool] | None = None,
                       timeout: float = 30.0, session_id: str | None = None) -> asyncio.Future:
        """Future for the next matching event's params; resolves to None on timeout or close.

        It registers immediately, so create it before sending the command that
        triggers the event.
        """
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        if self.closed:
            future.set_result(None)
            return future
        self._waiters.append((method, session_id, predicate, future))
        timer = loop.call_later(timeout, lambda: future.done() or future.set_result(None))
        future.add_done_callback(lambda _: timer.cancel())
        return future

    def close(self) -> None:
        try:
            self._transport.close()
        finally:
            self._closed()

    # ---- internals -------------------------------------------------------

    def _dispatch(self, message: dict) -> None:
        if "id" in message:
            future = self._pending.get(message["id"])
            if future is None or future.done():
                return
            if "error" in message:
                error = message["error"] or {}
                future.set_exception(CDPError(error.get("message", "CDP error"), error.get("code")))
            else:
                future.set_result(message.get("result") or {})
            return
        method = message.get("method", "")
        params = message.get("params") or {}
        session_id = message.get("sessionId")
        for callback in list(self._handlers.get(method, ())):
            try:
                result = callback(params, session_id)
                if asyncio.iscoroutine(result):
                    task = asyncio.ensure_future(result)
                    self._tasks.add(task)
                    task.add_done_callback(self._task_done)
            except Exception:
                log.exception("CDP event handler for %s failed", method)
        if self._waiters:
            remaining = []
            for waiter in self._waiters:
                name, sid, predicate, future = waiter
                if future.done():
                    continue
                if name == method and (sid is None or sid == session_id):
                    try:
                        matched = predicate is None or predicate(params)
                    except Exception:
                        matched = False
                    if matched:
                        future.set_result(params)
                        continue
                remaining.append(waiter)
            self._waiters = remaining

    def _task_done(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            log.error("CDP event task failed", exc_info=task.exception())

    def _closed(self) -> None:
        if self.closed:
            return
        self.closed = True
        for future in self._pending.values():
            if not future.done():
                future.set_exception(CDPError("Browser connection closed"))
        for *_, future in self._waiters:
            if not future.done():
                future.set_result(None)
        self._waiters = []
        for callback in self._close_callbacks:
            try:
                callback()
            except Exception:
                log.exception("CDP close callback failed")
