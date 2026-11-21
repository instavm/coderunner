"""The broker's launcher for browser.mode "computer": Google Chrome on the sandbox VM's own desktop.

Chrome runs inside the VM under its desktop user, controlled over --remote-debugging-pipe; the sandbox server
relays that pipe on the websocket /v1/browser/cdp behind the sandbox credential, so the browser opens no DevTools
port on any network. The owner watches the same screen (VNC) and can take over at any time. Logins live in the
VM's Chrome profile, never in the owner's everyday browser.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from typing import Callable

from .cdp import CDPError, CDPPipe

log = logging.getLogger(__name__)

NOT_READY = ("The sandbox has no desktop yet. Run `coderunner browser computer`, or set browser.mode back to "
             "dedicated or chrome.")


class WebSocketTransport:
    """CDP messages as websocket text frames (one JSON object each)."""

    def __init__(self, socket):
        self.socket = socket
        self.reader: asyncio.Task | None = None

    async def start(self, on_message: Callable[[dict], None], on_close: Callable[[], None]) -> None:
        async def read() -> None:
            try:
                async for frame in self.socket:
                    try:
                        on_message(json.loads(frame))
                    except ValueError:
                        log.warning("dropping malformed CDP frame from the sandbox")
            except Exception as exc:          # closed by the sandbox, or the VM went away
                log.info("sandbox browser connection ended: %s", exc)
            finally:
                on_close()
        self.reader = asyncio.create_task(read())

    def send(self, message: dict) -> None:
        if self.reader is None or self.reader.done():
            raise CDPError("Browser connection is closed")
        task = asyncio.ensure_future(self.socket.send(json.dumps(message, separators=(",", ":"))))
        task.add_done_callback(lambda t: t.cancelled() or t.exception())

    def close(self) -> None:
        with contextlib.suppress(Exception):
            asyncio.ensure_future(self.socket.close())
        if self.reader is not None:
            self.reader.cancel()


class VMProcess:
    """Stands in for the browser process: the broker checks poll() and pid, and waits on stop."""

    pid = 0

    def __init__(self, transport: WebSocketTransport):
        self.transport = transport

    def poll(self) -> int | None:
        return 0 if self.transport.reader is None or self.transport.reader.done() else None

    def wait(self, timeout: float | None = None) -> int:
        return 0


class VMLauncher:
    def __init__(self, sandbox_url: str, token: str):
        self.url = sandbox_url.rstrip("/").replace("http://", "ws://", 1).replace("https://", "wss://", 1) \
            + "/v1/browser/cdp"
        self.token = token

    def find_browser(self, config) -> str:
        return "sandbox-chrome"

    async def launch(self, executable: str, profile_dir, headless: bool = False) -> tuple[VMProcess, CDPPipe]:
        from websockets.asyncio.client import connect
        try:
            socket = await connect(self.url, additional_headers={"Authorization": f"Bearer {self.token}"},
                                   max_size=None, open_timeout=20,
                                   # a private link to the VM: a busy moment must not cut it
                                   ping_interval=None,
                                   proxy=None)
        except Exception as exc:
            raise CDPError(f"could not reach the sandbox's browser ({exc}). {NOT_READY}") from None
        transport = WebSocketTransport(socket)
        cdp = await CDPPipe(transport).start()
        try:
            await cdp.send("Browser.getVersion", timeout=60)       # the first start brings up the desktop too
        except CDPError as exc:
            cdp.close()
            reason = getattr(socket, "close_reason", None) or exc
            raise CDPError(f"the sandbox's browser did not answer: {reason}") from None
        return VMProcess(transport), cdp

    def terminate(self, process: VMProcess) -> None:
        """Nothing to stop: Chrome stays on the VM's screen with the owner's tabs, and the broker has already
        closed the connection (this runs in a worker thread, away from the event loop)."""
