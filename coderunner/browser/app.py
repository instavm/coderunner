"""The browser inside the CodeRunner app, on the owner's Mac (the computer panel's Browser tab).

Some sites turn the sandbox's Chrome away with a bot wall (walls.py): the VM honestly looks like ARM Linux with
software WebGL. The app's own browser is a real Mac with a real GPU, and is shown as such: nothing in it is
disguised (no other user agent, platform, WebGL strings or canvas noise). With the owner's OK a blocked site
continues there (switch.py).

How it is driven: the app's main process (app/browser.js) keeps the tabs, each an Electron WebContentsView on its
own persistent session (persist:coderunner-browser, apart from the app's page and from the owner's Chrome), and
relays the DevTools protocol through each tab's webContents.debugger. It speaks to the daemon exactly as the
Chrome extension's native host does (extension.py): length-prefixed JSON frames over a unix socket in the owner's
assistant folder (0600), opened with a hello that carries a token from a file only the owner can read. The app
answers the browser-level Target methods with its own tabs alone, as the extension does for its tab group, so
the broker runs unchanged on top: policy, the step judge, the document-request guard, dialogs, password and card
fields, hand-off. The model never gets the protocol.

When the app is not open, a site the owner let through continues in a Chrome window of the assistant's own
profile on the Mac (launcher.py), which is just as honest a Mac browser; with no Chrome either, the owner clears
the check in the sandbox (hand-off).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import os
import pathlib
import sys
import tempfile
from typing import Any

from ..core import paths
from . import launcher as host_launcher
from .cdp import CDPError
from .extension import ChromeBridge, ExtensionLauncher, ExtensionProcess

ORIGIN = "coderunner-app"
EXECUTABLE = "coderunner-app"       # what find_browser names the app's browser by
APP_WAIT = 3.0                      # seconds a step waits for the app to connect before Chrome is used instead
COOKIE_WAIT = 30.0
NOT_OPEN = "the CodeRunner app is not open, so its browser is not there"


def app_dir() -> pathlib.Path:
    folder = paths.assistant_dir() / "app-browser"
    folder.mkdir(parents=True, exist_ok=True)
    os.chmod(folder, 0o700)
    return folder


def token() -> str:
    return paths._read_or_create_secret(app_dir() / "token")


def socket_path() -> pathlib.Path:
    """Socket paths are limited to about 100 bytes, so a long CODERUNNER_HOME falls back to the per-user temporary
    folder. The app reads where it is from app-browser/socket, so it never has to work this out itself."""
    preferred = paths.assistant_dir() / "run" / "app-browser.sock"
    if len(os.fsencode(str(preferred))) < 100:
        return preferred
    digest = hashlib.sha256(os.fsencode(str(paths.home()))).hexdigest()[:12]
    base = None
    if sys.platform == "darwin":
        with contextlib.suppress(ValueError, OSError):
            base = os.confstr("CS_DARWIN_USER_TEMP_DIR")
    base = base or os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return pathlib.Path(base) / f"coderunner-app-{digest}.sock"


def used() -> bool:
    """Whether the Mac browser has been used before: the first time, the owner is offered their sign-ins."""
    return (app_dir() / "used").exists()


def mark_used() -> None:
    with contextlib.suppress(OSError):
        (app_dir() / "used").touch()


class AppBridge(ChromeBridge):
    """The socket the app connects to. One app at a time, as one extension at a time."""

    ORIGIN = ORIGIN
    NAME = "app_browser"

    def __init__(self, path: pathlib.Path | None = None, secret: str | None = None, audit=None):
        super().__init__(path or socket_path(), secret, audit)
        self._replies: dict[int, asyncio.Future] = {}
        self._numbers = itertools.count(1)

    @property
    def secret(self) -> str:
        if self._secret is None:
            self._secret = token()
        return self._secret

    async def start(self) -> None:
        self.secret                     # the token exists before the app looks for it
        await super().start()
        pointer = app_dir() / "socket"
        fd = os.open(pointer, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(str(self.path) + "\n")

    def status(self) -> dict:
        if not self.connected():
            return {"connected": False}
        return {"connected": True, "since": round(self.connection.since)}

    def _check_version(self, connection, version: str) -> None:
        """The app comes with this CodeRunner: there is no older copy to point out."""

    async def ask(self, message: dict, timeout: float = COOKIE_WAIT) -> dict:
        """A message to the app beside the protocol, and its answer (the cookie import)."""
        if not self.connected():
            raise CDPError(NOT_OPEN)
        connection = self.connection
        number = next(self._numbers)
        future = asyncio.get_running_loop().create_future()
        self._replies[number] = future
        connection.on_typed = self._answered
        try:
            connection.send({**message, "ref": number})
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            raise CDPError("the app did not answer") from None
        finally:
            self._replies.pop(number, None)

    def _answered(self, message: dict) -> None:
        future = self._replies.get(message.get("ref")) if isinstance(message.get("ref"), int) else None
        if future is not None and not future.done():
            future.set_result(message)

    async def set_cookies(self, cookies: list[dict]) -> dict:
        """Hand imported cookies to the app's browser session. Returns how many it took ({"set", "failed"})."""
        answer = await self.ask({"type": "cookies", "cookies": cookies})
        return {"set": int(answer.get("set") or 0), "failed": int(answer.get("failed") or 0)}


class AppLauncher(ExtensionLauncher):
    """The broker's launcher for the app's browser: waits briefly for the app instead of starting a browser, and
    never closes it (Browser.close lets go of the tabs)."""

    def __init__(self, bridge: AppBridge, wait: float | None = None):
        super().__init__(bridge, APP_WAIT if wait is None else wait)

    def find_browser(self, config) -> str:
        return EXECUTABLE

    async def launch(self, executable: str, profile_dir, headless: bool = False) -> tuple[ExtensionProcess, Any]:
        if await self.bridge.wait(self.timeout) is None:
            raise CDPError(NOT_OPEN)
        return await super().launch(executable, profile_dir, headless)


class MacLauncher:
    """The Mac browser for a blocked site: the app's browser when the app is open, else a Chrome window of the
    assistant's own profile (launcher.py). A Chrome window was preferred to giving up to a hand-off: the task
    goes on, on as honest a Mac browser, and the owner already said yes to their Mac."""

    def __init__(self, bridge: AppBridge | None, fallback=host_launcher):
        self.app = AppLauncher(bridge) if bridge is not None else None
        self.fallback = fallback
        self._used: Any = None

    def app_open(self) -> bool:
        return self.app is not None and self.app.bridge.connected()

    def available(self, config) -> bool:
        return self.app_open() or bool(self.fallback.find_browser(config))

    def find_browser(self, config) -> str | None:
        return EXECUTABLE if self.app_open() else self.fallback.find_browser(config)

    async def launch(self, executable: str, profile_dir, headless: bool = False):
        if executable == EXECUTABLE and self.app is not None:
            self._used = self.app
            return await self.app.launch(executable, profile_dir, headless)
        self._used = self.fallback
        return await self.fallback.launch(executable, profile_dir, False)    # a window the owner can see

    def terminate(self, process) -> None:
        (self._used or self.fallback).terminate(process)
