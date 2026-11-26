"""The owner's everyday browser, through the CodeRunner extension (browser.mode = "chrome").

Instead of starting a browser with a profile of its own, the broker drives the tabs of a
"CodeRunner" tab group in the owner's Chrome (or Brave, Edge, Chromium), where the owner's sign-ins
are. The extension (extension/background.js) attaches chrome.debugger to those tabs only and
answers the protocol's browser-level Target methods with the group's tabs alone. Its native
messaging host (chrome_host.py) relays messages to the daemon over a unix socket only the owner's
account can open, after presenting a token. The broker code is the same as for the dedicated
profile: policy, step judge, document-request guard, dialogs, password and card fields, hand-off.

What differs from the dedicated profile: the owner's cookies and sign-ins are available to the
assistant's tabs; site permission prompts (camera, location...) are shown to the owner instead of
being denied outright; downloads are refused where the browser supports it; and a popup's very
first request can leave before the guard is attached to it (later requests and redirects are
checked).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import pathlib
import shutil
import stat
import struct
import sys
import tempfile
import time
from typing import Any, Callable

from ..core import paths
from .cdp import CDPError, CDPPipe

HOST_NAME = "com.coderunner.browser"
EXTENSION_ID = "glpfnkaafpjpikcbbgopomgpbekfamak"     # fixed by the "key" in extension/manifest.json
ORIGIN = f"chrome-extension://{EXTENSION_ID}/"
EXTENSION_SOURCE = pathlib.Path(__file__).with_name("extension")
HOST_SOURCE = pathlib.Path(__file__).with_name("chrome_host.py")
CONNECT_WAIT = 15.0          # seconds to wait for the extension when the assistant needs the browser
HELLO_WAIT = 5.0
FRAME = struct.Struct("!I")
MAX_FRAME = 256 * 1024 * 1024
NOT_CONNECTED = ("the CodeRunner extension is not connected. Run `coderunner browser connect`, load the "
                 "extension in Chrome (chrome://extensions, Developer mode, Load unpacked) and keep Chrome open")

# User-level native messaging host folders, by browser (the folder exists once the browser has run).
BROWSER_DIRS = {
    "darwin": {
        "chrome": "~/Library/Application Support/Google/Chrome",
        "chrome-beta": "~/Library/Application Support/Google/Chrome Beta",
        "chrome-canary": "~/Library/Application Support/Google/Chrome Canary",
        "chromium": "~/Library/Application Support/Chromium",
        "brave": "~/Library/Application Support/BraveSoftware/Brave-Browser",
        "edge": "~/Library/Application Support/Microsoft Edge",
    },
    "linux": {
        "chrome": "~/.config/google-chrome",
        "chrome-beta": "~/.config/google-chrome-beta",
        "chromium": "~/.config/chromium",
        "brave": "~/.config/BraveSoftware/Brave-Browser",
        "edge": "~/.config/microsoft-edge",
    },
}


# ---- files ------------------------------------------------------------------------

def chrome_dir() -> pathlib.Path:
    return paths.assistant_dir() / "chrome"


def token_path() -> pathlib.Path:
    return chrome_dir() / "token"


def token() -> str:
    return paths._read_or_create_secret(token_path())


def socket_path() -> pathlib.Path:
    """The daemon's socket for the host. Socket paths are limited to about 100 bytes, so a long
    CODERUNNER_HOME falls back to the per-user temporary folder."""
    preferred = paths.assistant_dir() / "run" / "chrome.sock"
    if len(os.fsencode(str(preferred))) < 100:
        return preferred
    digest = hashlib.sha256(os.fsencode(str(paths.home()))).hexdigest()[:12]
    base = None
    if sys.platform == "darwin":
        with contextlib.suppress(ValueError, OSError):
            base = os.confstr("CS_DARWIN_USER_TEMP_DIR")      # per user, whatever TMPDIR says
    base = base or os.environ.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    return pathlib.Path(base) / f"coderunner-{digest}.sock"


def browser_roots(platform: str = sys.platform) -> dict[str, pathlib.Path]:
    key = "darwin" if platform == "darwin" else "linux"
    return {name: pathlib.Path(os.path.expanduser(path)) for name, path in BROWSER_DIRS[key].items()}


def host_manifest(wrapper: pathlib.Path) -> dict:
    return {"name": HOST_NAME, "description": "CodeRunner assistant", "path": str(wrapper),
            "type": "stdio", "allowed_origins": [ORIGIN]}


def install(browsers: list[str] | None = None, roots: dict[str, pathlib.Path] | None = None,
            python: str = sys.executable) -> dict:
    """Copy the extension and host into the assistant folder and register the host with each
    installed browser (or the named ones). Returns what was written."""
    roots = roots if roots is not None else browser_roots()
    base = chrome_dir()
    base.mkdir(parents=True, exist_ok=True)
    os.chmod(base, 0o700)
    extension = base / "extension"
    if extension.exists():
        shutil.rmtree(extension)
    shutil.copytree(EXTENSION_SOURCE, extension)
    host = base / "host.py"
    shutil.copyfile(HOST_SOURCE, host)
    token()
    wrapper = base / "coderunner-chrome-host"
    args = [python, "-I", str(host), "--socket", str(socket_path()), "--token-file", str(token_path()),
            "--extension", EXTENSION_ID]
    wrapper.write_text("#!/bin/sh\nexec " + " ".join(_quote(a) for a in args) + ' "$@"\n')
    os.chmod(wrapper, stat.S_IRWXU)
    chosen = [name for name in roots if (browsers is None and roots[name].is_dir())
              or (browsers is not None and name in browsers)]
    unknown = sorted(set(browsers or []) - set(roots))
    if unknown:
        raise ValueError(f"unknown browser {', '.join(unknown)}; choose from {', '.join(roots)}")
    manifests = []
    for name in chosen:
        folder = roots[name] / "NativeMessagingHosts"
        folder.mkdir(parents=True, exist_ok=True)
        target = folder / f"{HOST_NAME}.json"
        target.write_text(json.dumps(host_manifest(wrapper), indent=2) + "\n")
        manifests.append({"browser": name, "path": str(target)})
    return {"extension_dir": str(extension), "extension_id": EXTENSION_ID, "host": str(wrapper),
            "manifests": manifests}


def uninstall(roots: dict[str, pathlib.Path] | None = None) -> list[str]:
    removed = []
    for root in (roots if roots is not None else browser_roots()).values():
        target = root / "NativeMessagingHosts" / f"{HOST_NAME}.json"
        if target.exists():
            target.unlink()
            removed.append(str(target))
    return removed


def current_version() -> str:
    """The version of the extension this CodeRunner carries."""
    return json.loads((EXTENSION_SOURCE / "manifest.json").read_text())["version"]


def refresh() -> bool:
    """Bring an installed copy of the extension and host up to this version of CodeRunner, so a fix reaches
    the browser without installing again; Chrome reads an unpacked extension's files when its worker next
    starts. Nothing happens unless the extension was installed. Returns whether files were replaced."""
    base = chrome_dir()
    extension, host = base / "extension", base / "host.py"
    if not extension.is_dir():
        return False
    source = {p.relative_to(EXTENSION_SOURCE): p.read_bytes() for p in EXTENSION_SOURCE.rglob("*") if p.is_file()}
    copy = {p.relative_to(extension): p.read_bytes() for p in extension.rglob("*") if p.is_file()}
    stale = source != copy or not host.exists() or host.read_bytes() != HOST_SOURCE.read_bytes()
    if stale:
        shutil.rmtree(extension)
        shutil.copytree(EXTENSION_SOURCE, extension)
        shutil.copyfile(HOST_SOURCE, host)
    return stale


def installed(roots: dict[str, pathlib.Path] | None = None) -> dict[str, bool]:
    return {name: (root / "NativeMessagingHosts" / f"{HOST_NAME}.json").exists()
            for name, root in (roots if roots is not None else browser_roots()).items() if root.is_dir()}


def _quote(text: str) -> str:
    return "'" + text.replace("'", "'\\''") + "'"


# ---- the daemon's end of the connection ---------------------------------------------

async def read_frame(reader: asyncio.StreamReader) -> Any:
    header = await reader.readexactly(FRAME.size)
    size = FRAME.unpack(header)[0]
    if size > MAX_FRAME:
        raise ValueError("frame too large")
    return json.loads(await reader.readexactly(size))


def encode(message: dict) -> bytes:
    data = json.dumps(message, separators=(",", ":")).encode()
    return FRAME.pack(len(data)) + data


class ExtensionConnection:
    """One connected native host, i.e. one browser with the extension."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter, pid: int | None):
        self.reader, self.writer, self.pid = reader, writer, pid
        self.since = time.time()
        self.closed = False
        self.on_message: Callable[[dict], None] | None = None
        self.on_close: Callable[[], None] | None = None
        self.version: str | None = None    # the extension's manifest version, once it has said (1.0.2 and later)
        self.on_version: Callable[[str], None] | None = None
        # Messages of the connection itself, beside the protocol (the app's browser answers a cookie import).
        self.on_typed: Callable[[dict], None] | None = None

    def send(self, message: dict) -> None:
        if self.closed:
            raise CDPError("The browser extension disconnected")
        self.writer.write(encode(message))

    async def pump(self) -> None:
        try:
            while True:
                message = await read_frame(self.reader)
                if isinstance(message, dict) and message.get("type") == "extension":
                    self.version = str(message.get("version") or "")
                    if self.on_version is not None:
                        self.on_version(self.version)
                elif isinstance(message, dict) and message.get("type"):
                    if self.on_typed is not None:
                        self.on_typed(message)
                elif isinstance(message, dict) and self.on_message is not None:
                    self.on_message(message)
        except (asyncio.IncompleteReadError, ConnectionError, ValueError):
            pass
        finally:
            self.close()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        with contextlib.suppress(Exception):
            self.writer.close()
        if self.on_close is not None:
            self.on_close()


class ChromeBridge:
    """The unix socket native hosts connect to. One browser at a time: the first one connected
    keeps the assistant until it disconnects. (app.AppBridge is the same socket for the app's own
    browser, with its own origin, token and file.)"""

    ORIGIN = ORIGIN             # what the hello must say it comes from
    NAME = "extension"          # in the audit trail: browser.extension_connected, ...

    def __init__(self, path: pathlib.Path | None = None, secret: str | None = None, audit=None):
        self.path = path or socket_path()
        self._secret = secret
        self.audit = audit
        self.connection: ExtensionConnection | None = None
        self._server: asyncio.AbstractServer | None = None
        self._made: tuple[int, int] | None = None     # the socket file this bridge created (device, inode)
        self._noted: set[str] = set()
        self._changed = asyncio.Event()

    @property
    def secret(self) -> str:
        if self._secret is None:
            self._secret = token()
        return self._secret

    async def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(FileNotFoundError):
            if stat.S_ISSOCK(os.lstat(self.path).st_mode) and os.lstat(self.path).st_uid == os.getuid():
                self.path.unlink()          # left over from a previous run
        self._server = await asyncio.start_unix_server(self._accept, path=str(self.path))
        os.chmod(self.path, 0o600)
        made = os.lstat(self.path)
        self._made = (made.st_dev, made.st_ino)

    async def stop(self) -> None:
        if self.connection is not None:
            self.connection.close()
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
        with contextlib.suppress(FileNotFoundError):
            now = os.lstat(self.path)
            # Only the socket this daemon made: a daemon that took a few seconds to stop removed the socket of
            # the one started after it, and the extension could never connect again.
            if (now.st_dev, now.st_ino) == self._made:
                self.path.unlink()

    def connected(self) -> bool:
        return self.connection is not None and not self.connection.closed

    def status(self) -> dict:
        if not self.connected():
            return {"connected": False}
        version = self.connection.version
        found = {"connected": True, "since": round(self.connection.since), "host_pid": self.connection.pid,
                 "version": version}
        if version != current_version():       # None: from before 1.0.2, which did not say
            found["outdated"] = "reload the CodeRunner extension in chrome://extensions (the circular arrow)"
        return found

    async def wait(self, timeout: float) -> ExtensionConnection | None:
        deadline = time.monotonic() + timeout
        while True:
            self._changed.clear()
            if self.connected():
                return self.connection
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._changed.wait(), remaining)

    async def _accept(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            hello = await asyncio.wait_for(read_frame(reader), HELLO_WAIT)
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, ConnectionError, ValueError):
            writer.close()
            return
        if not (isinstance(hello, dict) and hello.get("type") == "hello" and hello.get("origin") == self.ORIGIN
                and hmac.compare_digest(str(hello.get("token", "")).encode(), self.secret.encode())):
            self._audit(f"browser.{self.NAME}_refused", reason="bad hello")
            writer.close()
            return
        if self.connected():
            writer.write(encode({"type": "busy"}))
            writer.close()
            return
        pid = hello.get("pid") if isinstance(hello.get("pid"), int) else None
        writer.write(encode({"type": "welcome"}))
        connection = self.connection = ExtensionConnection(reader, writer, pid)
        connection.on_version = lambda version: self._check_version(connection, version)
        self._changed.set()
        self._audit(f"browser.{self.NAME}_connected", host_pid=pid)
        await connection.pump()
        self._changed.set()
        self._audit(f"browser.{self.NAME}_disconnected", host_pid=pid)

    def _check_version(self, connection: ExtensionConnection, version: str) -> None:
        """An extension older than this CodeRunner is noted, once per version: the daemon brings its files up to
        date, but Chrome keeps running the worker script it has (also after a restart) until the owner reloads
        the extension. Asking it to reload itself was tried: an extension loaded from the command line went
        away instead of coming back."""
        current = current_version()
        if version != current and version not in self._noted:
            self._noted.add(version)
            self._audit("browser.extension_outdated", had=version, now=current)

    def _audit(self, event: str, **fields: Any) -> None:
        if self.audit is not None:
            self.audit.write(event, **fields)


class ExtensionTransport:
    """CDPPipe transport over an extension connection. Closing it lets go of the connection, which
    stays up for the next time the assistant needs the browser."""

    def __init__(self, connection: ExtensionConnection):
        self.connection = connection

    async def start(self, on_message: Callable[[dict], None], on_close: Callable[[], None]) -> None:
        self.connection.on_message, self.connection.on_close = on_message, on_close
        if self.connection.closed:
            on_close()

    def send(self, message: dict) -> None:
        self.connection.send(message)

    def close(self) -> None:
        if self.connection.on_message is not None:
            self.connection.on_message = None
            self.connection.on_close = None


class ExtensionProcess:
    """Stands in for the browser process the dedicated mode starts: alive while the extension is."""

    def __init__(self, connection: ExtensionConnection):
        self.connection = connection
        self.pid = connection.pid
        self.returncode: int | None = None

    def poll(self) -> int | None:
        if self.returncode is None and self.connection.closed:
            self.returncode = 0
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


class ExtensionLauncher:
    """The broker's launcher for browser.mode "chrome": waits for the extension instead of starting
    a browser, and never stops the owner's browser."""

    def __init__(self, bridge: ChromeBridge, wait: float | None = None):
        self.bridge = bridge
        self.timeout = CONNECT_WAIT if wait is None else wait

    def find_browser(self, config) -> str:
        return "chrome-extension"

    async def launch(self, executable: str, profile_dir, headless: bool = False) -> tuple[ExtensionProcess, CDPPipe]:
        connection = await self.bridge.wait(self.timeout)
        if connection is None:
            raise CDPError(NOT_CONNECTED)
        cdp = await CDPPipe(ExtensionTransport(connection)).start()
        try:
            await cdp.send("Browser.getVersion", timeout=20)
        except CDPError:
            cdp.close()
            raise
        return ExtensionProcess(connection), cdp

    def terminate(self, process: ExtensionProcess) -> None:
        process.wait()
