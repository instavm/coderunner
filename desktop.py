"""The sandbox's own computer: a Linux desktop (Xvnc + Xfce) the owner can watch and take over, with Google Chrome
on it for the assistant's browsing.

The screen is served over VNC (password from /v1/desktop) on the VM's private network; the daemon relays it to the
app. Chrome runs as the unprivileged desktop user with --remote-debugging-pipe, so it opens no DevTools port; the
pipe is relayed to the daemon's browser broker over the server's authenticated websocket /v1/browser/cdp, one
client at a time. Chrome outlives a client: the owner's tabs and logins stay on screen between tasks, and when a
client goes, the relay detaches its sessions and stops its request interception so nothing is left waiting.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import pathlib
import pwd
import secrets
import shutil
import subprocess
import sys

from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.websockets import WebSocket, WebSocketDisconnect

log = logging.getLogger("desktop")

USER = os.environ.get("CODERUNNER_DESKTOP_USER", "box")
DISPLAY_NUMBER = 1
VNC_PORT = 5900
SIZE = os.environ.get("CODERUNNER_DESKTOP_SIZE", "1440x900")
CHROME = "/usr/bin/google-chrome-stable"
PASSWORD_FILE = pathlib.Path("/app/desktop/vnc-password")       # root only; read back for /v1/desktop
TRAMPOLINE = ("import os,sys; os.dup2(int(sys.argv[1]),3); os.dup2(int(sys.argv[2]),4); "
              "os.execv(sys.argv[3], sys.argv[3:])")
INTERNAL_IDS = 1 << 40          # ids the relay uses for its own commands; their replies are not forwarded
# Chrome's disk cache, capped: uncapped it reached 1.7 GB of the VM's disk, which is a file on the owner's Mac.
DISK_CACHE_BYTES = 256 << 20


def _account() -> pwd.struct_passwd | None:
    try:
        return pwd.getpwnam(USER)
    except KeyError:
        return None


def available() -> bool:
    return bool(shutil.which("Xvnc")) and os.path.exists(CHROME) and _account() is not None


def _env(account: pwd.struct_passwd) -> dict:
    runtime = f"/tmp/runtime-{account.pw_name}"
    os.makedirs(runtime, mode=0o700, exist_ok=True)
    os.chown(runtime, account.pw_uid, account.pw_gid)
    raw_lang = (os.environ.get("LANG") or "").strip()
    lang = raw_lang if raw_lang and raw_lang.upper() not in ("C", "POSIX", "C.UTF-8") else "en_US.UTF-8"
    return {"HOME": account.pw_dir, "USER": account.pw_name, "LOGNAME": account.pw_name, "SHELL": "/bin/bash",
            "PATH": "/usr/local/bin:/usr/bin:/bin", "DISPLAY": f":{DISPLAY_NUMBER}", "XDG_RUNTIME_DIR": runtime,
            "LANG": lang}


def _as_user(argv: list[str], account: pwd.struct_passwd, **kwargs) -> subprocess.Popen:
    log_path = pathlib.Path(account.pw_dir) / ".cache" / f"{pathlib.Path(argv[0]).name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    os.chown(log_path.parent, account.pw_uid, account.pw_gid)
    out = open(log_path, "ab")
    try:
        return subprocess.Popen(argv, user=account.pw_uid, group=account.pw_gid, extra_groups=[],
                                env=_env(account), cwd=account.pw_dir, stdin=subprocess.DEVNULL, stdout=out,
                                stderr=subprocess.STDOUT, start_new_session=True, **kwargs)
    finally:
        out.close()


# The desktop's backdrop: a vertical gradient xfdesktop draws itself, no image file, from deep navy at the top to the
# app's blue at the bottom. desktop-setup.sh writes the same values.
BACKDROP_TOP = ("0.059", "0.090", "0.200", "1")
BACKDROP_BOTTOM = ("0.200", "0.380", "0.870", "1")
BACKDROP_MARK = ".coderunner-backdrop"           # beside the settings: applied once, then the owner's to change
STOCK_BACKDROP = "xfce-shapes"


def backdrop_xml() -> str:
    def rgba(name, values):
        items = "".join(f'\n            <value type="double" value="{v}"/>' for v in values)
        return f'          <property name="{name}" type="array">{items}\n          </property>\n'
    workspaces = "".join(
        f'        <property name="workspace{n}" type="empty">\n'
        '          <property name="image-style" type="int" value="0"/>\n'
        '          <property name="color-style" type="int" value="2"/>\n'
        f'{rgba("rgba1", BACKDROP_TOP)}{rgba("rgba2", BACKDROP_BOTTOM)}'
        '        </property>\n' for n in range(4))
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<channel name="xfce4-desktop" version="1.0">\n'
            '  <property name="backdrop" type="empty">\n    <property name="screen0" type="empty">\n'
            f'      <property name="monitorVNC-0" type="empty">\n{workspaces}      </property>\n'
            '    </property>\n  </property>\n</channel>\n')


def apply_backdrop(home: pathlib.Path, uid: int = -1, gid: int = -1) -> bool:
    """The gradient backdrop on a desktop that still has the stock one (no settings yet, or xfce-shapes), once:
    before the session starts, so xfdesktop reads it. An owner's own backdrop is never replaced."""
    folder = home / ".config" / "xfce4" / "xfconf" / "xfce-perchannel-xml"
    target, mark = folder / "xfce4-desktop.xml", home / ".config" / "xfce4" / BACKDROP_MARK
    if mark.exists():
        return False
    try:
        current = target.read_text(errors="replace") if target.exists() else ""
    except OSError:
        return False
    if current and STOCK_BACKDROP not in current:
        return False
    try:
        folder.mkdir(parents=True, exist_ok=True)
        target.write_text(backdrop_xml())
        mark.write_text("")
        for path in (target, mark):
            os.chown(path, uid, gid)
    except OSError:
        return False
    return True


def _alive(process: subprocess.Popen | None) -> bool:
    return process is not None and process.poll() is None


class Desktop:
    """Xvnc (the X server and the VNC server in one) and an Xfce session on display :1."""

    def __init__(self):
        self.server: subprocess.Popen | None = None
        self.session: subprocess.Popen | None = None
        self.lock = asyncio.Lock()

    def password(self) -> str:
        if not PASSWORD_FILE.exists():
            PASSWORD_FILE.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            # VNC passwords are 8 characters at most.
            PASSWORD_FILE.write_text(secrets.token_urlsafe(6)[:8])
            os.chmod(PASSWORD_FILE, 0o600)
        return PASSWORD_FILE.read_text().strip()

    def _write_vnc_password(self, account: pwd.struct_passwd) -> str:
        vnc_dir = pathlib.Path(account.pw_dir) / ".vnc"
        vnc_dir.mkdir(mode=0o700, exist_ok=True)
        os.chown(vnc_dir, account.pw_uid, account.pw_gid)
        target = vnc_dir / "passwd"
        hashed = subprocess.run(["vncpasswd", "-f"], input=self.password().encode(), capture_output=True,
                                check=True).stdout
        target.write_bytes(hashed)
        os.chmod(target, 0o600)
        os.chown(target, account.pw_uid, account.pw_gid)
        return str(target)

    def running(self) -> bool:
        return _alive(self.server) and _alive(self.session)

    async def ensure(self) -> None:
        async with self.lock:
            if self.running():
                return
            account = _account()
            if account is None or not available():
                raise RuntimeError("This sandbox has no desktop yet.")
            if not _alive(self.server):
                with contextlib.suppress(FileNotFoundError):       # a stale lock from a server that crashed
                    os.unlink(f"/tmp/.X{DISPLAY_NUMBER}-lock")
                with contextlib.suppress(FileNotFoundError):
                    os.unlink(f"/tmp/.X11-unix/X{DISPLAY_NUMBER}")
                os.makedirs("/tmp/.X11-unix", mode=0o1777, exist_ok=True)
                os.chmod("/tmp/.X11-unix", 0o1777)
                passwd = self._write_vnc_password(account)
                self.server = _as_user(["Xvnc", f":{DISPLAY_NUMBER}", "-geometry", SIZE, "-depth", "24",
                                        "-rfbport", str(VNC_PORT), "-rfbauth", passwd, "-SecurityTypes", "VncAuth",
                                        "-AlwaysShared", "-nolisten", "tcp", "-desktop", "coderunner"], account)
                for _ in range(100):
                    if os.path.exists(f"/tmp/.X11-unix/X{DISPLAY_NUMBER}"):
                        break
                    if self.server.poll() is not None:
                        raise RuntimeError("The desktop's display server exited at startup")
                    await asyncio.sleep(0.1)
            if not _alive(self.session):
                with contextlib.suppress(Exception):
                    apply_backdrop(pathlib.Path(account.pw_dir), account.pw_uid, account.pw_gid)
                self.session = _as_user(["dbus-launch", "--exit-with-session", "startxfce4"], account)
                await asyncio.sleep(1.5)        # the panel and window manager settle before the first window

    def info(self) -> dict:
        width, _, height = SIZE.partition("x")
        return {"available": available(), "running": self.running(), "port": VNC_PORT,
                "password": self.password() if available() else "", "width": int(width), "height": int(height),
                "chrome": CHROME_RELAY.running()}


class _PipeReader(asyncio.Protocol):
    def __init__(self, relay: "ChromeRelay"):
        self.relay = relay
        self.buffer = bytearray()

    def data_received(self, data: bytes) -> None:
        self.buffer += data
        while (end := self.buffer.find(b"\0")) >= 0:
            frame = bytes(self.buffer[:end])
            del self.buffer[:end + 1]
            self.relay.from_chrome(frame)

    def connection_lost(self, exc) -> None:
        self.relay.chrome_closed()


_CACHED_UA: str | None = None


def _desktop_user_agent() -> str:
    """Match the installed Chrome major version with a regular desktop macOS Chrome User-Agent."""
    global _CACHED_UA
    if _CACHED_UA is not None:
        return _CACHED_UA
    major = "136"
    try:
        if os.path.exists(CHROME):
            out = subprocess.check_output([CHROME, "--version"], text=True, timeout=5, stderr=subprocess.DEVNULL)
            for token in out.split():
                head = token.split(".", 1)[0]
                if head.isdigit() and int(head) >= 100:
                    major = head
                    break
    except Exception:
        pass
    _CACHED_UA = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        f"AppleWebKit/537.36 (KHTML, like Gecko) Chrome/{major}.0.0.0 Safari/537.36"
    )
    return _CACHED_UA


def chrome_argv(profile: pathlib.Path) -> list[str]:
    """Chrome as the assistant runs it on the desktop, driven over its DevTools pipe (fds 3 and 4)."""
    return [CHROME, f"--user-data-dir={profile}", "--remote-debugging-pipe", "--no-first-run",
            "--no-default-browser-check", "--hide-crash-restore-bubble", "--password-store=basic",
            "--disable-dev-shm-usage", "--start-maximized", f"--disk-cache-size={DISK_CACHE_BYTES}",
            "--lang=en-US", f"--user-agent={_desktop_user_agent()}",
            "--disable-infobars", "--disable-background-timer-throttling",
            "--disable-renderer-backgrounding", "--disable-backgrounding-occluded-windows",
            # DevTools control otherwise sets navigator.webdriver, which sites read as a bot.
            "--disable-blink-features=AutomationControlled", "about:blank"]


class ChromeRelay:
    def __init__(self, desktop: Desktop):
        self.desktop = desktop
        self.process: subprocess.Popen | None = None
        self.writer: asyncio.WriteTransport | None = None
        self.reader: asyncio.ReadTransport | None = None
        self.client: WebSocket | None = None
        self.outbox: asyncio.Queue | None = None
        self.pump: asyncio.Task | None = None
        self.sessions: set[str] = set()
        self.internal = INTERNAL_IDS
        self.lock = asyncio.Lock()
        self.closed = asyncio.Event()

    def running(self) -> bool:
        return _alive(self.process) and self.writer is not None and not self.writer.is_closing()

    async def ensure(self) -> None:
        async with self.lock:
            if self.running():
                return
            await self.desktop.ensure()
            account = _account()
            profile = pathlib.Path(account.pw_dir) / ".config" / "coderunner-chrome"
            profile.mkdir(parents=True, exist_ok=True)
            for part in (profile.parent, profile):
                os.chown(part, account.pw_uid, account.pw_gid)
            commands_r, commands_w = os.pipe()      # we write, Chrome reads on fd 3
            replies_r, replies_w = os.pipe()        # Chrome writes on fd 4, we read
            moved = []
            for fd in (commands_r, replies_w):      # the trampoline overwrites fds 3 and 4
                import fcntl
                new = fcntl.fcntl(fd, fcntl.F_DUPFD, 5)
                os.close(fd)
                moved.append(new)
            child_r, child_w = moved
            try:
                self.process = _as_user(
                    [sys.executable, "-I", "-c", TRAMPOLINE, str(child_r), str(child_w), *chrome_argv(profile)],
                    account, pass_fds=(child_r, child_w))
            finally:
                os.close(child_r)
                os.close(child_w)
            loop = asyncio.get_running_loop()
            self.closed = asyncio.Event()
            self.reader, _ = await loop.connect_read_pipe(lambda: _PipeReader(self),
                                                          os.fdopen(replies_r, "rb", buffering=0))
            self.writer, _ = await loop.connect_write_pipe(asyncio.Protocol, os.fdopen(commands_w, "wb", buffering=0))
            self.outbox = asyncio.Queue()
            self.pump = asyncio.create_task(self._pump())
            self.sessions.clear()

    # ---- Chrome -> client ----------------------------------------------------

    def from_chrome(self, frame: bytes) -> None:
        try:
            message = json.loads(frame)
        except ValueError:
            return
        if isinstance(message.get("id"), int) and message["id"] >= INTERNAL_IDS:
            return
        method = message.get("method")
        if method == "Target.attachedToTarget" and "sessionId" not in message:
            sid = (message.get("params") or {}).get("sessionId")
            if sid:
                self.sessions.add(sid)
        elif method == "Target.detachedFromTarget":
            self.sessions.discard((message.get("params") or {}).get("sessionId"))
        if self.client is not None and self.outbox is not None:
            self.outbox.put_nowait(frame.decode("utf-8", "replace"))

    async def _pump(self) -> None:
        while True:
            text = await self.outbox.get()
            client = self.client
            if client is None:
                continue
            with contextlib.suppress(Exception):
                await client.send_text(text)

    def chrome_closed(self) -> None:
        self.closed.set()
        if self.writer is not None and not self.writer.is_closing():
            self.writer.close()

    # ---- client -> Chrome ----------------------------------------------------

    def to_chrome(self, text: str) -> None:
        if not self.running():
            raise ConnectionError("Chrome is not running")
        self.writer.write(text.encode() + b"\0")

    def _command(self, method: str, params: dict | None = None) -> None:
        self.internal += 1
        with contextlib.suppress(ConnectionError):
            self.to_chrome(json.dumps({"id": self.internal, "method": method, "params": params or {}}))

    def release(self) -> None:
        """What a departed client set up must not outlive it: a paused request would wait for ever."""
        self._command("Fetch.disable")
        for sid in list(self.sessions):
            self._command("Target.detachFromTarget", {"sessionId": sid})
        self.sessions.clear()


DESKTOP = Desktop()
CHROME_RELAY = ChromeRelay(DESKTOP)


async def api_desktop(request: Request):
    if request.method == "POST":
        if not available():
            return JSONResponse({"error": "This sandbox has no desktop yet."},
                                status_code=409)
        try:
            await DESKTOP.ensure()
            if (await request.json() if await request.body() else {}).get("chrome"):
                await CHROME_RELAY.ensure()
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            return JSONResponse({"error": str(exc)}, status_code=500)
    return JSONResponse(DESKTOP.info())


async def cdp_socket(websocket: WebSocket):
    await websocket.accept()
    try:
        await CHROME_RELAY.ensure()
    except Exception as exc:
        await websocket.close(code=1011, reason=str(exc)[:120])
        return
    previous, CHROME_RELAY.client = CHROME_RELAY.client, websocket
    if previous is not None:                 # the newest client wins, e.g. after the daemon restarted
        with contextlib.suppress(Exception):
            await previous.close(code=1012, reason="replaced")
        CHROME_RELAY.release()
    closed = asyncio.create_task(CHROME_RELAY.closed.wait())
    try:
        while True:
            receive = asyncio.create_task(websocket.receive_text())
            done, _ = await asyncio.wait({receive, closed}, return_when=asyncio.FIRST_COMPLETED)
            if closed in done:
                receive.cancel()
                with contextlib.suppress(Exception):
                    await websocket.close(code=1011, reason="Chrome closed")
                return
            CHROME_RELAY.to_chrome(receive.result())
    except (WebSocketDisconnect, ConnectionError, RuntimeError):
        pass
    finally:
        closed.cancel()
        if CHROME_RELAY.client is websocket:
            CHROME_RELAY.client = None
            if CHROME_RELAY.running():
                CHROME_RELAY.release()
