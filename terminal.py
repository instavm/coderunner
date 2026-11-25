"""The owner's terminal on the sandbox computer: one bash on a real PTY, as the desktop user in /workspace.

One session lives in the server across reconnects: the shell keeps running when the app's tab closes, and the
last SCROLLBACK bytes of what it printed are replayed to a connection that attaches, so the owner comes back to
where they were. The assistant's own commands (run_shell, run_python) run in their own processes, never in this
shell; they are echoed into the same stream as they run, dim and marked as the assistant's, so the owner sees what
it does and a reconnect replays that too.

Websocket protocol (/v1/terminal): binary frames are bytes, both ways (keys in, output out). A text frame from the
client is a JSON control message: {"type": "resize", "cols": 120, "rows": 32}.

Standard library only (and Starlette for the route): copied into the sandbox image beside server.py and tested on
the host, where it runs a shell as the current user.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import json
import math
import os
import pty
import pwd
import struct
import subprocess
import termios
import time
from typing import Callable

SCROLLBACK = 200_000            # bytes of output kept for a connection that attaches
TYPING_S = 15                   # input this recent means the owner is using the terminal
QUEUE_LIMIT = 4000              # chunks waiting for a slow connection before it is dropped (it reconnects)
USER = os.environ.get("CODERUNNER_DESKTOP_USER", "box")
CWD = os.environ.get("CODERUNNER_TERMINAL_CWD", "/workspace")
DIM, RESET = b"\x1b[2m", b"\x1b[0m"
MARK = b"assistant \xe2\x80\xba "      # "assistant › ", before each of the assistant's commands


def _crlf(data: bytes) -> bytes:
    """Output from a process without a terminal has bare line feeds; the terminal needs carriage returns too."""
    return data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")


def first_line(text: str, longest: int = 160) -> str:
    lines = [line for line in str(text).splitlines() if line.strip()]
    line = "".join(ch for ch in lines[0] if ch >= " " and ch != "\x7f").strip() if lines else ""   # no escapes
    more = len(lines) > 1 or len(line) > longest
    return (line[:longest] + (" …" if more else "")).strip()


def shell_label(command: str) -> str:
    return first_line(command)


def python_label(code: str) -> str:
    return ("python  " + first_line(code)).strip()


def _controlling_tty() -> None:                   # in the child, after setsid: the PTY becomes its terminal
    with contextlib.suppress(OSError):
        fcntl.ioctl(0, termios.TIOCSCTTY, 0)


class Echo:
    """One of the assistant's commands as it shows in the terminal: its line, its output, and a non-zero exit."""

    def __init__(self, terminal: "Terminal", label: str):
        self.terminal = terminal
        self.ended = False
        terminal._line_start()
        terminal._feed(DIM + MARK + label.encode(errors="replace").replace(b"\x1b", b"") + RESET + b"\r\n")

    def output(self, data: bytes | str) -> None:
        if self.ended or not data:
            return
        if isinstance(data, str):
            data = data.encode(errors="replace")
        self.terminal._feed(_crlf(data))

    def end(self, exit_code: int | None = 0, output: bytes | str | None = None, note: str = "") -> None:
        if self.ended:
            return
        if output:
            self.output(output)
        self.ended = True
        self.terminal._line_start()
        words = note or (f"exit {exit_code}" if exit_code not in (0, None) else "")
        if words:
            self.terminal._feed(DIM + words.encode() + RESET + b"\r\n")


class Terminal:
    def __init__(self, argv: list[str] | None = None, cwd: str | None = None, user: str | None = None,
                 scrollback: int = SCROLLBACK, clock: Callable[[], float] = time.monotonic):
        self.argv = argv or ["/bin/bash", "-l"]
        self.cwd = cwd or CWD
        self.user = user or USER
        self.scrollback = scrollback
        self.clock = clock
        self.buffer = bytearray()
        self.listeners: set[asyncio.Queue] = set()
        self.cols, self.rows = 100, 30
        self.last_input = -math.inf
        self.fd: int | None = None
        self.proc: subprocess.Popen | None = None
        self._pending = bytearray()       # keys not yet taken by the PTY (a large paste)

    # ---- the shell ----------------------------------------------------------

    def running(self) -> bool:
        return self.fd is not None and self.proc is not None and self.proc.poll() is None

    def _account(self) -> pwd.struct_passwd | None:
        """The desktop user when the server can become them (it runs as root in the VM); otherwise None, and the
        shell runs as the server's own user (tests on the host)."""
        if os.geteuid() != 0:
            return None
        try:
            return pwd.getpwnam(self.user)
        except KeyError:
            return None

    def start(self) -> None:
        if self.running():
            return
        self._close()
        master, slave = pty.openpty()
        try:
            self._set_size(slave)
            account = self._account()
            env = {"TERM": "xterm-256color", "COLORTERM": "truecolor", "LANG": os.environ.get("LANG", "C.UTF-8")}
            extra: dict = {}
            if account is not None:
                os.fchown(slave, account.pw_uid, -1)
                env.update({"HOME": account.pw_dir, "USER": account.pw_name, "LOGNAME": account.pw_name,
                            "SHELL": "/bin/bash", "PATH": "/usr/local/bin:/usr/bin:/bin"})
                extra = {"user": account.pw_uid, "group": account.pw_gid, "extra_groups": []}
                home = account.pw_dir
            else:
                env = {**os.environ, **env}
                home = os.path.expanduser("~")
            cwd = self.cwd if os.path.isdir(self.cwd) else home
            self.proc = subprocess.Popen(self.argv, stdin=slave, stdout=slave, stderr=slave, cwd=cwd, env=env,
                                         start_new_session=True, preexec_fn=_controlling_tty, close_fds=True,
                                         **extra)
        except Exception:
            os.close(master)
            raise
        finally:
            os.close(slave)
        os.set_blocking(master, False)
        self.fd = master
        asyncio.get_running_loop().add_reader(master, self._readable)

    def _readable(self) -> None:
        try:
            data = os.read(self.fd, 65536)
        except BlockingIOError:
            return
        except OSError:                   # EIO: the shell and everything on the PTY are gone
            data = b""
        if data:
            self._feed(data)
            return
        self._close()
        self._line_start()
        self._feed(DIM + b"[shell ended]" + RESET + b"\r\n")

    def _close(self) -> None:
        if self.fd is not None:
            loop = asyncio.get_running_loop()
            loop.remove_reader(self.fd)
            loop.remove_writer(self.fd)
            with contextlib.suppress(OSError):
                os.close(self.fd)
            self.fd = None
        self._pending.clear()
        if self.proc is not None and self.proc.poll() is None:
            with contextlib.suppress(OSError):
                os.killpg(self.proc.pid, 9)
        if self.proc is not None:
            with contextlib.suppress(Exception):
                self.proc.wait(timeout=1)

    async def stop(self) -> None:
        self._close()
        self.proc = None

    # ---- the owner's side -----------------------------------------------------

    def write(self, data: bytes) -> None:
        """Keys from the owner. A shell that ended is started again by the next key."""
        self.last_input = self.clock()
        if not self.running():
            self.start()
            return
        self._pending += data
        self._flush()

    def _flush(self) -> None:
        if self.fd is None:
            return
        while self._pending:
            try:
                sent = os.write(self.fd, self._pending)
            except BlockingIOError:
                asyncio.get_running_loop().add_writer(self.fd, self._flush)
                return
            except OSError:
                self._pending.clear()
                return
            del self._pending[:sent]
        asyncio.get_running_loop().remove_writer(self.fd)

    def resize(self, cols: int, rows: int) -> None:
        self.cols, self.rows = max(2, min(int(cols), 1000)), max(2, min(int(rows), 500))
        if self.fd is not None:
            with contextlib.suppress(OSError):
                self._set_size(self.fd)

    def _set_size(self, fd: int) -> None:
        fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", self.rows, self.cols, 0, 0))

    def typing(self, window: float = TYPING_S) -> bool:
        return self.clock() - self.last_input < window

    # ---- the stream -------------------------------------------------------------

    def attach(self) -> asyncio.Queue:
        """A connection's queue: what is kept so far first, then everything as it comes."""
        queue: asyncio.Queue = asyncio.Queue()
        if self.buffer:
            queue.put_nowait(bytes(self.buffer))
        self.listeners.add(queue)
        return queue

    def detach(self, queue: asyncio.Queue) -> None:
        self.listeners.discard(queue)

    def _feed(self, data: bytes) -> None:
        self.buffer += data
        if len(self.buffer) > self.scrollback:
            cut = len(self.buffer) - self.scrollback
            line = self.buffer.find(b"\n", cut, cut + 4096)      # start the kept part on a line of its own
            del self.buffer[:line + 1 if line >= 0 else cut]
        for queue in list(self.listeners):
            if queue.qsize() > QUEUE_LIMIT:
                self.listeners.discard(queue)
                queue.put_nowait(None)        # too far behind: that connection closes and attaches again
            else:
                queue.put_nowait(data)

    def _line_start(self) -> None:
        if self.buffer and not self.buffer.endswith(b"\n"):
            self._feed(b"\r\n")

    def assistant(self, label: str) -> Echo:
        return Echo(self, label)


TERMINAL = Terminal()


async def terminal_socket(websocket) -> None:
    """The owner's terminal: attaches to the one session (starting its shell when needed) and relays it."""
    await websocket.accept()
    try:
        TERMINAL.start()
    except Exception as exc:
        await websocket.close(code=1011, reason=str(exc)[:120])
        return
    queue = TERMINAL.attach()

    async def pump() -> None:
        while True:
            data = await queue.get()
            if data is None:
                with contextlib.suppress(Exception):
                    await websocket.close(code=1013, reason="behind")
                return
            await websocket.send_bytes(data)

    sender = asyncio.create_task(pump())
    try:
        while True:
            receive = asyncio.ensure_future(websocket.receive())
            done, _ = await asyncio.wait({receive, sender}, return_when=asyncio.FIRST_COMPLETED)
            if sender in done:
                receive.cancel()
                return
            message = receive.result()
            if message["type"] == "websocket.disconnect":
                return
            if message.get("bytes") is not None:
                TERMINAL.write(message["bytes"])
            elif message.get("text") is not None:
                try:
                    control = json.loads(message["text"])
                except ValueError:
                    continue
                if isinstance(control, dict) and control.get("type") == "resize":
                    with contextlib.suppress(TypeError, ValueError):
                        TERMINAL.resize(control.get("cols"), control.get("rows"))
    except (RuntimeError, ConnectionError):
        pass
    finally:
        sender.cancel()
        TERMINAL.detach(queue)
