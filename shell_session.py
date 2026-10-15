"""Persistent shells for the sandbox VM.

Each session has one bash that keeps its working directory, variables, functions and installed tools between
commands, as a terminal does. A command runs in it with stdin closed; the shell then prints a marker with the
exit status and its directory, which ends the reply. A command that outlives its timeout is stopped (its
processes, not the shell). Work meant to keep running (a server, a long build) runs as a job: detached from the
shell, started in its directory with its variables, its output in a log that can be read while it runs.

Standard library only: copied into the sandbox image beside server.py and tested on the host.
"""

from __future__ import annotations

import asyncio
import os
import pathlib
import secrets
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass

DEFAULT_TIMEOUT = 120
MAX_TIMEOUT = 3600
HEAD = 64 * 1024            # bytes kept from the start of a long output
TAIL = 448 * 1024           # and from its end, where errors and results usually are
QUIET = {"TERM": "dumb", "PAGER": "cat", "GIT_PAGER": "cat", "DEBIAN_FRONTEND": "noninteractive",
         "PYTHONUNBUFFERED": "1", "PIP_PROGRESS_BAR": "off", "NO_COLOR": "1"}


@dataclass
class Result:
    output: str
    exit_code: int | None
    cwd: str
    timed_out: bool = False
    omitted: int = 0            # bytes left out of the middle of the output
    restarted: bool = False     # the shell ended (exit, a crash) and the next command gets a new one


def _descendants(pid: int) -> list[int]:
    """Processes started under pid, deepest first."""
    try:
        listing = subprocess.run(["ps", "-A", "-o", "pid=,ppid="], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    children: dict[int, list[int]] = {}
    for line in listing.split("\n"):
        parts = line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            children.setdefault(int(parts[1]), []).append(int(parts[0]))
    found: list[int] = []
    stack = list(children.get(pid, []))
    while stack:
        child = stack.pop()
        found.append(child)
        stack.extend(children.get(child, []))
    return found[::-1]


def _stop(pids: list[int], grace: float = 2.0) -> None:
    for sig in (signal.SIGTERM, signal.SIGKILL):
        alive = []
        for pid in pids:
            try:
                os.kill(pid, sig)
                alive.append(pid)
            except (ProcessLookupError, PermissionError):
                pass
        if not alive or sig == signal.SIGKILL:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline and any(_alive(pid) for pid in alive):
            time.sleep(0.05)
        pids = [pid for pid in alive if _alive(pid)]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:                                        # a finished child not yet reaped is not running
        done, _ = os.waitpid(pid, os.WNOHANG)
        return done == 0
    except ChildProcessError:
        return True


class Shell:
    def __init__(self, cwd: str, env: dict[str, str] | None = None):
        self.cwd = cwd
        self.env = {**os.environ, **QUIET, **(env or {})}
        self.marker = f"__coderunner_done_{secrets.token_hex(8)}__"
        self.proc: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def _start(self) -> None:
        pathlib.Path(self.cwd).mkdir(parents=True, exist_ok=True)
        self.proc = await asyncio.create_subprocess_exec(
            "bash", "--noprofile", "--norc", stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT, cwd=self.cwd, env=self.env, start_new_session=True)

    async def run(self, command: str, timeout: float = DEFAULT_TIMEOUT) -> Result:
        timeout = max(1.0, min(float(timeout or DEFAULT_TIMEOUT), MAX_TIMEOUT))
        async with self.lock:
            if not self.running:
                await self._start()
            proc = self.proc
            assert proc is not None and proc.stdin is not None and proc.stdout is not None
            script = (f"{{\n{command}\n}} < /dev/null\n"
                      f"printf '\\n{self.marker} %s %s\\n' \"$?\" \"$PWD\"\n")
            proc.stdin.write(script.encode())
            await proc.stdin.drain()
            data, omitted, timed_out = bytearray(), 0, False
            deadline = time.monotonic() + timeout
            marker = self.marker.encode()
            while True:
                left = deadline - time.monotonic()
                if left <= 0 and not timed_out:
                    timed_out = True
                    # Stop the command, keep the shell: it then prints the marker as usual.
                    await asyncio.to_thread(_stop, _descendants(proc.pid))
                    deadline = time.monotonic() + 5
                    continue
                if left <= 0:
                    proc.kill()                 # the command could not be stopped: start over next time
                    await proc.wait()
                    break
                try:
                    chunk = await asyncio.wait_for(proc.stdout.read(65536), left)
                except asyncio.TimeoutError:
                    continue
                if not chunk:                   # the shell ended (exit in the command)
                    await proc.wait()
                    break
                data += chunk
                if len(data) > HEAD + TAIL + 65536:
                    cut = len(data) - HEAD - TAIL
                    omitted += cut
                    del data[HEAD:HEAD + cut]
                end = data.rfind(marker, max(0, len(data) - len(chunk) - len(marker) - 4096))
                if end >= 0 and data.find(b"\n", end) >= 0:
                    tail = bytes(data[end:data.find(b"\n", end)]).decode(errors="replace").split(" ", 2)
                    output = bytes(data[:end]).decode(errors="replace")
                    output = output[:-1] if output.endswith("\n") else output     # the marker's own line break
                    output = output[:-1] if output.endswith("\n") else output     # the last line's, as $(...) does
                    code = int(tail[1]) if len(tail) > 1 and tail[1].lstrip("-").isdigit() else None
                    self.cwd = tail[2] if len(tail) > 2 else self.cwd
                    return Result(output, code, self.cwd, timed_out, omitted)
            return Result(bytes(data).decode(errors="replace"), proc.returncode, self.cwd, timed_out, omitted,
                          restarted=True)

    async def close(self) -> None:
        if self.running and self.proc is not None:
            await asyncio.to_thread(_stop, [*_descendants(self.proc.pid), self.proc.pid])
            await self.proc.wait()


class Shells:
    """The shells of all sessions, and the jobs started from them."""

    def __init__(self, home: str | None = None, jobs: str | None = None):
        self.home = home or os.environ.get("CODERUNNER_SHELL_HOME") or os.path.expanduser("~")
        self.jobs = pathlib.Path(jobs or os.environ.get("CODERUNNER_JOBS", "/tmp/coderunner-jobs"))
        self.shells: dict[str, Shell] = {}

    def shell(self, session: str) -> Shell:
        if session not in self.shells:
            self.shells[session] = Shell(self.home)
        return self.shells[session]

    async def run(self, session: str, command: str, timeout: float = DEFAULT_TIMEOUT) -> Result:
        return await self.shell(session).run(command, timeout)

    async def reset(self, session: str) -> bool:
        shell = self.shells.pop(session, None)
        if shell is not None:
            await shell.close()
        return shell is not None

    async def start_job(self, session: str, command: str) -> dict:
        """Run command detached from the shell, in its directory and with its variables."""
        job = secrets.token_hex(4)
        folder = self.jobs / job
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "command").write_text(command)
        q = shlex.quote
        inner = f"echo $$ > {q(str(folder / 'pid'))}; {{\n{command}\n}}; echo $? > {q(str(folder / 'exit'))}"
        result = await self.run(session, f"( bash -c {q(inner)} > {q(str(folder / 'log'))} 2>&1 < /dev/null & )",
                                timeout=15)
        for _ in range(50):                     # the pid file says it started
            if (folder / "pid").exists():
                break
            await asyncio.sleep(0.02)
        return {**self.job(job), "cwd": result.cwd}

    def job(self, job: str, tail: int = 20000) -> dict:
        folder = self.jobs / job
        if not job.isalnum() or not folder.is_dir():
            raise KeyError(job)
        exit_file, pid_file, log = folder / "exit", folder / "pid", folder / "log"
        exit_code = int(exit_file.read_text().strip() or "0") if exit_file.exists() else None
        pid = int(pid_file.read_text().strip()) if pid_file.exists() else None
        running = exit_code is None and pid is not None and _alive(pid)
        size = log.stat().st_size if log.exists() else 0
        output = ""
        if size:
            with open(log, "rb") as handle:
                handle.seek(max(0, size - tail))
                output = handle.read().decode(errors="replace")
        return {"job": job, "command": (folder / "command").read_text(), "running": running,
                "exit_code": exit_code, "output": output, "log": str(log), "log_bytes": size}

    def stop_job(self, job: str) -> dict:
        found = self.job(job)
        pid_file = self.jobs / job / "pid"
        if found["running"] and pid_file.exists():
            pid = int(pid_file.read_text().strip())
            _stop([*_descendants(pid), pid])
        return self.job(job)

    def list_jobs(self) -> list[dict]:
        if not self.jobs.is_dir():
            return []
        return [{k: v for k, v in self.job(p.name, tail=0).items() if k != "output"}
                for p in sorted(self.jobs.iterdir(), key=lambda p: p.stat().st_mtime) if p.is_dir()]
